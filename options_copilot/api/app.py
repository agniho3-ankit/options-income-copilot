"""FastAPI app factory for the local review cockpit.

Every route reads through an existing function (review.queue, review.brief_signals,
guardrails.risk_score, observability.metrics, db.spans, db.guardrail_events, db.tot_nodes,
journal.metrics, the stored evals). The API reshapes values for JSON; it never computes credit,
max loss, PoP, breakevens, risk or scores itself.

Two databases stay separate: `main` (data/copilot.db, the owner's real runs and briefs) and
`eval` (data/eval_snapshots/eval.db, sandboxed eval and ablation runs). Read routes take
`?db=main|eval`; decisions only ever touch `main`.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..config import Settings
from ..db import Database
from ..mask import Masker
from ..playbook import Playbook
from .security import (CSRF_HEADER, DEV_PORT, LOCAL_HOST, LocalOnlyMiddleware, allowed_origins, new_token,
                       scrub)

DbName = Query("main", pattern="^(main|eval)$")


@dataclass
class AppState:
    settings: Settings
    pb: Playbook
    main_db: Database
    eval_db: Database | None
    masker: Masker
    outputs_dir: Path
    dist_dir: Path | None
    port: int
    csrf_token: str = field(default_factory=new_token)
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def secrets(self) -> tuple[str, ...]:
        return (self.settings.sec_user_agent, self.settings.anthropic_api_key)

    @contextmanager
    def db(self, name: str = "main") -> Iterator[Database]:
        """One connection per database, used by one request at a time."""
        if name == "eval" and self.eval_db is None:
            raise HTTPException(404, "no eval database yet (run `eval` to create one)")
        with self.lock:
            yield self.main_db if name == "main" else self.eval_db


def create_app(settings: Settings, pb: Playbook, *, main_db: Database, eval_db: Database | None = None,
               outputs_dir: Path | None = None, dist_dir: Path | None = None, port: int = 8765) -> FastAPI:
    st = AppState(settings=settings, pb=pb, main_db=main_db, eval_db=eval_db,
                  masker=Masker(settings.synthetic_portfolio_value),
                  outputs_dir=outputs_dir or settings.repo_root / "demo_outputs", dist_dir=dist_dir, port=port)
    # no /docs, /redoc or /openapi.json: the Swagger page pulls assets from a CDN
    app = FastAPI(title="Options Income Copilot (local)", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ui = st
    app.add_middleware(CORSMiddleware, allow_origins=[f"http://{LOCAL_HOST}:{DEV_PORT}"],
                       allow_methods=["GET", "POST"], allow_headers=[CSRF_HEADER, "Content-Type"])
    app.add_middleware(LocalOnlyMiddleware, token=st.csrf_token, origins=allowed_origins(port))

    def out(obj: Any) -> Any:
        return scrub(obj, st.secrets)

    def masked(obj: Any) -> Any:
        """Trace content is stored masked; mask again on the way out (idempotent) and scrub secrets."""
        return out(st.masker.value(obj))

    # --- session and static reference data ---------------------------------------------------

    @app.get("/api/session")
    def session() -> dict:
        return {"csrf_token": st.csrf_token, "paper_only": True,
                "databases": {"main": True, "eval": st.eval_db is not None},
                "playbook": {"version": st.pb.version, "sha256": st.pb.sha256}}

    @app.get("/api/guardrails")
    def guardrails() -> dict:
        from ..guardrails import REGISTRY, SIGNALS, STAGES
        from ..review import REASONS
        return {"stages": list(STAGES), "signals": list(SIGNALS), "signal_text": REASONS,
                "reject_codes": list(st.pb["guardrails.reject_codes"]),
                "registry": [{"name": g.name, "stage": g.stage, "kind": g.kind, "what": g.what} for g in REGISTRY]}

    @app.get("/api/playbook")
    def playbook() -> dict:
        return {"version": st.pb.version, "sha256": st.pb.sha256, "source": st.pb.source,
                "editable_by": "owner only, in the file", "tree": _thaw(st.pb.tree)}

    # --- the review queue and briefs -----------------------------------------------------------

    @app.get("/api/queue")
    def queue() -> dict:
        from ..calculators import portfolio_heat_pct
        from ..review import queue as review_queue
        from ..broker import open_positions
        with st.db() as db:
            q = review_queue(db, st.pb)
            losses = [p.entry_max_loss * p.contracts for p in open_positions(db)]
        heat = portfolio_heat_pct(losses, st.settings.synthetic_portfolio_value)
        trades = [{**i, "top": [{"signal": s, "text": t} for s, t in i["top"]]} for i in q["trades"]]
        return out({"trades": trades, "info": q["info"],
                    "heat": {"pct": heat, "cap_pct": st.pb["entry.portfolio_heat_pct.block"],
                             "warn_pct": st.pb["entry.portfolio_heat_pct.warn"]}})

    @app.get("/api/briefs/{brief_id}")
    def brief(brief_id: str) -> dict:
        from ..guardrails import risk_score
        from ..review import brief_signals, explain
        with st.db() as db:
            b = db.brief(brief_id)
            if b is None:
                raise HTTPException(404, f"no brief {brief_id}")
            sig = brief_signals(db, b)
            events = db.guardrail_events(b["run_id"]) if b.get("run_id") else []
            row = next((r for r in db.briefs() if r["brief_id"] == brief_id), {})
        score, tier, parts = risk_score(sig, st.pb["guardrails.review"])
        payload = b.pop("payload")
        return out({"brief": {**b, "decision_code": row.get("decision_code"), "deferred_at": row.get("deferred_at")},
                    "payload": payload,
                    "review": {"score": score, "tier": tier, "signals": sig,
                               "top": [{"signal": s, "points": p, "text": explain(s, sig[s])} for s, p in parts[:3]]},
                    "guardrail_events": masked(events),
                    "reject_codes": list(st.pb["guardrails.reject_codes"])})

    # --- runs, spans, guardrail events, the ToT tree ------------------------------------------

    @app.get("/api/runs")
    def runs(db: str = DbName, limit: int = Query(30, ge=1, le=500)) -> dict:
        with st.db(db) as d:
            rows = [dict(r) for r in d.conn.execute(
                "SELECT run_id, kind, ticker, started_at, finished_at, outcome FROM runs "
                "ORDER BY started_at DESC LIMIT ?", (limit,))]
        return out({"db": db, "runs": rows})

    @app.get("/api/runs/{run_id}")
    def run(run_id: str, db: str = DbName) -> dict:
        with st.db(db) as d:
            r = d.run(run_id)
            if r is None:
                raise HTTPException(404, f"no run {run_id}")
            spans = [_decode(s, ("input_summary", "output_summary", "citations")) for s in d.spans(run_id)]
            events = [_decode(e, ("data",)) for e in d.guardrail_events(run_id)]
        return masked({"db": db, "run": r, "spans": spans, "guardrail_events": events})

    @app.get("/api/runs/{run_id}/tot")
    def tot(run_id: str, db: str = DbName) -> dict:
        with st.db(db) as d:
            if d.run(run_id) is None:
                raise HTTPException(404, f"no run {run_id}")
            nodes = d.tot_nodes(run_id)
        return masked({"db": db, "run_id": run_id, "nodes": nodes})

    # --- journal, metrics, evals ---------------------------------------------------------------

    @app.get("/api/journal")
    def journal() -> dict:
        from ..journal import metrics as journal_metrics
        with st.db() as db:
            rows = db.journal()
        months = sorted({r["date_opened"][:7] for r in rows})
        return out({"rows": rows, "all": journal_metrics(rows),
                    "months": [journal_metrics(rows, m) for m in months]})

    @app.get("/api/metrics")
    def metrics(db: str = DbName, since: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}")) -> dict:
        from ..observability import metrics as obs_metrics
        with st.db(db) as d:
            m = obs_metrics(d, st.pb, since)
        return out({"db": db, **m})

    @app.get("/api/evals")
    def evals() -> dict:
        with st.db() as db:
            rows = [{**dict(r), "summary": json.loads(r["summary"])} for r in db.conn.execute(
                "SELECT eval_id, created_at, mode, summary FROM evals ORDER BY created_at DESC, rowid DESC")]
        saved = []
        if st.outputs_dir.is_dir():
            for p in sorted(st.outputs_dir.glob("eval_*.json")):
                try:
                    saved.append({"name": p.stem, "result": json.loads(p.read_text())})
                except (OSError, json.JSONDecodeError):
                    saved.append({"name": p.stem, "error": "could not read this file"})
        return out({"stored": rows, "saved": saved})

    # --- the built front end (single-page app) ---------------------------------------------------

    if dist_dir is not None and (dist_dir / "index.html").is_file():
        root = dist_dir.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "no such API route")
            f = (root / path).resolve()
            if path and f.is_file() and root in f.parents:
                return FileResponse(f)
            return FileResponse(root / "index.html")

    return app


def _thaw(node: Any) -> Any:
    if hasattr(node, "items"):
        return {k: _thaw(v) for k, v in node.items()}
    if isinstance(node, tuple):
        return [_thaw(v) for v in node]
    return node


def _decode(row: dict, keys: tuple[str, ...]) -> dict:
    out = dict(row)
    for k in keys:
        v = out.get(k)
        if isinstance(v, str) and v[:1] in "[{":
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return out
