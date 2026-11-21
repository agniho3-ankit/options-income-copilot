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
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..db import Database
from ..mask import Masker
from ..playbook import Playbook
from .security import (CSRF_HEADER, DEV_PORT, LOCAL_HOST, LocalOnlyMiddleware, allowed_origins, new_token,
                       scrub)

DbName = Query("main", pattern="^(main|eval)$")
MARKET_TTL_SECONDS = 300
TELEMETRY_OFF = {"tracing": False, "metrics": False, "logs": False, "operation_spans": False,
                 "auto_configure": False}


class MarketCache:
    """Live market data with a short in-memory cache. If a fetch fails, the last good result
    is returned marked stale; if there is none, the caller gets the error."""

    def __init__(self, market_factory: Callable[[], Any], ttl: float = MARKET_TTL_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self._factory, self._market, self.ttl, self._clock = market_factory, None, ttl, clock
        self._rows: dict[str, tuple[float, Any, Any]] = {}
        self._lock = threading.Lock()

    def get(self, ticker: str, today: date) -> tuple[Any, Any, dict]:
        """(history, profile, freshness) for one ticker."""
        key = ticker.upper()
        with self._lock:
            hit = self._rows.get(key)
            now = self._clock()
            if hit and now - hit[0] < self.ttl:
                return hit[1], hit[2], {"stale": False, "age_s": round(now - hit[0])}
            try:
                if self._market is None:
                    self._market = self._factory()
                hist, prof = self._market.history(key, "1y"), self._market.profile(key, today)
            except Exception as exc:  # noqa: BLE001 - any data-source failure falls back to the cache
                if hit:
                    return hit[1], hit[2], {"stale": True, "age_s": round(now - hit[0]),
                                            "error": f"live fetch failed ({type(exc).__name__})"}
                raise HTTPException(503, f"market data unavailable for {key} ({type(exc).__name__})") from exc
            self._rows[key] = (now, hist, prof)
            return hist, prof, {"stale": False, "age_s": 0}


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
    market: MarketCache | None = None
    today: Callable[[], date] = date.today
    llm_factory: Callable[[], Any] | None = None     # local LLM for reject lessons (no cloud fallback)
    store_factory: Callable[[], Any] | None = None   # Chroma lesson store
    llm_lock: threading.Lock = field(default_factory=threading.Lock)   # one LLM job at a time
    _store: Any = None
    jobs: Any = None                                 # api.jobs.JobManager
    db_factory: Callable[[], Any] | None = None      # a separate connection for slow reads
    sources_factory: Callable[[date], Any] | None = None   # live market sources for Positions
    positions_cache: dict = field(default_factory=dict)    # position_id -> (time, row)
    poll_s: float = 0.5

    def store(self) -> Any:
        if self._store is None and self.store_factory is not None:
            self._store = self.store_factory()
        return self._store
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
               outputs_dir: Path | None = None, dist_dir: Path | None = None, port: int = 8765,
               market_factory: Callable[[], Any] | None = None, today: Callable[[], date] = date.today,
               llm_factory: Callable[[], Any] | None = None, store_factory: Callable[[], Any] | None = None,
               job_runner: Callable[..., dict] | None = None, db_factory: Callable[[], Any] | None = None,
               poll_s: float = 0.5, sources_factory: Callable[[date], Any] | None = None) -> FastAPI:
    if market_factory is None:
        def market_factory():
            from ..market import Market
            return Market()
    st = AppState(settings=settings, pb=pb, main_db=main_db, eval_db=eval_db,
                  masker=Masker(settings.synthetic_portfolio_value),
                  outputs_dir=outputs_dir or settings.repo_root / "demo_outputs", dist_dir=dist_dir, port=port,
                  market=MarketCache(market_factory), today=today,
                  llm_factory=llm_factory or (lambda: _local_llm(settings)),
                  store_factory=store_factory or (lambda: _chroma_store(settings)), poll_s=poll_s)
    from .jobs import JobManager, live_runner
    st.db_factory = db_factory or (lambda: Database(settings.db_path))
    st.sources_factory = sources_factory or (lambda day: _market_sources(pb, day))
    st.jobs = JobManager(st.llm_lock, st.db_factory, job_runner or live_runner(settings, pb))
    # no /docs, /redoc or /openapi.json: the Swagger page pulls assets from a CDN
    # FastAPI's built-in OpenTelemetry instrumentation: every part off, and no exporter is ever added
    app = FastAPI(title="Options Income Copilot (local)", docs_url=None, redoc_url=None, openapi_url=None,
                  telemetry=TELEMETRY_OFF)
    app.state.ui = st
    app.add_middleware(CORSMiddleware, allow_origins=[f"http://{LOCAL_HOST}:{DEV_PORT}"],
                       allow_methods=["GET", "POST"], allow_headers=[CSRF_HEADER, "Content-Type"])
    app.add_middleware(LocalOnlyMiddleware, token=st.csrf_token, origins=allowed_origins(port))

    def out(obj: Any) -> Any:
        return scrub(obj, st.secrets)

    def masked(obj: Any) -> Any:
        """Trace content is stored masked; mask it again on the way out and scrub secrets.

        Summaries and data blobs go through the full mask (account values become % of portfolio);
        every other string gets the free-text mask. Identifiers (`id`, `*_id`) and the stored ToT
        trade numbers pass through, so links and charts keep working."""
        return out(_remask(obj, st.masker))

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
        keys = []
        for path, label, unit in KEY_THRESHOLDS:
            try:
                keys.append({"path": path, "label": label, "unit": unit, "value": _thaw(st.pb[path])})
            except KeyError:
                continue
        return {"version": st.pb.version, "sha256": st.pb.sha256, "source": st.pb.source,
                "editable_by": "owner only, in the file", "key_thresholds": keys, "tree": _thaw(st.pb.tree)}

    @app.get("/api/lab")
    def lab() -> dict:
        """Saved evaluation evidence (lab.build): nothing is run from here."""
        from ..lab import build
        with st.db() as db:
            return out(build(db, st.outputs_dir))

    # --- the review queue and briefs -----------------------------------------------------------

    @app.get("/api/queue")
    def queue() -> dict:
        from ..calculators import portfolio_heat_pct
        from ..review import queue as review_queue
        from ..broker import open_positions
        with st.db() as db:
            q = review_queue(db, st.pb)
            losses = [p.entry_max_loss * p.contracts for p in open_positions(db)]
            summaries = {i["brief_id"]: trade_summary(i["kind"], db.brief(i["brief_id"])["payload"])
                         for i in q["trades"]}
        heat = portfolio_heat_pct(losses, st.settings.synthetic_portfolio_value)
        trades = [{**i, "top": [{"signal": s, "text": t} for s, t in i["top"]], "trade": summaries[i["brief_id"]]}
                  for i in q["trades"]]
        return out({"trades": trades, "info": q["info"],
                    "heat": {"pct": heat, "cap_pct": st.pb["entry.portfolio_heat_pct.block"],
                             "warn_pct": st.pb["entry.portfolio_heat_pct.warn"]}})

    @app.get("/api/briefs/{brief_id}")
    def brief(brief_id: str) -> dict:
        from ..guardrails import pipeline_summary, risk_score
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
        alts = sorted((a for a in payload.get("alternatives") or [] if a.get("score") is not None),
                      key=lambda a: -a["score"])
        return out({"brief": {**b, "decision_code": row.get("decision_code"), "deferred_at": row.get("deferred_at")},
                    "payload": payload,
                    "review": {"score": score, "tier": tier, "signals": sig,
                               "top": [{"signal": s, "points": p, "text": explain(s, sig[s])} for s, p in parts[:3]]},
                    "guardrail_events": masked(events),
                    "guardrail_strip": masked(pipeline_summary(events)),
                    "alternatives": alts[:6],
                    "reject_codes": list(st.pb["guardrails.reject_codes"])})

    @app.get("/api/briefs/{brief_id}/evidence")
    def evidence(brief_id: str) -> dict:
        """Citations (rebuilt from chunk metadata) and the masked excerpts the researcher graded.
        Excerpts are untrusted filing text: the UI shows them as quoted plain text only."""
        from ..window import excerpts_from_trace
        with st.db() as db:
            b = db.brief(brief_id)
            if b is None:
                raise HTTPException(404, f"no brief {brief_id}")
            excerpts = excerpts_from_trace(db.trace(b["run_id"])) if b.get("run_id") else []
        p = b["payload"]
        return masked({"brief_id": brief_id, "landmine_verdict": p.get("landmine_verdict"),
                       "citations": p.get("citations") or [], "excerpts": excerpts,
                       "note": None if excerpts else "no graded excerpts are stored for this run"})

    @app.get("/api/briefs/{brief_id}/window")
    def window(brief_id: str) -> dict:
        """Trade-window chart data, computed in Python from live prices (cached briefly)."""
        from ..window import WindowError, build_window, excerpts_from_trace
        with st.db() as db:
            b = db.brief(brief_id)
            if b is None:
                raise HTTPException(404, f"no brief {brief_id}")
            excerpts = excerpts_from_trace(db.trace(b["run_id"])) if b.get("run_id") else []
            today = st.today()
            ivs = [v for d, v in db.iv_history(b["ticker"]) if d <= today.isoformat()]
        hist, prof, fresh = st.market.get(b["ticker"], today)
        try:
            w = build_window(ticker=b["ticker"], kind=b["kind"], payload=b["payload"], history=hist, profile=prof,
                             today=today, atm_iv=ivs[-1] if ivs else None, excerpts=excerpts)
        except WindowError as exc:
            raise HTTPException(422, str(exc)) from exc
        return out({**w, "freshness": fresh})

    # --- decisions: approve, reject, defer (api/decisions.py; the only state-changing routes) ---
    from .decisions import register as register_decisions
    register_decisions(app, st, out)

    # --- background runs: start, status, live progress (Server-Sent Events) ---------------------

    @app.post("/api/runs")
    def start_run(body: RunBody) -> dict:
        if body.kind == "monitor":
            ticker = "ALL"   # one pass over every open paper position
        else:
            ticker = (body.ticker or "").upper()
            if st.pb.tier_of(ticker) is None:
                raise HTTPException(422, f"{ticker or 'a ticker'} is not in the playbook universe")
        try:
            job = st.jobs.start(body.kind, ticker, st.today())
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return out(job.view())

    @app.get("/api/jobs")
    def jobs() -> dict:
        active = st.jobs.active()
        return out({"active": active.view() if active else None,
                    "recent": [j.view() for j in list(st.jobs.jobs.values())[-10:]][::-1]})

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = st.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, f"no job {job_id}")
        return out(j.view())

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        from .jobs import stream
        j = st.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, f"no job {job_id}")

        def read(run_ids: list[str]):
            spans, events, n_tot = [], [], 0
            with st.db() as d:
                for rid in run_ids:
                    spans += [_decode(s, ("input_summary", "output_summary", "citations")) for s in d.spans(rid)]
                    events += [_decode(e, ("data",)) for e in d.guardrail_events(rid)]
                    n_tot += d.conn.execute("SELECT COUNT(*) FROM tot_nodes WHERE run_id = ?", (rid,)).fetchone()[0]
            return masked(spans), masked(events), n_tot

        return StreamingResponse(stream(j, read, poll_s=st.poll_s), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

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
            linked = _linked(d, r)
        return masked({"db": db, "run": r, "spans": spans, "guardrail_events": events, **linked})

    @app.get("/api/runs/{run_id}/tot")
    def tot(run_id: str, db: str = DbName) -> dict:
        with st.db(db) as d:
            if d.run(run_id) is None:
                raise HTTPException(404, f"no run {run_id}")
            nodes = d.tot_nodes(run_id)
        return masked({"db": db, "run_id": run_id, "nodes": nodes})

    # --- paper positions: live rule tracks (read-only) ---------------------------------------------

    @app.get("/api/positions")
    def positions() -> dict:
        """Open paper positions with each management rule's progress, from a live snapshot
        (monitor.snapshot + management.evaluate + management.rule_tracks). Cached briefly; a failed
        refresh shows the last good row marked stale, or an error for that position."""
        from ..broker import open_positions
        from ..calculators import portfolio_heat_pct
        from ..management import evaluate, heat_gauge, heat_review, rule_tracks
        from ..monitor import _pending_for, snapshot
        today = st.today()
        db = st.db_factory()          # its own connection: live fetches must not hold the shared lock
        try:
            pos_list = open_positions(db)
            heat = portfolio_heat_pct([p.entry_max_loss * p.contracts for p in pos_list],
                                      st.settings.synthetic_portfolio_value)
            sources, rows, now = None, [], time.monotonic()
            for pos in pos_list:
                base = {"position_id": pos.position_id, "ticker": pos.ticker, "strategy": pos.strategy.value,
                        "opened_on": pos.opened_on.isoformat(), "expiry": pos.expiry.isoformat(),
                        "contracts": pos.contracts, "entry_credit": pos.entry_credit,
                        "entry_max_loss": pos.entry_max_loss,
                        "legs": [{"side": g.side.value, "option_type": g.option_type.value, "strike": g.strike}
                                 for g in pos.legs],
                        "pending_brief": _pending_for(db, pos.position_id)}
                hit = st.positions_cache.get(pos.position_id)
                if hit and now - hit[0] < MARKET_TTL_SECONDS:
                    rows.append({**hit[1], **base, "freshness": {"stale": False, "age_s": round(now - hit[0])}})
                    continue
                try:
                    sources = sources or st.sources_factory(today)
                    snap, info = snapshot(pos, sources, today, db, st.pb)
                    ev = evaluate(pos, snap, st.pb)
                    live = {"spot": snap.spot, "close_cost": info.get("close_cost"), "profit_pct": ev.profit_pct,
                            "tracks": rule_tracks(pos, snap, ev, st.pb), "not_evaluated": list(ev.not_evaluated),
                            "as_of": today.isoformat()}
                    st.positions_cache[pos.position_id] = (now, live)
                    rows.append({**live, **base, "freshness": {"stale": False, "age_s": 0}})
                except Exception as exc:  # noqa: BLE001 - one position's data failure must not hide the others
                    err = f"live data unavailable ({type(exc).__name__})"
                    if hit:
                        rows.append({**hit[1], **base, "freshness": {"stale": True, "age_s": round(now - hit[0]),
                                                                    "error": err}})
                    else:
                        rows.append({**base, "tracks": [], "error": err})
        finally:
            db.close()
        review = heat_review(heat, today, st.pb)
        return out({"as_of": today.isoformat(), "positions": rows,
                    "heat": {**heat_gauge(heat, st.pb), "review": review.reason if review else None}})

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
                return FileResponse(f)   # hashed asset names, safe to cache
            # always revalidate the page so a rebuild is picked up
            return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _legs(legs: list[dict] | None) -> list[dict]:
    return [{"side": g.get("side"), "option_type": g.get("option_type", g.get("type")), "strike": g.get("strike"),
             "expiry": g.get("expiry")} for g in legs or []]


def trade_summary(kind: str, p: dict) -> dict:
    """What a queue card shows, copied from the stored payload (no values are derived)."""
    if kind == "entry":
        prop = p.get("proposal") or {}
        legs = _legs(prop.get("legs"))
        return {"strategy": prop.get("strategy"), "legs": legs,
                "expiry": prop.get("expiry") or next((g["expiry"] for g in legs if g["expiry"]), None)}
    legs = _legs(p.get("legs"))
    out = {"strategy": p.get("strategy"), "legs": legs, "expiry": next((g["expiry"] for g in legs if g["expiry"]), None),
           "exit_rule": p.get("exit_rule"), "action": (p.get("decision") or {}).get("action"), "dte": p.get("dte")}
    if kind == "roll" and p.get("new"):
        new = p["new"].get("proposal") or {}
        out["new"] = {"strategy": new.get("strategy"), "legs": _legs(new.get("legs")), "expiry": new.get("expiry")}
    return out


# The thresholds the Playbook screen lists first: (dotted path, label, unit). Values are read live.
KEY_THRESHOLDS: tuple[tuple[str, str, str], ...] = (
    ("entry.pop_min", "Minimum probability of profit", "fraction"),
    ("entry.position_risk_pct.block", "Position risk block", "%"),
    ("entry.portfolio_heat_pct.block", "Portfolio heat cap", "%"),
    ("entry.earnings.block_within_days", "Block when earnings are within", "days"),
    ("entry.dte_min", "Minimum days to expiry", "days"),
    ("entry.dte_max", "Maximum days to expiry", "days"),
    ("entry.open_interest_min", "Minimum open interest at each short strike", "contracts"),
    ("builder.short_delta_min", "Short delta, lowest", "delta"),
    ("builder.short_delta_max", "Short delta, highest", "delta"),
    ("management.r1_profit_target_pct", "Take profit at", "% of max profit"),
    ("management.r2_roll.dte_below", "Roll below", "DTE"),
    ("management.r3_stop_loss.loss_credit_multiple", "Stop loss at", "x credit"),
    ("guardrails.min_confidence", "Confidence floor", "fraction"),
)


class RunBody(BaseModel):
    ticker: str | None = Field(default=None, min_length=1, max_length=8, pattern=r"^[A-Za-z.]{1,8}$")
    kind: str = Field(default="multi_agent", pattern="^(multi_agent|tot|monitor)$")


def _market_sources(pb: Playbook, day: date):
    """Market data only (no EDGAR, no LLM, no Chroma): what the Positions snapshot needs."""
    from ..market import Market
    from ..tools import CachedSources, LiveSources
    return CachedSources(LiveSources(pb, day, Market(), None, None, None))


def _local_llm(settings: Settings):
    """The UI uses the local model only: no cloud fallback, even if a key is configured."""
    from ..llm import FallbackClient, OllamaClient
    return FallbackClient(OllamaClient(settings.ollama_url, settings.ollama_model, think=settings.ollama_think,
                                       temperature=settings.llm_temperature), None)


def _chroma_store(settings: Settings):
    from ..store import Store
    return Store(settings.chroma_dir)


def _linked(d: Database, run: dict) -> dict:
    """The ToT run and brief a multi-agent run produced (from its stored final state), if any."""
    if run["kind"] != "multi_agent":
        return {"tot_run_id": run["run_id"] if run["kind"] == "tot" else None, "brief_id": None}
    row = d.conn.execute("SELECT content FROM trace WHERE run_id = ? AND name = 'multi_agent.state' "
                         "ORDER BY seq DESC LIMIT 1", (run["run_id"],)).fetchone()
    try:
        state = json.loads(row[0]) if row else {}
    except json.JSONDecodeError:
        state = {}
    return {"tot_run_id": state.get("tot_run_id"), "brief_id": state.get("brief_id"),
            "reasons": list(state.get("reasons") or [])[:5]}


SUMMARY_KEYS = frozenset({"input_summary", "output_summary", "citations", "data"})


def _remask(o: Any, masker: Masker, key: str | None = None) -> Any:
    if isinstance(o, dict):
        return {k: masker.value(v) if k in SUMMARY_KEYS else _remask(v, masker, str(k)) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_remask(v, masker, key) for v in o]
    if isinstance(o, str):
        return o if key is not None and (key == "id" or key.endswith("_id")) else masker.text(o)
    return o


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
