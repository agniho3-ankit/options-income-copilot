"""Background runs for the UI: one LLM job at a time, progress read back from SQLite.

A job runs `run_multi_agent` or `tot.explore` on its own thread with its own database connection.
Progress is never pushed by the agents: the SSE stream polls the run's `spans` and
`guardrail_events` rows as they are written (the same rows `trace` and `metrics` read).

* Only one LLM job runs at a time. A second start is refused (409), and while a run holds the
  LLM, reject writes its lesson from the template.
* The run IDs a job creates are captured as `start_run` is called (the multi-agent run and the
  Tree-of-Thought run inside it), so the stream knows what to read while the run is still going.
* The UI's runs use the local model only (no cloud fallback).
"""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

KINDS = ("multi_agent", "tot", "monitor")


@dataclass
class Job:
    job_id: str
    kind: str
    ticker: str
    as_of: str
    status: str = "running"          # running | done | failed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str | None = None
    runs: dict[str, str] = field(default_factory=dict)   # run kind -> run_id, in start order
    result: dict[str, Any] | None = None
    error: str | None = None

    def view(self) -> dict:
        return {"job_id": self.job_id, "kind": self.kind, "ticker": self.ticker, "as_of": self.as_of,
                "status": self.status, "created_at": self.created_at, "finished_at": self.finished_at,
                "runs": dict(self.runs), "result": self.result, "error": self.error}


class RecordingDatabase:
    """Delegates to a Database and records the run IDs it starts."""

    def __init__(self, db: Any, job: Job):
        self._db, self._job = db, job

    def start_run(self, kind: str, *args: Any, **kw: Any) -> str:
        run_id = self._db.start_run(kind, *args, **kw)
        self._job.runs.setdefault(kind, run_id)
        return run_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._db, name)


Runner = Callable[[str, str, date, Any], dict]   # (kind, ticker, as_of, db) -> result dict


class JobManager:
    def __init__(self, llm_lock: threading.Lock, db_factory: Callable[[], Any], runner: Runner):
        self.llm_lock, self.db_factory, self.runner = llm_lock, db_factory, runner
        self.jobs: dict[str, Job] = {}
        self._mu = threading.Lock()

    def start(self, kind: str, ticker: str, as_of: date) -> Job:
        if kind not in KINDS:
            raise ValueError(f"unknown run kind {kind!r}")
        if not self.llm_lock.acquire(blocking=False):
            raise RuntimeError("another LLM job is running; wait for it to finish")
        job = Job(job_id=f"job-{secrets.token_hex(4)}", kind=kind, ticker=ticker, as_of=as_of.isoformat())
        with self._mu:
            self.jobs[job.job_id] = job
        try:
            threading.Thread(target=self._work, args=(job, as_of), name=job.job_id, daemon=True).start()
        except Exception:
            self.llm_lock.release()
            raise
        return job

    def _work(self, job: Job, as_of: date) -> None:
        db, status, result, error = None, "failed", None, None
        try:
            db = self.db_factory()
            result, status = self.runner(job.kind, job.ticker, as_of, RecordingDatabase(db, job)), "done"
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never swallowed silently
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            traceback.print_exc()
        finally:
            if db is not None and hasattr(db, "close"):
                db.close()
            self.llm_lock.release()
            # published last: once a client sees the job finished, the LLM is already free
            job.result, job.error = result, error
            job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            job.status = status

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def active(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.status == "running"), None)


def stream(job: Job, read: Callable[[list[str]], tuple[list[dict], list[dict], int]], *, poll_s: float = 0.5,
           heartbeat_s: float = 10.0, sleep: Callable[[float], None] = time.sleep):
    """Server-Sent Events for one job. `read(run_ids)` returns (spans, guardrail events, ToT node count)
    for those runs, already ordered and masked. Spans are sent when new or changed (a running span is
    sent again when it completes), guardrail events once each, in the order they were written."""
    sent_spans: dict[tuple[str, str], tuple] = {}
    sent_events: set[tuple[str, int]] = set()
    tot_count = -1
    seq = 0
    quiet = 0.0

    def msg(kind: str, data: dict) -> str:
        nonlocal seq
        seq += 1
        import json
        return f"id: {seq}\nevent: {kind}\ndata: {json.dumps(data, default=str)}\n\n"

    yield msg("job", job.view())
    while True:
        finished = job.status != "running"          # read the status before the rows: the last poll is complete
        spans, events, n_tot = read(list(job.runs.values()))
        out = []
        for s in spans:
            key, ver = (s["run_id"], s["span_id"]), (s.get("ended_at"), s.get("llm_calls"), s.get("error"))
            if sent_spans.get(key) != ver:
                sent_spans[key] = ver
                out.append(msg("span", s))
        for e in events:
            key = (e["run_id"], e["seq"])
            if key not in sent_events:
                sent_events.add(key)
                out.append(msg("guardrail", e))
        if n_tot != tot_count:
            tot_count = n_tot
            out.append(msg("tot", {"nodes": n_tot, "run_id": job.runs.get("tot")}))
        yield from out
        if finished:
            yield msg("done", job.view())
            return
        quiet = 0.0 if out else quiet + poll_s
        if quiet >= heartbeat_s:
            quiet = 0.0
            yield ": keep-alive\n\n"
        sleep(poll_s)


# --- the real runner (live data, local LLM) ---------------------------------------------------

def live_runner(settings: Any, pb: Any) -> Runner:
    """Runs exactly what the CLI's `ma`, `tot` and `monitor` commands run, with the local model only."""
    def run(kind: str, ticker: str, as_of: date, db: Any) -> dict:
        from ..agent_tools import ToolBox
        from ..broker import open_positions
        from ..edgar import EdgarClient
        from ..llm import FallbackClient, OllamaClient
        from ..mask import Masker
        from ..market import Market
        from ..store import Store
        from ..tools import CachedSources, LiveSources

        llm = FallbackClient(OllamaClient(settings.ollama_url, settings.ollama_model, think=settings.ollama_think,
                                          temperature=settings.llm_temperature, num_ctx=16384), None)
        store = Store(settings.chroma_dir)
        edgar = EdgarClient(settings.sec_user_agent, settings.raw_dir) if settings.sec_user_agent else None
        sources = CachedSources(LiveSources(pb, as_of, Market(), edgar, store, llm))
        masker = Masker(settings.synthetic_portfolio_value)
        losses = tuple(p.entry_max_loss * p.contracts for p in open_positions(db))
        if kind == "monitor":
            from ..monitor import run_monitor
            run_id, items, summary = run_monitor(db, sources, pb, as_of, llm=llm, masker=masker,
                                                 portfolio_value=settings.synthetic_portfolio_value, store=store,
                                                 tot_llm=llm)
            return {"run_id": run_id, "outcome": "done", "brief_id": None, "summary": summary,
                    "items": [{"position_id": i.position_id, "ticker": i.ticker,
                               "action": i.decision.action.value if i.decision.action else "hold",
                               "chosen_by": i.decision.chosen_by, "brief_id": i.brief_id, "note": i.note}
                              for i in items]}
        if kind == "multi_agent":
            from ..multi_agent.runner import run_multi_agent
            r = run_multi_agent(ticker, sources=sources, pb=pb, as_of=as_of, llm=llm, db=db, masker=masker,
                                portfolio_value=settings.synthetic_portfolio_value, store=store,
                                open_max_losses=losses, use_agents=True)
            return {"run_id": r.run_id, "outcome": r.outcome, "brief_id": r.state.get("brief_id"),
                    "tot_run_id": r.state.get("tot_run_id"), "reasons": list(r.state.get("reasons") or [])[:5],
                    "llm_calls": r.llm_calls, "latency_ms": r.latency_ms}
        from ..tot import explore, summary_text
        box = ToolBox(ticker, sources, pb, as_of, settings.synthetic_portfolio_value, open_max_losses=losses,
                      iv_history=tuple(v for _, v in db.iv_history(ticker)))
        t = explore(box, pb, llm, db, masker)
        return {"run_id": t.run_id, "outcome": t.outcome, "brief_id": None, "tot_run_id": t.run_id,
                "reasons": list(t.reasons)[:5], "summary": summary_text(t)}
    return run
