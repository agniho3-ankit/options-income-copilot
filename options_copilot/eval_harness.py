"""The evaluation harness (Phase 6): frozen snapshots, labeled cases, repeats, ablation.

Cases live in `eval/cases.local.yaml` (gitignored): each points at real past runs (provenance)
and holds the owner's label. Snapshots live in `data/eval_snapshots/<case>/` (gitignored):
    meta.json     ticker, as_of, capture time, IV history
    market.pkl    ticker data, listed expiries, option chains (written and read only by this tool)
    chunks.json   the ticker's in-window filing chunks with embeddings, copied from local Chroma
    lessons.json  the lessons the run could retrieve
A replay rebuilds an in-memory Chroma from the snapshot, so retrieval and every deterministic step
are identical across replays; only LLM sampling varies.

Every eval run is sandboxed: no brief, no broker call, no lesson. Committed tests use synthetic
fixtures only; downloaded filing text stays under data/.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from . import evaluation as E
from .guardrails import risk_score, run_config

DEFAULT_CASES = Path("eval/cases.local.yaml")
SNAPSHOT_ROOT = Path("data/eval_snapshots")
TRADE_CONTEXT = "selling an options-income trade"
PARAPHRASE_QUERY = ("{ticker}: recent 8-K and 10-Q disclosures that could move the stock sharply before an option "
                    "expires: deals, mergers, buyouts, financing or recapitalization, lawsuits or probes, leadership "
                    "exits, restatements, write-downs, guidance changes, delisting, bankruptcy, cyber incidents.")
OUTCOMES = ("trade", "NO_TRADE", "BLOCK", "INSUFFICIENT_EVIDENCE")
VERDICTS = ("CLEAR", "CAUTION", "BLOCK", "INSUFFICIENT_EVIDENCE")
ABLATIONS = ("coverage_guard", "grader_rewrite", "critic", "event_date_check", "confidence_gate", "judge", "verifier")


class CaseError(ValueError):
    pass


# --- cases ----------------------------------------------------------------------------------

def load_cases(path: Path) -> tuple[list[dict], list[dict]]:
    """(cases, judge_labels). Refuses unlabeled cases: labels are the owner's, never inferred."""
    if not path.is_file():
        raise CaseError(f"{path} not found; copy the proposed list and add your labels")
    raw = yaml.safe_load(path.read_text()) or {}
    cases, problems = [], []
    for c in raw.get("cases", []):
        lab = c.get("label") or {}
        if lab.get("expected_verdict") not in VERDICTS:
            problems.append(f"{c.get('id')}: expected_verdict must be one of {VERDICTS}")
        if lab.get("expected_outcome") not in OUTCOMES:
            problems.append(f"{c.get('id')}: expected_outcome must be one of {OUTCOMES}")
        cases.append(c)
    if problems:
        raise CaseError("unlabeled or invalid cases (labels are yours to set):\n  " + "\n  ".join(problems))
    judges = [j for j in raw.get("judge_labels", []) if isinstance(j.get("faithful"), bool)]
    return cases, judges


# --- snapshots ------------------------------------------------------------------------------

def capture_snapshot(case_id: str, ticker: str, as_of: date, *, live_sources, store, db, pb,
                     root: Path = SNAPSHOT_ROOT) -> Path:
    """Freeze everything a replay needs. Run during market hours so the chains have live bids."""
    from datetime import timedelta
    ticker = ticker.upper()
    out = root / case_id
    out.mkdir(parents=True, exist_ok=True)
    data = live_sources.ticker_data(ticker)
    expiries = live_sources.expiries(ticker)
    horizon = as_of + timedelta(days=pb["entry.dte_max"] + 7)
    chains = {e: live_sources.chain(ticker, e) for e in expiries if as_of < e <= horizon}
    with (out / "market.pkl").open("wb") as fh:
        pickle.dump({"ticker_data": data, "expiries": expiries, "chains": chains}, fh)
    cfg = store.cfg
    start = as_of - timedelta(days=cfg.window_days)
    from .models import yyyymmdd
    got = store.filings.get(where={"$and": [{"ticker": ticker}, {"filed_date": {"$gte": yyyymmdd(start)}},
                                            {"filed_date": {"$lte": yyyymmdd(as_of)}}]},
                            include=["documents", "metadatas", "embeddings"])
    chunks = {"ids": got["ids"], "documents": got["documents"], "metadatas": got["metadatas"],
              "embeddings": [[float(x) for x in e] for e in (got["embeddings"] if got["embeddings"] is not None else [])]}
    (out / "chunks.json").write_text(json.dumps(chunks))
    les = store.reflections.get(where={"ticker": {"$in": [ticker, "ANY"]}},
                                include=["documents", "metadatas", "embeddings"])
    lessons = {"ids": les["ids"], "documents": les["documents"], "metadatas": les["metadatas"],
               "embeddings": [[float(x) for x in e] for e in (les["embeddings"] if les["embeddings"] is not None else [])]}
    (out / "lessons.json").write_text(json.dumps(lessons))
    meta = {"case": case_id, "ticker": ticker, "as_of": as_of.isoformat(),
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chains": len(chains), "chunks": len(chunks["ids"]), "lessons": len(lessons["ids"]),
            "iv_history": db.iv_history(ticker)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


@dataclass
class Snapshot:
    meta: dict
    market: dict
    chunks: dict
    lessons: dict

    @property
    def as_of(self) -> date:
        return date.fromisoformat(self.meta["as_of"])

    @property
    def ticker(self) -> str:
        return self.meta["ticker"]


def load_snapshot(case_id: str, root: Path = SNAPSHOT_ROOT) -> Snapshot:
    d = root / case_id
    if not (d / "meta.json").is_file():
        raise CaseError(f"no snapshot for case {case_id}; run `eval-snapshot {case_id}` during market hours")
    with (d / "market.pkl").open("rb") as fh:   # written by capture_snapshot only (local, gitignored)
        market = pickle.load(fh)
    return Snapshot(json.loads((d / "meta.json").read_text()), market,
                    json.loads((d / "chunks.json").read_text()), json.loads((d / "lessons.json").read_text()))


def snapshot_store(snap: Snapshot, cfg, embedding_function=None):
    """An in-memory Chroma holding exactly the snapshot's chunks and lessons."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from .store import FILINGS, REFLECTIONS, Store
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True))
    for name in (FILINGS, REFLECTIONS):
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - absent on first use
            pass
    st = Store(None, embedding_function=embedding_function, cfg=cfg, client=client)
    for coll, part in ((st.filings, snap.chunks), (st.reflections, snap.lessons)):
        if part["ids"]:
            coll.add(ids=part["ids"], documents=part["documents"], metadatas=part["metadatas"],
                     embeddings=part["embeddings"])
    return st


class SnapshotSources:
    """The agent's data-source interface, answered from a frozen snapshot."""

    def __init__(self, snap: Snapshot, store, llm):
        self.snap, self.store, self.llm = snap, store, llm

    def ticker_data(self, ticker):
        return self.snap.market["ticker_data"]

    def expiries(self, ticker):
        return list(self.snap.market["expiries"])

    def chain(self, ticker, expiry):
        got = self.snap.market["chains"].get(expiry)
        return got if got is not None else ([], None)

    def landmine(self, ticker, context, query=None):
        from .agent import landmine_check
        return landmine_check(ticker, context, store=self.store, llm=self.llm, as_of=self.snap.as_of, query=query)

    def lessons(self, ticker, query):
        return self.store.search_lessons(ticker, query)


# --- running cases ----------------------------------------------------------------------------

@dataclass
class Harness:
    """`make_sources(case)` -> (sources, as_of, iv_history); `llm` the eval LLM client; `db` the eval
    database (never the owner's main one); `pb`, `masker`, `portfolio_value` as for a normal run."""
    make_sources: Callable[[dict], tuple[Any, date, list]]
    llm: Any
    db: Any
    pb: Any
    masker: Any
    portfolio_value: float
    use_agents: bool = True
    log: Callable[[str], None] = field(default=lambda s: None)

    def run_case(self, case: dict, repeat: int, disabled: frozenset = frozenset(), paraphrase: bool = False) -> dict:
        from .multi_agent.runner import run_multi_agent
        sources, as_of, iv = self.make_sources(case)
        for d, v in iv:
            self.db.add_iv_snapshot(case["ticker"], d, v, None, None)
        t0 = time.monotonic()
        r = run_multi_agent(case["ticker"], sources=sources, pb=self.pb, as_of=as_of, llm=self.llm, db=self.db,
                            masker=self.masker, portfolio_value=self.portfolio_value, use_agents=self.use_agents,
                            sandbox=True, disabled=disabled)
        st = r.state
        research = st.get("research") or {}
        conf = (st.get("confidence") or {}).get("landmine") or {}
        signals = st.get("signals") or {}
        _, tier, _ = risk_score(signals, self.pb["guardrails.review"])
        rec = {"case": case["id"], "repeat": repeat, "run_id": r.run_id, "verdict": research.get("verdict"),
               "outcome": r.outcome, "strategy": ((st.get("chosen") or {}).get("path") or "").split(" / ")[0] or None,
               "chosen": (st.get("chosen") or {}).get("path"), "confidence": conf.get("verbal"),
               "p_true": conf.get("p_true"), "p_true_method": conf.get("p_true_method"),
               "latency_ms": int((time.monotonic() - t0) * 1000), "llm_calls": r.llm_calls, "builds": r.builds,
               "fallbacks": r.fallbacks, "signals": signals, "tier": tier if r.outcome == "SELECTED" else None,
               "judge_flags": len((st.get("judge") or {}).get("flags", [])), "prose": st.get("prose"),
               "paraphrase_verdict": None, "disabled": sorted(disabled)}
        if paraphrase:
            with run_config(sandbox=True, disabled=disabled):
                pv = sources.landmine(case["ticker"], TRADE_CONTEXT, query=PARAPHRASE_QUERY.format(ticker=case["ticker"]))
            rec["paraphrase_verdict"] = pv.get("verdict")
        self.log(f"  {case['id']} #{repeat}{' -' + ','.join(sorted(disabled)) if disabled else ''}: "
                 f"{rec['verdict']} -> {rec['outcome']} {rec['chosen'] or ''} ({rec['latency_ms'] / 1000:.0f}s)")
        return rec

    def evaluate(self, cases: list[dict], repeats: int = 3) -> dict:
        labels = {c["id"]: c["label"] for c in cases}
        records = []
        for c in cases:
            for k in range(repeats):
                records.append(self.run_case(c, k + 1, paraphrase=(k == 0)))
        return {"mode": "eval", "repeats": repeats, "summary": E.summarize(records, labels), "records": records}

    def ablation(self, cases: list[dict], configs: tuple[str, ...] = ABLATIONS) -> dict:
        labels = {c["id"]: c["label"] for c in cases}
        rows = []
        for name in ("full", *configs):
            disabled = frozenset() if name == "full" else frozenset({name})
            recs = [self.run_case(c, 1, disabled=disabled) for c in cases]
            s = E.summarize(recs, labels)["aggregate"]
            unflagged = sum(1 for r in recs if E.unsafe_pass(r, labels[r["case"]]) and r["tier"] == "standard")
            rows.append({"config": name if name == "full" else f"-{name}", "unsafe_passed": s["unsafe_passed"],
                         "unsafe_standard_tier": unflagged, "false_blocks": s["false_blocks"],
                         "verdict_accuracy": s["verdict_accuracy"], "outcome_accuracy": s["outcome_accuracy"],
                         "llm_calls_mean": s["llm_calls_mean"], "latency_p50_s": s["latency_p50_s"],
                         "records": recs})
        return {"mode": "ablation", "rows": rows}


def judge_eval(judge_labels: list[dict], main_db, llm, masker) -> dict:
    """Run the judge on owner-labeled briefs (faithful: true/false) and score it."""
    from .judge import brief_facts, judge_brief
    results = []
    for j in judge_labels:
        b = main_db.brief(j["brief_id"])
        if b is None:
            results.append({"brief_id": j["brief_id"], "error": "no such brief"})
            continue
        prose = j.get("prose_override") or b["payload"].get("summary", "")
        out = judge_brief(prose, brief_facts(b["payload"]), llm, masker)
        results.append({"brief_id": j["brief_id"], "faithful": j["faithful"], "flagged": bool(out["flags"]),
                        "status": out["status"], "flags": out["flags"]})
    scored = [r for r in results if "error" not in r and r["status"] == "ok"]
    return {"results": results, "accuracy": E.judge_accuracy(scored)}


# --- rendering ------------------------------------------------------------------------------

def render_eval(res: dict) -> str:
    s = res["summary"]
    a = s["aggregate"]
    lines = [f"== Evaluation: {a['cases']} cases x {res['repeats']} repeats = {a['runs']} runs (multi-agent, "
             "sandboxed, frozen snapshots) ==",
             f"accuracy     verdict {a['verdict_accuracy']}  outcome {a['outcome_accuracy']}",
             f"safety       unsafe trades passed {a['unsafe_passed']}  false blocks {a['false_blocks']} "
             f"(rate {a['false_block_rate']})",
             f"robustness   verdict consistency {a['verdict_consistency']}  outcome consistency "
             f"{a['outcome_consistency']}  paraphrase agreement {a['paraphrase_agreement']}",
             f"efficiency   latency p50 {a['latency_p50_s']}s p95 {a['latency_p95_s']}s  LLM calls/run "
             f"{a['llm_calls_mean']}  builds/run {a['builds_mean']}  Claude fallbacks {a['fallbacks']}",
             "calibration (verdict correct vs stated probability; small sample, counts shown):"]
    for key, c in s["calibration"].items():
        lines.append(f"  {key:<10} n {c['n']}  ECE {c['ece']}  Brier {c['brier']}")
        lines.append("             " + "  ".join(f"{r['bin']}: n {r['n']}" + (f" conf {r['mean_conf']} acc {r['accuracy']}"
                                                                          if r["n"] else "") for r in c["reliability"]))
    lines.append("per case:")
    for cid, c in s["cases"].items():
        lines.append(f"  {cid:<22} verdicts {c['verdicts']}  outcomes {c['outcomes']}  unsafe {c['unsafe_passed']}  "
                     f"false blocks {c['false_blocks']}  paraphrase {c['paraphrase_agreement']}")
    if res.get("judge"):
        ja = res["judge"]["accuracy"]
        lines.append(f"judge        n {ja['n']}  accuracy {ja['accuracy']}  precision {ja['precision']}  "
                     f"recall {ja['recall']}  (tp {ja['tp']} fp {ja['fp']} fn {ja['fn']} tn {ja['tn']})")
    return "\n".join(lines)


def render_ablation(res: dict) -> str:
    head = (f"{'config':<20} {'unsafe':>6} {'unsafe@std':>10} {'false blk':>9} {'verdict acc':>11} "
            f"{'outcome acc':>11} {'LLM/run':>8} {'p50 s':>6}")
    lines = ["== Ablation: one guardrail off at a time (sandboxed; the verifier may be off only here) ==", head]
    for r in res["rows"]:
        lines.append(f"{r['config']:<20} {r['unsafe_passed']:>6} {r['unsafe_standard_tier']:>10} {r['false_blocks']:>9} "
                     f"{str(r['verdict_accuracy']):>11} {str(r['outcome_accuracy']):>11} "
                     f"{str(r['llm_calls_mean']):>8} {str(r['latency_p50_s']):>6}")
    lines.append("unsafe = a trade selected where your label says no trade; unsafe@std = of those, how many the "
                 "review queue would show at the standard tier (no escalation signal).")
    return "\n".join(lines)
