"""Local observability (Phase 6): `metrics` summarizes behavior over time from SQLite only.

No cloud dashboard and no exporter: LangSmith and OpenTelemetry stay off (multi_agent.privacy).
Spans carry operational data (duration, LLM calls, tokens, fallbacks, errors), cognitive data
(decision, confidence; guardrail events link by span_id) and contextual data (ticker, as_of,
masked summaries, citations).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .db import Database
from .evaluation import percentile
from .playbook import Playbook


def metrics(db: Database, pb: Playbook, since: str | None = None) -> dict:
    since = since or "0000"
    c = db.conn
    runs = c.execute("SELECT kind, outcome FROM runs WHERE started_at >= ?", (since,)).fetchall()
    by_outcome: dict[str, Counter] = defaultdict(Counter)
    for r in runs:
        by_outcome[r["kind"]][r["outcome"] or "unfinished"] += 1

    from .review import brief_signals
    from .guardrails import risk_score
    tiers: Counter = Counter()
    for row in c.execute("SELECT brief_id FROM briefs WHERE kind = 'entry' AND created_at >= ?", (since,)):
        b = db.brief(row["brief_id"])
        tiers[risk_score(brief_signals(db, b), pb["guardrails.review"])[1]] += 1
    info = sum(n for k in by_outcome.values() for o, n in k.items() if o in ("BLOCK", "INSUFFICIENT_EVIDENCE"))
    total_items = sum(tiers.values()) + info
    escalation = {t: _rate(tiers[t], total_items) for t in ("standard", "look_closely", "escalated")}
    escalation["info_block_or_insufficient"] = _rate(info, total_items)

    events = c.execute("SELECT g.guardrail, g.decision, g.signal FROM guardrail_events g JOIN runs r USING (run_id) "
                       "WHERE r.started_at >= ?", (since,)).fetchall()
    guard: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        guard[e["guardrail"]][e["decision"]] += 1
    judge_runs = guard.get("judge", Counter())
    judge_rate = _rate(judge_runs["flag"], sum(judge_runs.values()))

    spans = c.execute("SELECT s.* FROM spans s JOIN runs r USING (run_id) WHERE r.started_at >= ? "
                      "AND s.ended_at IS NOT NULL", (since,)).fetchall()   # a running span has no duration yet
    lat: dict[str, list[int]] = defaultdict(list)
    calls_per_run: Counter = Counter()
    fallbacks = errors = 0
    conf = []
    for s in spans:
        if s["agent"] == "session":
            fallbacks += s["fallbacks"] or 0
            continue
        lat[s["agent"]].append(s["duration_ms"] or 0)
        calls_per_run[s["run_id"]] += s["llm_calls"] or 0
        errors += 1 if s["error"] else 0
        if s["agent"] == "researcher" and s["confidence"] is not None:
            conf.append(s["confidence"])
    per_agent = {a: {"n": len(v), "p50_s": _s(percentile(v, 50)), "p95_s": _s(percentile(v, 95))}
                 for a, v in sorted(lat.items()) if a != "supervisor"}
    calls = list(calls_per_run.values())
    codes = Counter(r["decision_code"] for r in c.execute(
        "SELECT decision_code FROM briefs WHERE status = 'rejected' AND decided_at >= ?", (since,)))
    latest = db.latest_eval("eval")
    unsafe = None if latest is None else {"eval_id": latest["eval_id"], "created_at": latest["created_at"],
                                          "unsafe_passed": latest["summary"].get("unsafe_passed")}
    bins = Counter(_bin(x) for x in conf)
    return {
        "since": None if since == "0000" else since,
        "runs_by_outcome": {k: dict(v) for k, v in sorted(by_outcome.items())},
        "escalation_rate": escalation, "review_items": total_items,
        "guardrails": {k: dict(v) for k, v in sorted(guard.items())},
        "latest_eval_unsafe_passed": unsafe,
        "latency_per_agent": per_agent,
        "llm_calls_per_run": {"runs": len(calls), "mean": round(sum(calls) / len(calls), 2) if calls else None,
                              "p95": percentile(calls, 95)},
        "claude_fallbacks": fallbacks, "span_errors": errors,
        "reject_codes": {(k or "uncoded"): v for k, v in codes.most_common()},
        "judge_flag_rate": judge_rate,
        "landmine_confidence": {"n": len(conf), "bins": {b: bins.get(b, 0) for b in
                                                         ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")}},
    }


def render(m: dict) -> str:
    lines = [f"== Metrics{' since ' + m['since'] if m['since'] else ''} (local SQLite only; no exporters) =="]
    lines.append("runs by outcome:")
    for kind, outs in m["runs_by_outcome"].items():
        lines.append(f"  {kind:<12} " + ", ".join(f"{o} {n}" for o, n in sorted(outs.items())))
    e = m["escalation_rate"]
    lines.append(f"review items {m['review_items']}: standard {e['standard']}, look_closely {e['look_closely']}, "
                 f"escalated {e['escalated']}, info (BLOCK/INSUFFICIENT) {e['info_block_or_insufficient']}")
    lines.append("guardrail decisions:")
    for g, d in m["guardrails"].items():
        lines.append(f"  {g:<24} " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())))
    u = m["latest_eval_unsafe_passed"]
    lines.append("latest eval: " + (f"unsafe trades passed {u['unsafe_passed']} ({u['eval_id']}, {u['created_at']})"
                                    if u else "none recorded yet"))
    lines.append("latency per agent: " + "; ".join(f"{a} n {v['n']} p50 {v['p50_s']}s p95 {v['p95_s']}s"
                                                   for a, v in m["latency_per_agent"].items()))
    lc = m["llm_calls_per_run"]
    lines.append(f"LLM calls per multi-agent run: mean {lc['mean']} p95 {lc['p95']} over {lc['runs']} run(s)")
    lines.append(f"Claude fallbacks {m['claude_fallbacks']}; span errors {m['span_errors']}; "
                 f"judge flag rate {m['judge_flag_rate']}")
    lines.append("reject codes: " + (", ".join(f"{k} {v}" for k, v in m["reject_codes"].items()) or "none"))
    lc2 = m["landmine_confidence"]
    lines.append(f"landmine confidence (n {lc2['n']}): " + ", ".join(f"{b} {n}" for b, n in lc2["bins"].items()))
    rc = {k: v for k, v in m["reject_codes"].items() if k not in ("uncoded", "other")}
    if rc and max(rc.values()) >= 3:
        top = max(rc, key=rc.get)
        lines.append(f"note: '{top}' was used {rc[top]} times; consider tightening the related prompt or guardrail "
                     "(this report never changes the playbook)")
    return "\n".join(lines)


def _bin(x: float) -> str:
    edges = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
    return edges[min(4, max(0, int(x * 5)))]


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _s(ms) -> float | None:
    return None if ms is None else round(ms / 1000, 1)
