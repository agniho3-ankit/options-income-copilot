"""The owner's review queue (Phase 6).

Every trade still needs the owner's approval; nothing here approves anything. Pending briefs are
ranked by a deterministic risk score computed from escalation signals (weights and tiers in
playbook.yaml `guardrails.review`) into three tiers: standard, look_closely, escalated. Runs that
ended in BLOCK or INSUFFICIENT_EVIDENCE appear as info items: no trade, just why and what to do.
The reviewer view fits one screen and ends with three ready commands: approve, reject with a
reason code, or defer.
"""
from __future__ import annotations

import json
from typing import Any

from .db import Database
from .guardrails import SIGNALS, risk_score
from .playbook import Playbook

REASONS = {
    "verifier_cautions": "the verifier passed it with {n} caution(s)",
    "landmine_caution": "the filings check said CAUTION",
    "low_confidence": "a model verdict or critic action was below the confidence floor",
    "grader_zero_relevant": "the grader found no relevant filing passage, so CLEAR rests on weak evidence",
    "critic_citation_unsupported": "the critic cited a check that passed, a missing lesson, or one generic reason",
    "judge_flag": "the judge found a summary claim the facts do not support",
    "budget_exhausted": "a budget ran out before the search finished",
    "revisions_all_duplicates": "every revision the critic asked for repeated an earlier candidate",
}
INFO_OUTCOMES = ("BLOCK", "INSUFFICIENT_EVIDENCE")
NEXT_STEP = {
    "BLOCK": "no trade; read the cited filing and wait for the event to resolve",
    "INSUFFICIENT_EVIDENCE": "no trade; run `ingest TICKER` for recent filings, then re-run",
}


def brief_signals(db: Database, brief: dict) -> dict[str, int]:
    """Signal counts for one brief: the ones stored with it, the run's guardrail events, and what
    the payload itself shows (for briefs written before Phase 6)."""
    p = brief["payload"]
    out: dict[str, int] = {s: 0 for s in SIGNALS}
    for k, v in (p.get("signals") or {}).items():
        if k in out:
            out[k] = max(out[k], int(v))
    if brief.get("run_id"):
        for e in db.guardrail_events(brief["run_id"]):
            if e["signal"] in out:
                out[e["signal"]] = max(out[e["signal"]], 1)
    out["verifier_cautions"] = max(out["verifier_cautions"], len(p.get("risks", [])))
    if p.get("landmine_verdict") == "CAUTION":
        out["landmine_caution"] = 1
    if (p.get("judge") or {}).get("flags"):
        out["judge_flag"] = 1
    return {k: v for k, v in out.items() if v}


def explain(signal: str, n: int) -> str:
    return REASONS[signal].format(n=n)


def queue(db: Database, pb: Playbook, info_limit: int = 10) -> dict[str, list[dict]]:
    """{"trades": pending briefs ranked, "info": recent BLOCK / INSUFFICIENT_EVIDENCE runs}."""
    cfg = pb["guardrails.review"]
    items = []
    for row in db.briefs("pending"):
        b = db.brief(row["brief_id"])
        sig = brief_signals(db, b)
        score, tier, parts = risk_score(sig, cfg)
        items.append({"brief_id": b["brief_id"], "ticker": b["ticker"], "kind": b["kind"], "verdict": b["verdict"],
                      "created_at": b["created_at"], "deferred": bool(row.get("deferred_at")), "score": score,
                      "tier": tier, "top": [(s, explain(s, sig[s])) for s, _ in parts[:3]], "signals": sig})
    order = {"escalated": 0, "look_closely": 1, "standard": 2}
    items.sort(key=lambda i: (i["deferred"], order[i["tier"]], -i["score"], i["created_at"]))
    info = []
    rows = db.conn.execute("SELECT run_id, kind, ticker, outcome, started_at FROM runs WHERE outcome IN (?, ?) "
                           "ORDER BY started_at DESC LIMIT ?", (*INFO_OUTCOMES, info_limit)).fetchall()
    for r in rows:
        why = _escalation_reason(db, r["run_id"])
        info.append({"run_id": r["run_id"], "ticker": r["ticker"], "kind": r["kind"], "outcome": r["outcome"],
                     "when": r["started_at"], "why": why, "next": NEXT_STEP[r["outcome"]]})
    return {"trades": items, "info": info}


def _escalation_reason(db: Database, run_id: str) -> str:
    for e in db.guardrail_events(run_id):
        if e["decision"] == "block":
            return e["reason"]
    for t in reversed(db.trace(run_id)):
        if t["role"] == "summary":
            try:
                body = json.loads(t["content"])
            except json.JSONDecodeError:
                continue
            reasons = body.get("reasons") or (body.get("research") or {}).get("escalation")
            if reasons:
                return reasons[0] if isinstance(reasons, list) else str(reasons)
    return "see `trace " + run_id + "`"


def render_queue(q: dict[str, list[dict]]) -> str:
    lines = ["Review queue (every trade needs your approval; tiers only set the order and how closely to look)"]
    if not q["trades"]:
        lines.append("  no pending trade briefs")
    for i in q["trades"]:
        flag = " (deferred)" if i["deferred"] else ""
        lines.append(f"  [{i['tier']:<12}] score {i['score']:>2}  {i['brief_id']}  {i['ticker']:<5} {i['kind']:<5} "
                     f"verifier {i['verdict']}{flag}")
        for _, why in i["top"]:
            lines.append(f"      - {why}")
    if q["info"]:
        lines.append("Info items (no trade):")
        for i in q["info"]:
            lines.append(f"  {i['outcome']:<22} {i['ticker']:<5} {i['run_id']}  {i['why'][:90]}")
            lines.append(f"      next: {i['next']}")
    return "\n".join(lines)


def render_view(db: Database, pb: Playbook, brief_id: str) -> str:
    """One brief on one screen: trade, verifier, top signals, citations, alternatives, choices."""
    b = db.brief(brief_id)
    if b is None:
        raise ValueError(f"no brief {brief_id}")
    p: dict[str, Any] = b["payload"]
    sig = brief_signals(db, b)
    score, tier, parts = risk_score(sig, pb["guardrails.review"])
    prop, n = p.get("proposal") or {}, p.get("numbers") or {}
    legs = ", ".join(f"{g.get('side')} {g.get('option_type', g.get('type'))} {g.get('strike')}"
                     for g in prop.get("legs", []))
    expiry = prop.get("expiry") or next((g.get("expiry") for g in prop.get("legs", []) if g.get("expiry")), None)
    lines = [f"{brief_id}  {b['ticker']}  {b['kind']}  [{tier}, score {score}]  status {b['status']}",
             f"trade     {prop.get('strategy')} exp {expiry}: {legs}",
             f"numbers   credit {n.get('credit')}  max loss {n.get('max_loss')}  PoP {n.get('pop')}  "
             f"breakevens {n.get('breakevens')}  risk {n.get('risk_pct_of_portfolio')}% of portfolio",
             f"verifier  {b['verdict']}" + (f"; cautions: {'; '.join(p.get('risks', []))[:200]}" if p.get("risks") else "")]
    lines.append("signals   " + ("; ".join(explain(s, sig[s]) for s, _ in parts[:3]) if parts else "none"))
    cites = p.get("citations", [])
    lines.append("citations " + ("; ".join(f"{c.get('form')} {c.get('date')} {c.get('item')}" for c in cites[:3])
                                 if cites else "none") + f"  (filings {p.get('landmine_verdict')})")
    alts = sorted((a for a in p.get("alternatives") or [] if a.get("score") is not None),
                  key=lambda a: -a["score"])
    if alts:
        lines.append("also considered " + "; ".join(f"{a.get('path')} (score {a['score']}, {a.get('status')})"
                                                    for a in alts[:3]))
    else:
        lines.append("also considered: no other priced candidate")
    codes = "|".join(pb["guardrails.reject_codes"])
    if b["status"] == "pending":
        lines += ["choose one:",
                  f"  approve   python -m options_copilot.cli approve {brief_id}",
                  f"  reject    python -m options_copilot.cli reject {brief_id} --code <{codes}> --reason \"...\"",
                  f"  defer     python -m options_copilot.cli defer {brief_id}"]
    return "\n".join(lines)
