"""Approval briefs: create them from an agent run, and approve or reject them.

Approval is a human act through the CLI. It writes an order ticket to SQLite and never
places an order. The PaperBroker (Phase 4) turns tickets into synthetic positions.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from .agent_tools import ToolBox
from .db import Database, new_id
from .llm import LLMClient
from .mask import Masker
from .models import ReflectionTrigger
from .react import AgentResult
from .reflexion import write_lesson


def make_brief(result: AgentResult, box: ToolBox, db: Database, as_of: date) -> str | None:
    """Store a pending entry brief if the verifier accepted a trade. Returns the brief ID."""
    parts = box.accepted_brief_parts()
    if parts is None:
        return None
    if not box.lessons_used:
        box.search_lessons()  # every brief shows relevant lessons, even if the agent skipped the tool
    lm = box.landmine_result or {}
    risks = [c["reason"] for c in parts["verifier"] if c["verdict"] == "CAUTION"]
    brief_id = new_id("brf")
    payload: dict[str, Any] = {
        **parts, "as_of": as_of.isoformat(), "run_id": result.run_id, "summary": result.final_text.strip(),
        "citations": lm.get("citations", []), "landmine_verdict": lm.get("verdict"),
        "lessons": box.lessons_used, "risks": risks, "synthetic_portfolio": True,
        "submissions": [{"proposal_id": pid, "verdict": r.verdict.value} for pid, r in box.submissions],
    }
    db.save_brief(brief_id, result.run_id, "entry", box.ticker, "pending", parts["verdict"], payload)
    return brief_id


def reflect_on_failures(box: ToolBox, llm: LLMClient | None, store: Any, masker: Masker):
    """Reflexion after a run whose submissions failed the verifier (one lesson per run)."""
    failed = [r for _, r in box.submissions if not r.passed]
    if not failed:
        return None
    checks = sorted({c.name for r in failed for c in r.failures})
    reasons = sorted({c.reason for r in failed for c in r.failures})[:5]
    return write_lesson(ReflectionTrigger.VERIFIER, box.ticker,
                        {"failed_checks": checks, "reasons": reasons, "accepted_after": box.accepted is not None},
                        llm=llm, store=store, masker=masker, tags=["verifier", *checks])


def approve(db: Database, brief_id: str, note: str = "") -> str:
    b = db.brief(brief_id)
    if b is None:
        raise ValueError(f"no brief {brief_id}")
    if b["status"] != "pending":
        raise ValueError(f"brief {brief_id} is {b['status']}, not pending")
    if b["verdict"] not in ("CLEAR", "CAUTION"):
        raise ValueError(f"brief {brief_id} did not pass the verifier ({b['verdict']})")
    p = b["payload"]
    prop = p["proposal"]
    ticket = {"brief_id": brief_id, "ticker": b["ticker"], "strategy": prop["strategy"],
              "contracts": prop["contracts"], "order_type": "limit", "limit_credit_per_share":
              round(p["numbers"]["credit"] / 100, 2), "legs": prop["legs"], "paper_only": True,
              "synthetic": True, "note": note}
    db.decide(brief_id, "approved", note or "approved")
    return db.add_ticket(brief_id, ticket)


def reject(db: Database, brief_id: str, reason: str, *, llm: LLMClient | None, store: Any, masker: Masker):
    if not reason.strip():
        raise ValueError("a rejection needs a reason (it becomes a lesson)")
    b = db.brief(brief_id)
    if b is None:
        raise ValueError(f"no brief {brief_id}")
    db.decide(brief_id, "rejected", reason)
    p = b["payload"]
    return write_lesson(ReflectionTrigger.REJECTION, b["ticker"],
                        {"strategy": p["proposal"]["strategy"], "reason": reason, "risks": p.get("risks", [])},
                        llm=llm, store=store, masker=masker, tags=["rejection"])
