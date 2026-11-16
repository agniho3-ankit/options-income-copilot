"""Typed shared state of one multi-agent run. Plain JSON values only, so it round-trips through
json and can be stored with the run. Non-serializable runtime objects (the ToolBox, the MCP
bridge, the agents) live in graph.Runtime, not here."""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict


class Research(TypedDict, total=False):
    verdict: str                 # final, after the merge rule
    citations: list[dict]
    coverage: dict | None        # of the attempt the verdict came from
    attempts: list[dict]         # [{query, verdict, coverage_ok, relevant, graded}]
    rewrites: int
    reasons: str | None
    escalation: str | None


class Budgets(TypedDict):
    llm_calls: int
    builds: int
    critic_rounds: int
    started_at: float
    max_llm_calls: int
    max_builds: int
    max_critic_rounds: int
    max_wall_seconds: int


class AgentState(TypedDict, total=False):
    run_id: str
    ticker: str
    as_of: str
    research: Research | None
    tot_run_id: str | None
    strategist_done: bool
    candidates: list[dict]       # leaf summaries: node, path, score, status, critic
    pending_review: list[str]    # leaf node ids the critic has not reviewed yet
    revise: list[str]            # expiry node ids the critic sent back for new deltas
    critic_feedback: str
    critic_rounds: int
    critic_enabled: bool
    verifier: dict | None        # the gate's final verification of the chosen trade
    chosen: dict | None
    gate_done: bool
    outcome: str | None          # SELECTED, NO_TRADE, BLOCK, INSUFFICIENT_EVIDENCE
    reasons: list[str]
    brief_id: str | None
    lesson: str | None
    budgets: Budgets
    messages: list[dict]         # {from, to, pattern, summary}
    errors: list[str]
    route_log: list[str]
    # Phase 6: guardrail signals (counts), confidence, the judge, and the evaluation sandbox
    signals: dict[str, int]
    confidence: dict
    prose: str | None
    judge: dict | None
    sandbox: bool
    disabled: list[str]


def new_state(run_id: str, ticker: str, as_of: str, cfg: dict, started_at: float | None = None,
              critic_enabled: bool = True, sandbox: bool = False, disabled: list[str] | None = None) -> AgentState:
    return AgentState(
        run_id=run_id, ticker=ticker.upper(), as_of=as_of, research=None, tot_run_id=None, strategist_done=False,
        candidates=[], pending_review=[], revise=[], critic_feedback="", critic_rounds=0,
        critic_enabled=critic_enabled, verifier=None, chosen=None,
        gate_done=False, outcome=None, reasons=[], brief_id=None, lesson=None,
        budgets=Budgets(llm_calls=0, builds=0, critic_rounds=0,
                        started_at=time.time() if started_at is None else started_at,
                        max_llm_calls=cfg["max_llm_calls"], max_builds=cfg["max_builds"],
                        max_critic_rounds=cfg["max_critic_rounds"], max_wall_seconds=cfg["max_wall_seconds"]),
        messages=[], errors=[], route_log=[], signals={}, confidence={}, prose=None, judge=None,
        sandbox=sandbox, disabled=sorted(disabled or []))


def to_json(state: AgentState) -> str:
    return json.dumps(state, sort_keys=True, default=_strict)


def from_json(text: str) -> AgentState:
    return AgentState(**json.loads(text))


def _strict(obj: Any) -> Any:
    raise TypeError(f"state holds a non-JSON value of type {type(obj).__name__}")


def exhausted(state: AgentState, now: float | None = None) -> str | None:
    """Which run-level budget is used up, if any (deterministic given `now`)."""
    b = state["budgets"]
    if b["llm_calls"] >= b["max_llm_calls"]:
        return f"LLM call budget ({b['max_llm_calls']}) used up"
    if b["builds"] >= b["max_builds"]:
        return f"build budget ({b['max_builds']}) used up"
    if (time.time() if now is None else now) - b["started_at"] >= b["max_wall_seconds"]:
        return f"wall-clock budget ({b['max_wall_seconds']} s) used up"
    return None
