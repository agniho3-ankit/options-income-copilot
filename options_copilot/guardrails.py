"""One registry and one logged pipeline for every guardrail (Phase 6, Module 6).

The checks themselves live where they always have (store, agent, tot, verifier, briefs); this
module names them, orders them, and records every decision as an event on the current run.

    pre     coverage_guard, grader_rewrite, playbook_bands
    during  budgets, critic_lower_only, critic_citation, confidence_gate
    post    verifier, event_date_check, citation_from_metadata, brief_numbers, judge

A decision is `pass`, `flag` (an escalation signal for the review queue; the run continues) or
`block` (the guardrail stopped or removed something). Signals can only add caution: no event
here ever relaxes a verdict, raises a score, or approves a trade.

Two process-wide switches exist for evaluation only:
  * `run_config(sandbox=True)`: briefs, broker calls and lesson writes raise SandboxViolation;
  * `run_config(disabled={...})`: switches named guardrails off (the ablation study). Switching
    off the verifier is refused unless the sandbox is on.
"""
from __future__ import annotations

import contextlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Iterator

STAGES = ("pre", "during", "post")
KINDS = ("static", "runtime", "risk_score")
DECISIONS = ("pass", "flag", "block")


@dataclass(frozen=True)
class Guardrail:
    name: str
    stage: str
    kind: str
    what: str
    where: str                 # the code that does the check (unchanged by this module)
    can_disable: bool = True   # in ablation mode


REGISTRY: tuple[Guardrail, ...] = (
    Guardrail("coverage_guard", "pre", "static",
              "too little retrieved filing evidence returns INSUFFICIENT_EVIDENCE before any LLM call",
              "store.assess_coverage via agent.landmine_check"),
    Guardrail("grader_rewrite", "pre", "runtime",
              "LLM relevance grade of the excerpts; one query rewrite if too few are relevant",
              "multi_agent.graph researcher"),
    Guardrail("playbook_bands", "pre", "static",
              "generator candidates outside the playbook strategies, DTE window or delta bands are rejected",
              "tot.Explorer generator validation"),
    Guardrail("budgets", "during", "static",
              "LLM-call, build, critic-round and wall-clock budgets end the run with NO_TRADE",
              "multi_agent.state.exhausted, tot budgets", can_disable=False),
    Guardrail("critic_lower_only", "during", "static",
              "critic reviews may only keep, lower or veto; raises and unknown actions are ignored",
              "tot.apply_critique", can_disable=True),
    Guardrail("critic_citation", "during", "runtime",
              "a critic citation must name a check that did not pass or a lesson that exists, and not be "
              "one generic reason pasted on every candidate", "guardrails.critic_citation_problems"),
    Guardrail("confidence_gate", "during", "risk_score",
              "landmine or critic confidence below guardrails.min_confidence raises low_confidence",
              "guardrails.confidence_gate"),
    Guardrail("verifier", "post", "static",
              "the deterministic verifier and the gate's re-verify; final authority",
              "verifier.verify, multi_agent.graph verifier_gate"),
    Guardrail("event_date_check", "post", "static",
              "cited filing event dates vs the trade window (Python reads the dates, never the LLM)",
              "verifier.check_event_window"),
    Guardrail("citation_from_metadata", "post", "static",
              "citations are rebuilt from stored chunk metadata; invented chunk IDs are dropped",
              "agent.enforce"),
    Guardrail("brief_numbers", "post", "static",
              "every number in the brief equals the tool numbers; prose sentences with digits are dropped",
              "guardrails.brief_number_problems, graph.write_prose"),
    Guardrail("judge", "post", "risk_score",
              "LLM-as-judge lists prose claims unsupported by or contradicting the facts; flag only",
              "judge.judge_brief"),
)
BY_NAME = {g.name: g for g in REGISTRY}
ABLATABLE = ("coverage_guard", "grader_rewrite", "critic", "event_date_check", "confidence_gate", "judge",
             "verifier")

# escalation signals the review queue scores (playbook guardrails.review.weights)
SIGNALS = ("verifier_cautions", "landmine_caution", "low_confidence", "grader_zero_relevant",
           "critic_citation_unsupported", "judge_flag", "budget_exhausted", "revisions_all_duplicates")


class SandboxViolation(RuntimeError):
    pass


# --- process-wide run configuration (evaluation only) -------------------------------------

_lock = threading.Lock()
_config: dict[str, Any] = {"sandbox": False, "disabled": frozenset()}


@contextlib.contextmanager
def run_config(*, sandbox: bool = False, disabled: frozenset[str] | set[str] = frozenset()) -> Iterator[None]:
    disabled = frozenset(disabled)
    unknown = disabled - set(ABLATABLE)
    if unknown:
        raise ValueError(f"unknown guardrail(s) to disable: {', '.join(sorted(unknown))}")
    if disabled and not sandbox:
        raise ValueError("guardrails can only be switched off inside the evaluation sandbox")
    with _lock:
        prev = dict(_config)
        _config.update(sandbox=sandbox or prev["sandbox"], disabled=disabled | prev["disabled"])
    try:
        yield
    finally:
        with _lock:
            _config.clear()
            _config.update(prev)


def sandboxed() -> bool:
    return bool(_config["sandbox"])


def disabled(name: str) -> bool:
    return name in _config["disabled"]


def forbid_in_sandbox(action: str) -> None:
    """Called by every side effect an evaluation must never cause."""
    if sandboxed():
        raise SandboxViolation(f"{action} is not allowed in an evaluation sandbox")


# --- the event log ------------------------------------------------------------------------

class GuardrailLog:
    """Records guardrail decisions of one run in SQLite (`guardrail_events`), linked to spans."""

    def __init__(self, db, run_id: str, masker=None):
        self.db, self.run_id, self.masker = db, run_id, masker
        self.events: list[dict] = []
        self.span_id: str | None = None   # set by the span wrapper while a node runs

    def emit(self, name: str, decision: str, reason: str, *, signal: str | None = None,
             data: Any = None) -> dict:
        g = BY_NAME[name]   # KeyError: every guardrail must be registered
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r}")
        if signal is not None and signal not in SIGNALS:
            raise ValueError(f"unknown signal {signal!r}")
        if self.masker is not None:
            reason = self.masker.text(reason)
            data = self.masker.value(data) if data is not None else None
        ev = {"seq": len(self.events) + 1, "span_id": self.span_id, "guardrail": name, "stage": g.stage,
              "kind": g.kind, "decision": decision, "signal": signal, "reason": reason[:400], "data": data}
        self.events.append(ev)
        if self.db is not None:
            self.db.add_guardrail_event(self.run_id, **ev)
        return ev

    def signals(self) -> list[str]:
        return [e["signal"] for e in self.events if e["signal"]]


def stage_order_ok(events: list[dict]) -> bool:
    """True if the events run pre, then during, then post (the pipeline's sequencing)."""
    ranks = [STAGES.index(e["stage"]) for e in events]
    return ranks == sorted(ranks)


# --- pure checks added in Phase 6 (the existing ones are wrapped where they run) ------------

def confidence_gate(confidence: float | None, min_confidence: float) -> tuple[str, str]:
    """(decision, reason). Missing or malformed confidence escalates; it is never 'safe'."""
    if confidence is None:
        return "flag", "no usable confidence was returned; escalated"
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "flag", f"confidence {confidence!r} is not a number; escalated"
    if not 0.0 <= c <= 1.0:
        return "flag", f"confidence {c} is outside 0..1; escalated"
    if c < min_confidence:
        return "flag", f"confidence {c:.2f} is below {min_confidence:.2f}"
    return "pass", f"confidence {c:.2f}"


def critic_citation_problems(reviews: list[dict], check_verdicts: dict[str, dict[str, str]],
                             n_lessons: int) -> list[str]:
    """Problems with applied critic actions (keep is ignored).

    `reviews`: [{node_id, action, cite, reason}]; `check_verdicts`: node_id -> {check name: verdict}.
    A cited check must not have passed; a cited lesson must exist; and one identical reason and cite
    on three or more candidates is generic, not a review of each.
    """
    problems = []
    acted = [r for r in reviews if r.get("action") in ("lower_minor", "lower_major", "veto")]
    for r in acted:
        cite = str(r.get("cite", ""))
        if cite.startswith("check:"):
            v = check_verdicts.get(str(r.get("node_id")), {}).get(cite[6:])
            if v is None:
                problems.append(f"{r.get('node_id')}: cites {cite}, which was not run on this candidate")
            elif v == "CLEAR":   # the check passed, so it cannot support added caution
                problems.append(f"{r.get('node_id')}: cites {cite}, which passed")
        elif cite.startswith("lesson:"):
            try:
                k = int(cite[7:])
            except ValueError:
                k = 0
            if not 1 <= k <= n_lessons:
                problems.append(f"{r.get('node_id')}: cites {cite}, but only {n_lessons} lesson(s) were shown")
    if len(acted) >= 3:
        keys = {(" ".join(str(r.get("reason", "")).lower().split()), r.get("cite")) for r in acted}
        if len(keys) == 1:
            problems.append(f"the same reason and cite ({acted[0].get('cite')}) on all {len(acted)} candidates")
    return problems


def brief_number_problems(brief_numbers: dict, tool_numbers: dict) -> list[str]:
    """Hard check: each number in the brief equals the verified tool number."""
    out = []
    for k, v in tool_numbers.items():
        if k not in brief_numbers:
            out.append(f"{k} missing from the brief")
        elif json.dumps(brief_numbers[k], default=str) != json.dumps(v, default=str):
            out.append(f"{k}: brief {brief_numbers[k]!r} != tool {v!r}")
    return out


# --- the deterministic review risk score -------------------------------------------------

def risk_score(signals: dict[str, int], review_cfg: dict) -> tuple[int, str, list[tuple[str, int]]]:
    """(score, tier, [(signal, points)] highest first). `signals` maps a signal to its count."""
    w = review_cfg["weights"]
    parts = []
    for s in SIGNALS:
        n = int(signals.get(s, 0))
        if n <= 0:
            continue
        if s == "verifier_cautions":
            n = min(n, review_cfg["max_verifier_cautions"])
        else:
            n = 1
        parts.append((s, n * w[s]))
    score = sum(p for _, p in parts)
    tiers = review_cfg["tiers"]
    tier = ("escalated" if score >= tiers["escalated_at"] else
            "look_closely" if score >= tiers["look_closely_at"] else "standard")
    return score, tier, sorted(parts, key=lambda x: (-x[1], SIGNALS.index(x[0])))
