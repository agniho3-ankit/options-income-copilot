"""Tree-of-Thought explorer: generator, critic, controller, state. Synthetic data, no network, no LLM."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta

import pytest
import yaml

from agent_fakes import AS_OF, EXP, TICKER, FakeSources
from options_copilot.agent_tools import ToolBox
from options_copilot.db import Database
from options_copilot.events import extract_event_dates
from options_copilot.llm import LLMError, LLMResponse
from options_copilot.mask import Masker
from options_copilot.models import Strategy, Verdict
from options_copilot.playbook import PlaybookError, load_playbook
from options_copilot.tot import (NO_TRADE, SELECTED, Node, Spec, Status, Tree, allowed_strategies,
                                 alternatives, apply_critique, explore, fallback_deltas, render_tree, score_trade)
from options_copilot.verifier import InsiderActivity, MarketFacts, check_event_window

PV = 87_654.0


def pb_with(tmp_path, repo_root, **overrides):
    """The real playbook with dotted-key overrides (e.g. tot__beam_width=2)."""
    raw = yaml.safe_load((repo_root / "playbook.yaml").read_text())
    for key, value in overrides.items():
        node = raw
        *parents, leaf = key.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    path = tmp_path / "playbook.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_playbook(path)


def run(pb, sources=None, llm=None, **kw):
    db = Database(":memory:")
    box = ToolBox(TICKER, sources or FakeSources(), pb, AS_OF, PV)
    return explore(box, pb, llm, db, Masker(PV), **kw), box, db


class RoutingLLM:
    """Answers generator calls by level and critic calls by a handler; counts calls."""
    name = "routing"

    def __init__(self, strategy=None, expiry=None, delta=None, critic=None, fail=False):
        self.handlers = {"strategy": strategy, "expiry": expiry, "target_delta": delta}
        self.critic, self.fail, self.calls = critic, fail, 0

    def chat(self, messages, *, json_schema=None, tools=None):
        self.calls += 1
        if self.fail:
            raise LLMError("ollama down")
        user = json.loads(messages[-1]["content"])
        if "decide" in user:
            h = self.handlers[user["decide"]]
            out = h(user) if h else {"candidates": []}
        else:
            out = self.critic(user) if self.critic else {"reviews": []}
        return LLMResponse(json.dumps(out), self.name, [], 3)


# --- state -----------------------------------------------------------------------------------

def test_tree_json_round_trips(pb):
    result, _, db = run(pb)
    text = result.tree.to_json()
    again = Tree.from_json(text)
    assert again.to_json() == text
    assert all(isinstance(n.status, Status) and isinstance(n.spec, Spec) for n in again.nodes)
    # the state manager stored every node under the run
    stored = db.tot_nodes(result.run_id)
    assert {n["id"] for n in stored} == {n.id for n in result.tree.nodes}
    assert db.run(result.run_id)["kind"] == "tot" and db.run(result.run_id)["outcome"] == SELECTED


def test_node_from_dict_keeps_every_field():
    n = Node("n3", "n1", 2, Spec("iron_condor", "2026-12-14"), "why", 0.5, Status.PRUNED, "beam", "llm", 1,
             {"x": [1, 2]})
    assert Node.from_dict(json.loads(json.dumps(n.to_dict()))) == n


# --- critic: deterministic prune and score ---------------------------------------------------

class LowOISources(FakeSources):
    """Open interest at the 89 put is below the playbook floor."""

    def chain(self, ticker, expiry):
        quotes, flow = super().chain(ticker, expiry)
        return [replace(q, open_interest=100) if q.strike == 89 else q for q in quotes], flow


def test_verifier_failing_leaf_is_pruned_with_reason(pb):
    result, _, db = run(pb, LowOISources())
    failed = [n for n in result.tree.nodes if n.depth == 3 and "open_interest" in n.prune_reason]
    assert failed and all(n.status is Status.PRUNED for n in failed)
    assert "below 500" in failed[0].prune_reason
    stored = {n["id"]: n for n in db.tot_nodes(result.run_id)}
    assert stored[failed[0].id]["prune_reason"] == failed[0].prune_reason
    # the brief can list it as a losing alternative with its reason
    alt = {a["node"]: a for a in alternatives(result)}
    assert alt[failed[0].id]["reason"] == failed[0].prune_reason


def test_build_error_prunes_branch(pb):
    class HighSupport(FakeSources):
        def ticker_data(self, ticker):
            return replace(super().ticker_data(ticker), support=81.0)  # no put in band below 81
    result, _, _ = run(pb, HighSupport())
    puts = [n for n in result.tree.nodes if n.depth == 3 and n.spec.strategy == "put_credit_spread"]
    assert puts and all(n.prune_reason.startswith("build: no put") for n in puts)


def test_score_is_deterministic_and_uses_tool_numbers(pb):
    box = ToolBox(TICKER, FakeSources(), pb, AS_OF, PV)
    pid, _ = box.build_and_verify(Strategy.IRON_CONDOR, EXP, 0.16)
    prop = box.proposals[pid]
    a = score_trade(pb, prop.proposal, prop.numbers, 100.0, 93.0, 107.0, 60.0)
    b = score_trade(pb, prop.proposal, prop.numbers, 100.0, 93.0, 107.0, 60.0)
    assert a == b
    score, comps = a
    assert set(comps) == set(pb["tot.weights"]) and 0 <= score <= 1
    assert score == round(sum(pb["tot.weights"][k] * v for k, v in comps.items()), 4)
    # unknown inputs are never rewarded
    assert score_trade(pb, prop.proposal, prop.numbers, 100.0, None, 107.0, None)[1]["sr_distance"] == 0
    # two identical searches give identical trees
    assert run(pb)[0].tree.to_json() == run(pb)[0].tree.to_json()


def test_llm_critic_cannot_raise_or_revive():
    leaf = Node("n5", "n2", 3, Spec(), score=0.5, status=Status.LEAF)
    pruned = Node("n6", "n2", 3, Spec(), status=Status.PRUNED, prune_reason="pop: too low")
    nodes = {"n5": leaf, "n6": pruned}
    cites = {"check:iv_rank"}
    notes = apply_critique(nodes, [
        {"node_id": "n5", "action": "raise", "reason": "great trade", "cite": "check:iv_rank", "score": 0.99},
        {"node_id": "n5", "action": "keep", "reason": "", "cite": "", "score": 0.99},
        {"node_id": "n6", "action": "keep", "reason": "revive it", "cite": "check:iv_rank"},
        {"node_id": "n6", "action": "lower_minor", "reason": "x", "cite": "check:iv_rank"},
    ], {"minor": 0.9, "major": 0.7}, cites)
    assert leaf.score == 0.5 and leaf.status is Status.LEAF
    assert pruned.status is Status.PRUNED and pruned.score is None and pruned.prune_reason == "pop: too low"
    assert len(notes) == 3
    # even a misconfigured penalty above 1 cannot raise the score
    apply_critique(nodes, [{"node_id": "n5", "action": "lower_major", "reason": "IV thin", "cite": "check:iv_rank"}],
                   {"minor": 1.5, "major": 1.5}, cites)
    assert leaf.score == 0.5
    # an uncited lower or veto is ignored; a cited one applies
    apply_critique(nodes, [{"node_id": "n5", "action": "veto", "reason": "no", "cite": "made-up"}],
                   {"minor": 0.9, "major": 0.7}, cites)
    assert leaf.status is Status.LEAF
    apply_critique(nodes, [{"node_id": "n5", "action": "lower_minor", "reason": "IV thin", "cite": "check:iv_rank"}],
                   {"minor": 0.9, "major": 0.7}, cites)
    assert leaf.score == 0.45 and leaf.detail["critic"]["score_before"] == 0.5


def test_llm_critic_in_search_only_lowers_or_vetoes(pb):
    base, _, _ = run(pb)
    base_scores = {n.id: n.score for n in base.tree.nodes if n.score is not None}

    def critic(user):
        ids = [c["node_id"] for c in user["candidates"]]
        return {"reviews": [{"node_id": ids[0], "action": "veto", "reason": "lesson says avoid", "cite": "lesson:1"}]
                + [{"node_id": i, "action": "raise", "reason": "better", "cite": "lesson:1"} for i in ids[1:]]
                + [{"node_id": "n1", "action": "lower_major", "reason": "interior", "cite": "lesson:1"}]}
    result, _, _ = run(pb, llm=RoutingLLM(critic=critic))
    for n in result.tree.nodes:
        if n.score is not None:
            assert n.score <= base_scores[n.id]
    vetoed = [n for n in result.tree.nodes if n.prune_reason.startswith("critic veto")]
    assert len(vetoed) == 1 and vetoed[0].status is Status.PRUNED
    assert result.selected is not None and result.selected.id != vetoed[0].id


# --- generator --------------------------------------------------------------------------------

def test_out_of_playbook_candidates_are_rejected(pb):
    llm = RoutingLLM(
        strategy=lambda u: {"candidates": [{"strategy": "naked_call", "rationale": "yolo"},
                                           {"strategy": "iron_condor", "rationale": "range-bound"},
                                           {"strategy": "iron_condor", "rationale": "again"}]},
        expiry=lambda u: {"candidates": [{"expiry": "2027-06-18", "rationale": "far"},
                                         {"expiry": EXP.isoformat(), "rationale": "45 DTE"}]},
        delta=lambda u: {"candidates": [{"target_delta": 0.45, "rationale": "rich"},
                                        {"target_delta": "abc", "rationale": "?"},
                                        {"target_delta": 0.16, "rationale": "inside band"}]})
    result, _, db = run(pb, llm=llm)
    root = result.tree.nodes[0]
    assert [c.spec.strategy for c in result.tree.children("n0")] == ["iron_condor"]
    assert any("naked_call" in r for r in root.detail["rejected_candidates"])
    assert any("duplicate" in r for r in root.detail["rejected_candidates"])
    exp_node = result.tree.children("n1")
    assert [n.spec.expiry for n in exp_node] == [EXP.isoformat()]
    leaves = result.tree.children(exp_node[0].id)
    assert [n.spec.target_delta for n in leaves] == [0.16]
    assert any("outside the iron_condor band" in r for r in exp_node[0].detail["rejected_candidates"])
    assert all(n.source == "llm" for n in result.tree.nodes[1:])
    assert result.stats["candidates_rejected"] >= 5 and result.outcome == SELECTED
    assert any(r["name"] == "validator" for r in db.trace(result.run_id))


def test_fallback_runs_when_llm_fails(pb):
    llm = RoutingLLM(fail=True)
    result, _, db = run(pb, llm=llm)
    assert result.outcome == SELECTED
    assert all(n.source == "fallback" for n in result.tree.nodes[1:])
    assert result.stats["llm_errors"] >= 1 and result.stats["fallback_expansions"] >= 3
    assert any(r["role"] == "error" for r in db.trace(result.run_id))


def test_fallback_when_llm_returns_nothing_valid(pb):
    llm = RoutingLLM(strategy=lambda u: {"candidates": [{"strategy": "straddle", "rationale": "no"}]})
    result, _, _ = run(pb, llm=llm)
    assert {c.source for c in result.tree.children("n0")} == {"fallback"}
    assert "llm returned no valid candidate" in result.tree.nodes[0].detail["generator"]


def test_allowed_strategies_and_fallback_deltas(pb):
    allowed, _ = allowed_strategies(pb, TICKER, 100.0, AS_OF + timedelta(days=70), AS_OF)
    assert Strategy.CASH_SECURED_PUT in allowed
    allowed, why = allowed_strategies(pb, TICKER, 100.0, AS_OF + timedelta(days=20), AS_OF)
    assert Strategy.CASH_SECURED_PUT not in allowed and "spreads-only" in why["cash_secured_put"]
    assert allowed_strategies(pb, "SCHW", 100.0, None, AS_OF)[0] == []
    assert fallback_deltas(0.20, 0.30, 3) == [0.25, 0.225, 0.275]


# --- controller --------------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1, 2, 3])
def test_beam_width_respected_at_every_depth(tmp_path, repo_root, width):
    pb = pb_with(tmp_path, repo_root, tot__beam_width=width)
    result, _, _ = run(pb)
    for depth in (1, 2, 3):
        alive = [n for n in result.tree.nodes if n.depth == depth and n.status is not Status.PRUNED]
        extra = pb["tot.exploration_slots"] if depth == 2 else 0   # the exploration node is extra
        assert len(alive) <= width + extra
    assert any("beam:" in n.prune_reason for n in result.tree.nodes) or width == 3


def test_build_budget_stops_the_search(tmp_path, repo_root):
    pb = pb_with(tmp_path, repo_root, tot__max_builds=2)
    result, _, _ = run(pb)
    assert result.stats["builds"] == 2
    assert "build budget (2)" in result.stats["stopped_by"]
    assert any(n.prune_reason.startswith("search stopped") for n in result.tree.nodes if n.depth == 3)
    assert result.outcome in (SELECTED, NO_TRADE)


def test_llm_budget_stops_the_search(tmp_path, repo_root):
    pb = pb_with(tmp_path, repo_root, tot__max_llm_calls=2)
    llm = RoutingLLM()
    result, _, _ = run(pb, llm=llm)
    assert llm.calls == 2 and result.stats["llm_calls"] == 2
    assert "LLM call budget (2)" in result.stats["stopped_by"]
    assert result.stats["builds"] == 0 and result.outcome == NO_TRADE


def test_all_branches_pruned_gives_no_trade(pb):
    sources = FakeSources(insider=InsiderActivity(bought=100, sold=10_000, filings=5))
    result, box, _ = run(pb, sources)
    assert result.outcome == NO_TRADE and result.selected is None and box.accepted is None
    assert any("insider" in r for r in result.reasons)
    assert not [n for n in result.tree.nodes if n.status in (Status.LEAF, Status.SELECTED)]


def test_no_expiry_before_earnings_gives_no_trade(pb):
    class EarningsSoon(FakeSources):
        def ticker_data(self, ticker):
            d = super().ticker_data(ticker)
            return replace(d, profile=replace(d.profile, earnings_date=AS_OF + timedelta(days=20)))
    result, _, _ = run(pb, EarningsSoon())
    assert result.outcome == NO_TRADE and result.stats["builds"] == 0
    assert any("before earnings" in r for r in result.reasons)


@pytest.mark.parametrize("verdict", ["BLOCK", "INSUFFICIENT_EVIDENCE"])
def test_landmine_stops_search_before_branching(pb, verdict):
    sources = FakeSources(landmine_verdict=verdict)
    llm = RoutingLLM(fail=True)
    result, _, _ = run(pb, sources, llm=llm)
    assert result.outcome == verdict and len(result.tree.nodes) == 1
    assert result.stats["builds"] == 0 and llm.calls == 0 and sources.landmine_calls == 1
    assert result.tree.nodes[0].prune_reason.startswith(f"landmine {verdict}")


def test_landmine_runs_once_per_search(pb):
    sources = FakeSources()
    run(pb, sources)
    assert sources.landmine_calls == 1


def test_render_tree_shows_scores_and_reasons(pb):
    result, _, _ = run(pb, LowOISources())
    text = render_tree(result.tree, result.stats)
    assert "SELECTED score" in text and "PRUNED: open_interest" in text and "builds" in text


def test_depth_limit_below_three_never_trades(tmp_path, repo_root):
    pb = pb_with(tmp_path, repo_root, tot__max_depth=2)
    result, _, _ = run(pb)
    assert result.outcome == NO_TRADE and result.stats["builds"] == 0


def test_tot_weights_are_validated(tmp_path, repo_root):
    with pytest.raises(PlaybookError, match="tot.weights must sum to 1"):
        pb_with(tmp_path, repo_root, tot__weights={"pop": 0.5, "return_on_risk": 0.5, "risk_headroom": 0.5,
                                                   "iv_rank": 0, "sr_distance": 0})
    with pytest.raises(PlaybookError, match="critic_penalty.major"):
        pb_with(tmp_path, repo_root, tot__critic_penalty={"minor": 0.7, "major": 0.9})


# --- event dates vs expiry ------------------------------------------------------------------

def test_extract_event_dates_reads_dates_next_to_event_words():
    text = ("On October 21, 2026, the Company entered into an agreement. The amendment becomes effective "
            "on December 1, 2026. The record date is 2027-01-15; holders vote at a special meeting on "
            "02/20/2027. Revenue grew in March 3, 2026 alone.")
    got = extract_event_dates(text)
    assert (date(2026, 12, 1), "effective") in got
    assert (date(2027, 1, 15), "record date") in got
    assert (date(2027, 2, 20), "special meeting") in got
    assert all(d != date(2026, 10, 21) and d != date(2026, 3, 3) for d, _ in got)


def _facts(events):
    return MarketFacts(as_of=AS_OF, spot=100.0, portfolio_value=PV, event_dates=tuple(events))


def test_event_inside_trade_window_blocks(pb):
    box = ToolBox(TICKER, FakeSources(), pb, AS_OF, PV)
    pid, _ = box.build_and_verify(Strategy.IRON_CONDOR, EXP, 0.16)
    p = box.proposals[pid].proposal
    inside = check_event_window(p, _facts([(EXP - timedelta(days=5), "effective (8-K)")]))
    assert inside.verdict is Verdict.BLOCK and "inside the trade window" in inside.reason
    on_expiry = check_event_window(p, _facts([(EXP, "closing (8-K)")]))
    assert on_expiry.verdict is Verdict.BLOCK
    after = check_event_window(p, _facts([(EXP + timedelta(days=1), "effective (8-K)")]))
    assert after.verdict is Verdict.CAUTION and "after expiry" in after.reason
    past = check_event_window(p, _facts([(AS_OF - timedelta(days=1), "effective (8-K)")]))
    assert past.verdict is Verdict.CLEAR
    assert check_event_window(p, _facts([])).verdict is Verdict.CLEAR


class EventSources(FakeSources):
    def __init__(self, event_day, **kw):
        super().__init__(**kw)
        self.event_day = event_day

    def landmine(self, ticker, context):
        lm = super().landmine(ticker, context)
        lm["event_dates"] = [{"date": self.event_day.isoformat(), "event": "effective", "source": "8-K 2026-10-01 Item 1.01"}]
        return lm


def test_cited_event_inside_window_prunes_every_leaf(pb):
    result, _, _ = run(pb, EventSources(EXP - timedelta(days=3)))
    assert result.outcome == NO_TRADE
    assert any("event_window" in r for r in result.reasons)


def test_cited_event_after_expiry_is_only_caution(pb):
    result, box, _ = run(pb, EventSources(EXP + timedelta(days=30)))
    assert result.outcome == SELECTED
    checks = {c.name: c for c in box.accepted[1].checks}
    assert checks["event_window"].verdict is Verdict.CAUTION


def test_chain_is_fetched_once_per_expiry(pb):
    class Counting(FakeSources):
        n = 0

        def chain(self, ticker, expiry):
            Counting.n += 1
            return super().chain(ticker, expiry)
    run(pb, Counting())
    assert Counting.n <= 2  # one per distinct expiry the search touched


def test_tot_brief_lists_alternatives(pb):
    from options_copilot.briefs import make_brief
    from options_copilot.cli import _render_brief
    from options_copilot.react import AgentResult
    from options_copilot.tot import summary_text

    result, box, db = run(pb, LowOISources())
    brief_id = make_brief(AgentResult(result.run_id, "accepted", summary_text(result)), box, db, AS_OF,
                          extra={"planner": "tot", "alternatives": alternatives(result)})
    b = db.brief(brief_id)
    assert b["status"] == "pending" and b["payload"]["planner"] == "tot"
    text = _render_brief(b)
    assert "alternatives considered" in text and "open_interest" in text
    assert result.selected.path() in summary_text(result)


# --- depth-2 mitigation: deterministic expiry score + exploration slot ----------------------

EXP2 = AS_OF + timedelta(days=38)


class TwoExpiries(FakeSources):
    """A second listed expiry (38 DTE) with a thinner market than the 45-DTE one."""

    def expiries(self, ticker):
        return sorted({*super().expiries(ticker), EXP2})

    def chain(self, ticker, expiry):
        if expiry == EXP2:
            quotes, flow = super().chain(ticker, EXP)
            return [replace(q, expiry=EXP2, open_interest=600) for q in quotes], flow
        return super().chain(ticker, expiry)


def test_expiry_score_is_deterministic(pb):
    from options_copilot.tot import score_expiry
    quotes = FakeSources().chain(TICKER, EXP)[0]
    earnings = AS_OF + timedelta(days=70)
    a = score_expiry(pb, EXP, AS_OF, earnings, quotes, 100.0)
    assert a == score_expiry(pb, EXP, AS_OF, earnings, quotes, 100.0)
    score, c = a
    assert c["dte_fit"] == 1.0 and c["earnings_clearance"] == round(min(25 / 30, 1), 4)
    assert c["liquidity"] == round(1000 / (500 * 4), 4)        # median OI of in-band strikes / 2,000
    assert score == round(0.5 * c["dte_fit"] + 0.25 * c["earnings_clearance"] + 0.25 * c["liquidity"], 4)
    edge = score_expiry(pb, AS_OF + timedelta(days=30), AS_OF, None, quotes, 100.0)[1]
    assert edge["dte_fit"] == 0.0 and edge["earnings_clearance"] == 0.0   # unknown earnings never rewarded
    assert score_expiry(pb, EXP, AS_OF, earnings, [], 100.0)[1]["liquidity"] == 0.0
    thin = [replace(q, open_interest=10) if 0.12 <= abs(q.delta) <= 0.30 else q for q in quotes]
    assert score_expiry(pb, EXP, AS_OF, earnings, thin, 100.0)[1]["liquidity"] == round(10 / 2000, 4)


def test_depth2_ranks_by_expiry_score_and_explores_a_second_expiry(pb):
    result, _, _ = run(pb, TwoExpiries())
    depth2 = [n for n in result.tree.nodes if n.depth == 2]
    alive = [n for n in depth2 if n.status is not Status.PRUNED]
    assert len(alive) == pb["tot.beam_width"] + pb["tot.exploration_slots"]
    assert {n.spec.expiry for n in alive} == {EXP.isoformat(), EXP2.isoformat()}
    explored = [n for n in alive if n.detail.get("exploration")]
    assert len(explored) == 1 and explored[0].spec.expiry == EXP2.isoformat()
    best = [n for n in alive if not n.detail.get("exploration")]
    assert all(n.detail["expiry_score"] >= explored[0].detail["expiry_score"] for n in best)
    priced = {n.spec.expiry for n in result.tree.nodes if n.depth == 3 and "numbers" in n.detail}
    assert EXP2.isoformat() in priced            # the exploration branch was actually built


def test_exploration_can_be_switched_off(tmp_path, repo_root):
    pb0 = pb_with(tmp_path, repo_root, tot__exploration_slots=0)
    result, _, _ = run(pb0, TwoExpiries())
    alive = [n for n in result.tree.nodes if n.depth == 2 and n.status is not Status.PRUNED]
    assert {n.spec.expiry for n in alive} == {EXP.isoformat()}


def test_exhaustive_mode_bounds_the_beam(pb):
    beam, _, _ = run(pb, TwoExpiries())
    full, _, _ = run(pb, TwoExpiries(), exhaustive=True)
    assert full.stats["builds"] > beam.stats["builds"] and full.stats["llm_calls"] == 0
    assert full.selected.score >= beam.selected.score
    assert not any(n.prune_reason.startswith("beam:") for n in full.tree.nodes if n.depth < 3)


def test_expiry_weights_are_validated(tmp_path, repo_root):
    with pytest.raises(PlaybookError, match="tot.expiry_weights must sum to 1"):
        pb_with(tmp_path, repo_root, tot__expiry_weights={"dte_fit": 0.5, "earnings_clearance": 0.5, "liquidity": 0.5})


def test_depth2_keeps_every_strategy_and_a_second_expiry(pb):
    result, _, _ = run(pb, TwoExpiries())
    strategies = {n.spec.strategy for n in result.tree.nodes if n.depth == 1 and n.status is not Status.PRUNED}
    alive = [n for n in result.tree.nodes if n.depth == 2 and n.status is not Status.PRUNED]
    assert {n.spec.strategy for n in alive} == strategies          # no strategy loses all its expiries
    assert len({n.spec.expiry for n in alive}) == 2
