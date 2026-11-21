"""Phase 5 multi-agent workflow. Synthetic data, fake LLM, no network (conftest blocks sockets)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace

import pytest
import yaml

from agent_fakes import AS_OF, EXP, TICKER, FakeSources
from options_copilot.db import Database
from options_copilot.llm import LLMError, LLMResponse
from options_copilot.mask import Masker
from options_copilot.multi_agent import privacy
from options_copilot.multi_agent.graph import (GRADER_SYSTEM, REWRITE_SYSTEM, WRITER_SYSTEM, build_graph, merge_research,
                                               route)
from options_copilot.multi_agent.runner import run_multi_agent
from options_copilot.multi_agent.state import exhausted, from_json, new_state, to_json
from options_copilot.playbook import load_playbook


PV = 87_654.0
END = "__end__"


def pb_with(tmp_path, repo_root, **overrides):
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


class RoleLLM:
    """Answers each role by its prompt. CrewAI prompts get a 'Final Answer:' reply."""
    name = "role-fake"

    def __init__(self, relevant=2, rewrite="material agreements financing officer changes",
                 critic=None, prose="An iron condor fits a range-bound stock. Insider selling is the main caution.",
                 fail=()):
        self.relevant, self.rewrite, self.critic, self.prose, self.fail = relevant, rewrite, critic, prose, set(fail)
        self.calls: list[str] = []

    def chat(self, messages, *, json_schema=None, tools=None):
        text = " ".join(str(m.get("content", "")) for m in messages)
        role = ("grader" if GRADER_SYSTEM[:40] in text else "rewrite" if REWRITE_SYSTEM[:40] in text
                else "writer" if WRITER_SYSTEM[:40] in text else "critic" if "critic of a Tree-of-Thought" in text
                else "generator" if "thought generator" in text else "other")
        self.calls.append(role)
        if role in self.fail:
            raise LLMError(f"{role} down")
        if role == "grader":
            ids = re.findall(r'"id": "(C\d+)"', text)
            return LLMResponse(json.dumps({"relevant_ids": ids[:self.relevant]}), self.name, [], 1)
        if role == "rewrite":
            return LLMResponse(json.dumps({"query": self.rewrite}), self.name, [], 1)
        if role == "writer":
            return LLMResponse(self.prose, self.name, [], 1)
        if role == "critic":
            ids = re.findall(r'"node_id": "(n\d+)"', text)
            reviews = self.critic(ids) if self.critic else [{"node_id": i, "action": "keep", "reason": "", "cite": ""}
                                                            for i in ids]
            return LLMResponse("Thought: reviewed.\nFinal Answer: " + json.dumps({"reviews": reviews}), self.name, [], 1)
        if role == "generator":
            return LLMResponse("Thought: done.\nFinal Answer: " + json.dumps({"candidates": []}), self.name, [], 1)
        return LLMResponse("Synthetic lesson: the multi-agent run found no trade that passed every rule.", self.name, [], 1)


class EvidenceSources(FakeSources):
    """Landmine results with evidence and coverage, optionally different for a rewritten query."""

    def __init__(self, first="CLEAR", second="CLEAR", cov1=True, cov2=True, **kw):
        super().__init__(**kw)
        self.first, self.second, self.cov1, self.cov2 = first, second, cov1, cov2
        self.queries = []

    def landmine(self, ticker, context, query=None):
        self.landmine_calls += 1
        self.queries.append(query)
        verdict, cov = (self.first, self.cov1) if query is None else (self.second, self.cov2)
        return {"verdict": verdict, "query": query or "default query", "reasons": ["Synthetic reasons."],
                "coverage": {"ok": cov, "reason": "synthetic"},
                "citations": [{"form": "8-K", "date": "2026-10-01", "item": "Item 8.01", "url": "https://example.com/8k"}]
                if cov else [],
                "evidence": [{"id": f"C{i}", "form": "8-K", "date": "2026-10-01", "item": "Item 8.01", "score": 0.4,
                              "excerpt": "Synthetic excerpt."} for i in range(1, 4)] if cov else []}


def run(pb, sources=None, llm=None, **kw):
    db = Database(":memory:")
    r = run_multi_agent(TICKER, sources=sources or FakeSources(), pb=pb, as_of=AS_OF, llm=llm, db=db,
                        masker=Masker(PV), portfolio_value=PV, **kw)
    return r, db


def state(pb, **kw):
    s = new_state("run-x", TICKER, AS_OF.isoformat(), dict(pb["multi_agent"]), started_at=0.0)
    s.update(kw)
    return s


# --- supervisor routing (deterministic) ------------------------------------------------------

def test_route_each_verdict_path(pb):
    now = 1.0
    assert route(state(pb), now) == "researcher"
    for v in ("BLOCK", "INSUFFICIENT_EVIDENCE"):
        assert route(state(pb, research={"verdict": v}), now) == END
    clear = {"verdict": "CLEAR"}
    assert route(state(pb, research=clear), now) == "strategist"
    assert route(state(pb, research={"verdict": "CAUTION"}), now) == "strategist"
    assert route(state(pb, research=clear, strategist_done=True, pending_review=["n9"]), now) == "risk_critic"
    assert route(state(pb, research=clear, strategist_done=True, revise=["n4"], critic_rounds=1), now) == "strategist"
    assert route(state(pb, research=clear, strategist_done=True, revise=["n4"], critic_rounds=2), now) == "verifier_gate"
    assert route(state(pb, research=clear, strategist_done=True, pending_review=["n9"], critic_enabled=False), now) \
        == "verifier_gate"
    assert route(state(pb, research=clear, strategist_done=True), now) == "verifier_gate"
    assert route(state(pb, research=clear, gate_done=True), now) == END


def test_route_exhausted_budget_goes_to_the_gate(pb):
    s = state(pb, research={"verdict": "CLEAR"})
    s["budgets"]["builds"] = s["budgets"]["max_builds"]
    assert exhausted(s, 1.0) and route(s, 1.0) == "verifier_gate"
    s = state(pb, research={"verdict": "CLEAR"})
    assert route(s, s["budgets"]["max_wall_seconds"] + 1.0) == "verifier_gate"


def test_no_path_reaches_the_brief_without_the_verifier(pb):
    from options_copilot.mcp_server import MCPBridge, MCPContext, build_server
    from options_copilot.multi_agent.graph import Runtime
    ctx = MCPContext(FakeSources(), pb, AS_OF, PV, Database(":memory:"))
    with MCPBridge(build_server(ctx)) as bridge:
        g = build_graph(Runtime(box=ctx.box(TICKER), bridge=bridge, pb=pb, db=ctx.db, masker=Masker(PV), llm=None)).get_graph()
    edges = [(e.source, e.target) for e in g.edges]
    assert {s for s, t in edges if t == "brief_writer"} == {"verifier_gate"}
    # and no path at all from START to brief_writer once the gate is removed
    adj: dict[str, set] = {}
    for s, t in edges:
        if "verifier_gate" not in (s, t):
            adj.setdefault(s, set()).add(t)
    seen, todo = set(), ["__start__"]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += adj.get(n, [])
    assert "brief_writer" not in seen and "supervisor" in seen


# --- researcher: grader, one rewrite, coverage guard ------------------------------------------

def test_merge_rule_cures_coverage_but_never_loosens():
    cov_fail = {"verdict": "INSUFFICIENT_EVIDENCE", "coverage": {"ok": False}}
    assert merge_research(cov_fail, {"verdict": "CLEAR", "coverage": {"ok": True}})[0] == "CLEAR"
    assert merge_research(cov_fail, {"verdict": "INSUFFICIENT_EVIDENCE", "coverage": {"ok": False}})[0] \
        == "INSUFFICIENT_EVIDENCE"
    assert merge_research({"verdict": "BLOCK", "coverage": {"ok": True}},
                          {"verdict": "CLEAR", "coverage": {"ok": True}})[0] == "BLOCK"
    assert merge_research({"verdict": "CAUTION", "coverage": {"ok": True}},
                          {"verdict": "CLEAR", "coverage": {"ok": True}})[0] == "CAUTION"
    llm_fail = {"verdict": "INSUFFICIENT_EVIDENCE", "coverage": {"ok": True}}   # not a coverage miss
    assert merge_research(llm_fail, {"verdict": "CLEAR", "coverage": {"ok": True}})[0] == "INSUFFICIENT_EVIDENCE"


def test_rewrite_happens_at_most_once(pb):
    sources = EvidenceSources()
    r, db = run(pb, sources, llm=RoleLLM(relevant=0), use_agents=False)
    research = r.state["research"]
    assert research["rewrites"] == 1 and len(research["attempts"]) == 2
    assert sources.queries.count(None) == 1 and len([q for q in sources.queries if q]) == 1
    assert any(t["name"] == "researcher.rewrite" and t["role"] == "tool" for t in db.trace(r.run_id))


def test_no_rewrite_when_enough_chunks_are_relevant(pb):
    r, _ = run(pb, EvidenceSources(), llm=RoleLLM(relevant=3), use_agents=False)
    assert r.state["research"]["rewrites"] == 0


def test_coverage_guard_unchanged(pb):
    sources = EvidenceSources(first="INSUFFICIENT_EVIDENCE", second="INSUFFICIENT_EVIDENCE", cov1=False, cov2=False)
    r, _ = run(pb, sources, llm=RoleLLM(relevant=0), use_agents=False)
    assert r.outcome == "INSUFFICIENT_EVIDENCE" and r.state["route_log"] == ["researcher", "END"]
    assert r.state["brief_id"] is None and r.builds == 0
    cured = EvidenceSources(first="INSUFFICIENT_EVIDENCE", second="CAUTION", cov1=False, cov2=True)
    r, _ = run(pb, cured, llm=RoleLLM(relevant=0), use_agents=False)
    assert r.state["research"]["verdict"] == "CAUTION" and r.outcome == "SELECTED"


def test_block_escalates_before_any_strategist(pb):
    r, db = run(pb, FakeSources(landmine_verdict="BLOCK"), llm=RoleLLM())
    assert r.outcome == "BLOCK" and "strategist" not in r.state["route_log"]
    assert r.state["research"]["escalation"].startswith("BLOCK") and r.builds == 0
    assert db.tot_nodes(r.state.get("tot_run_id") or "none") == []


# --- strategist <-> risk critic ---------------------------------------------------------------

def veto_first(ids):
    return [{"node_id": ids[0], "action": "veto", "reason": "a lesson warns about this", "cite": "lesson:1"}] + \
           [{"node_id": i, "action": "keep", "reason": "", "cite": ""} for i in ids[1:]]


class DenseSources(FakeSources):
    """A strike every point, so a revised delta reaches strikes the first round did not try."""

    def chain(self, ticker, expiry):
        from options_copilot.calculators import Quote
        from options_copilot.models import OptionType
        if expiry != EXP:
            return [], 1.2
        q = []
        for k in range(80, 100):
            d = -round(0.03 + (k - 80) * 0.02, 2)
            q.append(Quote(OptionType.PUT, k, EXP, bid=round(abs(d) * 8, 2), ask=round(abs(d) * 8 + 0.1, 2), iv=0.30,
                           delta=d, open_interest=1000))
        for k in range(101, 121):
            d = round(0.43 - (k - 101) * 0.02, 2)
            q.append(Quote(OptionType.CALL, k, EXP, bid=round(d * 8, 2), ask=round(d * 8 + 0.1, 2), iv=0.30, delta=d,
                           open_interest=1000))
        return q, 1.2


def test_critic_loop_stops_at_the_round_cap(pb):
    r, _ = run(pb, DenseSources(), llm=RoleLLM(critic=veto_first))
    assert r.state["critic_rounds"] == pb["multi_agent.max_critic_rounds"] == 2
    log = r.state["route_log"]
    assert log.count("risk_critic") == 2 and log.count("strategist") == 2 and log[-1] == "verifier_gate"
    assert any(m["pattern"] == "two-way" for m in r.state["messages"])
    assert any(m["pattern"] == "brainstorming" for m in r.state["messages"])


def test_round_cap_zero_skips_the_critic(tmp_path, repo_root):
    pb0 = pb_with(tmp_path, repo_root, multi_agent__max_critic_rounds=0)
    r, _ = run(pb0, llm=RoleLLM(critic=veto_first))
    assert "risk_critic" not in r.state["route_log"] and r.state["critic_rounds"] == 0


def test_critic_can_never_raise_a_score(pb):
    base, _ = run(pb, llm=RoleLLM())
    base_scores = {c["node"]: c["score"] for c in base.state["candidates"]}

    def greedy(ids):
        return [{"node_id": i, "action": "raise", "reason": "great", "cite": "lesson:1", "score": 0.99} for i in ids] + \
               [{"node_id": ids[-1], "action": "lower_major", "reason": "thin premium", "cite": "check:iv_rank"}]
    r, _ = run(pb, llm=RoleLLM(critic=greedy))
    for c in r.state["candidates"]:
        if c["node"] in base_scores and c["score"] is not None:
            assert c["score"] <= base_scores[c["node"]]
    assert any(c["critic"] and c["critic"]["action"] == "lower_major" for c in r.state["candidates"])


def test_critic_failure_leaves_scores_unchanged(pb):
    r, _ = run(pb, llm=RoleLLM(fail={"critic"}))
    assert r.outcome == "SELECTED" and any("critic failed" in e for e in r.state["errors"])


# --- budgets -----------------------------------------------------------------------------------

def test_budgets_end_the_graph(tmp_path, repo_root):
    pb3 = pb_with(tmp_path, repo_root, multi_agent__max_builds=3)
    r, _ = run(pb3, llm=RoleLLM(critic=veto_first))
    assert r.state["route_log"][-1] == "verifier_gate" and r.outcome in ("SELECTED", "NO_TRADE")
    assert r.state["critic_rounds"] < 2 or r.builds <= 3 + 9   # the exploration already in flight finishes its tier
    pbt = pb_with(tmp_path, repo_root, multi_agent__max_wall_seconds=10)
    db = Database(":memory:")
    from options_copilot.multi_agent import runner
    orig = runner.new_state
    runner.new_state = lambda *a, **k: orig(*a, **{**k, "started_at": 0.0})
    try:
        r = run_multi_agent(TICKER, sources=FakeSources(), pb=pbt, as_of=AS_OF, llm=None, db=db, masker=Masker(PV),
                            portfolio_value=PV)
    finally:
        runner.new_state = orig
    assert r.state["route_log"] == ["researcher", "verifier_gate"] and r.outcome == "NO_TRADE"
    assert "wall-clock" in r.state["reasons"][0]


# --- verifier gate and brief writer -----------------------------------------------------------

def test_brief_numbers_equal_the_tool_numbers(pb):
    r, db = run(pb, llm=RoleLLM(prose="It fits. It earns 2.5 dollars with a 84% chance. Insider selling is a caution."))
    assert r.outcome == "SELECTED" and r.state["brief_id"]
    b = db.brief(r.state["brief_id"])
    gate = r.state["verifier"]
    assert b["payload"]["numbers"]["credit"] == gate["numbers"]["credit"]
    assert b["payload"]["numbers"]["max_loss"] == gate["numbers"]["max_loss"]
    assert b["payload"]["numbers"]["pop"] == gate["numbers"]["pop"]
    assert [g["strike"] for g in b["payload"]["proposal"]["legs"]] == [g["strike"] for g in gate["legs"]]
    assert not re.search(r"\d", b["payload"]["summary"])          # the LLM's sentence with numbers was dropped
    assert "Insider selling" in b["payload"]["summary"] and b["payload"]["planner"] == "multi_agent"
    assert b["verdict"] in ("CLEAR", "CAUTION") and b["status"] == "pending"   # the owner still approves


def test_all_pruned_gives_no_trade_and_a_lesson(pb):
    from options_copilot.verifier import InsiderActivity

    class Store:
        added = []

        def add_reflection(self, r):
            self.added.append(r)
    store = Store()
    r, _ = run(pb, FakeSources(insider=InsiderActivity(100, 10_000, 5)), llm=RoleLLM(), store=store)
    assert r.outcome == "NO_TRADE" and r.state["brief_id"] is None
    assert r.state["route_log"][-1] == "verifier_gate" and r.state["lesson"] and store.added


# --- state, spans, mermaid ---------------------------------------------------------------------

def test_state_json_round_trips(pb):
    r, _ = run(pb, llm=RoleLLM())
    text = to_json(r.state)
    assert to_json(from_json(text)) == text
    bad = dict(r.state, oops=object())
    with pytest.raises(TypeError):
        to_json(bad)


def test_every_node_is_a_span_with_a_parent(pb):
    r, db = run(pb, llm=RoleLLM(critic=veto_first))
    spans = db.spans(r.run_id)
    assert spans[0]["agent"] == "session" and spans[0]["parent_id"] is None
    agents = {s["agent"] for s in spans}
    assert {"supervisor", "researcher", "strategist", "risk_critic", "verifier_gate", "brief_writer"} <= agents
    assert all(s["parent_id"] for s in spans[1:])
    patterns = {s["pattern"] for s in spans}
    assert {"routing", "one-way", "two-way", "one-way + brainstorming"} <= patterns
    assert sum(s["llm_calls"] for s in spans) == r.llm_calls


def test_mermaid_export(pb, tmp_path):
    out = tmp_path / "agent_graph.md"
    run(pb, graph_out=out)
    text = out.read_text()
    for n in ("supervisor", "researcher", "strategist", "risk_critic", "verifier_gate", "brief_writer", "learner"):
        assert n in text


# --- MCP server -------------------------------------------------------------------------------

def test_mcp_tools_match_direct_calls(pb):
    from options_copilot.mcp_server import MCPBridge, MCPContext, build_server, filings_view, verification_view
    from options_copilot.models import Strategy
    from options_copilot.tot import score_proposal
    ctx = MCPContext(FakeSources(), pb, AS_OF, PV, Database(":memory:"))
    direct = MCPContext(FakeSources(), pb, AS_OF, PV, Database(":memory:")).box(TICKER)
    with MCPBridge(build_server(ctx)) as b:
        assert set(b.list_tools()) == {"get_quote", "get_option_chain", "get_earnings_date", "search_filings",
                                       "search_lessons", "build_and_verify", "score_trade"}
        assert b.call("get_quote", {"ticker": TICKER}) == direct.get_quote()
        assert b.call("get_option_chain", {"ticker": TICKER}) == direct.get_option_chain()
        assert b.call("get_earnings_date", {"ticker": TICKER}) == direct.get_earnings_date()
        assert b.call("search_lessons", {"ticker": TICKER}) == direct.search_lessons()
        direct.search_filings()
        assert b.call("search_filings", {"ticker": TICKER}) == filings_view(direct.landmine_result)
        v = b.call("build_and_verify", {"ticker": TICKER, "strategy": "iron_condor", "expiry": EXP.isoformat(),
                                        "target_delta": 0.16})
        pid, _ = direct.build_and_verify(Strategy.IRON_CONDOR, EXP, 0.16)
        assert v == verification_view(direct, pid)
        s = b.call("score_trade", {"ticker": TICKER, "proposal_id": v["proposal_id"]})
        assert (s["score"], s["components"]) == score_proposal(direct, pb, pid)
        err = b.call("build_and_verify", {"ticker": TICKER, "strategy": "iron_condor", "expiry": "2026-11-30",
                                          "target_delta": 0.16})
        assert err["kind"] == "BuildError"
        assert b.read("tot://run-none") == []
    assert "<<<DATA" in json.dumps(filings_view(direct.landmine_result))   # fetched text stays wrapped


def test_mcp_server_has_no_network_transport_in_use(pb):
    import inspect
    from options_copilot import mcp_server
    src = inspect.getsource(mcp_server)
    assert 'run(transport="stdio")' in src
    assert not re.search(r'transport\s*=\s*"(sse|streamable-http|http)"', src)
    assert src.count(".run(") == 1                       # the stdio server is the only listener of any kind


# --- privacy and LLM safety -------------------------------------------------------------------

def test_telemetry_and_tracing_flags_are_set():
    assert privacy.problems() == []
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true" and os.environ["OTEL_SDK_DISABLED"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false" and os.environ["LANGSMITH_TRACING"] == "false"
    os.environ["LANGSMITH_API_KEY"] = "x"
    os.environ["LANGSMITH_TRACING"] = "true"
    assert privacy.problems()
    privacy.configure()
    assert privacy.problems() == []


def test_env_example_lists_the_flags(repo_root):
    text = (repo_root / ".env.example").read_text()
    for k in ("CREWAI_DISABLE_TELEMETRY=true", "OTEL_SDK_DISABLED=true", "LANGCHAIN_TRACING_V2=false",
              "LANGSMITH_TRACING=false"):
        assert k in text


def test_no_openai_or_default_llm_fallback():
    from crewai import LLM
    from options_copilot.multi_agent.agents import make_agent
    from options_copilot.multi_agent.llm_adapter import CopilotLLM
    for bad in (None, "gpt-4o-mini", LLM(model="gpt-4o-mini")):
        with pytest.raises(RuntimeError, match="refusing"):
            make_agent("X", "goal", "backstory", bad)
    assert os.environ["OPENAI_BASE_URL"].startswith("http://127.0.0.1:9/") and "OPENAI_API_KEY" not in os.environ
    agent = make_agent("X", "goal", "backstory", CopilotLLM(RoleLLM()))
    assert agent.llm.model == "ollama/qwen3:8b" and isinstance(agent.llm, CopilotLLM)


def test_run_agents_use_only_the_local_llm(pb):
    llm = RoleLLM(critic=veto_first)
    r, db = run(pb, llm=llm)
    assert "generator" in llm.calls and "critic" in llm.calls      # every agent call reached our client
    assert any(t["name"].startswith("crewai.") for t in db.trace(r.run_id))


def test_spans_are_written_when_a_node_starts_and_completed_when_it_ends(pb):
    """The live UI timeline needs the running node: its span row exists (ended_at NULL) while it runs."""
    seen = []

    class WatchingLLM(RoleLLM):
        def chat(self, messages, **kw):
            seen.append([(s["agent"], s["ended_at"]) for s in self._db.spans(self._db._run)])
            return super().chat(messages, **kw)

    db = Database(":memory:")
    llm = WatchingLLM(critic=veto_first)
    real_start = db.start_run

    def start_run(*a, **k):
        rid = real_start(*a, **k)
        if a[0] == "multi_agent":
            db._run = rid
        return rid
    db.start_run = start_run
    llm._db = db
    r = run_multi_agent(TICKER, sources=FakeSources(), pb=pb, as_of=AS_OF, llm=llm, db=db, masker=Masker(PV),
                        portfolio_value=PV)
    # while an LLM call was in flight, the calling node's span was already present and open
    assert any(any(agent == "researcher" and ended is None for agent, ended in snap) for snap in seen)
    spans = db.spans(r.run_id)
    assert all(s["ended_at"] is not None for s in spans)          # all completed at the end
    assert len({s["span_id"] for s in spans}) == len(spans)        # updated in place, never duplicated


def test_metrics_ignore_a_span_that_is_still_running(pb):
    from options_copilot.observability import metrics
    db = Database(":memory:")
    rid = db.start_run("multi_agent", TICKER, "fake", pb.sha256)
    db.add_span(rid, "s1", None, "researcher", "one-way", "2026-11-20T10:00:00", None, None, 0, None, None)
    assert metrics(db, pb)["latency_per_agent"] == {}
    db.add_span(rid, "s1", None, "researcher", "one-way", "2026-11-20T10:00:00", "2026-11-20T10:00:05", 5000, 2,
                None, None)
    assert metrics(db, pb)["latency_per_agent"]["researcher"]["n"] == 1
    assert len(db.spans(rid)) == 1
