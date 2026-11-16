"""Run the multi-agent entry workflow for one ticker."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from . import privacy
from ..db import Database
from ..mask import Masker
from ..playbook import Playbook
from .graph import CountingLLM, Runtime, build_graph, mermaid
from .state import AgentState, new_state, to_json


@dataclass
class MAResult:
    run_id: str
    outcome: str
    state: AgentState
    latency_ms: int
    llm_calls: int
    builds: int
    mcp_calls: int
    agents: list[str]
    spans: list[dict]
    events: list[dict] = None
    fallbacks: int = 0


def run_multi_agent(ticker: str, *, sources: Any, pb: Playbook, as_of: date, llm, db: Database, masker: Masker,
                    portfolio_value: float, store=None, open_max_losses: tuple[float, ...] = (),
                    use_agents: bool = True, graph_out: Path | None = None, sandbox: bool = False,
                    disabled: frozenset[str] = frozenset()) -> MAResult:
    """`llm` is our LLMClient (or None for a rules-only run). With `use_agents`, the Strategist and
    Risk Critic are CrewAI agents on CopilotLLM; without, the strategist uses the deterministic
    enumerator and there is no critic.

    `sandbox` (evaluation) makes briefs, broker calls and lesson writes impossible; `disabled` switches
    named guardrails off for the ablation study and is only accepted together with `sandbox`."""
    from .. import guardrails as G
    privacy.configure()  # again: load_settings() may have read .env since import
    with G.run_config(sandbox=sandbox, disabled=disabled):
        return _run(ticker, sources=sources, pb=pb, as_of=as_of, llm=llm, db=db, masker=masker,
                    portfolio_value=portfolio_value, store=store, open_max_losses=open_max_losses,
                    use_agents=use_agents, graph_out=graph_out, sandbox=sandbox,
                    disabled=disabled, critic_off="critic" in disabled)


def _run(ticker, *, sources, pb, as_of, llm, db, masker, portfolio_value, store, open_max_losses, use_agents,
         graph_out, sandbox, disabled, critic_off) -> MAResult:
    from .. import guardrails as G
    from ..mcp_server import MCPBridge, MCPContext, build_server

    run_id = db.start_run("multi_agent", ticker.upper(), llm.name if llm else "rules-only", pb.sha256)
    ctx = MCPContext(sources, pb, as_of, portfolio_value, db, open_max_losses)
    box = ctx.box(ticker)
    t0 = time.monotonic()
    with MCPBridge(build_server(ctx)) as bridge:
        rt = Runtime(box=box, bridge=bridge, pb=pb, db=db, masker=masker, llm=None, store=store, db_run_id=run_id,
                     guard=G.GuardrailLog(db, run_id, masker))
        if llm is not None:
            rt.llm = CountingLLM(llm, rt)
        if llm is not None and use_agents:
            from .agents import json_task, risk_critic, strategist
            from .llm_adapter import CopilotLLM

            def counted(agent_role: str, tokens: int, text: str) -> None:
                rt.llm_calls += 1
                rt.trace("assistant", text, f"crewai.{agent_role}", eval_tokens=tokens)

            agent_llm = CopilotLLM(llm, on_call=counted)
            n = pb["multi_agent.agent_max_iter"]
            rt.strategist, rt.critic, rt.json_task = strategist(agent_llm, bridge, n), risk_critic(agent_llm, bridge, n), json_task
        if critic_off:   # ablation: the Strategist agent stays, the Risk Critic is removed
            rt.critic = None
        rt.session_span = f"session-{run_id}"
        db.add_span(run_id, rt.session_span, None, "session", "supervisor", "", "", 0, 0, {"ticker": box.ticker}, None)
        graph = build_graph(rt)
        if graph_out is not None:
            graph_out.parent.mkdir(parents=True, exist_ok=True)
            graph_out.write_text("# Multi-agent entry workflow (LangGraph)\n\n```mermaid\n" + mermaid(graph) + "```\n")
        state = new_state(run_id, box.ticker, as_of.isoformat(), dict(pb["multi_agent"]),
                          critic_enabled=rt.critic is not None, sandbox=sandbox, disabled=sorted(disabled))
        clients = {id(c): c for c in (llm, getattr(sources, "llm", None)) if c is not None}
        fb0 = sum(getattr(c, "fallbacks", 0) for c in clients.values())
        state = graph.invoke(state, config={"recursion_limit": 80})
        mcp_calls = bridge.calls
    fallbacks = sum(getattr(c, "fallbacks", 0) for c in clients.values()) - fb0   # Claude-fallback answers
    latency = int((time.monotonic() - t0) * 1000)
    db.add_trace(run_id, "summary", to_json(state), name="multi_agent.state")
    outcome = state.get("outcome") or "NO_TRADE"
    db.finish_run(run_id, outcome, steps=len(rt.spans), tool_calls=rt.builds, eval_tokens=rt.tokens)
    with db.conn:   # the session span carries the run's fallback count
        db.conn.execute("UPDATE spans SET fallbacks = ?, tokens = ?, ticker = ?, as_of = ? WHERE run_id = ? AND span_id = ?",
                        (fallbacks, rt.tokens, box.ticker, as_of.isoformat(), run_id, rt.session_span))
    agents = sorted({s["agent"] for s in rt.spans})
    return MAResult(run_id, outcome, state, latency, rt.llm_calls, rt.builds, mcp_calls, agents, rt.spans,
                    rt.guard.events, max(0, fallbacks))
