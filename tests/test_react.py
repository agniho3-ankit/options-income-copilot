"""The ReAct loop with a scripted LLM: tool execution, recovery, budget, masking, briefs."""
import json

import pytest

from agent_fakes import AS_OF, TICKER, FakeSources, ScriptedLLM, call
from options_copilot.agent_tools import ToolBox
from options_copilot.briefs import approve, make_brief, reflect_on_failures, reject
from options_copilot.db import Database
from options_copilot.llm import LLMError
from options_copilot.mask import Masker, find_leaks
from options_copilot.react import parse_inline_call, run_agent, system_prompt

PV = 100_000.0


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


def box(pb, **kw):
    return ToolBox(TICKER, FakeSources(**kw), pb, AS_OF, PV)


HAPPY = [
    [call("get_quote", ticker=TICKER), call("get_earnings_date", ticker=TICKER)],
    [call("build_trade", ticker=TICKER, strategy="put_credit_spread")],
    [call("submit_trade", proposal_id="P1", rationale="fits the playbook")],
    "Put credit spread on MRVL below support; the verifier passed it.",
]


def test_happy_path_writes_a_pending_brief(pb, db):
    b = box(pb)
    llm = ScriptedLLM(HAPPY)
    res = run_agent("Plan one entry.", b, llm, db, Masker(PV), pb)
    assert res.outcome == "accepted" and res.stats["tool_calls"] == 4 and res.stats["bad_calls"] == 0
    assert llm.sent[0]["tools"] and "Never do arithmetic" in llm.sent[0]["messages"][0]["content"]
    brief_id = make_brief(res, b, db, AS_OF)
    brief = db.brief(brief_id)
    assert brief["status"] == "pending" and brief["verdict"] in ("CLEAR", "CAUTION")
    assert brief["payload"]["lessons"] and brief["payload"]["citations"]
    assert db.run(res.run_id)["outcome"] == "accepted"


def test_observations_and_trace_are_masked(pb, db):
    pv = 87_654.0   # not a round number, so a risk % cannot echo a dollar figure's digits by chance
    b = ToolBox(TICKER, FakeSources(), pb, AS_OF, pv)
    llm = ScriptedLLM(HAPPY)
    res = run_agent("Plan one entry.", b, llm, db, Masker(pv), pb)
    numbers = b.proposals["P1"].numbers
    # content only: run ids and timestamps are random/clock digits, not trade values
    trace_text = json.dumps([r["content"] for r in db.trace(res.run_id)])
    sent_text = json.dumps(llm.sent)
    for text in (trace_text, sent_text):
        assert find_leaks(text, [numbers.credit, numbers.max_loss, pv]) == []
        assert "% of portfolio" in text


def test_bad_calls_are_reported_back_and_counted(pb, db):
    llm = ScriptedLLM([
        [call("approve_trade", brief="x")],
        [{"function": {"name": "get_quote", "arguments": "{not json"}}],
        [call("get_quote", ticker="NVDA")],
        "I could not find a trade.",
        "Still no trade.",
    ])
    res = run_agent("Plan.", box(pb), llm, db, Masker(PV), pb)
    assert res.stats["bad_calls"] == 3 and res.outcome == "no_trade"
    errors = [m for m in llm.sent[-1]["messages"] if m["role"] == "tool"]
    assert all("error" in json.loads(m["content"]) for m in errors)
    assert any(m["role"] == "user" and "not submitted" in m["content"] for m in llm.sent[-1]["messages"])


def test_inline_tool_call_is_recovered(pb, db):
    llm = ScriptedLLM(['{"name": "get_quote", "arguments": {"ticker": "MRVL"}}', "done", "done"])
    res = run_agent("Plan.", box(pb), llm, db, Masker(PV), pb)
    assert res.stats["tool_calls"] == 1 and res.stats["bad_calls"] == 1
    assert parse_inline_call("no json") is None and parse_inline_call('{"x": 1}') is None


def test_step_budget(pb, db):
    llm = ScriptedLLM([[call("get_quote", ticker=TICKER)]] * 5)
    res = run_agent("Plan.", box(pb), llm, db, Masker(PV), pb, max_steps=3)
    assert res.outcome == "budget_exhausted" and res.stats["steps"] == 3


def test_llm_outage(pb, db):
    res = run_agent("Plan.", box(pb), ScriptedLLM([LLMError("down")]), db, Masker(PV), pb)
    assert res.outcome == "llm_error"


def test_verifier_rejection_then_revision_and_reflexion(pb, db):
    b = box(pb)
    llm = ScriptedLLM([
        [call("price_trade", ticker=TICKER, strategy="put_credit_spread", short_put=98, long_put=95)],
        [call("submit_trade", proposal_id="P1", rationale="closer to the money")],
        [call("build_trade", ticker=TICKER, strategy="put_credit_spread")],
        [call("submit_trade", proposal_id="P2", rationale="revised")],
        "Revised to the playbook strikes.",
    ])
    res = run_agent("Plan.", b, llm, db, Masker(PV), pb)
    assert res.outcome == "accepted" and len(b.submissions) == 2 and not b.submissions[0][1].passed
    obs = json.loads([m for m in llm.sent[2]["messages"] if m["role"] == "tool"][-1]["content"])
    assert obs["passed"] is False and obs["failures"]
    lesson = reflect_on_failures(b, None, None, Masker(PV))   # template fallback, no store
    assert "short_delta" in lesson.lesson or "support" in lesson.lesson
    assert lesson.trigger.value == "verifier" and lesson.synthetic


def test_approve_writes_ticket_and_reject_writes_lesson(pb, db, store):
    b = box(pb)
    res = run_agent("Plan.", b, ScriptedLLM(HAPPY), db, Masker(PV), pb)
    brief_id = make_brief(res, b, db, AS_OF)
    ticket_id = approve(db, brief_id, "ok")
    t = db.tickets()[0]
    assert t["ticket_id"] == ticket_id and t["payload"]["paper_only"] and t["payload"]["synthetic"]
    assert db.brief(brief_id)["status"] == "approved"
    with pytest.raises(ValueError, match="not pending"):
        approve(db, brief_id)

    b2 = box(pb)
    res2 = run_agent("Plan.", b2, ScriptedLLM(HAPPY), db, Masker(PV), pb)
    brief2 = make_brief(res2, b2, db, AS_OF)
    with pytest.raises(ValueError, match="needs a reason"):
        reject(db, brief2, " ", llm=None, store=store, masker=Masker(PV))
    lesson = reject(db, brief2, "Too close to a $12,500 loss limit", llm=None, store=store, masker=Masker(PV))
    assert "[amount masked]" in lesson.lesson and db.brief(brief2)["status"] == "rejected"
    assert store.reflections.count() == 1


def test_no_brief_without_acceptance(pb, db):
    b = box(pb, landmine_verdict="BLOCK")
    llm = ScriptedLLM([[call("build_trade", ticker=TICKER, strategy="put_credit_spread")],
                       [call("submit_trade", proposal_id="P1", rationale="x")], "No trade can pass.", "No trade."])
    res = run_agent("Plan.", b, llm, db, Masker(PV), pb)
    assert res.outcome == "no_trade" and make_brief(res, b, db, AS_OF) is None


def test_failed_brief_cannot_be_approved(pb, db):
    db.save_brief("brf-x", None, "entry", TICKER, "pending", "BLOCK", {"proposal": {}, "numbers": {}})
    with pytest.raises(ValueError, match="did not pass"):
        approve(db, "brf-x")


def test_system_prompt_is_built_from_the_playbook(pb):
    text = system_prompt(pb)
    assert f"v{pb.version}" in text and "MRVL" in text and "untrusted data" in text
