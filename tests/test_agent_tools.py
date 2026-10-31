import pytest

from agent_fakes import AS_OF, EXP, TICKER, FakeSources
from options_copilot.agent_tools import SCHEMAS, ToolBox, ToolError
from options_copilot.verifier import InsiderActivity


@pytest.fixture
def box(pb):
    return ToolBox(TICKER, FakeSources(), pb, AS_OF, 100_000.0)


def test_schema_has_no_approval_tool():
    names = {s["function"]["name"] for s in SCHEMAS}
    assert "submit_trade" in names and not any("approve" in n or "order" in n for n in names)


def test_ticker_lock_and_unknown_tools(box):
    with pytest.raises(ToolError, match="only for MRVL"):
        box.call("get_quote", {"ticker": "NVDA"})
    with pytest.raises(ToolError, match="unknown tool"):
        box.call("approve_trade", {})
    with pytest.raises(ToolError, match="bad arguments"):
        box.call("get_quote", {"ticker": TICKER, "surprise": 1})


def test_data_tools(box):
    q = box.call("get_quote", {"ticker": "mrvl"})
    assert q["tier"] == "tier_1" and q["support"] == 93.0
    assert box.call("get_earnings_date", {"ticker": TICKER})["days_away"] == 70
    c = box.call("get_option_chain", {"ticker": TICKER})
    assert c["expiry"] == EXP.isoformat() and c["dte"] == 45
    assert all(r["strike"] <= 100 for r in c["puts_below_spot"]) and all(r["strike"] >= 100 for r in c["calls_above_spot"])


def test_filings_and_lessons_are_wrapped_as_data(box):
    f = box.call("search_filings", {"ticker": TICKER})
    assert f["summary"].startswith("<<<DATA") and "<<<x>>>" not in f["summary"]
    lessons = box.call("search_lessons", {"ticker": TICKER})["lessons"]
    assert lessons and all(item.startswith("<<<DATA") for item in lessons)


def test_build_respects_bands_and_levels(box):
    out = box.call("build_trade", {"ticker": TICKER, "strategy": "put_credit_spread"})
    assert out["proposal_id"] == "P1" and out["legs"][0]["strike"] <= 93
    with pytest.raises(ToolError, match="band"):
        box.call("build_trade", {"ticker": TICKER, "strategy": "iron_condor", "target_delta": 0.25})
    with pytest.raises(ToolError, match="strategy must be"):
        box.call("build_trade", {"ticker": TICKER, "strategy": "naked_call"})


def test_price_trade_uses_listed_strikes_only(box):
    out = box.call("price_trade", {"ticker": TICKER, "strategy": "put_credit_spread", "short_put": 89, "long_put": 86})
    assert out["breakevens"] and out["pop"] > 0
    with pytest.raises(ToolError, match="not listed"):
        box.call("price_trade", {"ticker": TICKER, "strategy": "put_credit_spread", "short_put": 90, "long_put": 86})
    with pytest.raises(ToolError, match="needs long_put"):
        box.call("price_trade", {"ticker": TICKER, "strategy": "put_credit_spread", "short_put": 89})


def test_submit_runs_the_gate_and_records_acceptance(box):
    pid = box.call("build_trade", {"ticker": TICKER, "strategy": "put_credit_spread"})["proposal_id"]
    res = box.call("submit_trade", {"proposal_id": pid, "rationale": "fits"})
    assert res["passed"] and box.accepted[0] == pid
    assert box.sources.landmine_calls == 1   # the filings check ran even though the agent never asked
    with pytest.raises(ToolError, match="unknown proposal_id"):
        box.call("submit_trade", {"proposal_id": "P9", "rationale": ""})


def test_submit_failure_returns_reasons(pb):
    box = ToolBox(TICKER, FakeSources(landmine_verdict="BLOCK"), pb, AS_OF, 100_000.0)
    pid = box.call("build_trade", {"ticker": TICKER, "strategy": "put_credit_spread"})["proposal_id"]
    res = box.call("submit_trade", {"proposal_id": pid, "rationale": "x"})
    assert not res["passed"] and {f["check"] for f in res["failures"]} == {"landmine"}
    assert box.accepted is None and res["next"].startswith("STOP")


def test_earnings_inside_block_window_tells_the_agent_to_stop(pb):
    from datetime import timedelta

    from agent_fakes import FakeSources as FS
    src = FS()
    original = src.ticker_data

    def soon(ticker):
        d = original(ticker)
        d.profile = d.profile.__class__(**{**d.profile.__dict__, "earnings_date": AS_OF + timedelta(days=3)})
        return d
    src.ticker_data = soon
    box = ToolBox(TICKER, src, pb, AS_OF, 100_000.0)
    pid = box.call("build_trade", {"ticker": TICKER, "strategy": "put_credit_spread"})["proposal_id"]
    res = box.call("submit_trade", {"proposal_id": pid, "rationale": "x"})
    assert res["next"].startswith("STOP") and "earnings in 3 days" in res["next"]


def test_flag_concern_only_tightens(box):
    with pytest.raises(ToolError):
        box.call("flag_concern", {"reason": "x"})
    pid = box.call("build_trade", {"ticker": TICKER, "strategy": "put_credit_spread"})["proposal_id"]
    box.call("submit_trade", {"proposal_id": pid, "rationale": "x"})
    assert box.call("flag_concern", {"reason": "soft worry"})["verdict"] == "CAUTION"
    assert box.call("flag_concern", {"reason": "veto", "block": True})["passed"] is False
    assert box.accepted is None


def test_caution_from_insider_sales_still_passes(pb):
    box = ToolBox(TICKER, FakeSources(insider=InsiderActivity(0, 500)), pb, AS_OF, 100_000.0)
    pid = box.call("build_trade", {"ticker": TICKER, "strategy": "put_credit_spread"})["proposal_id"]
    res = box.call("submit_trade", {"proposal_id": pid, "rationale": "x"})
    assert res["verdict"] == "CAUTION" and res["passed"] and res["cautions"]
