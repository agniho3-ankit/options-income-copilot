"""Trade-window chart data (UI slice 3): pure Python, no network."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from options_copilot.guardrails import REGISTRY, pipeline_summary
from options_copilot.management import one_sigma
from options_copilot.market import History, Profile
from options_copilot.window import WindowError, build_window, excerpts_from_trace, zones

TODAY = date(2026, 11, 20)


def history(n: int = 120, last: float = 500.0) -> History:
    days = [TODAY - timedelta(days=n - 1 - i) for i in range(n)]
    closes = [last - (n - 1 - i) * 0.5 for i in range(n)]
    return History(closes=closes, highs=closes, lows=closes, dates=days)


def profile(earnings: date | None = date(2027, 1, 27), exdiv: date | None = date(2026, 11, 18)) -> Profile:
    return Profile(price=500.0, market_cap=None, sector=None, country=None, analyst_target=None, analyst_count=None,
                   analyst_buy_ratio=None, earnings_date=earnings, ex_dividend_date=exdiv, dividend_rate=None)


def entry_payload(strategy="call_credit_spread", breakevens=(548.48,), legs=None) -> dict:
    legs = legs or [{"option_type": "call", "side": "short", "strike": 545.0, "expiry": "2027-01-15", "iv": 0.2995},
                    {"option_type": "call", "side": "long", "strike": 560.0, "expiry": "2027-01-15", "iv": 0.29}]
    return {"as_of": "2026-11-19", "proposal": {"strategy": strategy, "legs": legs},
            "numbers": {"credit": 347.5, "breakevens": list(breakevens)},
            "citations": [{"form": "8-K", "date": "2026-10-28", "item": "Item 8.01", "url": "u"}]}


def test_cone_uses_the_playbook_one_sigma_from_today_to_expiry():
    w = build_window(ticker="MSFT", kind="entry", payload=entry_payload(), history=history(), profile=profile(),
                     today=TODAY, atm_iv=0.32, excerpts=[])
    assert w["today"] == "2026-11-20" and w["expiry"] == "2027-01-15" and w["spot"] == 500.0
    assert w["cone"][0] == {"date": "2026-11-20", "low": 500.0, "high": 500.0}
    last = w["cone"][-1]
    band = one_sigma(500.0, 0.32, (date(2027, 1, 15) - TODAY).days)
    assert last["date"] == "2027-01-15" and last == {"date": "2027-01-15", "low": round(500 - band, 2),
                                                      "high": round(500 + band, 2)}
    assert w["sigma"]["iv"] == 0.32 and "ATM" in w["sigma"]["source"]
    assert len(w["history"]) == 90 and w["history"][-1] == {"date": "2026-11-20", "close": 500.0}


def test_breakevens_and_strikes_are_the_stored_values():
    w = build_window(ticker="MSFT", kind="entry", payload=entry_payload(), history=history(), profile=profile(),
                     today=TODAY, atm_iv=0.32, excerpts=[])
    assert w["breakevens"] == [548.48]
    assert [s["strike"] for s in w["strikes"]] == [545.0, 560.0]
    assert w["zones"] == [{"kind": "profit", "low": None, "high": 548.48}, {"kind": "loss", "low": 548.48, "high": None}]
    lo, hi = w["y_domain"]
    assert lo < min(c["low"] for c in w["cone"]) and hi > 560.0


def test_short_leg_iv_is_the_labelled_fallback():
    w = build_window(ticker="MSFT", kind="entry", payload=entry_payload(), history=history(), profile=profile(),
                     today=TODAY, atm_iv=None, excerpts=[])
    assert w["sigma"]["iv"] == 0.2995 and "short-leg" in w["sigma"]["source"]


@pytest.mark.parametrize("strategy,be,expected", [
    ("put_credit_spread", [446.5], [("profit", 446.5, None), ("loss", None, 446.5)]),
    ("cash_secured_put", [440.0], [("profit", 440.0, None), ("loss", None, 440.0)]),
    ("iron_condor", [560.0, 440.0], [("profit", 440.0, 560.0), ("loss", None, 440.0), ("loss", 560.0, None)]),
    ("iron_condor", [440.0], []),          # inconsistent data draws no zones rather than a guess
    ("call_credit_spread", [], []),
])
def test_zones(strategy, be, expected):
    assert [(z["kind"], z["low"], z["high"]) for z in zones(strategy, be)] == expected


def test_event_pins_and_in_window_flags():
    ex = [{"id": "c1", "form": "8-K", "item": "Item 1.01",
           "text": "The merger is expected to close on December 15, 2026, subject to approval."}]
    w = build_window(ticker="META", kind="entry", payload=entry_payload(), history=history(), profile=profile(),
                     today=TODAY, atm_iv=0.3, excerpts=ex)
    by = {e["kind"]: e for e in w["events"]}
    assert by["earnings"]["date"] == "2027-01-27" and by["earnings"]["in_window"] is False
    assert by["ex_dividend"]["in_window"] is False          # 2026-11-18 is before the brief's as-of
    assert by["filing"]["date"] == "2026-10-28"
    assert by["filing_event"]["date"] == "2026-12-15" and by["filing_event"]["in_window"] is True
    assert w["x_domain"] == [w["history"][0]["date"], "2027-01-27"]   # widened to show earnings just after expiry


def test_exit_brief_draws_the_open_position_without_breakevens():
    p = {"as_of": "2026-11-20", "strategy": "put_credit_spread",
         "legs": [{"option_type": "put", "side": "short", "strike": 480.0, "expiry": "2026-12-18"},
                  {"option_type": "put", "side": "long", "strike": 465.0, "expiry": "2026-12-18"}]}
    w = build_window(ticker="MSFT", kind="exit", payload=p, history=history(), profile=profile(),
                     today=TODAY, atm_iv=0.3, excerpts=[])
    assert w["expiry"] == "2026-12-18" and w["breakevens"] == [] and w["zones"] == []


def test_missing_dates_or_expiry_raise():
    with pytest.raises(WindowError):
        build_window(ticker="X", kind="entry", payload=entry_payload(), history=History([1.0], [1.0], [1.0]),
                     profile=profile(), today=TODAY, atm_iv=0.3, excerpts=[])
    with pytest.raises(WindowError):
        build_window(ticker="X", kind="entry", payload={"proposal": {"legs": []}}, history=history(),
                     profile=profile(), today=TODAY, atm_iv=0.3, excerpts=[])


def test_excerpts_come_from_grader_messages_deduplicated():
    rows = [{"role": "user", "name": "researcher.grader",
             "content": json.dumps({"excerpts": [{"id": "a", "form": "8-K", "item": "1.01", "text": "one"},
                                                 {"id": "b", "form": "10-Q", "item": "1A", "text": "two"}]})},
            {"role": "assistant", "name": "researcher.grader", "content": "{}"},
            {"role": "user", "name": "researcher.grader",
             "content": json.dumps({"excerpts": [{"id": "a", "text": "one again"}, {"id": "c", "text": "three"}]})},
            {"role": "user", "name": "other", "content": "not json"}]
    assert [(e["id"], e["text"]) for e in excerpts_from_trace(rows)] == [("a", "one"), ("b", "two"), ("c", "three")]


def test_pipeline_summary_keeps_order_and_takes_the_most_severe_decision():
    events = [{"seq": 1, "guardrail": "coverage_guard", "decision": "pass", "reason": "ok"},
              {"seq": 2, "guardrail": "judge", "decision": "flag", "signal": "judge_flag", "reason": "claim"},
              {"seq": 3, "guardrail": "verifier", "decision": "pass", "reason": "ok"},
              {"seq": 4, "guardrail": "verifier", "decision": "block", "reason": "pop"}]
    s = pipeline_summary(events)
    assert [g["name"] for g in s] == [g.name for g in REGISTRY]
    d = {g["name"]: g for g in s}
    assert d["verifier"]["decision"] == "block" and len(d["verifier"]["events"]) == 2
    assert d["judge"]["decision"] == "flag" and d["budgets"]["decision"] is None


def test_nan_closes_are_skipped_so_spot_is_the_last_real_close():
    h = history()
    closes = h.closes[:-1] + [float("nan")]
    w = build_window(ticker="MSFT", kind="entry", payload=entry_payload(),
                     history=History(closes=closes, highs=closes, lows=closes, dates=h.dates), profile=profile(),
                     today=TODAY, atm_iv=0.3, excerpts=[])
    assert w["spot"] == 499.5 and w["today"] == "2026-11-19"
    assert all(c["low"] == c["low"] for c in w["cone"])   # no NaN anywhere in the cone
