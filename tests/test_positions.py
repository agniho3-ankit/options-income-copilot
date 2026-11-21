"""Rule tracks (Positions screen) and the journal's monthly summary (UI slice 6). No network."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from options_copilot.journal import metrics
from options_copilot.management import Snapshot, evaluate, rule_tracks
from options_copilot.models import OptionLeg, OptionType, Position, Side, Strategy
from options_copilot.playbook import load_playbook
from options_copilot.config import load_settings

OPENED = date(2026, 10, 20)
EXP = OPENED + timedelta(days=45)


@pytest.fixture(scope="module")
def pb():
    return load_playbook(load_settings(env={}).playbook_path)


def position(**kw) -> Position:
    legs = (OptionLeg(OptionType.PUT, Side.SHORT, 90.0, EXP, 1.6, 1.7), OptionLeg(OptionType.PUT, Side.LONG, 85.0, EXP, 0.2, 0.3))
    base = dict(position_id="pos-1", ticker="MSFT", strategy=Strategy.PUT_CREDIT_SPREAD, legs=legs, opened_on=OPENED,
                entry_credit=150.0, entry_max_loss=350.0, entry_spot=100.0, entry_iv=0.30)
    return Position(**{**base, **kw})


def tracks(pb, pos, **snap):
    s = Snapshot(**{"as_of": OPENED + timedelta(days=10), "spot": 101.0, "close_cost": 90.0,
                    "earnings_known_none": True, **snap})
    return {t["rule"]: t for t in rule_tracks(pos, s, evaluate(pos, s, pb), pb)}


def test_profit_and_dte_tracks_measure_progress_toward_the_playbook_targets(pb):
    t = tracks(pb, position())
    assert t["r1_profit_target"]["value"] == "40.0%" and t["r1_profit_target"]["progress_pct"] == 80.0
    assert t["r1_profit_target"]["state"] == "ok" and t["r1_profit_target"]["target"] == "50%"
    # 45 DTE at open, 35 now, roll below 21: 10 of 24 days elapsed
    assert t["r2_roll"]["value"] == "35 DTE" and t["r2_roll"]["progress_pct"] == round(10 / 24 * 100, 1)
    assert t["r3_stop_loss"]["value"] == "0.00x" and t["r3_stop_loss"]["progress_pct"] == 0.0


def test_a_triggered_profit_target_is_due_and_a_stop_is_a_breach(pb):
    won = tracks(pb, position(), close_cost=60.0)
    assert won["r1_profit_target"]["state"] == "due" and won["r1_profit_target"]["progress_pct"] == 100.0
    lost = tracks(pb, position(), close_cost=300.0)
    assert lost["r3_stop_loss"]["state"] == "breach" and lost["r3_stop_loss"]["value"] == "1.00x"
    assert lost["r3_stop_loss"]["action"] == "close"


def test_missing_data_is_unknown_never_ok(pb):
    t = tracks(pb, position(entry_spot=None), close_cost=None, earnings_known_none=False)
    for rule in ("r1_profit_target", "r3_stop_loss", "r4_big_move", "r5_earnings"):
        assert t[rule]["state"] == "unknown" and t[rule]["progress_pct"] is None, rule


def test_big_move_uses_the_one_sigma_band(pb):
    t = tracks(pb, position(), spot=80.0)
    assert t["r4_big_move"]["state"] == "due" and t["r4_big_move"]["progress_pct"] == 100.0
    assert t["r4_big_move"]["value"] == "-20.00"


def row(opened, closed=None, pnl=0.0, ror=None, rule=None, pnl_pct=0.0, days=10):
    return {"date_opened": opened, "date_closed": closed, "pnl": pnl, "pnl_pct": pnl_pct, "days_held": days,
            "return_on_risk_pct": ror, "exit_rule": rule}


def test_journal_summary_adds_return_on_risk_and_exit_rules():
    rows = [row("2026-10-01", "2026-10-20", 100.0, 25.0, "r1_profit_target", 50.0),
            row("2026-10-05", "2026-10-30", -150.0, -40.0, "r3_stop_loss", -100.0),
            row("2026-10-07", "2026-10-25", 80.0, 20.0, "r1_profit_target", 40.0),
            row("2026-11-02")]
    m = metrics(rows, "2026-10")
    assert m["closed_trades"] == 3 and m["total_pnl"] == 30.0 and m["win_rate_pct"] == 66.7
    assert m["avg_return_on_risk_pct"] == 1.67
    assert m["exit_rules"] == {"r1_profit_target": 2, "r3_stop_loss": 1}
    empty = metrics(rows, "2026-11")
    assert empty["avg_return_on_risk_pct"] is None and empty["exit_rules"] == {} and empty["active_trades"] == 1
