from datetime import date, timedelta

import pytest

from options_copilot.management import Action, Snapshot, evaluate, heat_review, one_sigma, profit_pct
from options_copilot.models import OptionLeg, OptionType, Position, Side, Strategy

AS_OF = date(2026, 10, 30)   # a Friday
P, C, S, L = OptionType.PUT, OptionType.CALL, Side.SHORT, Side.LONG


def condor(dte=30, opened_days_ago=10, credit=100.0, entry_spot=100.0, entry_iv=0.35):
    exp = AS_OF + timedelta(days=dte)
    legs = (OptionLeg(P, S, 90, exp, 1, 1.1), OptionLeg(P, L, 87, exp, 0.5, 0.6),
            OptionLeg(C, S, 110, exp, 1, 1.1), OptionLeg(C, L, 113, exp, 0.5, 0.6))
    return Position("SYN-1", "TICKER_A", Strategy.IRON_CONDOR, legs, AS_OF - timedelta(days=opened_days_ago),
                    credit, 300 - credit, entry_spot=entry_spot, entry_iv=entry_iv)


def put_spread(**kw):
    pos = condor(**kw)
    pos.legs = pos.legs[:2]
    pos.strategy = Strategy.PUT_CREDIT_SPREAD
    return pos


def snap(**kw):
    base = dict(as_of=AS_OF, spot=100.0, close_cost=80.0, sma200=90.0, earnings_known_none=True,
                iv_rank_now=50.0, iv_rank_window_ago=50.0)
    base.update(kw)
    return Snapshot(**base)


def rules(ev):
    return {(t.rule, t.action) for t in ev.triggers}


def test_quiet_position_fires_nothing(pb):
    ev = evaluate(condor(), snap(), pb)
    assert ev.triggers == () and ev.not_evaluated == () and ev.profit_pct == 20.0


@pytest.mark.parametrize("close_cost, fires", [(50.0, True), (50.01, False)])
def test_r1_profit_target_boundary(pb, close_cost, fires):
    assert (("r1_profit_target", Action.CLOSE) in rules(evaluate(condor(), snap(close_cost=close_cost), pb))) is fires


@pytest.mark.parametrize("dte, fires", [(20, True), (21, False)])
def test_r2_roll_boundary(pb, dte, fires):
    ev = evaluate(condor(dte=dte), snap(), pb)
    assert (("r2_roll", Action.ROLL) in rules(ev)) is fires


def test_r2_does_not_roll_a_winner(pb):
    assert ("r2_roll", Action.ROLL) not in rules(evaluate(condor(dte=10), snap(close_cost=40.0), pb))


@pytest.mark.parametrize("close_cost, fires", [(200.0, True), (199.99, False)])
def test_r3_stop_at_one_times_credit(pb, close_cost, fires):
    ev = evaluate(condor(), snap(close_cost=close_cost), pb)
    assert (("r3_stop_loss", Action.CLOSE) in rules(ev)) is fires


def test_r4_sigma_band_widens_with_time(pb):
    band10 = one_sigma(100, 0.35, 10)
    assert band10 == pytest.approx(100 * 0.35 * (10 / 365) ** 0.5)
    assert one_sigma(100, 0.35, 0) == one_sigma(100, 0.35, 1)
    up = evaluate(condor(), snap(spot=100 + band10 + 0.01), pb)
    assert ("r4_big_move", Action.BUY_BACK_CALLS) in rules(up)
    assert rules(evaluate(condor(), snap(spot=100 + band10 - 0.01), pb)) == set()
    down = evaluate(condor(), snap(spot=100 - band10 - 0.01, sma200=80.0), pb)
    assert rules(down) == {("r4_big_move", Action.BUY_BACK_PUTS)}
    broken = evaluate(condor(), snap(spot=100 - band10 - 0.01, sma200=99.0), pb)
    assert ("r4_big_move", Action.CLOSE) in rules(broken)


def test_r4_up_move_on_put_spread_has_no_call_to_buy(pb):
    assert rules(evaluate(put_spread(), snap(spot=120.0, close_cost=10.0), pb)) == {("r1_profit_target", Action.CLOSE)}


@pytest.mark.parametrize("days, expected", [(7, Action.CLOSE), (8, Action.CLOSE_CALLS), (30, Action.CLOSE_CALLS)])
def test_r5_earnings(pb, days, expected):
    ev = evaluate(condor(), snap(earnings_date=AS_OF + timedelta(days=days), earnings_known_none=False), pb)
    assert ("r5_earnings", expected) in rules(ev)


def test_r5_put_spread_keeps_puts(pb):
    ev = evaluate(put_spread(), snap(earnings_date=AS_OF + timedelta(days=20), earnings_known_none=False), pb)
    assert rules(ev) == set()


def test_r6_iv_crush_branches(pb):
    crush = dict(iv_rank_now=30.0, iv_rank_window_ago=50.0)
    assert ("r6_iv_crush", Action.CLOSE) in rules(evaluate(condor(), snap(close_cost=40.0, **crush), pb))
    assert ("r6_iv_crush", Action.HOLD_OR_ROLL_OUT) in rules(evaluate(condor(), snap(**crush), pb))
    assert rules(evaluate(condor(), snap(iv_rank_now=31.0, iv_rank_window_ago=50.0), pb)) == set()


def test_r7_ex_dividend_on_itm_short_call(pb):
    base = dict(spot=112.0, ex_dividend_date=AS_OF + timedelta(days=3), dividend=0.5)
    ev = evaluate(condor(entry_iv=2.0), snap(short_call_extrinsic=0.2, close_cost=60.0, **base), pb)
    assert ("r7_ex_dividend", Action.CLOSE_CALLS) in rules(ev)
    ev = evaluate(condor(entry_iv=2.0), snap(short_call_extrinsic=0.2, close_cost=150.0, **base), pb)
    assert ("r7_ex_dividend", Action.HOLD_OR_ROLL) in rules(ev)
    ev = evaluate(condor(entry_iv=2.0), snap(short_call_extrinsic=0.6, close_cost=60.0, **base), pb)
    assert not any(r == "r7_ex_dividend" for r, _ in rules(ev))
    ev = evaluate(condor(entry_iv=2.0), snap(close_cost=60.0, spot=112.0, ex_dividend_date=AS_OF + timedelta(days=3)), pb)
    assert "r7_ex_dividend" in ev.not_evaluated


def test_conflicts_are_all_returned(pb):
    ev = evaluate(condor(dte=10), snap(close_cost=40.0, earnings_date=AS_OF + timedelta(days=5),
                                       earnings_known_none=False, iv_rank_now=20.0, iv_rank_window_ago=45.0), pb)
    assert {"r1_profit_target", "r5_earnings", "r6_iv_crush"} <= {t.rule for t in ev.triggers}
    assert ev.actions == [Action.CLOSE]


def test_missing_inputs_are_named_not_passed(pb):
    ev = evaluate(condor(entry_spot=None), Snapshot(as_of=AS_OF, spot=100.0), pb)
    assert set(ev.not_evaluated) >= {"r1_profit_target", "r3_stop_loss", "r4_big_move", "r5_earnings"}
    assert ev.profit_pct is None


def test_emergency_exits(pb):
    ev = evaluate(condor(), snap(sector_change_pct=-5.0, insider_stake_sold_pct=50.0), pb)
    assert {"emergency_sector", "emergency_insider"} <= {t.rule for t in ev.triggers}
    assert not evaluate(condor(), snap(sector_change_pct=-4.9), pb).triggers


def test_r8_heat_review_only_on_friday(pb):
    assert heat_review(30.01, AS_OF, pb).action is Action.REVIEW_WEAKEST
    assert heat_review(30.0, AS_OF, pb) is None
    assert heat_review(45.0, AS_OF + timedelta(days=1), pb) is None


def test_profit_pct():
    assert profit_pct(100, 25) == 75 and profit_pct(100, 150) == -50
