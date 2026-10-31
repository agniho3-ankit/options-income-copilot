"""Calculator math on hand-checkable synthetic trades (TICKER_A)."""
from datetime import date, timedelta

import pytest

from options_copilot.calculators import (BuildError, BuildRules, CalcError, Quote, bs_delta, build, build_for_pop,
                                         close_cost_per_contract, compute, portfolio_heat_pct, prob_above,
                                         prob_between)
from options_copilot.models import OptionLeg, OptionType, Side, Strategy, TradeProposal

AS_OF = date(2026, 10, 30)
EXP = AS_OF + timedelta(days=45)
P, C, S, L = OptionType.PUT, OptionType.CALL, Side.SHORT, Side.LONG


def leg(t, side, k, bid, ask, iv=0.30):
    return OptionLeg(t, side, k, EXP, bid, ask, iv=iv)


def condor():
    # credit = 1.05 - 0.55 + 1.05 - 0.55 = 1.00 per share
    return TradeProposal("TICKER_A", Strategy.IRON_CONDOR, (
        leg(P, S, 90, 1.0, 1.1), leg(P, L, 87, 0.5, 0.6), leg(C, S, 110, 1.0, 1.1), leg(C, L, 115, 0.5, 0.6)))


def test_condor_numbers():
    n, x = compute(condor(), spot=100, as_of=AS_OF, portfolio_value=100_000)
    assert n.credit == 100.0
    assert n.max_loss == 400.0           # wider (call) wing 5 x 100 - 100 credit
    assert n.breakevens == (89.0, 111.0)
    assert n.risk_pct_of_portfolio == 0.4
    assert x.profit_zone_width == 22.0 and x.profit_zone_pct == 22.0 and x.risk_reward == 4.0
    expected = prob_between(100, 89, 111, 0.30, 45 / 365)
    assert n.pop == round(expected, 4)


def test_put_and_call_spreads_and_csp():
    pcs = TradeProposal("TICKER_A", Strategy.PUT_CREDIT_SPREAD, (leg(P, S, 95, 2.0, 2.2), leg(P, L, 90, 0.9, 1.1)))
    n, _ = compute(pcs, 100, AS_OF, 100_000)
    assert (n.credit, n.max_loss, n.breakevens) == (110.0, 390.0, (93.9,))
    assert n.pop == round(prob_above(100, 93.9, 0.30, 45 / 365), 4)

    ccs = TradeProposal("TICKER_A", Strategy.CALL_CREDIT_SPREAD, (leg(C, S, 105, 2.0, 2.2), leg(C, L, 110, 0.9, 1.1)))
    n, _ = compute(ccs, 100, AS_OF, 100_000)
    assert (n.credit, n.max_loss, n.breakevens) == (110.0, 390.0, (106.1,))

    csp = TradeProposal("TICKER_A", Strategy.CASH_SECURED_PUT, (leg(P, S, 40, 1.0, 1.0),), contracts=2)
    n, _ = compute(csp, 45, AS_OF, 100_000)
    assert n.max_loss == 3900.0 and n.collateral == 8000.0 and n.risk_pct_of_portfolio == 7.8


def test_compute_rejects_bad_trades():
    debit = TradeProposal("TICKER_A", Strategy.PUT_CREDIT_SPREAD, (leg(P, S, 95, 0.5, 0.5), leg(P, L, 90, 1.0, 1.0)))
    with pytest.raises(CalcError, match="not a credit"):
        compute(debit, 100, AS_OF, 100_000)
    inverted = TradeProposal("TICKER_A", Strategy.PUT_CREDIT_SPREAD, (leg(P, S, 90, 2.0, 2.0), leg(P, L, 95, 1.0, 1.0)))
    with pytest.raises(CalcError):
        compute(inverted, 100, AS_OF, 100_000)
    no_iv = TradeProposal("TICKER_A", Strategy.CASH_SECURED_PUT, (leg(P, S, 40, 1.0, 1.0, iv=None),))
    with pytest.raises(CalcError, match="volatility"):
        compute(no_iv, 45, AS_OF, 100_000)
    with pytest.raises(CalcError):
        compute(condor(), 100, AS_OF, 0)


def test_black_scholes_sanity():
    assert bs_delta(100, 100, 0.3, 0.5, C) == pytest.approx(0.542, abs=0.002)
    assert bs_delta(100, 100, 0.3, 0.5, P) == pytest.approx(0.542 - 1, abs=0.002)
    assert bs_delta(100, 90, 0.3, 0, C) == 1.0 and bs_delta(100, 90, 0.3, 0, P) == 0.0
    assert prob_above(100, 100, 0.3, 1) < 0.5          # lognormal median is below the spot
    assert prob_between(100, 110, 90, 0.3, 1) == 0.0
    assert prob_above(100, 0, 0.3, 1) == 1.0


def test_close_cost_and_heat():
    p = condor()
    mids = {("put", "short", 90): 0.5, ("put", "long", 87): 0.2, ("call", "short", 110): 0.4, ("call", "long", 115): 0.1}
    assert close_cost_per_contract(p.legs, mids) == 60.0
    with pytest.raises(CalcError):
        close_cost_per_contract(p.legs, {})
    assert portfolio_heat_pct([1000, 2000], 100_000) == 3.0


def chain():
    """Synthetic chain with explicit deltas: puts 80-98, calls 102-120."""
    q = []
    for k, d in [(80, -0.05), (83, -0.10), (86, -0.16), (89, -0.21), (92, -0.25), (95, -0.31), (98, -0.42)]:
        q.append(Quote(P, k, EXP, bid=round(abs(d) * 8, 2), ask=round(abs(d) * 8 + 0.1, 2), iv=0.3, delta=d,
                       open_interest=1000))
    for k, d in [(102, 0.42), (105, 0.31), (108, 0.25), (111, 0.21), (114, 0.16), (117, 0.10), (120, 0.05)]:
        q.append(Quote(C, k, EXP, bid=round(d * 8, 2), ask=round(d * 8 + 0.1, 2), iv=0.3, delta=d, open_interest=1000))
    return q


RULES = BuildRules(0.20, 0.30, 3.0)


def test_build_condor_picks_delta_and_wings():
    p = build(Strategy.IRON_CONDOR, chain(), 100, AS_OF, RULES, ticker="ticker_a")
    strikes = {(g.option_type.value, g.side.value): g.strike for g in p.legs}
    assert strikes == {("put", "short"): 92, ("put", "long"): 89, ("call", "short"): 108, ("call", "long"): 111}
    assert p.ticker == "TICKER_A"


def test_build_respects_support_resistance_and_target():
    p = build(Strategy.IRON_CONDOR, chain(), 100, AS_OF, RULES, ticker="TICKER_A", support=90, resistance=110)
    assert p.leg(P, S).strike == 89 and p.leg(C, S).strike == 111
    with pytest.raises(BuildError, match="strike <= 104"):
        build(Strategy.CALL_CREDIT_SPREAD, chain(), 100, AS_OF, RULES, ticker="TICKER_A", analyst_target=104)


def test_build_needs_one_expiry_and_a_wing():
    mixed = chain() + [Quote(P, 50, EXP + timedelta(days=7), 1, 1, delta=-0.25)]
    with pytest.raises(BuildError, match="one expiry"):
        build(Strategy.PUT_CREDIT_SPREAD, mixed, 100, AS_OF, RULES, ticker="TICKER_A")
    lonely = [Quote(P, 92, EXP, 2.0, 2.1, delta=-0.25)]
    with pytest.raises(BuildError, match="wing"):
        build(Strategy.PUT_CREDIT_SPREAD, lonely, 100, AS_OF, RULES, ticker="TICKER_A")


def test_build_computes_missing_delta_from_iv():
    q = [Quote(P, k, EXP, 1.0, 1.2, iv=0.3) for k in (85, 88, 91, 94, 97)]
    p = build(Strategy.PUT_CREDIT_SPREAD, q, 100, AS_OF, RULES, ticker="TICKER_A")
    assert 0.20 <= abs(p.leg(P, S).delta) <= 0.30


def test_widen_until_pop_passes():
    p, n, _, steps = build_for_pop(Strategy.IRON_CONDOR, chain(), 100, AS_OF, RULES, 100_000, 0.10,
                                   ticker="TICKER_A")
    assert steps == 0 and n.pop >= 0.10
    p, n, _, steps = build_for_pop(Strategy.IRON_CONDOR, chain(), 100, AS_OF, RULES, 100_000, 0.99,
                                   ticker="TICKER_A")
    assert steps > 0 and abs(p.leg(P, S).delta) == 0.21  # widest inside the band, still failing


def test_condors_get_their_own_delta_band(pb):
    from options_copilot.calculators import delta_band
    assert delta_band(pb, Strategy.IRON_CONDOR) == (0.12, 0.20)
    assert delta_band(pb, Strategy.PUT_CREDIT_SPREAD) == (0.20, 0.30)
    rules = BuildRules.from_playbook(pb, Strategy.IRON_CONDOR)
    p = build(Strategy.IRON_CONDOR, chain(), 100, AS_OF, rules, ticker="TICKER_A")
    assert abs(p.leg(P, S).delta) == 0.16 and p.leg(C, S).delta == 0.16
