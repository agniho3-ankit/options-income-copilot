"""Verifier checks, including the boundary values of every threshold."""
from dataclasses import replace
from datetime import date, timedelta

import pytest

from options_copilot.calculators import compute
from options_copilot.models import OptionLeg, OptionType, Side, Strategy, TradeNumbers, TradeProposal, Verdict
from options_copilot.verifier import InsiderActivity, MarketFacts, verify, with_veto

AS_OF = date(2026, 10, 30)
EXP = AS_OF + timedelta(days=45)
P, C, S, L = OptionType.PUT, OptionType.CALL, Side.SHORT, Side.LONG


def leg(t, side, k, bid, ask, delta, oi=1000):
    return OptionLeg(t, side, k, EXP, bid, ask, delta=delta, iv=0.30, open_interest=oi)


def pcs(ticker="MRVL", **kw):
    """A put credit spread on a Tier-1 name (the universe is real; the trade is synthetic)."""
    return TradeProposal(ticker, Strategy.PUT_CREDIT_SPREAD,
                         (leg(P, S, 90, 1.0, 1.1, -0.22, **kw), leg(P, L, 87, 0.45, 0.55, -0.15)))


def condor(short_call=110, delta=0.16):
    return TradeProposal("MRVL", Strategy.IRON_CONDOR, (
        leg(P, S, 90, 1.0, 1.1, -0.16), leg(P, L, 87, 0.45, 0.55, -0.15),
        leg(C, S, short_call, 1.0, 1.1, delta), leg(C, L, short_call + 3, 0.45, 0.55, 0.15)))


def facts(**kw):
    base = dict(as_of=AS_OF, spot=100.0, portfolio_value=100_000.0, earnings_date=AS_OF + timedelta(days=60),
                support=92.0, resistance=108.0, analyst_target=120.0, insider=InsiderActivity(0, 0, 3),
                landmine=Verdict.CLEAR, rsi=50.0, iv_rank=50.0, sma200=95.0)
    base.update(kw)
    return MarketFacts(**base)


def by_name(result):
    return {c.name: c for c in result.checks}


def nums(pop=0.70, risk=1.0, max_loss=1000.0):
    return TradeNumbers(credit=50, max_loss=max_loss, breakevens=(89.5,), pop=pop, risk_pct_of_portfolio=risk,
                        collateral=max_loss)


def test_clean_trade_passes(pb):
    r = verify(pcs(), facts(), pb, nums())
    assert r.verdict is Verdict.CLEAR and r.passed and r.failures == []
    assert r.playbook_sha256 == pb.sha256


@pytest.mark.parametrize("pop, verdict", [(0.6499, Verdict.BLOCK), (0.65, Verdict.CLEAR)])
def test_pop_boundary(pb, pop, verdict):
    assert by_name(verify(pcs(), facts(), pb, nums(pop=pop)))["pop"].verdict is verdict


@pytest.mark.parametrize("risk, verdict", [(2.0, Verdict.CLEAR), (2.01, Verdict.CAUTION), (5.0, Verdict.CAUTION),
                                           (5.01, Verdict.BLOCK)])
def test_position_risk_bands(pb, risk, verdict):
    assert by_name(verify(pcs(), facts(), pb, nums(risk=risk)))["position_risk"].verdict is verdict


@pytest.mark.parametrize("existing, verdict", [(19_000, Verdict.CLEAR), (19_500, Verdict.CAUTION),
                                               (29_000, Verdict.CAUTION), (29_500, Verdict.BLOCK)])
def test_heat_bands_include_this_trade(pb, existing, verdict):
    r = verify(pcs(), facts(open_max_losses=(existing,)), pb, nums(max_loss=1000))
    assert by_name(r)["portfolio_heat"].verdict is verdict


@pytest.mark.parametrize("days, verdict", [(7, Verdict.BLOCK), (8, Verdict.BLOCK), (46, Verdict.CLEAR)])
def test_earnings(pb, days, verdict):
    # 7 days: inside the block window; 8 days: expiry (45 DTE) is after earnings; 46: expiry precedes it
    r = verify(pcs(), facts(earnings_date=AS_OF + timedelta(days=days)), pb, nums())
    assert by_name(r)["earnings"].verdict is verdict


def test_earnings_missing_is_insufficient_unless_confirmed_none(pb):
    assert by_name(verify(pcs(), facts(earnings_date=None), pb, nums()))["earnings"].verdict \
        is Verdict.INSUFFICIENT_EVIDENCE
    r = verify(pcs(), facts(earnings_date=None, earnings_known_none=True), pb, nums())
    assert by_name(r)["earnings"].verdict is Verdict.CLEAR


def test_csp_rules(pb):
    csp = TradeProposal("MRVL", Strategy.CASH_SECURED_PUT, (leg(P, S, 90, 1.0, 1.1, -0.22),))
    assert by_name(verify(csp, facts(spot=301.0, support=400.0), pb, nums()))["price_range"].verdict is Verdict.BLOCK
    assert by_name(verify(csp, facts(spot=300.0), pb, nums()))["price_range"].verdict is Verdict.CLEAR
    assert by_name(verify(pcs(), facts(spot=19.99, support=95), pb, nums()))["price_range"].verdict is Verdict.BLOCK
    # earnings 20 days out: spreads only, so a CSP is blocked (expiry rule aside)
    near = verify(csp, facts(earnings_date=AS_OF + timedelta(days=20)), pb, nums())
    assert by_name(near)["earnings"].verdict is Verdict.BLOCK


def test_tier_rules(pb):
    assert by_name(verify(pcs("SCHW"), facts(), pb, nums()))["tier"].verdict is Verdict.BLOCK   # watch list
    assert by_name(verify(pcs("TICKER_A"), facts(), pb, nums()))["tier"].verdict is Verdict.BLOCK


@pytest.mark.parametrize("support, verdict", [(90.0, Verdict.CLEAR), (89.99, Verdict.BLOCK),
                                              (None, Verdict.INSUFFICIENT_EVIDENCE)])
def test_short_put_at_or_below_support(pb, support, verdict):
    assert by_name(verify(pcs(), facts(support=support), pb, nums()))["support_resistance"].verdict is verdict


def test_short_call_resistance_and_target(pb):
    r = by_name(verify(condor(110), facts(resistance=110.01), pb, nums()))
    assert r["support_resistance"].verdict is Verdict.BLOCK
    r = by_name(verify(condor(110), facts(resistance=110, analyst_target=109), pb, nums()))
    assert r["support_resistance"].verdict is Verdict.CLEAR and r["short_call_vs_target"].verdict is Verdict.BLOCK
    r = by_name(verify(condor(110), facts(analyst_target=None), pb, nums()))
    assert r["short_call_vs_target"].verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert "short_call_vs_target" not in by_name(verify(pcs(), facts(), pb, nums()))


@pytest.mark.parametrize("oi, verdict", [(500, Verdict.CLEAR), (499, Verdict.BLOCK),
                                         (None, Verdict.INSUFFICIENT_EVIDENCE)])
def test_open_interest(pb, oi, verdict):
    assert by_name(verify(pcs(oi=oi), facts(), pb, nums()))["open_interest"].verdict is verdict


def test_delta_band(pb):
    # condors use their own band (0.12-0.20); spreads keep 0.20-0.30
    assert by_name(verify(condor(delta=0.21), facts(), pb, nums()))["short_delta"].verdict is Verdict.BLOCK
    assert by_name(verify(condor(delta=0.20), facts(), pb, nums()))["short_delta"].verdict is Verdict.CLEAR
    assert by_name(verify(condor(delta=0.11), facts(), pb, nums()))["short_delta"].verdict is Verdict.BLOCK
    assert by_name(verify(pcs(), facts(), pb, nums()))["short_delta"].verdict is Verdict.CLEAR  # 0.22 put spread


@pytest.mark.parametrize("days, verdict", [(29, Verdict.BLOCK), (30, Verdict.CLEAR), (60, Verdict.CLEAR),
                                           (61, Verdict.BLOCK)])
def test_dte_window(pb, days, verdict):
    r = verify(pcs(), facts(as_of=EXP - timedelta(days=days), earnings_date=EXP + timedelta(days=5)), pb, nums())
    assert by_name(r)["dte"].verdict is verdict


@pytest.mark.parametrize("insider, verdict", [
    (None, Verdict.INSUFFICIENT_EVIDENCE), (InsiderActivity(0, 0), Verdict.CLEAR),
    (InsiderActivity(0, 100), Verdict.CAUTION), (InsiderActivity(100, 200), Verdict.CLEAR),
    (InsiderActivity(100, 201), Verdict.BLOCK)])
def test_insider_rule(pb, insider, verdict):
    assert by_name(verify(pcs(), facts(insider=insider), pb, nums()))["insider"].verdict is verdict


@pytest.mark.parametrize("landmine, verdict, passed", [
    (None, Verdict.INSUFFICIENT_EVIDENCE, False), (Verdict.CAUTION, Verdict.CAUTION, True),
    (Verdict.BLOCK, Verdict.BLOCK, False), (Verdict.INSUFFICIENT_EVIDENCE, Verdict.INSUFFICIENT_EVIDENCE, False)])
def test_landmine_feeds_the_gate(pb, landmine, verdict, passed):
    r = verify(pcs(), facts(landmine=landmine), pb, nums())
    assert by_name(r)["landmine"].verdict is verdict and r.passed is passed


def test_advisories_only_caution(pb):
    r = verify(pcs(), facts(rsi=25, iv_rank=10, sma200=120, ex_dividend_date=AS_OF + timedelta(days=3),
                            max_vol_oi_ratio=5.0), pb, nums())
    adv = [c for c in r.checks if not c.hard]
    assert {c.verdict for c in adv} <= {Verdict.CAUTION, Verdict.CLEAR}
    assert by_name(r)["rsi"].verdict is Verdict.CAUTION and by_name(r)["iv_rank"].verdict is Verdict.CAUTION
    assert by_name(r)["ex_dividend"].verdict is Verdict.CLEAR       # no short call in a put spread
    assert r.verdict is Verdict.CAUTION and r.passed
    missing = verify(pcs(), facts(rsi=None, iv_rank=None, sma200=None), pb, nums())
    assert missing.verdict is Verdict.CAUTION                       # missing advisory data is never "safe"


def test_veto_only_tightens(pb):
    r = verify(pcs(), facts(), pb, nums())
    assert with_veto(r, "LLM sees a sector risk").verdict is Verdict.BLOCK
    assert with_veto(r, "minor", block=False).verdict is Verdict.CAUTION
    blocked = verify(pcs(), facts(landmine=Verdict.BLOCK), pb, nums())
    assert with_veto(blocked, "fine", block=False).verdict is Verdict.BLOCK


def test_numbers_are_computed_when_not_supplied(pb):
    r = verify(pcs(), facts(), pb)
    n, _ = compute(pcs(), 100, AS_OF, 100_000)
    assert by_name(r)["pop"].reason == f"PoP {n.pop:.1%}" or by_name(r)["pop"].verdict is Verdict.BLOCK


def test_verifier_is_pure(pb):
    f, p, n = facts(), pcs(), nums()
    assert verify(p, f, pb, n) == verify(p, f, pb, n)
    assert replace(f) == f
