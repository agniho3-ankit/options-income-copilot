"""The verifier gate: a pure rule engine over a proposal, its numbers, and market facts.

Hard checks BLOCK on failure. Advisory checks can only add CAUTION. Any fact the rules
need but that is missing gives INSUFFICIENT_EVIDENCE, which fails the gate: missing
evidence is never safe. The LLM may add a veto (`with_veto`); nothing here can be
loosened by the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

from .calculators import compute, delta_band, years_between
from .models import (CheckResult, OptionType, Side, Strategy, TradeNumbers, TradeProposal, Verdict,
                     VerifierResult)
from .playbook import Playbook

B, C, I, OK = Verdict.BLOCK, Verdict.CAUTION, Verdict.INSUFFICIENT_EVIDENCE, Verdict.CLEAR


@dataclass(frozen=True)
class InsiderActivity:
    """Discretionary open-market Form 4 activity over the lookback window (shares)."""
    bought: float
    sold: float
    filings: int = 0


@dataclass(frozen=True)
class MarketFacts:
    """Everything the checks need besides the proposal. None means "not known"."""
    as_of: date
    spot: float
    portfolio_value: float
    open_max_losses: tuple[float, ...] = ()        # existing positions, dollars
    earnings_date: date | None = None
    earnings_known_none: bool = False              # the data source confirmed no upcoming date
    support: float | None = None
    resistance: float | None = None
    analyst_target: float | None = None
    insider: InsiderActivity | None = None
    landmine: Verdict | None = None
    landmine_reason: str = ""
    event_dates: tuple[tuple[date, str], ...] = ()  # (date, "event (source)") read by Python from cited filings
    # advisory inputs
    rsi: float | None = None
    iv_rank: float | None = None
    iv_rank_label: str = ""
    sma50: float | None = None
    sma200: float | None = None
    ex_dividend_date: date | None = None
    max_vol_oi_ratio: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _check(name: str, verdict: Verdict, reason: str, hard: bool = True) -> CheckResult:
    return CheckResult(name, verdict, reason, hard)


def _shorts(p: TradeProposal):
    return [g for g in p.legs if g.side is Side.SHORT]


# --- hard checks ---------------------------------------------------------------------------

def check_tier(p: TradeProposal, pb: Playbook) -> CheckResult:
    tier = pb.tier_of(p.ticker)
    if pb.is_enterable(p.ticker):
        return _check("tier", OK, f"{p.ticker} is {tier}")
    why = "on the watch list (not enterable until promoted)" if tier == "watch" else "outside the universe"
    return _check("tier", B, f"{p.ticker} is {why}")


def check_price(p: TradeProposal, f: MarketFacts, pb: Playbook) -> CheckResult:
    lo = pb["screener.price_min_usd"]
    if f.spot < lo:
        return _check("price_range", B, f"price {f.spot:.2f} is below the {lo:g} minimum")
    cap = pb["entry.cash_secured_put_price_max_usd"]
    if p.strategy is Strategy.CASH_SECURED_PUT and f.spot > cap:
        return _check("price_range", B, f"price {f.spot:.2f} is above the {cap:g} cash-secured-put cap")
    return _check("price_range", OK, f"price {f.spot:.2f} is in range")


def check_dte(p: TradeProposal, f: MarketFacts, pb: Playbook) -> CheckResult:
    dte = (p.expiry - f.as_of).days
    lo, hi = pb["entry.dte_min"], pb["entry.dte_max"]
    if not lo <= dte <= hi:
        return _check("dte", B, f"{dte} DTE is outside {lo}-{hi}")
    return _check("dte", OK, f"{dte} DTE (target {pb['entry.dte_target']})")


def check_delta(p: TradeProposal, pb: Playbook) -> CheckResult:
    lo, hi = delta_band(pb, p.strategy)
    for g in _shorts(p):
        if g.delta is None:
            return _check("short_delta", I, f"no delta for the short {g.option_type.value} {g.strike:g}")
        if not lo <= abs(g.delta) <= hi:
            return _check("short_delta", B, f"short {g.option_type.value} {g.strike:g} |delta| "
                                            f"{abs(g.delta):.2f} is outside {lo}-{hi}")
    return _check("short_delta", OK, "short deltas inside the band")


def check_pop(n: TradeNumbers, pb: Playbook) -> CheckResult:
    floor = pb["entry.pop_min"]
    if n.pop < floor:
        return _check("pop", B, f"PoP {n.pop:.1%} is below {floor:.0%}")
    return _check("pop", OK, f"PoP {n.pop:.1%}")


def check_position_risk(n: TradeNumbers, pb: Playbook) -> CheckResult:
    warn, block = pb["entry.position_risk_pct.warn"], pb["entry.position_risk_pct.block"]
    r = n.risk_pct_of_portfolio
    if r > block:
        return _check("position_risk", B, f"position risk {r:.2f}% is above {block}%")
    if r > warn:
        return _check("position_risk", C, f"position risk {r:.2f}% is above the {warn}% warning level")
    return _check("position_risk", OK, f"position risk {r:.2f}%")


def check_heat(p: TradeProposal, n: TradeNumbers, f: MarketFacts, pb: Playbook) -> CheckResult:
    warn, block = pb["entry.portfolio_heat_pct.warn"], pb["entry.portfolio_heat_pct.block"]
    heat = (sum(f.open_max_losses) + n.max_loss * p.contracts) / f.portfolio_value * 100
    if heat > block:
        return _check("portfolio_heat", B, f"heat after this trade {heat:.2f}% is above {block}%")
    if heat > warn:
        return _check("portfolio_heat", C, f"heat after this trade {heat:.2f}% is above the {warn}% warning")
    return _check("portfolio_heat", OK, f"heat after this trade {heat:.2f}%")


def check_earnings(p: TradeProposal, f: MarketFacts, pb: Playbook) -> CheckResult:
    if f.earnings_date is None:
        if f.earnings_known_none:
            return _check("earnings", OK, "no upcoming earnings date reported")
        return _check("earnings", I, "earnings date unknown")
    days = (f.earnings_date - f.as_of).days
    if days < 0:
        return _check("earnings", I, f"earnings date {f.earnings_date} is in the past; next date unknown")
    if days <= pb["entry.earnings.block_within_days"]:
        return _check("earnings", B, f"earnings in {days} days")
    if pb["entry.earnings.expiry_must_precede"] and p.expiry >= f.earnings_date:
        return _check("earnings", B, f"expiry {p.expiry} is on/after earnings {f.earnings_date}")
    if days <= pb["entry.earnings.spreads_only_within_days"] and p.strategy is Strategy.CASH_SECURED_PUT:
        return _check("earnings", B, f"earnings in {days} days: defined-risk spreads only")
    return _check("earnings", OK, f"earnings {f.earnings_date} is after expiry {p.expiry}")


def check_support_resistance(p: TradeProposal, f: MarketFacts) -> CheckResult:
    sp, sc = p.leg(OptionType.PUT, Side.SHORT), p.leg(OptionType.CALL, Side.SHORT)
    if sp is not None:
        if f.support is None:
            return _check("support_resistance", I, "support level unknown")
        if sp.strike > f.support:
            return _check("support_resistance", B, f"short put {sp.strike:g} is above support {f.support:.2f}")
    if sc is not None:
        if f.resistance is None:
            return _check("support_resistance", I, "resistance level unknown")
        if sc.strike < f.resistance:
            return _check("support_resistance", B,
                          f"short call {sc.strike:g} is below resistance {f.resistance:.2f}")
    return _check("support_resistance", OK, "short strikes sit beyond support/resistance")


def check_analyst_target(p: TradeProposal, f: MarketFacts) -> CheckResult | None:
    sc = p.leg(OptionType.CALL, Side.SHORT)
    if sc is None:
        return None
    if f.analyst_target is None:
        return _check("short_call_vs_target", I, "analyst price target unknown")
    if sc.strike > f.analyst_target:
        return _check("short_call_vs_target", B,
                      f"short call {sc.strike:g} is above the analyst target {f.analyst_target:.2f}")
    return _check("short_call_vs_target", OK, f"short call {sc.strike:g} <= target {f.analyst_target:.2f}")


def check_open_interest(p: TradeProposal, pb: Playbook) -> CheckResult:
    floor = pb["entry.open_interest_min"]
    for g in _shorts(p):
        if g.open_interest is None:
            return _check("open_interest", I, f"open interest unknown at {g.option_type.value} {g.strike:g}")
        if g.open_interest < floor:
            return _check("open_interest", B, f"open interest {g.open_interest} at {g.option_type.value} "
                                              f"{g.strike:g} is below {floor}")
    return _check("open_interest", OK, f"open interest >= {floor} at each short strike")


def check_insider(f: MarketFacts, pb: Playbook) -> CheckResult:
    if f.insider is None:
        return _check("insider", I, "Form 4 activity unknown")
    ratio_max = pb["screener.exclusions.insider_sell_buy_ratio_max"]
    a = f.insider
    if a.sold <= 0:
        return _check("insider", OK, "no discretionary insider sales")
    if a.bought <= 0:
        return _check("insider", C, f"discretionary insider sales ({a.sold:,.0f} sh) with no open-market buys")
    if a.sold > ratio_max * a.bought:
        return _check("insider", B, f"insider sales {a.sold:,.0f} sh are over {ratio_max:g}x buys {a.bought:,.0f} sh")
    return _check("insider", OK, f"insider sales {a.sold:,.0f} sh vs buys {a.bought:,.0f} sh")


def check_landmine(f: MarketFacts) -> CheckResult:
    if f.landmine is None:
        return _check("landmine", I, "filings landmine check not run")
    return _check("landmine", f.landmine, f.landmine_reason or f"landmine check: {f.landmine.value}")


def check_event_window(p: TradeProposal, f: MarketFacts) -> CheckResult:
    """Cited event dates vs the trade window (as-of to expiry). Dates come from events.py,
    never from the LLM's reading of the filing."""
    upcoming = sorted((d, why) for d, why in f.event_dates if d >= f.as_of)
    inside = [(d, why) for d, why in upcoming if d <= p.expiry]
    if inside:
        d, why = inside[0]
        return _check("event_window", B, f"cited event {why} on {d} falls inside the trade window "
                                         f"(expiry {p.expiry})")
    if upcoming:
        d, why = upcoming[0]
        return _check("event_window", C, f"cited event {why} on {d} is after expiry {p.expiry}")
    return _check("event_window", OK, "no cited event date inside the trade window")


# --- advisory checks (CAUTION at most) ------------------------------------------------------

def advisory(p: TradeProposal, f: MarketFacts, pb: Playbook) -> list[CheckResult]:
    out: list[CheckResult] = []
    has_put = p.leg(OptionType.PUT, Side.SHORT) is not None
    has_call = p.leg(OptionType.CALL, Side.SHORT) is not None

    def add(name: str, caution: bool, reason: str) -> None:
        out.append(_check(name, C if caution else OK, reason, hard=False))

    if f.rsi is None:
        add("rsi", True, "RSI unavailable")
    else:
        lo, hi = pb["entry.advisory.rsi_oversold"], pb["entry.advisory.rsi_overbought"]
        risky = (has_put and f.rsi < lo) or (has_call and f.rsi > hi)
        add("rsi", risky, f"RSI {f.rsi:.0f}" + (" (stretched against a short side)" if risky else ""))

    if f.iv_rank is None:
        add("iv_rank", True, "IV rank unavailable")
    else:
        low = pb["entry.advisory.iv_rank_low"]
        label = f" ({f.iv_rank_label})" if f.iv_rank_label else ""
        add("iv_rank", f.iv_rank < low, f"IV rank {f.iv_rank:.0f}{label}"
            + (f": below {low}, thin premium" if f.iv_rank < low else ""))

    if f.sma200 is None:
        add("trend_200dma", True, "200-day average unavailable")
    else:
        below = f.spot < f.sma200
        add("trend_200dma", below and has_put, f"price {'below' if below else 'above'} the 200-day average "
                                               f"{f.sma200:.2f}")

    within = pb["entry.advisory.dividend_exdate_within_days"]
    if f.ex_dividend_date is not None and has_call and 0 <= (f.ex_dividend_date - f.as_of).days <= within:
        add("ex_dividend", True, f"ex-dividend {f.ex_dividend_date} with a short call open")
    else:
        add("ex_dividend", False, "no ex-dividend date near a short call")

    if f.max_vol_oi_ratio is not None:
        hot = f.max_vol_oi_ratio > pb["entry.advisory.unusual_flow_vol_oi_ratio"]
        add("unusual_flow", hot, f"max volume/OI {f.max_vol_oi_ratio:.1f}")
    return out


# --- entry point ---------------------------------------------------------------------------

def verify(p: TradeProposal, f: MarketFacts, pb: Playbook, numbers: TradeNumbers | None = None) -> VerifierResult:
    """Run every check. Numbers are recomputed here unless supplied by the calculators."""
    if numbers is None:
        numbers, _ = compute(p, f.spot, f.as_of, f.portfolio_value)
    checks = [
        check_tier(p, pb), check_price(p, f, pb), check_dte(p, f, pb), check_delta(p, pb),
        check_pop(numbers, pb), check_position_risk(numbers, pb), check_heat(p, numbers, f, pb),
        check_earnings(p, f, pb), check_support_resistance(p, f), check_open_interest(p, pb),
        check_insider(f, pb), check_landmine(f), check_event_window(p, f),
    ]
    target = check_analyst_target(p, f)
    if target is not None:
        checks.insert(9, target)
    checks += advisory(p, f, pb)
    return VerifierResult(tuple(checks), pb.sha256)


def with_veto(result: VerifierResult, reason: str, block: bool = True) -> VerifierResult:
    """Add an LLM veto. It can only make the result stricter."""
    veto = _check("llm_veto", B if block else C, reason, hard=False)
    return replace(result, checks=result.checks + (veto,))


__all__ = ["InsiderActivity", "MarketFacts", "verify", "with_veto", "years_between"]
