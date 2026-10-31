"""Deterministic trade math: Black-Scholes, credit, max loss, breakevens, PoP, risk, strikes.

The LLM never does arithmetic. Every number in a brief comes from these functions.
Dollar figures are per contract (100 shares) unless the name says otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .models import OptionLeg, OptionType, Side, Strategy, TradeNumbers, TradeProposal

MULTIPLIER = 100


class CalcError(ValueError):
    pass


# --- Black-Scholes -------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def years_between(start: date, end: date) -> float:
    return max((end - start).days, 0) / 365.0


def bs_delta(spot: float, strike: float, iv: float, t: float, option_type: OptionType, r: float = 0.0) -> float:
    """Black-Scholes delta (calls positive, puts negative)."""
    if spot <= 0 or strike <= 0:
        raise CalcError("spot and strike must be positive")
    if t <= 0 or iv <= 0:
        itm = spot > strike if option_type is OptionType.CALL else spot < strike
        return (1.0 if itm else 0.0) * (1 if option_type is OptionType.CALL else -1)
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    return norm_cdf(d1) if option_type is OptionType.CALL else norm_cdf(d1) - 1.0


def bs_price(spot: float, strike: float, iv: float, t: float, option_type: OptionType, r: float = 0.0) -> float:
    """Black-Scholes premium per share (intrinsic value at or after expiry)."""
    if spot <= 0 or strike <= 0:
        raise CalcError("spot and strike must be positive")
    if t <= 0 or iv <= 0:
        return max(spot - strike, 0.0) if option_type is OptionType.CALL else max(strike - spot, 0.0)
    sq = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * t) / sq
    d2 = d1 - sq
    disc = math.exp(-r * t)
    if option_type is OptionType.CALL:
        return spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    return strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)


def limit_credit_per_share(credit_per_contract: float) -> float:
    """Limit price for a credit order: the mid rounded DOWN to the cent, so it sits at or
    below the market and can fill at the price the brief showed."""
    return math.floor(credit_per_contract / MULTIPLIER * 100 + 1e-6) / 100


def limit_debit_per_share(debit_per_contract: float) -> float:
    """Limit price for a debit (closing) order: the mid rounded UP to the cent."""
    return math.ceil(debit_per_contract / MULTIPLIER * 100 - 1e-6) / 100


def intrinsic_close_cost(legs: tuple[OptionLeg, ...], spot: float) -> float:
    """Debit per contract to settle legs at expiry: shorts' intrinsic minus longs'."""
    total = 0.0
    for g in legs:
        v = max(spot - g.strike, 0.0) if g.option_type is OptionType.CALL else max(g.strike - spot, 0.0)
        total += v if g.side is Side.SHORT else -v
    return round(total * MULTIPLIER, 2)


def prob_above(spot: float, level: float, iv: float, t: float, r: float = 0.0) -> float:
    """Risk-neutral P(S_T > level) under a lognormal model (N(d2))."""
    if level <= 0:
        return 1.0
    if t <= 0 or iv <= 0:
        return 1.0 if spot > level else 0.0
    d2 = (math.log(spot / level) + (r - iv * iv / 2) * t) / (iv * math.sqrt(t))
    return norm_cdf(d2)


def prob_between(spot: float, lo: float, hi: float, iv: float, t: float, r: float = 0.0) -> float:
    if hi <= lo:
        return 0.0
    return max(prob_above(spot, lo, iv, t, r) - prob_above(spot, hi, iv, t, r), 0.0)


# --- trade numbers --------------------------------------------------------------------------

def _legs(p: TradeProposal) -> dict[tuple[OptionType, Side], OptionLeg]:
    return {(g.option_type, g.side): g for g in p.legs}


def net_credit_per_share(p: TradeProposal) -> float:
    """Sum of short mids minus long mids (playbook: credit priced at mid)."""
    return sum(g.mid if g.side is Side.SHORT else -g.mid for g in p.legs)


@dataclass(frozen=True)
class Extras:
    """Brief-only figures that are not verifier inputs."""
    max_profit: float
    profit_zone_width: float | None
    profit_zone_pct: float | None
    risk_reward: float | None


def compute(p: TradeProposal, spot: float, as_of: date, portfolio_value: float,
            iv: float | None = None, r: float = 0.0) -> tuple[TradeNumbers, Extras]:
    """Credit, max loss, breakevens, PoP, and risk for a proposal.

    PoP is P(the stock finishes between the breakevens) with a lognormal model. `iv`
    defaults to the average IV of the short legs.
    """
    if portfolio_value <= 0 or spot <= 0:
        raise CalcError("portfolio value and spot must be positive")
    legs = _legs(p)
    credit_ps = net_credit_per_share(p)
    if credit_ps <= 0:
        raise CalcError(f"not a credit trade: net credit {credit_ps:.2f} per share")
    shorts = [g for g in p.legs if g.side is Side.SHORT]
    if iv is None:
        ivs = [g.iv for g in shorts if g.iv]
        if not ivs:
            raise CalcError("no implied volatility on the short legs")
        iv = sum(ivs) / len(ivs)
    t = years_between(as_of, p.expiry)
    sp, lp = legs.get((OptionType.PUT, Side.SHORT)), legs.get((OptionType.PUT, Side.LONG))
    sc, lc = legs.get((OptionType.CALL, Side.SHORT)), legs.get((OptionType.CALL, Side.LONG))

    if p.strategy is Strategy.PUT_CREDIT_SPREAD:
        width = sp.strike - lp.strike
        if width <= 0:
            raise CalcError("long put must be below the short put")
        be = (sp.strike - credit_ps,)
        pop = prob_above(spot, be[0], iv, t, r)
        max_loss_ps = width - credit_ps
        collateral_ps = max_loss_ps
    elif p.strategy is Strategy.CALL_CREDIT_SPREAD:
        width = lc.strike - sc.strike
        if width <= 0:
            raise CalcError("long call must be above the short call")
        be = (sc.strike + credit_ps,)
        pop = 1.0 - prob_above(spot, be[0], iv, t, r)
        max_loss_ps = width - credit_ps
        collateral_ps = max_loss_ps
    elif p.strategy is Strategy.IRON_CONDOR:
        put_w, call_w = sp.strike - lp.strike, lc.strike - sc.strike
        if put_w <= 0 or call_w <= 0 or sp.strike >= sc.strike:
            raise CalcError("condor needs long put < short put < short call < long call")
        be = (sp.strike - credit_ps, sc.strike + credit_ps)
        pop = prob_between(spot, be[0], be[1], iv, t, r)
        max_loss_ps = max(put_w, call_w) - credit_ps  # only one side can finish in the money
        collateral_ps = max_loss_ps
    elif p.strategy is Strategy.CASH_SECURED_PUT:
        be = (sp.strike - credit_ps,)
        pop = prob_above(spot, be[0], iv, t, r)
        max_loss_ps = sp.strike - credit_ps            # assignment with the stock at zero
        collateral_ps = sp.strike
    else:  # pragma: no cover - enum is exhaustive
        raise CalcError(f"unsupported strategy {p.strategy}")
    if max_loss_ps <= 0:
        raise CalcError("credit is at least the spread width; quotes look wrong")

    n = p.contracts
    credit, max_loss = credit_ps * MULTIPLIER, max_loss_ps * MULTIPLIER
    numbers = TradeNumbers(
        credit=round(credit, 2),
        max_loss=round(max_loss, 2),
        breakevens=tuple(round(b, 2) for b in be),
        pop=round(pop, 4),
        risk_pct_of_portfolio=round(max_loss * n / portfolio_value * 100, 3),
        collateral=round(collateral_ps * MULTIPLIER * n, 2),
    )
    zone = (be[1] - be[0]) if len(be) == 2 else None
    extras = Extras(max_profit=round(credit * n, 2), profit_zone_width=round(zone, 2) if zone else None,
                    profit_zone_pct=round(zone / spot * 100, 2) if zone else None,
                    risk_reward=round(max_loss / credit, 2))
    return numbers, extras


def close_cost_per_contract(p_legs: tuple[OptionLeg, ...], current_mids: dict[tuple[str, str, float], float]) -> float:
    """Debit to close now: buy back shorts, sell longs, at current mids. Keys are
    (option_type, side, strike)."""
    total = 0.0
    for g in p_legs:
        key = (g.option_type.value, g.side.value, g.strike)
        if key not in current_mids:
            raise CalcError(f"no current quote for {key}")
        total += current_mids[key] if g.side is Side.SHORT else -current_mids[key]
    return round(total * MULTIPLIER, 2)


def portfolio_heat_pct(max_losses: list[float], portfolio_value: float) -> float:
    """Sum of open max losses as % of the portfolio."""
    if portfolio_value <= 0:
        raise CalcError("portfolio value must be positive")
    return round(sum(max_losses) / portfolio_value * 100, 3)


# --- strike selection (Skill #3) -----------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    """One listed option from a chain (public market data)."""
    option_type: OptionType
    strike: float
    expiry: date
    bid: float
    ask: float
    iv: float | None = None
    delta: float | None = None
    open_interest: int | None = None

    def leg(self, side: Side, spot: float | None = None, as_of: date | None = None) -> OptionLeg:
        delta = self.delta
        if delta is None and self.iv and spot and as_of:
            delta = bs_delta(spot, self.strike, self.iv, years_between(as_of, self.expiry), self.option_type)
        return OptionLeg(self.option_type, side, self.strike, self.expiry, self.bid, self.ask,
                         delta=None if delta is None else round(delta, 4), iv=self.iv,
                         open_interest=self.open_interest)


@dataclass(frozen=True)
class BuildRules:
    short_delta_min: float
    short_delta_max: float
    wing_width_pct: float

    @classmethod
    def from_playbook(cls, pb, strategy: Strategy) -> "BuildRules":
        lo, hi = delta_band(pb, strategy)
        return cls(lo, hi, pb["builder.wing_width_pct"])


def delta_band(pb, strategy: Strategy) -> tuple[float, float]:
    """Short-strike |delta| band: condors have their own, further-out band."""
    if strategy is Strategy.IRON_CONDOR:
        return pb["builder.condor_short_delta_min"], pb["builder.condor_short_delta_max"]
    return pb["builder.short_delta_min"], pb["builder.short_delta_max"]


class BuildError(ValueError):
    pass


def _delta(q: Quote, spot: float, as_of: date) -> float | None:
    if q.delta is not None:
        return q.delta
    if q.iv:
        return bs_delta(spot, q.strike, q.iv, years_between(as_of, q.expiry), q.option_type)
    return None


def pick_short(chain: list[Quote], option_type: OptionType, spot: float, as_of: date, rules: BuildRules,
               *, max_strike: float | None = None, min_strike: float | None = None,
               target: float | None = None) -> Quote:
    """The quote whose |delta| is inside the band and closest to `target` (default: the
    band midpoint), within bounds.

    Puts are bounded above (`max_strike`, e.g. support); calls below (`min_strike`, e.g.
    resistance) and above (`max_strike`, e.g. the analyst target).
    """
    if target is None:
        target = (rules.short_delta_min + rules.short_delta_max) / 2
    best, best_gap = None, None
    for q in chain:
        if q.option_type is not option_type or q.bid <= 0:
            continue
        if max_strike is not None and q.strike > max_strike:
            continue
        if min_strike is not None and q.strike < min_strike:
            continue
        d = _delta(q, spot, as_of)
        if d is None or not rules.short_delta_min <= abs(d) <= rules.short_delta_max:
            continue
        gap = abs(abs(d) - target)
        if best_gap is None or gap < best_gap or (gap == best_gap and q.strike < best.strike):
            best, best_gap = q, gap
    if best is None:
        bounds = []
        if max_strike is not None:
            bounds.append(f"strike <= {max_strike:g}")
        if min_strike is not None:
            bounds.append(f"strike >= {min_strike:g}")
        raise BuildError(f"no {option_type.value} with |delta| {rules.short_delta_min}-{rules.short_delta_max}"
                         + (f" and {' and '.join(bounds)}" if bounds else ""))
    return best


def pick_wing(chain: list[Quote], short: Quote, spot: float, rules: BuildRules) -> Quote:
    """Listed strike nearest short -/+ wing_width_pct of spot, strictly beyond the short."""
    offset = spot * rules.wing_width_pct / 100
    want = short.strike - offset if short.option_type is OptionType.PUT else short.strike + offset
    beyond = [q for q in chain if q.option_type is short.option_type and q.expiry == short.expiry
              and (q.strike < short.strike if short.option_type is OptionType.PUT else q.strike > short.strike)]
    if not beyond:
        raise BuildError(f"no listed {short.option_type.value} strike beyond {short.strike:g} for the wing")
    return min(beyond, key=lambda q: (abs(q.strike - want), q.strike))


def build(strategy: Strategy, chain: list[Quote], spot: float, as_of: date, rules: BuildRules, *,
          ticker: str, support: float | None = None, resistance: float | None = None,
          analyst_target: float | None = None, contracts: int = 1,
          target_delta: float | None = None) -> TradeProposal:
    """Pick strikes for one expiry's chain (Skill #3) and return a proposal.

    Delta governs; support, resistance, and the analyst target are bounds. Missing levels
    are not bounds here: the verifier reports them as missing evidence.
    """
    expiries = {q.expiry for q in chain}
    if len(expiries) != 1:
        raise BuildError("pass the chain for exactly one expiry")
    legs: list[OptionLeg] = []
    if strategy in (Strategy.PUT_CREDIT_SPREAD, Strategy.IRON_CONDOR, Strategy.CASH_SECURED_PUT):
        sp = pick_short(chain, OptionType.PUT, spot, as_of, rules, max_strike=support, target=target_delta)
        legs.append(sp.leg(Side.SHORT, spot, as_of))
        if strategy is not Strategy.CASH_SECURED_PUT:
            legs.append(pick_wing(chain, sp, spot, rules).leg(Side.LONG, spot, as_of))
    if strategy in (Strategy.CALL_CREDIT_SPREAD, Strategy.IRON_CONDOR):
        sc = pick_short(chain, OptionType.CALL, spot, as_of, rules, min_strike=resistance,
                        max_strike=analyst_target, target=target_delta)
        legs.append(sc.leg(Side.SHORT, spot, as_of))
        legs.append(pick_wing(chain, sc, spot, rules).leg(Side.LONG, spot, as_of))
    return TradeProposal(ticker=ticker.upper(), strategy=strategy, legs=tuple(legs), contracts=contracts)


def build_for_pop(strategy: Strategy, chain: list[Quote], spot: float, as_of: date, rules: BuildRules,
                  portfolio_value: float, pop_min: float, *, step: float = 0.01,
                  **bounds) -> tuple[TradeProposal, TradeNumbers, Extras, int]:
    """Skill #3 "widen until PoP passes": start at the delta-band midpoint and move the
    short strikes out (lower delta) one step at a time, never below the band minimum.

    Returns the first build whose PoP clears `pop_min`, or the widest one tried (which the
    verifier will then BLOCK on PoP), plus how many widening steps were taken.
    """
    start = (rules.short_delta_min + rules.short_delta_max) / 2
    targets, t = [], start
    while t >= rules.short_delta_min - 1e-9:
        targets.append(round(t, 4))
        t -= step
    last, seen = None, set()
    for steps, target in enumerate(targets):
        try:
            p = build(strategy, chain, spot, as_of, rules, target_delta=target, **bounds)
        except BuildError:
            if last is None and steps == len(targets) - 1:
                raise
            continue
        key = tuple(g.strike for g in p.legs)
        if key in seen:
            continue
        seen.add(key)
        numbers, extras = compute(p, spot, as_of, portfolio_value)
        last = (p, numbers, extras, steps)
        if numbers.pop >= pop_min:
            return last
    if last is None:
        raise BuildError("no strikes inside the delta band satisfy the bounds")
    return last
