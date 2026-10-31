"""Skill #1: the transparent 0-10 quality composite and the screen gates.

Four components, each worth its playbook weight (2.5 by default). Every component says
why it scored what it did. A missing input scores 0 and is named, so a ticker never
passes the screen on missing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Candidate
from .playbook import Playbook
from .verifier import InsiderActivity


@dataclass(frozen=True)
class ScreenInputs:
    ticker: str
    price: float
    market_cap: float | None
    sector: str | None
    country: str | None
    analyst_buy_ratio: float | None
    analyst_count: int | None
    analyst_target: float | None
    sma200: float | None
    net_income_positive: bool | None
    revenue_growth: float | None       # latest fiscal year vs prior, as a fraction
    insider: InsiderActivity | None
    reverse_split_recent: bool | None = None


def score_analyst(x: ScreenInputs, pb: Playbook, w: float) -> tuple[float, str]:
    if x.analyst_buy_ratio is None or x.analyst_count is None:
        return 0.0, "analyst data missing"
    if x.analyst_count < pb["screener.analyst_count_min"]:
        return 0.0, f"only {x.analyst_count} analysts"
    floor = pb["screener.analyst_buy_ratio_min"]
    s = w * min(x.analyst_buy_ratio / floor, 1.0)
    return s, f"{x.analyst_buy_ratio:.0%} buy ratings from {x.analyst_count} analysts"


def score_fundamentals(x: ScreenInputs, w: float) -> tuple[float, str]:
    parts, s = [], 0.0
    if x.net_income_positive is None:
        parts.append("net income missing")
    elif x.net_income_positive:
        s += w / 2
        parts.append("profitable")
    else:
        parts.append("unprofitable")
    if x.revenue_growth is None:
        parts.append("revenue growth missing")
    elif x.revenue_growth > 0:
        s += w / 2
        parts.append(f"revenue +{x.revenue_growth:.0%}")
    else:
        parts.append(f"revenue {x.revenue_growth:.0%}")
    return s, ", ".join(parts)


def score_trend(x: ScreenInputs, w: float) -> tuple[float, str]:
    if x.sma200 is None:
        return 0.0, "200-day average missing"
    ratio = x.price / x.sma200
    if ratio >= 1.0:
        return w, f"{(ratio - 1):.0%} above the 200-day average"
    if ratio >= 0.95:
        return w / 2, f"{(1 - ratio):.0%} below the 200-day average"
    return 0.0, f"{(1 - ratio):.0%} below the 200-day average"


def score_insider(x: ScreenInputs, pb: Playbook, w: float) -> tuple[float, str]:
    a = x.insider
    if a is None:
        return 0.0, "Form 4 data missing"
    if a.sold <= 0:
        return w, f"no discretionary insider sales ({a.filings} Form 4s)"
    if a.bought <= 0:
        return w / 2, f"discretionary sales {a.sold:,.0f} sh, no open-market buys"
    if a.sold > pb["screener.exclusions.insider_sell_buy_ratio_max"] * a.bought:
        return 0.0, f"sales {a.sold:,.0f} sh > 2x buys {a.bought:,.0f} sh"
    return w, f"sales {a.sold:,.0f} sh vs buys {a.bought:,.0f} sh"


def gates(x: ScreenInputs, pb: Playbook) -> list[str]:
    """Reasons the ticker fails the hard screen (empty list = passes the gates)."""
    fails = []
    if pb.tier_of(x.ticker) is None:
        fails.append("outside the universe")
    if x.market_cap is None:
        fails.append("market cap missing")
    elif x.market_cap < pb["screener.market_cap_min_usd"]:
        fails.append(f"market cap {x.market_cap / 1e9:.0f}B below minimum")
    if x.price < pb["screener.price_min_usd"]:
        fails.append("price below minimum")
    if x.sector not in pb["screener.sectors"]:
        fails.append(f"sector {x.sector!r} not in the list")
    if x.country != pb["screener.country"]:
        fails.append(f"country {x.country!r}")
    if x.reverse_split_recent:
        fails.append("reverse split in the lookback window")
    return fails


def screen(x: ScreenInputs, pb: Playbook) -> Candidate:
    w = pb["screener.quality_weights"]
    parts = {
        "analyst": score_analyst(x, pb, w["analyst"]),
        "fundamentals": score_fundamentals(x, w["fundamentals"]),
        "trend": score_trend(x, w["trend"]),
        "insider": score_insider(x, pb, w["insider"]),
    }
    total = round(sum(s for s, _ in parts.values()), 2)
    fails = gates(x, pb)
    passed = not fails and total >= pb["screener.quality_score_min"]
    reasons = tuple(f"{k}: {s:.2f} ({why})" for k, (s, why) in parts.items()) + tuple(f"gate: {f}" for f in fails)
    return Candidate(ticker=x.ticker, price=x.price, tier=pb.tier_of(x.ticker), quality_score=total,
                     score_components={k: round(s, 2) for k, (s, _) in parts.items()},
                     passed_screen=passed, reasons=reasons)


# --- EDGAR XBRL fundamentals ----------------------------------------------------------------

REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense")


def _annual(facts: dict[str, Any], tag: str) -> list[tuple[str, float]]:
    """(frame, value) for calendar-year annual USD values, oldest first."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
    seen = {}
    for row in node:
        frame = row.get("frame", "")
        if len(frame) == 6 and frame.startswith("CY") and frame[2:].isdigit():
            seen[frame] = float(row["val"])
    return sorted(seen.items())


def fundamentals_from_facts(facts: dict[str, Any]) -> tuple[bool | None, float | None]:
    ni = _annual(facts, "NetIncomeLoss")
    income_ok = ni[-1][1] > 0 if ni else None
    # Companies switch revenue tags over the years: use the tag with the most recent year.
    series = [r for r in (_annual(facts, t) for t in REVENUE_TAGS) if len(r) >= 2 and r[-2][1] > 0]
    if not series:
        return income_ok, None
    rev = max(series, key=lambda r: r[-1][0])
    return income_ok, rev[-1][1] / rev[-2][1] - 1
