"""Glue that gathers public facts for one ticker and runs build -> compute -> verify.

The Phase 3 agent will call these same functions as tools. Every number still comes from
the calculators and the verifier; this module only fetches inputs and wires them together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .calculators import BuildError, BuildRules, CalcError, Extras, build_for_pop
from .edgar import SEC_DATA, EdgarClient, EdgarError
from .form4 import insider_activity
from .indicators import realized_vol, rsi, sma, support_resistance
from .market import Market, pick_expiry
from .models import Strategy, TradeNumbers, TradeProposal, Verdict, VerifierResult
from .playbook import Playbook
from .screener import ScreenInputs, fundamentals_from_facts, screen
from .verifier import InsiderActivity, MarketFacts, verify


@dataclass
class TickerData:
    ticker: str
    as_of: date
    profile: Any
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    support: float | None
    resistance: float | None
    insider: InsiderActivity | None
    notes: list[str] = field(default_factory=list)
    rv30: float | None = None  # realized vol, for the IV-rank proxy


def gather(ticker: str, as_of: date, pb: Playbook, market: Market, edgar: EdgarClient | None) -> TickerData:
    prof = market.profile(ticker, as_of)
    hist = market.history(ticker)
    lookback = pb["entry.support_resistance.lookback_trading_days"]
    lv = support_resistance(hist.highs, hist.lows, prof.price, lookback,
                            pb["entry.support_resistance.pivot_window_bars"])
    notes = []
    insider = None
    if edgar is None:
        notes.append("no SEC access: Form 4 insider data unavailable")
    else:
        days = pb["screener.exclusions.insider_lookback_days"]
        try:
            insider = insider_activity(edgar, ticker, as_of - timedelta(days=days), as_of)
        except EdgarError as exc:
            notes.append(f"Form 4: {exc}")
    return TickerData(ticker.upper(), as_of, prof, sma(hist.closes, 50), sma(hist.closes, 200),
                      rsi(hist.closes), lv.support, lv.resistance, insider, notes,
                      realized_vol(hist.closes, pb["iv_rank.realized_vol_window_days"]))


@dataclass
class BuildOutcome:
    ticker: str
    as_of: date
    strategy: Strategy
    data: TickerData
    expiry: date | None = None
    proposal: TradeProposal | None = None
    numbers: TradeNumbers | None = None
    extras: Extras | None = None
    verifier: VerifierResult | None = None
    widen_steps: int = 0
    error: str = ""


def build_and_verify(ticker: str, strategy: Strategy, as_of: date, pb: Playbook, market: Market,
                     edgar: EdgarClient | None, portfolio_value: float,
                     open_max_losses: tuple[float, ...] = (), landmine: tuple[Verdict, str] | None = None,
                     iv_rank: tuple[float, str] | None = None,
                     event_dates: tuple[tuple[date, str], ...] = ()) -> BuildOutcome:
    data = gather(ticker, as_of, pb, market, edgar)
    out = BuildOutcome(ticker.upper(), as_of, strategy, data)
    prof = data.profile
    out.expiry = pick_expiry(market.expiries(ticker), as_of, pb["entry.dte_min"], pb["entry.dte_max"],
                             pb["entry.dte_target"], prof.earnings_date)
    if out.expiry is None:
        out.error = "no listed expiry inside the DTE window"
        return out
    chain, flow = market.chain(ticker, out.expiry)
    try:
        out.proposal, out.numbers, out.extras, out.widen_steps = build_for_pop(
            strategy, chain, prof.price, as_of, BuildRules.from_playbook(pb, strategy), portfolio_value,
            pb["entry.pop_min"], ticker=ticker, support=data.support, resistance=data.resistance,
            analyst_target=prof.analyst_target)
    except (BuildError, CalcError) as exc:
        out.error = str(exc)
        return out
    facts = MarketFacts(
        as_of=as_of, spot=prof.price, portfolio_value=portfolio_value, open_max_losses=open_max_losses,
        earnings_date=prof.earnings_date, support=data.support, resistance=data.resistance,
        analyst_target=prof.analyst_target, insider=data.insider,
        landmine=landmine[0] if landmine else None, landmine_reason=landmine[1] if landmine else "",
        event_dates=event_dates,
        rsi=data.rsi14, iv_rank=iv_rank[0] if iv_rank else None, iv_rank_label=iv_rank[1] if iv_rank else "",
        sma50=data.sma50, sma200=data.sma200, ex_dividend_date=prof.ex_dividend_date, max_vol_oi_ratio=flow)
    out.verifier = verify(out.proposal, facts, pb, out.numbers)
    return out


def screen_ticker(ticker: str, as_of: date, pb: Playbook, market: Market, edgar: EdgarClient | None):
    data = gather(ticker, as_of, pb, market, edgar)
    income_ok = growth = None
    if edgar is not None:
        try:
            cik = edgar.cik_for(ticker)
            facts = edgar.fetch_json(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik:010d}.json", mutable=True)
            income_ok, growth = fundamentals_from_facts(facts)
        except EdgarError as exc:
            data.notes.append(f"XBRL: {exc}")
    p = data.profile
    lookback_days = pb["screener.exclusions.reverse_split_lookback_months"] * 31
    reverse = any(r < 1 and (as_of - d).days <= lookback_days for d, r in p.recent_splits)
    inputs = ScreenInputs(ticker=ticker.upper(), price=p.price, market_cap=p.market_cap, sector=p.sector,
                          country=p.country, analyst_buy_ratio=p.analyst_buy_ratio, analyst_count=p.analyst_count,
                          analyst_target=p.analyst_target, sma200=data.sma200, net_income_positive=income_ok,
                          revenue_growth=growth, insider=data.insider, reverse_split_recent=reverse)
    return screen(inputs, pb), data


class CachedSources:
    """Memoizes another source so two planners (ReAct vs ToT) see identical data and one
    landmine verdict. Lessons pass through."""

    def __init__(self, inner):
        self.inner, self._cache = inner, {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def ticker_data(self, ticker):
        return self._get(("data", ticker), lambda: self.inner.ticker_data(ticker))

    def expiries(self, ticker):
        return self._get(("exp", ticker), lambda: self.inner.expiries(ticker))

    def chain(self, ticker, expiry):
        return self._get(("chain", ticker, expiry), lambda: self.inner.chain(ticker, expiry))

    def landmine(self, ticker, context, query=None):
        if query is None:
            return self._get(("lm", ticker), lambda: self.inner.landmine(ticker, context))
        return self._get(("lm", ticker, query), lambda: self.inner.landmine(ticker, context, query=query))

    def lessons(self, ticker, query):
        return self.inner.lessons(ticker, query)


class LiveSources:
    """Public data for the agent's tools: yfinance, SEC EDGAR, and the local Chroma index."""

    def __init__(self, pb: Playbook, as_of: date, market: Market, edgar: EdgarClient | None, store, llm):
        self.pb, self.as_of, self.market, self.edgar, self.store, self.llm = pb, as_of, market, edgar, store, llm

    def ticker_data(self, ticker: str) -> TickerData:
        return gather(ticker, self.as_of, self.pb, self.market, self.edgar)

    def expiries(self, ticker: str) -> list[date]:
        return self.market.expiries(ticker)

    def chain(self, ticker: str, expiry: date):
        return self.market.chain(ticker, expiry)

    def landmine(self, ticker: str, context: str, query: str | None = None) -> dict:
        from .agent import landmine_check
        return landmine_check(ticker, context, store=self.store, llm=self.llm, as_of=self.as_of, query=query)

    def lessons(self, ticker: str, query: str):
        return self.store.search_lessons(ticker, query)
