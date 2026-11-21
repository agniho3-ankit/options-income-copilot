"""Public market data from yfinance, turned into the plain types the calculators use.

This adapter is thin on purpose: all logic lives in pure, tested modules. Nothing here
touches an account. Option chains, prices, and analyst data are third-party data and are
never committed.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .calculators import Quote
from .models import OptionType



@dataclass(frozen=True)
class History:
    closes: list[float]
    highs: list[float]
    lows: list[float]
    dates: list[date] = field(default_factory=list)   # trading days, same order as closes


@dataclass(frozen=True)
class Profile:
    price: float
    market_cap: float | None
    sector: str | None
    country: str | None
    analyst_target: float | None
    analyst_count: int | None
    analyst_buy_ratio: float | None
    earnings_date: date | None
    ex_dividend_date: date | None
    dividend_rate: float | None
    recent_splits: list[tuple[date, float]] = field(default_factory=list)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _date(v: Any) -> date | None:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return None
    if hasattr(v, "date") and not isinstance(v, date):
        return v.date()
    return v if isinstance(v, date) else None


class Market:
    def __init__(self, yf_module: Any = None):
        if yf_module is None:
            import yfinance as yf_module  # noqa: N813
            # yfinance re-enables DeprecationWarnings for itself on import; silence pandas noise after that.
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="^yfinance")
        self._yf = yf_module

    def _t(self, ticker: str):
        return self._yf.Ticker(ticker.upper())

    def history(self, ticker: str, period: str = "2y") -> History:
        df = self._t(ticker).history(period=period, auto_adjust=False)
        # pandas Timestamps are datetimes; keep plain dates so they compare with date objects
        days = [i.date() if isinstance(i, datetime) else _date(i) for i in df.index]
        return History(closes=[float(x) for x in df["Close"]], highs=[float(x) for x in df["High"]],
                       lows=[float(x) for x in df["Low"]], dates=days)

    def profile(self, ticker: str, as_of: date) -> Profile:
        t = self._t(ticker)
        info = t.info or {}
        cal = t.calendar or {}
        price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
        if price is None:
            price = float(t.fast_info["last_price"])
        buy_ratio, count = None, None
        rec = t.recommendations_summary
        if rec is not None and len(rec):
            row = rec.iloc[0]
            total = sum(int(row.get(k, 0)) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
            if total:
                buy_ratio = (int(row.get("strongBuy", 0)) + int(row.get("buy", 0))) / total
                count = total
        splits = []
        try:
            for ts, ratio in t.splits.items():
                d = ts.date() if hasattr(ts, "date") else ts
                if (as_of - d).days <= 400:
                    splits.append((d, float(ratio)))
        except Exception:  # yfinance raises varied errors for tickers without splits
            pass
        earnings = _date(cal.get("Earnings Date"))
        return Profile(
            price=price, market_cap=_num(info.get("marketCap")), sector=info.get("sector"),
            country=info.get("country"), analyst_target=_num(info.get("targetMeanPrice")),
            analyst_count=count or (int(info["numberOfAnalystOpinions"]) if info.get("numberOfAnalystOpinions") else None),
            analyst_buy_ratio=buy_ratio, earnings_date=earnings if earnings and earnings >= as_of else None,
            ex_dividend_date=_date(cal.get("Ex-Dividend Date")), dividend_rate=_num(info.get("dividendRate")),
            recent_splits=splits)

    def expiries(self, ticker: str) -> list[date]:
        return [date.fromisoformat(s) for s in self._t(ticker).options]

    def chain(self, ticker: str, expiry: date) -> tuple[list[Quote], float | None]:
        """Quotes for one expiry, plus the max volume/open-interest ratio (unusual flow)."""
        oc = self._t(ticker).option_chain(expiry.isoformat())
        quotes, ratios = [], []
        for option_type, df in ((OptionType.PUT, oc.puts), (OptionType.CALL, oc.calls)):
            for row in df.to_dict("records"):
                bid, ask = _num(row.get("bid")) or 0.0, _num(row.get("ask")) or 0.0
                if ask < bid:
                    continue
                iv = _num(row.get("impliedVolatility"))
                oi = _num(row.get("openInterest"))
                vol = _num(row.get("volume"))
                if oi and vol:
                    ratios.append(vol / oi)
                quotes.append(Quote(option_type, float(row["strike"]), expiry, bid, ask,
                                    iv=iv if iv and iv > 0.01 else None,
                                    open_interest=int(oi) if oi is not None else None))
        return quotes, (max(ratios) if ratios else None)


def pick_expiry(expiries: list[date], as_of: date, dte_min: int, dte_max: int, dte_target: int,
                earnings: date | None) -> date | None:
    """Expiry inside [dte_min, dte_max] nearest the target, preferring ones before earnings."""
    inside = [e for e in expiries if dte_min <= (e - as_of).days <= dte_max]
    if not inside:
        return None
    pool = eligible_expiries(inside, as_of, dte_min, dte_max, dte_target, earnings) or inside
    return min(pool, key=lambda e: (abs((e - as_of).days - dte_target), e))


def eligible_expiries(expiries: list[date], as_of: date, dte_min: int, dte_max: int, dte_target: int,
                      earnings: date | None) -> list[date]:
    """Expiries inside [dte_min, dte_max] and before earnings (if known), nearest the target first.
    Unlike pick_expiry there is no after-earnings fallback: those would fail the verifier."""
    ok = [e for e in expiries if dte_min <= (e - as_of).days <= dte_max and (earnings is None or e < earnings)]
    return sorted(ok, key=lambda e: (abs((e - as_of).days - dte_target), e))
