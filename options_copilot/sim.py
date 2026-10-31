"""Simulated clock for the end-to-end demo (SYNTHETIC market data).

Wraps a real source captured on `start` and moves the clock to `as_of` with the stock at
`spot x (1 + move_pct/100)`. Each option is re-priced as its live mid plus the change in its
Black-Scholes value (same IV, less time, new spot), so on day 0 the quotes equal the market
exactly and later days show theta decay and the price move. The original bid/ask spread is
kept. Nothing here is a forecast; it only lets the demo reach a management rule without
waiting real days, and every output that uses it is labelled simulated.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

from .calculators import Quote, bs_price, years_between


class SimulatedSources:
    def __init__(self, inner, start: date, as_of: date, move_pct: float = 0.0):
        if as_of < start:
            raise ValueError("the simulated clock only moves forward")
        self.inner, self.start, self.as_of, self.move_pct = inner, start, as_of, move_pct
        self.label = f"SIMULATED: {start} + {(as_of - start).days} days, stock {move_pct:+.1f}%"

    def _spot(self, ticker: str) -> tuple[float, float]:
        base = self.inner.ticker_data(ticker).profile.price
        return base, base * (1 + self.move_pct / 100)

    def ticker_data(self, ticker: str):
        d = self.inner.ticker_data(ticker)
        _, spot = self._spot(ticker)
        return replace(d, as_of=self.as_of, profile=replace(d.profile, price=round(spot, 2)))

    def expiries(self, ticker: str) -> list[date]:
        return [e for e in self.inner.expiries(ticker) if e > self.as_of]

    def chain(self, ticker: str, expiry: date) -> tuple[list[Quote], float | None]:
        quotes, flow = self.inner.chain(ticker, expiry)
        s0, s1 = self._spot(ticker)
        t0, t1 = years_between(self.start, expiry), years_between(self.as_of, expiry)
        out = []
        for q in quotes:
            if not q.iv:
                continue
            mid = (q.bid + q.ask) / 2 + bs_price(s1, q.strike, q.iv, t1, q.option_type) \
                - bs_price(s0, q.strike, q.iv, t0, q.option_type)
            mid = max(mid, 0.0)
            half = (q.ask - q.bid) / 2
            out.append(replace(q, bid=round(max(mid - half, 0.0), 2), ask=round(mid + half, 2), delta=None))
        return out, flow

    def landmine(self, ticker: str, context: str):
        return self.inner.landmine(ticker, context)

    def lessons(self, ticker: str, query: str):
        return self.inner.lessons(ticker, query)
