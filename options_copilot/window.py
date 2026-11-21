"""Data for the trade-window chart (UI phase): time x price around one brief.

Everything the chart draws is computed here, in Python: recent closes, a ±1σ expected-move cone
from today to expiry (the playbook's 1σ convention, `management.one_sigma`), the strikes, the
breakevens stored with the brief, the profit and loss zones at expiry, and event pins (earnings,
ex-dividend, cited filing dates, and event dates that `events.py` reads from the cited excerpts).
The front end only maps these values to pixels.

Prices and dates are fetched live (market.py); the caller passes them in, so this module stays
pure and testable without a network.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from typing import Any

from .events import extract_event_dates
from .management import one_sigma
from .market import History, Profile

LOOKBACK_TRADING_DAYS = 90
EVENT_HORIZON_AFTER_EXPIRY_DAYS = 45   # show events this soon after expiry (e.g. earnings just after)
CONE_STEP_DAYS = 3
PAD = 0.03                              # 3% headroom above and below the drawn values


class WindowError(ValueError):
    pass


def trade_of(kind: str, payload: dict) -> dict:
    """The trade the chart describes: the new trade for a roll, the open position for an exit."""
    if kind == "entry":
        prop, numbers = payload.get("proposal") or {}, payload.get("numbers") or {}
    elif kind == "roll" and payload.get("new"):
        prop, numbers = payload["new"].get("proposal") or {}, payload["new"].get("numbers") or {}
    else:
        prop, numbers = {"strategy": payload.get("strategy"), "legs": payload.get("legs") or []}, {}
    legs = [{"side": g.get("side"), "option_type": g.get("option_type", g.get("type")), "strike": g.get("strike"),
             "expiry": g.get("expiry"), "iv": g.get("iv")} for g in prop.get("legs") or []]
    expiry = prop.get("expiry") or next((g["expiry"] for g in legs if g["expiry"]), None)
    return {"strategy": prop.get("strategy"), "legs": legs, "expiry": expiry,
            "breakevens": list(numbers.get("breakevens") or [])}


def zones(strategy: str | None, breakevens: list[float]) -> list[dict]:
    """Profit and loss zones at expiry, from the stored breakevens. None means unbounded."""
    be = sorted(breakevens)
    if strategy in ("put_credit_spread", "cash_secured_put") and len(be) == 1:
        return [{"kind": "profit", "low": be[0], "high": None}, {"kind": "loss", "low": None, "high": be[0]}]
    if strategy == "call_credit_spread" and len(be) == 1:
        return [{"kind": "profit", "low": None, "high": be[0]}, {"kind": "loss", "low": be[0], "high": None}]
    if strategy == "iron_condor" and len(be) == 2:
        return [{"kind": "profit", "low": be[0], "high": be[1]}, {"kind": "loss", "low": None, "high": be[0]},
                {"kind": "loss", "low": be[1], "high": None}]
    return []


def excerpts_from_trace(trace_rows: list[dict]) -> list[dict]:
    """Masked filing excerpts the researcher graded (multi-agent runs), first appearance first."""
    seen: dict[str, dict] = {}
    for row in trace_rows:
        if row.get("role") != "user" or row.get("name") != "researcher.grader":
            continue
        try:
            body = json.loads(row["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        for e in body.get("excerpts") or []:
            key = str(e.get("id"))
            if key not in seen and isinstance(e.get("text"), str):
                seen[key] = {"id": key, "form": e.get("form"), "item": e.get("item"), "text": e["text"]}
    return list(seen.values())


def sigma_source(atm_iv: float | None, legs: list[dict]) -> tuple[float | None, str]:
    if atm_iv:
        return atm_iv, "ATM implied volatility (latest IV snapshot)"
    short = next((g["iv"] for g in legs if g["side"] == "short" and g.get("iv")), None)
    if short:
        return short, "short-leg implied volatility (no ATM snapshot stored)"
    return None, "no implied volatility available"


def build_window(*, ticker: str, kind: str, payload: dict, history: History, profile: Profile, today: date,
                 atm_iv: float | None, excerpts: list[dict]) -> dict:
    if not history.dates or len(history.dates) != len(history.closes):
        raise WindowError("price history has no dates")
    t = trade_of(kind, payload)
    if not t["expiry"]:
        raise WindowError("the brief has no expiry")
    expiry = date.fromisoformat(str(t["expiry"])[:10])
    as_of = date.fromisoformat(str(payload.get("as_of") or today.isoformat())[:10])

    # skip missing days and non-finite closes (yfinance's intraday row can be NaN)
    points = [(d, c) for d, c in zip(history.dates, history.closes)
              if d is not None and d <= today and c is not None and math.isfinite(c)]
    points = points[-LOOKBACK_TRADING_DAYS:]
    if not points:
        raise WindowError("no closes up to today")
    last_day, spot = points[-1]

    iv, iv_label = sigma_source(atm_iv, t["legs"])
    cone = []
    if iv and expiry > last_day:
        days = [0, *range(CONE_STEP_DAYS, (expiry - last_day).days, CONE_STEP_DAYS), (expiry - last_day).days]
        for n in days:
            band = 0.0 if n == 0 else one_sigma(spot, iv, n)
            cone.append({"date": (last_day + timedelta(days=n)).isoformat(), "low": round(spot - band, 2),
                         "high": round(spot + band, 2)})

    events = _events(profile, payload, excerpts, as_of, expiry)
    end = max([expiry, *(date.fromisoformat(e["date"]) for e in events
                         if date.fromisoformat(e["date"]) <= expiry + timedelta(days=EVENT_HORIZON_AFTER_EXPIRY_DAYS))])
    start = points[0][0]
    events = [e for e in events if start <= date.fromisoformat(e["date"]) <= end]

    prices = [c for _, c in points] + [p for c in cone for p in (c["low"], c["high"])]
    prices += [g["strike"] for g in t["legs"] if g["strike"] is not None] + t["breakevens"]
    lo, hi = min(prices), max(prices)
    return {
        "ticker": ticker, "kind": kind, "strategy": t["strategy"], "brief_as_of": as_of.isoformat(),
        "today": last_day.isoformat(), "expiry": expiry.isoformat(), "spot": round(spot, 2),
        "sigma": {"iv": iv, "source": iv_label, "convention": "spot x IV x sqrt(days / 365)",
                  "at_expiry": cone[-1] if cone else None},
        "history": [{"date": d.isoformat(), "close": round(c, 2)} for d, c in points],
        "cone": cone,
        "strikes": [{k: g[k] for k in ("side", "option_type", "strike")} for g in t["legs"]],
        "breakevens": t["breakevens"],
        "zones": zones(t["strategy"], t["breakevens"]),
        "events": events,
        "x_domain": [start.isoformat(), end.isoformat()],
        "y_domain": [round(lo * (1 - PAD), 2), round(hi * (1 + PAD), 2)],
    }


def _events(profile: Profile, payload: dict, excerpts: list[dict], as_of: date, expiry: date) -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}

    def add(d: date | None, kind: str, label: str, source: str) -> None:
        if d is None:
            return
        rows.setdefault((d.isoformat(), label), {
            "date": d.isoformat(), "kind": kind, "label": label, "source": source,
            "in_window": as_of <= d <= expiry})

    add(profile.earnings_date, "earnings", "Earnings", "market data")
    add(profile.ex_dividend_date, "ex_dividend", "Ex-dividend", "market data")
    for c in payload.get("citations") or []:
        try:
            d = date.fromisoformat(str(c.get("date"))[:10])
        except ValueError:
            continue
        add(d, "filing", f"{c.get('form')} filed", f"{c.get('form')} {c.get('date')} {c.get('item') or ''}".strip())
    for e in excerpts:
        for d, word in extract_event_dates(e["text"]):
            add(d, "filing_event", f"Filing event: {word}", f"{e.get('form')} {e.get('item') or ''}".strip())
    return sorted(rows.values(), key=lambda r: (r["date"], r["label"]))
