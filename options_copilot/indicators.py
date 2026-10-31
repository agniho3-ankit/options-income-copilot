"""Deterministic price indicators: moving averages, RSI, realized vol, support/resistance.

Inputs are plain lists of floats, oldest first. Nothing here touches the network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def sma(values: Sequence[float], window: int) -> float | None:
    if window <= 0 or len(values) < window:
        return None
    return sum(values[-window:]) / window


def rsi(closes: Sequence[float], period: int = 14) -> float | None:
    """Wilder's RSI on closing prices."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(closes, closes[1:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def realized_vol(closes: Sequence[float], window: int = 30) -> float | None:
    """Annualized close-to-close volatility over the last `window` returns."""
    if len(closes) < window + 1:
        return None
    rets = [math.log(b / a) for a, b in zip(closes[-window - 1:], closes[-window:])]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


@dataclass(frozen=True)
class Levels:
    support: float | None     # nearest swing low below price
    resistance: float | None  # nearest swing high above price
    swing_lows: tuple[float, ...]
    swing_highs: tuple[float, ...]


def swing_pivots(highs: Sequence[float], lows: Sequence[float], window: int) -> tuple[list[float], list[float]]:
    """Bars whose high (low) is the strict max (min) of the `window` bars on each side."""
    piv_hi, piv_lo = [], []
    for i in range(window, len(highs) - window):
        left_h, right_h = highs[i - window:i], highs[i + 1:i + window + 1]
        if highs[i] > max(left_h) and highs[i] > max(right_h):
            piv_hi.append(highs[i])
        left_l, right_l = lows[i - window:i], lows[i + 1:i + window + 1]
        if lows[i] < min(left_l) and lows[i] < min(right_l):
            piv_lo.append(lows[i])
    return piv_hi, piv_lo


def support_resistance(highs: Sequence[float], lows: Sequence[float], price: float,
                       lookback: int, window: int) -> Levels:
    """Nearest swing-pivot support below and resistance above `price` over `lookback` bars."""
    if len(highs) != len(lows):
        raise ValueError("highs and lows must be the same length")
    highs, lows = list(highs)[-lookback:], list(lows)[-lookback:]
    piv_hi, piv_lo = swing_pivots(highs, lows, window)
    below = [lo for lo in piv_lo if lo < price]
    above = [hi for hi in piv_hi if hi > price]
    return Levels(support=max(below) if below else None, resistance=min(above) if above else None,
                  swing_lows=tuple(piv_lo), swing_highs=tuple(piv_hi))


def iv_rank_proxy(atm_iv: float, realized: float) -> float:
    """Stand-in for IV Rank before enough IV history exists: ATM IV / realized vol,
    mapped linearly so a ratio of 0.75 -> 0 and 1.5 -> 100 (clamped)."""
    if realized <= 0:
        raise ValueError("realized vol must be positive")
    return max(0.0, min(100.0, (atm_iv / realized - 0.75) / 0.75 * 100))


def atm_iv(quotes, spot: float, n: int = 4) -> float | None:
    """Average IV of the `n` strikes nearest the money (puts and calls together)."""
    near = sorted((q for q in quotes if q.iv), key=lambda q: abs(q.strike - spot))[:n]
    return sum(q.iv for q in near) / len(near) if near else None


def iv_rank_from_history(current: float, history: list[float], realized: float | None, pb) -> tuple[float, str] | None:
    """IV Rank once a full year of daily snapshots exists, IV percentile once
    `percentile_min_snapshots` exist, else the ATM-IV / realized-vol proxy. Returns
    (0-100 value, label) or None when not even the proxy is available."""
    full, min_n = pb["iv_rank.full_history_days"], pb["iv_rank.percentile_min_snapshots"]
    hist = list(history)[-full:]
    if len(hist) >= full:
        lo, hi = min(hist), max(hist)
        rank = 50.0 if hi == lo else (current - lo) / (hi - lo) * 100
        return max(0.0, min(100.0, rank)), f"IV rank over {len(hist)} daily snapshots"
    if len(hist) >= min_n:
        pct = sum(v < current for v in hist) / len(hist) * 100
        return pct, f"IV percentile over {len(hist)} daily snapshots (rank needs {full})"
    if realized:
        return (iv_rank_proxy(current, realized),
                f"proxy: ATM IV {current:.0%} / 30d RV {realized:.0%}, {len(hist)} IV snapshots so far")
    return None
