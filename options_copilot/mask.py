"""The mask layer: what the LLM, logs, and traces may see about the account.

Account values (portfolio value, dollar P&L, max loss, collateral, credits in dollars,
contract counts) become % of portfolio. Position, brief, account, and order IDs become
stable opaque tokens. Public market data (tickers, prices, strikes, deltas, IV) passes
through, so the agent can still reason about strikes versus support and resistance.

Free text is scrubbed too: dollar amounts of 1,000 or more and long digit runs (account
or order numbers) are replaced. Smaller dollar figures such as option premiums and share
prices are public market data and stay.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

DOLLAR_KEYS = frozenset({
    "credit", "max_loss", "collateral", "max_profit", "pnl", "entry_credit", "entry_max_loss",
    "close_cost", "portfolio_value", "balance", "cash", "buying_power", "equity", "loss", "profit",
    "open_max_losses", "synthetic_portfolio_value",
})
ID_KEYS = frozenset({"position_id", "brief_id", "account_id", "order_id", "account_number"})
COUNT_KEYS = frozenset({"contracts", "quantity", "shares_held"})

_BIG_DOLLARS = re.compile(
    r"(?<![\w.])-?\$\s?(?:"
    r"\d+(?:\.\d+)?\s?(?:[kKmMbB]|thousand|million|billion)\b"   # $1.2M, $50k, $3 million
    r"|(?:\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?)")               # $12,500 or $12500.00
_LONG_DIGITS = re.compile(r"(?<![\d.])\d{6,}(?![\d.])")
MASKED_AMOUNT = "[amount masked]"
MASKED_ID = "[id masked]"


class Masker:
    def __init__(self, portfolio_value: float, salt: bytes = b"options-copilot"):
        if portfolio_value <= 0:
            raise ValueError("portfolio value must be positive")
        self._pv = float(portfolio_value)
        self._salt = salt

    def __repr__(self) -> str:  # the portfolio value itself is masked data
        return "Masker()"

    def pct(self, dollars: float) -> str:
        return f"{dollars / self._pv * 100:.2f}% of portfolio"

    def token(self, value: Any, prefix: str = "id") -> str:
        digest = hmac.new(self._salt, str(value).encode(), hashlib.sha256).hexdigest()[:10]
        return f"{prefix}_{digest}"

    def text(self, s: str) -> str:
        s = _BIG_DOLLARS.sub(MASKED_AMOUNT, s)
        return _LONG_DIGITS.sub(MASKED_ID, s)

    def value(self, obj: Any, key: str | None = None) -> Any:
        """Mask a JSON-like structure (dicts, lists, scalars), keyed by field names."""
        if isinstance(obj, dict):
            return {k: self.value(v, str(k)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.value(v, key) for v in obj]
        if key in ID_KEYS and obj is not None:
            return self.token(obj, key.split("_")[0])
        if key in COUNT_KEYS and obj is not None:
            return "masked"
        if key in DOLLAR_KEYS and isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if key == "portfolio_value" or key == "synthetic_portfolio_value":
                return "100.00% of portfolio"
            return self.pct(obj)
        if isinstance(obj, str):
            return self.text(obj)
        return obj


def find_leaks(text: str, raw_values: list[float | str]) -> list[str]:
    """Raw values (formatted a few common ways) that still appear in `text`."""
    leaks = []
    for v in raw_values:
        forms = {str(v)}
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            forms |= {f"{v:.2f}", f"{v:,.2f}", f"{v:,.0f}", f"{v:.0f}"}
        leaks += [f for f in forms if f and f in text]
    return sorted(set(leaks))
