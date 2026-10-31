"""Insider activity from SEC Form 4 filings (public), for the insider check and score.

Only discretionary open-market trades count: code P (purchase) buys and code S (sale)
sales that are not marked as made under a Rule 10b5-1 plan. Tax withholding (F), option
exercises (M), grants (A), and gifts (G) are ignored.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date

from .edgar import EdgarClient, EdgarError, FilingRef
from .verifier import InsiderActivity


@dataclass(frozen=True)
class Form4Trades:
    bought: float = 0.0
    sold: float = 0.0
    planned_sold: float = 0.0   # 10b5-1 sales, reported but not counted


def _text(node: ET.Element | None, path: str) -> str:
    found = node.find(path) if node is not None else None
    return (found.text or "").strip() if found is not None and found.text else ""


def parse_form4(xml: bytes | str) -> Form4Trades:
    root = ET.fromstring(xml)
    plan_flag = _text(root, "aff10b5One").lower() in ("1", "true")
    bought = sold = planned = 0.0
    for tx in root.iter("nonDerivativeTransaction"):
        code = _text(tx, "transactionCoding/transactionCode").upper()
        try:
            shares = float(_text(tx, "transactionAmounts/transactionShares/value") or 0)
        except ValueError:
            continue
        if code == "P":
            bought += shares
        elif code == "S":
            if plan_flag:
                planned += shares
            else:
                sold += shares
    return Form4Trades(bought, sold, planned)


def raw_xml_url(ref: FilingRef) -> str:
    """The primary document is often the XSL-rendered view, e.g. 'xslF345X05/form4.xml'."""
    name = ref.primary_doc.rsplit("/", 1)[-1]
    return f"{ref.base_url}/{name}"


def insider_activity(edgar: EdgarClient, ticker: str, since: date, until: date) -> InsiderActivity:
    refs = edgar.recent_filings(ticker, since, until, forms=("4",))
    bought = sold = 0.0
    parsed = 0
    for ref in refs:
        try:
            trades = parse_form4(edgar.fetch(raw_xml_url(ref)))
        except (EdgarError, ET.ParseError):
            continue
        bought += trades.bought
        sold += trades.sold
        parsed += 1
    if refs and parsed == 0:
        raise EdgarError(f"could not read any of {len(refs)} Form 4 filings for {ticker}")
    return InsiderActivity(bought=bought, sold=sold, filings=parsed)
