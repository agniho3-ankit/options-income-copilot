"""Deterministic event-date extraction from cited filing text.

The landmine LLM may say an event is "after expiry", but that statement is never trusted.
Python reads dates next to event words (effective, closing, record date, ...) straight out
of the chunks the landmine check cited, and the verifier compares them with the expiry.
Dates written without a day ("in the first quarter of 2027") are not extracted.
"""
from __future__ import annotations

import re
from datetime import date

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

DATE_RE = re.compile(
    r"\b(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b"
    r"|\b(?P<iy>\d{4})-(?P<im>\d{2})-(?P<id>\d{2})\b"
    r"|\b(?P<um>\d{1,2})/(?P<ud>\d{1,2})/(?P<uy>\d{4})\b", re.I)

EVENT_WORDS = re.compile(
    r"\b(effective(?:ness)?|closing|to close|consummat\w*|complet\w*|expir\w*|terminat\w*|record date|"
    r"payable|matur\w*|special meeting|annual meeting|vote|deadline|outside date|end date)\b", re.I)

LOOKBACK_CHARS = 90


def _to_date(m: re.Match) -> date | None:
    try:
        if m.group("mon"):
            return date(int(m.group("year")), _MONTHS[m.group("mon").lower()[:3]], int(m.group("day")))
        if m.group("iy"):
            return date(int(m.group("iy")), int(m.group("im")), int(m.group("id")))
        return date(int(m.group("uy")), int(m.group("um")), int(m.group("ud")))
    except (ValueError, KeyError):
        return None


def extract_event_dates(text: str) -> list[tuple[date, str]]:
    """(date, event word) pairs where an event word precedes the date in the same sentence."""
    out: set[tuple[date, str]] = set()
    for m in DATE_RE.finditer(text or ""):
        d = _to_date(m)
        if d is None:
            continue
        before = text[max(0, m.start() - LOOKBACK_CHARS):m.start()]
        before = re.split(r"[.;]\s", before)[-1]  # stay inside the sentence
        words = EVENT_WORDS.findall(before)
        if words:
            out.add((d, words[-1].lower()))
    return sorted(out)


def event_dates_from_chunks(chunks) -> list[dict]:
    """Event dates from cited chunks, each tagged with the citation it came from."""
    rows: dict[tuple, dict] = {}
    for c in chunks:
        cite = c.citation()
        source = f"{cite['form']} {cite['date']} {cite['item']}".strip()
        for d, word in extract_event_dates(c.text):
            rows.setdefault((d, word), {"date": d.isoformat(), "event": word, "source": source})
    return sorted(rows.values(), key=lambda r: (r["date"], r["event"]))
