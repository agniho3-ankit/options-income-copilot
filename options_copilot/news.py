"""Recent headlines from yfinance, kept local-only.

Headline text is third-party content. It is cached under data/raw/news/ and indexed for
retrieval, but never committed. Per the playbook, headlines can raise CAUTION at most:
only filings can BLOCK, and CLEAR needs filings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Headline:
    ticker: str
    title: str
    summary: str
    publisher: str
    url: str
    published: int  # yyyymmdd

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip(" .") + "." if self.summary else self.title


def _to_yyyymmdd(value: Any) -> int | None:
    if isinstance(value, (int, float)) and value > 0:
        d = datetime.fromtimestamp(value, tz=timezone.utc).date()
    elif isinstance(value, str) and value:
        try:
            d = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    else:
        return None
    return d.year * 10000 + d.month * 100 + d.day


def parse_news(ticker: str, items: list[dict[str, Any]]) -> list[Headline]:
    """Accept both yfinance news shapes: the flat legacy one and the nested `content` one."""
    out = []
    for item in items or []:
        c = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (c.get("title") or "").strip()
        published = _to_yyyymmdd(c.get("pubDate") or c.get("providerPublishTime"))
        if not title or published is None:
            continue
        provider = c.get("provider") or {}
        url_obj = c.get("canonicalUrl") or c.get("clickThroughUrl") or {}
        out.append(Headline(
            ticker=ticker.upper(),
            title=title,
            summary=(c.get("summary") or "").strip(),
            publisher=(provider.get("displayName") if isinstance(provider, dict) else None)
                      or c.get("publisher") or "",
            url=(url_obj.get("url") if isinstance(url_obj, dict) else None) or c.get("link") or "",
            published=published,
        ))
    return out


def fetch_headlines(ticker: str, raw_dir: Path, today: date, yf_module: Any = None) -> list[Headline]:
    """Fetch (or read today's cached) headlines for a ticker."""
    path = raw_dir / "news" / f"{ticker.upper()}-{today.isoformat()}.json"
    if path.exists():
        items = json.loads(path.read_text())
    else:
        if yf_module is None:
            import yfinance as yf_module  # noqa: N813
        items = yf_module.Ticker(ticker).news or []
        if items:  # an empty reply is often a transient Yahoo-side gap, so retry next run
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(items, default=str))
    return parse_news(ticker, items)
