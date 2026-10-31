"""SEC EDGAR access: ticker to CIK, recent filings, documents, and HTML to text.

Every request carries the owner's SEC_USER_AGENT (from .env), stays under SEC's limit of
10 requests per second, and is cached under data/raw/. The User-Agent value is never
logged, printed, or included in exceptions or repr().
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .models import from_yyyymmdd

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"
TICKERS_URL = f"{SEC_WWW}/files/company_tickers.json"
FORMS = ("8-K", "10-Q", "10-K")
MUTABLE_TTL_SECONDS = 12 * 3600  # submissions and the ticker map change; filings never do


class EdgarError(RuntimeError):
    pass


class Throttle:
    """Enforces a minimum gap between requests (default ~6.7/s, under SEC's 10/s)."""

    def __init__(self, min_interval: float = 0.15, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            gap = self._last + self.min_interval - now
            if gap > 0:
                self._sleep(gap)
                now += gap
        self._last = now


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    form: str
    accession: str
    filed_date: int  # yyyymmdd
    primary_doc: str
    items: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"{SEC_WWW}/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.primary_doc}"

    @property
    def index_url(self) -> str:
        return f"{self.base_url}/{self.accession}-index.htm"


@dataclass(frozen=True)
class FilingDocument:
    ref: FilingRef
    doc_type: str  # the form for the main document, or the exhibit type, e.g. "EX-99.1"
    url: str
    text: str


def cache_path(raw_dir: Path, url: str) -> Path:
    """Map a URL to a file under raw_dir/edgar/<host>/<path>, refusing path tricks."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if parsed.scheme != "https" or not parsed.netloc or any(p in (".", "..") for p in parts):
        raise EdgarError(f"refusing to cache unexpected URL: {url}")
    if not parts:
        parts = ["index"]
    return raw_dir.joinpath("edgar", parsed.netloc, *parts)


class EdgarClient:
    def __init__(self, user_agent: str, raw_dir: Path, session: Any = None,
                 throttle: Throttle | None = None, clock: Callable[[], float] = time.time):
        if not user_agent.strip():
            raise EdgarError("SEC_USER_AGENT is not set in .env (format: \"Your Name your-email\")")
        self._user_agent = user_agent.strip()
        self.raw_dir = raw_dir
        self._session = session
        self._throttle = throttle or Throttle()
        self._clock = clock
        self.requests_made = 0

    def __repr__(self) -> str:  # keep the User-Agent out of logs and tracebacks
        return f"EdgarClient(raw_dir={self.raw_dir!s}, requests_made={self.requests_made})"

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def fetch(self, url: str, *, mutable: bool = False) -> bytes:
        """Return the body of `url`, from the cache when possible."""
        path = cache_path(self.raw_dir, url)
        if path.exists():
            fresh = not mutable or self._clock() - path.stat().st_mtime < MUTABLE_TTL_SECONDS
            if fresh:
                return path.read_bytes()
        self._throttle.wait()
        self.requests_made += 1
        try:
            resp = self._http().get(url, headers={"User-Agent": self._user_agent,
                                                  "Accept-Encoding": "gzip, deflate"}, timeout=30)
        except Exception as exc:  # network errors name the URL, never the headers
            raise EdgarError(f"request failed for {url}: {type(exc).__name__}") from None
        if resp.status_code != 200:
            raise EdgarError(f"SEC returned HTTP {resp.status_code} for {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return resp.content

    def fetch_json(self, url: str, *, mutable: bool = False) -> Any:
        return json.loads(self.fetch(url, mutable=mutable))

    def cik_for(self, ticker: str) -> int:
        table = self.fetch_json(TICKERS_URL, mutable=True)
        return cik_from_table(table, ticker)

    def recent_filings(self, ticker: str, since: date, until: date,
                       forms: tuple[str, ...] = FORMS) -> list[FilingRef]:
        cik = self.cik_for(ticker)
        subs = self.fetch_json(f"{SEC_DATA}/submissions/CIK{cik:010d}.json", mutable=True)
        return filings_from_submissions(subs, ticker, cik, since, until, forms)

    def documents(self, ref: FilingRef) -> list[FilingDocument]:
        """The main document, plus EX-99.x exhibits for 8-Ks."""
        docs = [FilingDocument(ref, ref.form, ref.url, html_to_text(self.fetch(ref.url)))]
        if ref.form == "8-K":
            for name, doc_type in exhibits_from_index(self.fetch(ref.index_url)):
                url = f"{ref.base_url}/{name}"
                docs.append(FilingDocument(ref, doc_type, url, html_to_text(self.fetch(url))))
        return docs


def cik_from_table(table: dict[str, Any], ticker: str) -> int:
    wanted = ticker.upper()
    for row in table.values():
        if str(row.get("ticker", "")).upper() == wanted:
            return int(row["cik_str"])
    raise EdgarError(f"ticker {ticker} not found in SEC company_tickers.json")


def filings_from_submissions(subs: dict[str, Any], ticker: str, cik: int, since: date,
                             until: date, forms: tuple[str, ...] = FORMS) -> list[FilingRef]:
    """Filings of the given forms filed in [since, until], newest first."""
    recent = subs.get("filings", {}).get("recent", {})
    columns = ("form", "accessionNumber", "filingDate", "primaryDocument", "items")
    rows = zip(*(recent.get(c, []) for c in columns))
    out = []
    for form, accession, filed, primary, items in rows:
        if form not in forms or not primary:
            continue
        filed_on = date.fromisoformat(filed)
        if not since <= filed_on <= until:
            continue
        item_list = tuple(i.strip() for i in (items or "").split(",") if i.strip())
        out.append(FilingRef(ticker.upper(), cik, form, accession,
                             int(filed.replace("-", "")), primary, item_list))
    out.sort(key=lambda r: r.filed_date, reverse=True)
    return out


def exhibits_from_index(index_html: bytes | str) -> list[tuple[str, str]]:
    """(document name, type) for EX-99.x exhibits listed in a filing's -index.htm."""
    soup = BeautifulSoup(index_html, "html.parser")
    out = []
    for table in soup.find_all("table", class_="tableFile"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            doc_type = cells[3].get_text(strip=True).upper()
            link = cells[2].find("a")
            if not doc_type.startswith("EX-99") or link is None:
                continue
            name = link.get_text(strip=True)
            if re.fullmatch(r"[\w.-]+\.(htm|html|txt)", name, flags=re.I):
                out.append((name, doc_type))
    return out


_HIDDEN = re.compile(r"display\s*:\s*none", re.I)


def html_to_text(html: bytes | str) -> str:
    """Readable text, one block per line, without scripts, styles, or hidden iXBRL data."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN):
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def describe(ref: FilingRef) -> str:
    items = f" items {','.join(ref.items)}" if ref.items else ""
    return f"{ref.form} {from_yyyymmdd(ref.filed_date).isoformat()}{items}"


__all__ = ["EdgarClient", "EdgarError", "FilingDocument", "FilingRef", "Throttle", "cache_path",
           "cik_from_table", "describe", "exhibits_from_index", "filings_from_submissions",
           "html_to_text"]
