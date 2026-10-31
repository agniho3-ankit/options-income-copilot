"""Ingest recent filings and headlines for a ticker into the `filings` collection."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .chunker import chunk_filing, chunk_headline
from .config import RetrievalConfig
from .edgar import EdgarClient, EdgarError, describe
from .news import fetch_headlines
from .store import Store


@dataclass
class IngestReport:
    ticker: str
    filings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks: int = 0
    headlines: int = 0
    errors: list[str] = field(default_factory=list)


def ingest_ticker(ticker: str, *, edgar: EdgarClient, store: Store, raw_dir: Path, as_of: date,
                  cfg: RetrievalConfig = RetrievalConfig(), headlines: bool = True,
                  reindex: bool = False, yf_module: Any = None) -> IngestReport:
    ticker = ticker.upper()
    report = IngestReport(ticker)
    since = as_of - timedelta(days=cfg.window_days)
    try:
        refs = edgar.recent_filings(ticker, since, as_of)
    except EdgarError as exc:
        report.errors.append(str(exc))
        refs = []
    for ref in refs:
        label = describe(ref)
        if store.has_accession(ticker, ref.accession):
            if not reindex:
                report.skipped.append(label + " (already indexed)")
                continue
            store.delete_accession(ticker, ref.accession)  # re-chunk from the cached documents
        try:
            docs = edgar.documents(ref)
        except EdgarError as exc:
            report.errors.append(f"{label}: {exc}")
            continue
        chunks = []
        for doc in docs:
            chunks += chunk_filing(doc.text, ticker=ticker, form=ref.form, filed_date=ref.filed_date,
                                   source_url=doc.url, accession=ref.accession, doc_type=doc.doc_type,
                                   filing_items=ref.items, cfg=cfg)
        report.chunks += store.add_chunks(chunks)
        report.filings.append(f"{label} ({len(docs)} doc(s), {len(chunks)} chunks)")

    if headlines:
        try:
            items = fetch_headlines(ticker, raw_dir, as_of, yf_module=yf_module)
        except Exception as exc:  # yfinance raises many types; headlines are optional
            report.errors.append(f"headlines: {type(exc).__name__}")
            items = []
        start = int(since.strftime("%Y%m%d"))
        fresh = [h for h in items if h.published >= start]
        chunks = [chunk_headline(ticker=ticker, text=h.text, published=h.published, url=h.url,
                                 key=hashlib.sha256(h.title.encode()).hexdigest()[:16]) for h in fresh]
        store.add_chunks(chunks)
        report.headlines = len(chunks)
    return report
