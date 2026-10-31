"""Section-aware chunking for SEC filings and headlines.

Filings are split on their Item headings first (8-K "Item 2.02", 10-Q "Part I Item 2",
10-K "Item 1A"), and then each section is cut into ~400-token windows with ~50 tokens of
overlap. A chunk never spans two Items.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import RetrievalConfig

_8K_ITEM = re.compile(r"^\s*item\s+(\d{1,2}\.\d{2})\b", re.I)
_PERIODIC_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-—–]?(\s|$)", re.I)
_PART = re.compile(r"^\s*part\s+(iv|iii|ii|i)\b", re.I)
_SIGNATURES = re.compile(r"^\s*signatures?\s*$", re.I)
TEN_K_ITEMS = ("1A", "3", "7")  # risk factors, legal proceedings, MD&A


@dataclass(frozen=True)
class Section:
    item: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    form: str
    filed_date: int
    item: str
    source_url: str
    chunk_idx: int
    accession: str = ""
    doc_type: str = ""
    filing_items: str = ""
    source_kind: str = "filing"  # "filing" or "headline"

    def metadata(self) -> dict[str, str | int]:
        return {"ticker": self.ticker, "form": self.form, "filed_date": self.filed_date,
                "item": self.item, "source_url": self.source_url, "chunk_idx": self.chunk_idx,
                "accession": self.accession, "doc_type": self.doc_type,
                "filing_items": self.filing_items, "source_kind": self.source_kind}


def _trim_signatures(lines: list[str]) -> list[str]:
    """Drop the signature block: from the last "SIGNATURES" line, if little text follows it.

    Tables of contents also list "Signatures", so the first match is not the real block.
    """
    for i in range(len(lines) - 1, -1, -1):
        if _SIGNATURES.match(lines[i]):
            tail_words = sum(len(line.split()) for line in lines[i + 1:])
            return lines[:i] if tail_words <= 300 else lines
    return lines


def _sections(lines: list[str], label_for) -> list[Section]:
    """Group lines under the most recent heading. `label_for(line)` returns a label or None."""
    sections: list[Section] = []
    label, buf = None, []
    for line in lines:
        new = label_for(line)
        if new is not None:
            if label is not None and buf:
                sections.append(Section(label, "\n".join(buf)))
            label, buf = new, [line]
        elif label is not None:
            buf.append(line)
    if label is not None and buf:
        sections.append(Section(label, "\n".join(buf)))
    return sections


def split_8k(text: str) -> list[Section]:
    """Sections per 8-K Item. The cover page before the first Item is dropped."""
    lines = _trim_signatures(text.splitlines())

    def label(line: str) -> str | None:
        m = _8K_ITEM.match(line)
        return f"Item {m.group(1)}" if m else None

    return _sections(lines, label)


def split_periodic(text: str, form: str) -> list[Section]:
    """Sections per 10-Q Part/Item or 10-K Item. For 10-Ks only Items 1A, 3, and 7 are kept."""
    lines = _trim_signatures(text.splitlines())
    part = {"value": ""}

    def label(line: str) -> str | None:
        p = _PART.match(line)
        if p:
            part["value"] = p.group(1).upper()
            return None
        m = _PERIODIC_ITEM.match(line)
        if not m:
            return None
        num = m.group(1).upper()
        if form == "10-Q" and part["value"]:
            return f"Part {part['value']} Item {num}"
        return f"Item {num}"

    sections = _sections(lines, label)
    if form == "10-K":
        sections = [s for s in sections if s.item.removeprefix("Item ") in TEN_K_ITEMS]
    return sections


def window(text: str, size_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping word windows. The last window may be shorter."""
    if size_words <= 0 or not 0 <= overlap_words < size_words:
        raise ValueError("need size_words > 0 and 0 <= overlap_words < size_words")
    words = text.split()
    if not words:
        return []
    step = size_words - overlap_words
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size_words]))
        if start + size_words >= len(words):
            break
    return out


def sections_for(text: str, doc_type: str, form: str) -> list[Section]:
    if doc_type.startswith("EX-"):
        return [Section(doc_type, text)]  # exhibits are one unit, labelled by exhibit type
    if form == "8-K":
        return split_8k(text)
    if form in ("10-Q", "10-K"):
        return split_periodic(text, form)
    return [Section("body", text)]


def chunk_filing(text: str, *, ticker: str, form: str, filed_date: int, source_url: str,
                 accession: str, doc_type: str, filing_items: tuple[str, ...] = (),
                 cfg: RetrievalConfig = RetrievalConfig()) -> list[Chunk]:
    size = round(cfg.chunk_tokens * cfg.words_per_token)
    overlap = round(cfg.overlap_tokens * cfg.words_per_token)
    chunks: list[Chunk] = []
    for section in sections_for(text, doc_type, form):
        if len(section.text.split()) < cfg.min_section_words:
            continue  # tables of contents and stub headings
        for piece in window(section.text, size, overlap):
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{ticker}:{accession}:{doc_type}:{idx}",
                text=piece, ticker=ticker, form=form, filed_date=filed_date,
                item=section.item, source_url=source_url, chunk_idx=idx, accession=accession,
                doc_type=doc_type, filing_items=",".join(filing_items), source_kind="filing"))
    return chunks


def chunk_headline(*, ticker: str, text: str, published: int, url: str, key: str) -> Chunk:
    return Chunk(chunk_id=f"{ticker}:news:{key}", text=text, ticker=ticker, form="HEADLINE",
                 filed_date=published, item="news", source_url=url, chunk_idx=0,
                 doc_type="HEADLINE", source_kind="headline")
