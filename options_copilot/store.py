"""Chroma-backed retrieval with hard metadata filters, MMR selection, and a coverage check.

Two collections live under data/chroma/: `filings` (filing and headline chunks) and
`reflections` (lessons). Every filings query is hard-filtered by ticker and a date window
before ranking. MMR and the coverage check are pure functions so they can be unit-tested.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from .chunker import Chunk
from .config import RetrievalConfig
from .models import Reflection, ReflectionTrigger, RetrievedChunk, from_yyyymmdd, yyyymmdd

FILINGS = "filings"
REFLECTIONS = "reflections"


# --- pure retrieval math ------------------------------------------------------------------

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mmr(query: Sequence[float], candidates: Sequence[Sequence[float]], k: int,
        lambda_: float = 0.5) -> list[int]:
    """Maximal Marginal Relevance: indices of up to k candidates, in selection order.

    Each step picks the candidate maximizing
        lambda * sim(query, c) - (1 - lambda) * max(sim(c, already_selected))
    so a near-duplicate of a selected chunk loses to a slightly less relevant new one.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    relevance = [cosine(query, c) for c in candidates]
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        def score(i: int) -> float:
            redundancy = max((cosine(candidates[i], candidates[j]) for j in selected), default=0.0)
            return lambda_ * relevance[i] - (1 - lambda_) * redundancy
        best = max(remaining, key=lambda i: (score(i), -i))
        selected.append(best)
        remaining.remove(best)
    return selected


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float
    embedding: tuple[float, ...]

    @property
    def is_filing(self) -> bool:
        return self.metadata.get("source_kind") == "filing"


@dataclass(frozen=True)
class Coverage:
    ok: bool
    reason: str
    candidates: int
    passing: int
    passing_filings: int


def assess_coverage(candidates: Sequence[Candidate], cfg: RetrievalConfig, window: str) -> Coverage:
    """Decide whether retrieval found enough evidence. Missing evidence is never safe."""
    passing = [c for c in candidates if c.similarity >= cfg.min_similarity]
    filings = [c for c in passing if c.is_filing]
    n, p, f = len(candidates), len(passing), len(filings)
    if n == 0:
        return Coverage(False, f"no filings or headlines indexed in {window}", n, p, f)
    if f == 0:
        why = "only headlines matched" if p else "nothing passed the similarity threshold"
        return Coverage(False, f"no filing evidence in {window} ({why})", n, p, f)
    if p < cfg.min_chunks:
        return Coverage(False, f"only {p} chunk(s) passed the similarity threshold in {window}, "
                               f"need {cfg.min_chunks}", n, p, f)
    return Coverage(True, f"{p} of {n} chunks passed, {f} from filings", n, p, f)


def select(query_emb: Sequence[float], candidates: Sequence[Candidate],
           cfg: RetrievalConfig) -> list[Candidate]:
    """MMR over the passing candidates, guaranteeing at least one filing chunk is kept."""
    passing = [c for c in candidates if c.similarity >= cfg.min_similarity]
    picked = [passing[i] for i in mmr(query_emb, [c.embedding for c in passing], cfg.k, cfg.mmr_lambda)]
    if picked and not any(c.is_filing for c in picked):
        best_filing = next((c for c in passing if c.is_filing), None)
        if best_filing is not None:
            picked[-1] = best_filing
    return picked


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    coverage: Coverage
    window_start: date
    window_end: date


# --- Chroma wrapper -----------------------------------------------------------------------

class Store:
    def __init__(self, path: Path, embedding_function: Any = None, cfg: RetrievalConfig = RetrievalConfig()):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.cfg = cfg
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        if embedding_function is None:  # all-MiniLM-L6-v2 via ONNX, downloaded once on first use
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embedding_function = DefaultEmbeddingFunction()
        self._embed = embedding_function
        kwargs: dict[str, Any] = {"configuration": {"hnsw": {"space": "cosine"}},
                                  "embedding_function": embedding_function}
        self.filings = self._client.get_or_create_collection(FILINGS, **kwargs)
        self.reflections = self._client.get_or_create_collection(REFLECTIONS, **kwargs)

    # filings
    def add_chunks(self, chunks: Sequence[Chunk], batch: int = 64) -> int:
        for start in range(0, len(chunks), batch):
            part = chunks[start:start + batch]
            self.filings.upsert(ids=[c.chunk_id for c in part], documents=[c.text for c in part],
                                metadatas=[c.metadata() for c in part])
        return len(chunks)

    def delete_accession(self, ticker: str, accession: str) -> None:
        self.filings.delete(where={"$and": [{"ticker": ticker}, {"accession": accession}]})

    def has_accession(self, ticker: str, accession: str) -> bool:
        got = self.filings.get(where={"$and": [{"ticker": ticker}, {"accession": accession}]},
                               limit=1, include=[])
        return bool(got["ids"])

    def search_filings(self, ticker: str, query: str, as_of: date) -> RetrievalResult:
        cfg = self.cfg
        start = as_of - timedelta(days=cfg.window_days)
        window = f"the {cfg.window_days}-day window {start.isoformat()} to {as_of.isoformat()}"
        where = {"$and": [{"ticker": ticker.upper()},
                          {"filed_date": {"$gte": yyyymmdd(start)}},
                          {"filed_date": {"$lte": yyyymmdd(as_of)}}]}
        candidates, query_emb = self._query(self.filings, query, where, cfg.fetch_k)
        coverage = assess_coverage(candidates, cfg, window)
        picked = select(query_emb, candidates, cfg) if coverage.ok else []
        return RetrievalResult([_to_retrieved(c) for c in picked], coverage, start, as_of)

    def list_filings(self, ticker: str) -> list[dict[str, Any]]:
        got = self.filings.get(where={"ticker": ticker.upper()}, include=["metadatas"])
        seen: dict[str, dict[str, Any]] = {}
        for m in got["metadatas"]:
            key = m.get("accession") or m.get("source_url")
            row = seen.setdefault(key, {"form": m["form"], "date": from_yyyymmdd(m["filed_date"]).isoformat(),
                                        "items": m.get("filing_items", ""), "chunks": 0,
                                        "url": m["source_url"], "kind": m.get("source_kind", "")})
            row["chunks"] += 1
        return sorted(seen.values(), key=lambda r: r["date"], reverse=True)

    # reflections
    def add_reflection(self, r: Reflection) -> str:
        if not r.lesson.strip():
            raise ValueError("lesson text is empty")
        rid = "lesson:" + hashlib.sha256(f"{r.ticker}|{r.lesson}".encode()).hexdigest()[:16]
        self.reflections.upsert(ids=[rid], documents=[r.lesson.strip()], metadatas=[{
            "ticker": r.ticker.upper(), "trigger": r.trigger.value,
            "created_on": yyyymmdd(r.created_on), "tags": ",".join(r.tags),
            "synthetic": bool(r.synthetic)}])
        return rid

    def search_lessons(self, ticker: str, query: str) -> list[Reflection]:
        cfg = self.cfg
        where = {"ticker": {"$in": [ticker.upper(), "ANY"]}}
        candidates, query_emb = self._query(self.reflections, query, where, cfg.lesson_fetch_k)
        order = mmr(query_emb, [c.embedding for c in candidates], cfg.lesson_k, cfg.mmr_lambda)
        out = []
        for i in order:
            m = candidates[i].metadata
            out.append(Reflection(
                lesson=candidates[i].text, trigger=ReflectionTrigger(m["trigger"]), ticker=m["ticker"],
                tags=[t for t in str(m.get("tags", "")).split(",") if t],
                created_on=from_yyyymmdd(int(m["created_on"])), synthetic=bool(m.get("synthetic", True))))
        return out

    # internals
    def _query(self, collection, query: str, where: dict, n: int) -> tuple[list[Candidate], list[float]]:
        if collection.count() == 0:
            return [], []
        query_emb = [float(x) for x in self._embed([query])[0]]
        res = collection.query(query_embeddings=[query_emb], n_results=n, where=where,
                               include=["documents", "metadatas", "distances", "embeddings"])
        ids = res["ids"][0]
        if not ids:
            return [], query_emb
        cands = [Candidate(chunk_id=i, text=d, metadata=dict(m), similarity=1.0 - float(dist),
                           embedding=tuple(float(x) for x in e))
                 for i, d, m, dist, e in zip(ids, res["documents"][0], res["metadatas"][0],
                                             res["distances"][0], res["embeddings"][0])]
        return cands, query_emb


def _to_retrieved(c: Candidate) -> RetrievedChunk:
    m = c.metadata
    item = m["item"]
    if m.get("doc_type", "").startswith("EX-") and m.get("filing_items"):
        item = f"Item {m['filing_items']} ({m['doc_type']})"
    return RetrievedChunk(text=c.text, ticker=m["ticker"], form=m["form"], filed_date=int(m["filed_date"]),
                          item=item, source_url=m["source_url"], chunk_idx=int(m["chunk_idx"]),
                          score=round(c.similarity, 4))
