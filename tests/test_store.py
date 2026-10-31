"""MMR, the coverage check, and the Chroma wrapper (offline, fake embeddings)."""
from datetime import date

import pytest

from options_copilot.chunker import Chunk, chunk_headline
from options_copilot.config import RetrievalConfig
from options_copilot.models import Reflection, ReflectionTrigger
from options_copilot.store import Candidate, assess_coverage, mmr, select


def test_mmr_prefers_a_new_angle_over_a_near_duplicate():
    query = [0.9, 0.44]
    cands = [[1.0, 0.0], [0.999, 0.04], [0.6, 0.8]]  # 0 and 1 are near-duplicates
    assert mmr(query, cands, k=2, lambda_=0.5) == [1, 2]
    assert mmr(query, cands, k=2, lambda_=1.0) == [1, 0]  # pure relevance keeps the duplicate


def test_mmr_handles_small_inputs_and_bad_lambda():
    assert mmr([1.0], [], k=5) == []
    assert mmr([1.0, 0.0], [[1.0, 0.0]], k=5) == [0]
    with pytest.raises(ValueError):
        mmr([1.0], [[1.0]], k=1, lambda_=1.5)


def cand(cid, sim, kind="filing", emb=(1.0, 0.0)):
    return Candidate(cid, "text", {"source_kind": kind}, sim, emb)


CFG = RetrievalConfig(min_similarity=0.3, min_chunks=2, k=2)


@pytest.mark.parametrize("cands, ok, phrase", [
    ([], False, "no filings or headlines"),
    ([cand("a", 0.9, "headline"), cand("b", 0.8, "headline")], False, "only headlines matched"),
    ([cand("a", 0.1), cand("b", 0.2)], False, "nothing passed"),
    ([cand("a", 0.9), cand("b", 0.1)], False, "only 1 chunk"),
    ([cand("a", 0.9), cand("b", 0.4, "headline")], True, "2 of 2"),
])
def test_coverage_never_treats_missing_evidence_as_enough(cands, ok, phrase):
    cov = assess_coverage(cands, CFG, "the window")
    assert cov.ok is ok and phrase in cov.reason


def test_selection_always_keeps_a_filing():
    cands = [cand("h1", 0.95, "headline", (1.0, 0.0)), cand("h2", 0.9, "headline", (0.0, 1.0)),
             cand("f1", 0.35, "filing", (0.7, 0.7))]
    picked = select([1.0, 0.2], cands, CFG)
    assert any(c.is_filing for c in picked) and len(picked) == 2


def chunk(cid, ticker, filed, text, item="Item 8.01", kind="filing"):
    return Chunk(chunk_id=cid, text=text, ticker=ticker, form="8-K", filed_date=filed, item=item,
                 source_url=f"https://example.com/{cid}", chunk_idx=0, accession=f"acc-{cid}",
                 doc_type="8-K", source_kind=kind)


def test_hard_filters_on_ticker_and_window(store):
    text = "merger agreement acquisition tender offer pending deal"
    store.add_chunks([
        chunk("in", "TICKER_A", 20261001, text),
        chunk("old", "TICKER_A", 20260101, text),         # outside the 90-day window
        chunk("other", "TICKER_B", 20261001, text),       # other ticker
        chunk("future", "TICKER_A", 20261105, text),      # after as_of
        chunk("in2", "TICKER_A", 20261015, text + " shareholders vote", item="Item 1.01"),
    ])
    res = store.search_filings("ticker_a", "merger agreement acquisition", date(2026, 10, 30))
    urls = {c.source_url.rsplit("/", 1)[1] for c in res.chunks}
    assert res.coverage.ok and urls == {"in", "in2"}
    assert all(c.ticker == "TICKER_A" for c in res.chunks)


def test_empty_window_is_insufficient(store):
    store.add_chunks([chunk("old", "TICKER_A", 20250101, "merger agreement")])
    res = store.search_filings("TICKER_A", "merger agreement", date(2026, 10, 30))
    assert not res.coverage.ok and res.chunks == []
    empty = store.search_filings("TICKER_Z", "merger", date(2026, 10, 30))
    assert empty.coverage.candidates == 0


def test_exhibit_citation_names_the_filing_items(store):
    c = Chunk(chunk_id="x", text="results of operations revenue guidance quarter", ticker="TICKER_A",
              form="8-K", filed_date=20261020, item="EX-99.1", source_url="https://example.com/ex991.htm",
              chunk_idx=0, accession="acc-x", doc_type="EX-99.1", filing_items="2.02,9.01")
    store.add_chunks([c, chunk("y", "TICKER_A", 20261020, "results of operations revenue quarter guidance")])
    res = store.search_filings("TICKER_A", "results of operations revenue guidance", date(2026, 10, 30))
    assert "Item 2.02,9.01 (EX-99.1)" in {r.item for r in res.chunks}


def test_upsert_is_idempotent_and_listing_groups_by_filing(store):
    c = [chunk("a", "TICKER_A", 20261001, "one"), chunk_headline(ticker="TICKER_A", text="two", published=20261002,
                                                                  url="https://example.com/h", key="k")]
    store.add_chunks(c)
    store.add_chunks(c)
    assert store.filings.count() == 2
    assert store.has_accession("TICKER_A", "acc-a") and not store.has_accession("TICKER_A", "acc-zz")
    rows = store.list_filings("TICKER_A")
    assert [r["form"] for r in rows] == ["HEADLINE", "8-K"]


def test_lessons_filter_to_ticker_or_any(store):
    for ticker, lesson in [("ANY", "officer departure cfo exit caution"), ("TICKER_A", "cfo exit hurt this one"),
                           ("TICKER_B", "cfo exit unrelated ticker"), ("ANY", "merger agreement block")]:
        store.add_reflection(Reflection(lesson=lesson, trigger=ReflectionTrigger.REJECTION, ticker=ticker))
    got = store.search_lessons("TICKER_A", "cfo exit officer departure")
    assert {r.ticker for r in got} <= {"ANY", "TICKER_A"}
    assert len(got) == 3 and all(r.synthetic for r in got)
    with pytest.raises(ValueError):
        store.add_reflection(Reflection(lesson="  ", trigger=ReflectionTrigger.VERIFIER))
