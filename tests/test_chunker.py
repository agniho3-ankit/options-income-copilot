import pytest

from options_copilot.chunker import chunk_filing, chunk_headline, split_8k, split_periodic, window
from options_copilot.config import RetrievalConfig


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


EIGHT_K = "\n".join([
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION", "FORM 8-K", "cover page boilerplate",
    "Item 1.01 Entry into a Material Definitive Agreement", words("alpha", 500),
    "Item 5.02 Departure of Directors or Certain Officers", words("beta", 120),
    "Item 9.01 Financial Statements and Exhibits", words("gamma", 60),
    "SIGNATURES", "Pursuant to the requirements signed by someone",
])


def test_8k_splits_on_items_and_drops_cover_and_signatures():
    sections = split_8k(EIGHT_K)
    assert [s.item for s in sections] == ["Item 1.01", "Item 5.02", "Item 9.01"]
    assert "cover page" not in " ".join(s.text for s in sections)
    assert "Pursuant" not in sections[-1].text


def test_chunks_never_cross_items():
    chunks = chunk_filing(EIGHT_K, ticker="TICKER_A", form="8-K", filed_date=20261020,
                          source_url="https://example.com/a8k.htm", accession="0000000001-26-000010",
                          doc_type="8-K", filing_items=("1.01", "5.02", "9.01"))
    prefixes = {"Item 1.01": "alpha", "Item 5.02": "beta", "Item 9.01": "gamma"}
    for c in chunks:
        others = [p for item, p in prefixes.items() if item != c.item]
        assert not any(o in c.text for o in others), c.item
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    m = chunks[0].metadata()
    assert m["filed_date"] == 20261020 and m["filing_items"] == "1.01,5.02,9.01" and m["source_kind"] == "filing"


def test_window_size_and_overlap():
    cfg = RetrievalConfig()
    size, overlap = round(cfg.chunk_tokens * cfg.words_per_token), round(cfg.overlap_tokens * cfg.words_per_token)
    pieces = window(words("w", 700), size, overlap)
    assert all(len(p.split()) <= size for p in pieces)
    first, second = pieces[0].split(), pieces[1].split()
    assert first[-overlap:] == second[:overlap]
    assert pieces[-1].split()[-1] == "w699"
    assert window("", 10, 2) == []
    with pytest.raises(ValueError):
        window("a b", 5, 5)


def test_10q_labels_part_and_item_and_drops_table_of_contents():
    text = "\n".join([
        "PART I", "Item 1. Financial Statements 3", "Item 2. MD&A 20",          # table of contents
        "PART I - FINANCIAL INFORMATION", "Item 1. Financial Statements", words("fin", 80),
        "Item 2. Management's Discussion and Analysis", words("mda", 80),
        "PART II - OTHER INFORMATION", "Item 1A. Risk Factors", words("risk", 80),
    ])
    chunks = chunk_filing(text, ticker="TICKER_A", form="10-Q", filed_date=20260801, source_url="u",
                          accession="acc", doc_type="10-Q", cfg=RetrievalConfig(min_section_words=40))
    assert [c.item for c in chunks] == ["Part I Item 1", "Part I Item 2", "Part II Item 1A"]


def test_10k_keeps_only_risk_factors_legal_and_mda():
    text = "\n".join(["Item 1. Business", words("biz", 60), "Item 1A. Risk Factors", words("risk", 60),
                      "Item 3. Legal Proceedings", words("legal", 60), "Item 7. MD&A", words("mda", 60),
                      "Item 8. Financial Statements", words("fs", 60)])
    assert [s.item for s in split_periodic(text, "10-K")] == ["Item 1A", "Item 3", "Item 7"]


def test_exhibit_is_one_section_labelled_by_type():
    chunks = chunk_filing(words("pr", 100), ticker="TICKER_A", form="8-K", filed_date=20261020,
                          source_url="u", accession="acc", doc_type="EX-99.1", filing_items=("2.02",))
    assert {c.item for c in chunks} == {"EX-99.1"}


def test_headline_chunk():
    c = chunk_headline(ticker="TICKER_A", text="Synthetic headline.", published=20261020, url="u", key="k")
    assert c.form == "HEADLINE" and c.source_kind == "headline" and c.chunk_id == "TICKER_A:news:k"


def test_signatures_in_table_of_contents_do_not_truncate():
    text = "\n".join(["PART I", "Item 1. Financial Statements", "Signatures",          # table of contents
                      "PART I", "Item 2. MD&A", words("mda", 400), "SIGNATURES", "signed"])
    sections = split_periodic(text, "10-Q")
    assert sections[-1].item == "Part I Item 2" and "mda399" in sections[-1].text
    assert "signed" not in sections[-1].text
