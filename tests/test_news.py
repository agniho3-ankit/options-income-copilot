from datetime import date

from options_copilot.news import fetch_headlines, parse_news

NESTED = [{"id": "1", "content": {"title": "TICKER_A wins contract", "summary": "Synthetic summary.",
                                  "pubDate": "2026-10-20T12:00:00Z",
                                  "provider": {"displayName": "Example Wire"},
                                  "canonicalUrl": {"url": "https://example.com/a"}}}]
FLAT = [{"title": "TICKER_A probe rumor", "publisher": "Example News", "link": "https://example.com/b",
         "providerPublishTime": 1_792_000_000}]


def test_parses_both_yfinance_shapes():
    nested, flat = parse_news("ticker_a", NESTED), parse_news("TICKER_A", FLAT)
    assert nested[0].ticker == "TICKER_A" and nested[0].published == 20261020
    assert nested[0].publisher == "Example Wire" and nested[0].url == "https://example.com/a"
    assert "Synthetic summary" in nested[0].text
    assert flat[0].publisher == "Example News" and flat[0].published > 20260000


def test_skips_items_without_title_or_date():
    assert parse_news("X", [{"content": {"title": ""}}, {"title": "no date"}, {}]) == []


def test_fetch_caches_per_day(tmp_path):
    class FakeTicker:
        calls = 0

        def __init__(self, t):
            FakeTicker.calls += 1
            self.news = NESTED

    class FakeYF:
        Ticker = FakeTicker

    today = date(2026, 10, 30)
    assert len(fetch_headlines("TICKER_A", tmp_path, today, yf_module=FakeYF)) == 1
    assert len(fetch_headlines("TICKER_A", tmp_path, today, yf_module=FakeYF)) == 1
    assert FakeTicker.calls == 1
    assert (tmp_path / "news" / "TICKER_A-2026-10-30.json").exists()


def test_empty_reply_is_not_cached(tmp_path):
    class Empty:
        def __init__(self, t):
            self.news = []

    class FakeYF:
        Ticker = Empty

    assert fetch_headlines("TICKER_A", tmp_path, date(2026, 10, 30), yf_module=FakeYF) == []
    assert not (tmp_path / "news").exists()
