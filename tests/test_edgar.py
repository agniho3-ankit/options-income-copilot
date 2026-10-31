"""EDGAR helpers, with synthetic data and fake HTTP. No network."""
from datetime import date

import pytest

from options_copilot.edgar import (EdgarClient, EdgarError, Throttle, cache_path, cik_from_table,
                                   exhibits_from_index, filings_from_submissions, html_to_text)

AGENT = "Test Agent test@example.com"  # privacy-audit: allow (example.com test value)


class FakeResp:
    def __init__(self, body: bytes, status: int = 200):
        self.content, self.status_code = body, status


class FakeSession:
    def __init__(self, body=b"{}", status=200, exc=None):
        self.body, self.status, self.exc = body, status, exc
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        if self.exc:
            raise self.exc
        return FakeResp(self.body, self.status)


def no_wait():
    return Throttle(min_interval=0, sleep=lambda s: None)


SUBMISSIONS = {"filings": {"recent": {
    "form": ["8-K", "4", "10-Q", "8-K", "10-K", "8-K"],
    "accessionNumber": ["0000000001-26-000010", "x", "0000000001-26-000008", "0000000001-26-000005",
                        "0000000001-25-000001", "0000000001-26-000011"],
    "filingDate": ["2026-10-20", "2026-10-19", "2026-08-01", "2026-06-01", "2025-11-01", "2026-10-25"],
    "primaryDocument": ["a8k.htm", "f4.xml", "q.htm", "old8k.htm", "k.htm", ""],
    "items": ["2.02,9.01", "", "", "5.02", "", "8.01"],
}}}


def test_filings_filtered_by_form_and_window_newest_first():
    refs = filings_from_submissions(SUBMISSIONS, "ticker_a", 1, date(2026, 7, 30), date(2026, 10, 28))
    assert [(r.form, r.filed_date) for r in refs] == [("8-K", 20261020), ("10-Q", 20260801)]
    assert refs[0].ticker == "TICKER_A"
    assert refs[0].items == ("2.02", "9.01")
    assert refs[0].url == "https://www.sec.gov/Archives/edgar/data/1/000000000126000010/a8k.htm"  # privacy-audit: allow (synthetic path)
    assert refs[0].index_url.endswith("/000000000126000010/0000000001-26-000010-index.htm")


def test_filings_skip_rows_without_primary_document():
    refs = filings_from_submissions(SUBMISSIONS, "TICKER_A", 1, date(2026, 10, 21), date(2026, 10, 30))
    assert refs == []  # the 2026-10-25 8-K has no primary document


def test_cik_lookup_is_case_insensitive_and_errors_on_unknown():
    table = {"0": {"cik_str": 42, "ticker": "TICKER_A", "title": "Example Corp"}}
    assert cik_from_table(table, "ticker_a") == 42
    with pytest.raises(EdgarError):
        cik_from_table(table, "TICKER_B")


def test_html_to_text_drops_scripts_styles_and_hidden_ixbrl():
    html = """<html><head><title>t</title><style>.x{}</style></head><body>
      <ix:header><ix:hidden>dei:SecretTag 123</ix:hidden></ix:header>
      <div style="display: none">hidden text</div><script>alert(1)</script>
      <p>Item&nbsp;2.02   Results of   Operations</p><p></p><table><tr><td>Revenue</td><td>10</td></tr></table>
    </body></html>"""
    text = html_to_text(html)
    assert "Item 2.02 Results of Operations" in text
    assert "Revenue" in text and "10" in text
    for gone in ("SecretTag", "hidden text", "alert", ".x{}"):
        assert gone not in text


def test_exhibits_from_index_keeps_only_ex99_documents():
    index = """<table class="tableFile"><tr><th>Seq</th></tr>
      <tr><td>1</td><td>8-K</td><td><a href="/x/a8k.htm">a8k.htm</a></td><td>8-K</td><td>1</td></tr>
      <tr><td>2</td><td>Press release</td><td><a href="/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>1</td></tr>
      <tr><td>3</td><td>Deck</td><td><a href="/x/ex992.htm">ex992.htm</a></td><td>EX-99.2</td><td>1</td></tr>
      <tr><td>4</td><td>Agreement</td><td><a href="/x/ex101.htm">ex101.htm</a></td><td>EX-10.1</td><td>1</td></tr>
      <tr><td>5</td><td>Bad</td><td><a href="/x/../evil">../evil</a></td><td>EX-99.3</td><td>1</td></tr>
    </table>"""
    assert exhibits_from_index(index) == [("ex991.htm", "EX-99.1"), ("ex992.htm", "EX-99.2")]


def test_cache_path_maps_url_and_refuses_traversal(tmp_path):
    p = cache_path(tmp_path, "https://www.sec.gov/Archives/edgar/data/1/2/a.htm")
    assert p == tmp_path / "edgar" / "www.sec.gov" / "Archives" / "edgar" / "data" / "1" / "2" / "a.htm"
    for bad in ("http://www.sec.gov/a", "https://www.sec.gov/a/../../b", "file:///etc/passwd"):
        with pytest.raises(EdgarError):
            cache_path(tmp_path, bad)


def test_client_requires_user_agent(tmp_path):
    with pytest.raises(EdgarError, match="SEC_USER_AGENT"):
        EdgarClient("  ", tmp_path)


def test_client_sends_user_agent_caches_and_hides_it(tmp_path):
    session = FakeSession(b'{"ok": 1}')
    client = EdgarClient(AGENT, tmp_path, session=session, throttle=no_wait())
    url = "https://www.sec.gov/Archives/edgar/data/1/2/a.htm"
    assert client.fetch(url) == b'{"ok": 1}'
    assert client.fetch(url) == b'{"ok": 1}'  # second read comes from the cache
    assert len(session.calls) == 1 and session.calls[0][1]["User-Agent"] == AGENT
    assert AGENT not in repr(client)


def test_mutable_cache_expires(tmp_path):
    now = {"t": 1_000_000.0}
    session = FakeSession(b"{}")
    client = EdgarClient(AGENT, tmp_path, session=session, throttle=no_wait(), clock=lambda: now["t"])
    url = "https://data.sec.gov/submissions/CIK0000000001.json"
    client.fetch(url, mutable=True)
    path = cache_path(tmp_path, url)
    import os
    os.utime(path, (now["t"], now["t"]))
    now["t"] += 3600
    client.fetch(url, mutable=True)
    assert len(session.calls) == 1  # still fresh after an hour
    now["t"] += 13 * 3600
    client.fetch(url, mutable=True)
    assert len(session.calls) == 2  # stale after 12 hours


def test_errors_never_include_the_user_agent(tmp_path):
    for session in (FakeSession(exc=ConnectionError(f"boom {AGENT}")), FakeSession(status=403)):
        client = EdgarClient(AGENT, tmp_path, session=session, throttle=no_wait())
        with pytest.raises(EdgarError) as err:
            client.fetch("https://www.sec.gov/files/company_tickers.json")
        assert AGENT not in str(err.value)
        assert err.value.__cause__ is None


def test_throttle_spaces_requests():
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    t = Throttle(min_interval=0.15, clock=lambda: clock["t"], sleep=sleep)
    for _ in range(3):
        t.wait()
    assert slept == pytest.approx([0.15, 0.15])
    clock["t"] += 1.0
    t.wait()
    assert len(slept) == 2  # no wait after a long gap
