"""UI API (slice 1): read-only routes, local binding, CSRF, masking. No network, no LLM."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from options_copilot.api.app import create_app  # noqa: E402
from options_copilot.api.security import BindError, check_bind  # noqa: E402
from options_copilot.config import load_settings  # noqa: E402
from options_copilot.db import Database  # noqa: E402
from options_copilot.guardrails import risk_score  # noqa: E402
from options_copilot.playbook import load_playbook  # noqa: E402
from options_copilot.review import brief_signals, queue as review_queue  # noqa: E402

UA = "Test Copilot tester@example.com"
BASE = "http://127.0.0.1:8765"
ORIGIN = {"Origin": BASE}


def _payload(credit: float, signals: dict, risks: list[str] | None = None, judge: bool = False) -> dict:
    return {"proposal": {"ticker": "MSFT", "strategy": "put_credit_spread", "contracts": 1, "expiry": "2027-01-15",
                         "legs": [{"option_type": "put", "side": "short", "strike": 450.0, "expiry": "2027-01-15"},
                                  {"option_type": "put", "side": "long", "strike": 440.0, "expiry": "2027-01-15"}]},
            "numbers": {"credit": credit, "max_loss": 1000 - credit, "breakevens": [446.53], "pop": 0.7712,
                        "risk_pct_of_portfolio": 0.653, "collateral": 1000 - credit},
            "signals": signals, "risks": risks or [], "landmine_verdict": "CLEAR",
            "judge": {"flags": [{"claim": "x", "issue": "unsupported"}]} if judge else {"flags": []},
            "citations": [{"form": "8-K", "date": "2026-10-28", "item": "Item 8.01", "url": "https://www.sec.gov/x"}]}


@pytest.fixture()
def seeded():
    db = Database(":memory:")
    pb = load_playbook(load_settings(env={}).playbook_path)
    r1 = db.start_run("multi_agent", "MSFT", "fake", pb.sha256)
    db.save_brief("brf-a", r1, "entry", "MSFT", "pending", "CLEAR", _payload(347.5, {}))
    db.save_brief("brf-b", None, "entry", "META", "pending", "CAUTION",
                  _payload(212.0, {"judge_flag": 1, "grader_zero_relevant": 1}, ["insider sales"], judge=True))
    db.save_brief("brf-c", None, "entry", "BAC", "pending", "CAUTION", _payload(98.25, {}, ["iv rank low"]))
    db.save_brief("brf-d", None, "entry", "AAPL", "pending", "CLEAR", _payload(150.0, {}))
    db.defer("brf-d")
    r2 = db.start_run("multi_agent", "AAPL", "fake", pb.sha256)
    db.add_guardrail_event(r2, 1, None, "coverage_guard", "pre", "static", "block", None,
                           "0 of 20 chunks passed")
    db.finish_run(r2, "INSUFFICIENT_EVIDENCE")
    # trace content that was (wrongly) stored raw: the API must still mask it and drop the User-Agent
    db.add_span(r1, "sp1", None, "researcher", "react", "2026-11-19T19:03:12", "2026-11-19T19:04:29", 77125, 3,
                {"ticker": "MSFT"}, {"note": f"asked EDGAR as {UA}; account 12345678 holds $12,500"})
    db.add_guardrail_event(r1, 1, "sp1", "judge", "post", "risk_score", "flag", "judge_flag",
                           f"claim mentions $25,000 and {UA}")
    db.save_tot_nodes(r1, [{"id": "n1", "parent_id": None, "depth": 1, "status": "kept", "score": 0.4,
                            "prune_reason": None, "source": "generator", "note": "loss of $40,000"}])
    db.add_eval("eval", {"unsafe_passed": 0, "false_blocks": 0})
    settings = load_settings(env={"SEC_USER_AGENT": UA})
    return db, pb, settings, r1, r2


@pytest.fixture()
def client(seeded, tmp_path):
    db, pb, settings, *_ = seeded
    (tmp_path / "eval_2026-11-16_x3.json").write_text(json.dumps({"unsafe_passed": 0}))
    app = create_app(settings, pb, main_db=db, outputs_dir=tmp_path)
    return TestClient(app, base_url=BASE)


def test_bind_refuses_anything_but_loopback():
    assert check_bind("127.0.0.1") == "127.0.0.1"
    for host in ("0.0.0.0", "localhost", "::", "::1", "192.168.1.5", ""):
        with pytest.raises(BindError):
            check_bind(host)


def test_ui_command_refuses_public_host(capsys):
    from options_copilot.cli import main
    assert main(["ui", "--host", "0.0.0.0"]) == 2
    assert "127.0.0.1 only" in capsys.readouterr().err


def test_host_header_must_be_loopback(client):
    assert client.get("/api/session").status_code == 200
    r = client.get("/api/session", headers={"Host": "evil.example:8765"})
    assert r.status_code == 400


def test_unsafe_methods_need_token_and_local_origin(client):
    token = client.get("/api/session").json()["csrf_token"]
    assert len(token) >= 32
    # rejected before routing, so even an unknown path never runs without the token
    assert client.post("/api/anything", headers=ORIGIN).status_code == 403
    assert client.post("/api/anything", headers={**ORIGIN, "X-CSRF-Token": "wrong"}).status_code == 403
    assert client.post("/api/anything", headers={"Origin": "http://evil.example", "X-CSRF-Token": token}
                       ).status_code == 403
    assert client.post("/api/anything", headers={"X-CSRF-Token": token}).status_code == 403   # no Origin
    # a valid token from the local origin gets through the guard (and then 404s: no such route)
    assert client.post("/api/anything", headers={**ORIGIN, "X-CSRF-Token": token}).status_code in (404, 405)


WRITE_ROUTES = {"/api/briefs/{brief_id}/approve", "/api/briefs/{brief_id}/reject", "/api/briefs/{brief_id}/defer",
                "/api/runs"}


def test_the_only_write_routes_are_the_three_decisions_and_starting_a_run(client):
    writes = {r.path for r in client.app.routes if set(getattr(r, "methods", set())) - {"GET", "HEAD"}}
    assert writes == WRITE_ROUTES


def test_no_cdn_docs_and_strict_csp(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    csp = client.get("/api/session").headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "http" not in csp


def test_session_is_paper_only(client):
    s = client.get("/api/session").json()
    assert s["paper_only"] is True and s["databases"] == {"main": True, "eval": False}


def test_queue_order_matches_review_queue(client, seeded):
    db, pb, *_ = seeded
    expected = review_queue(db, pb)
    got = client.get("/api/queue").json()
    assert [i["brief_id"] for i in got["trades"]] == [i["brief_id"] for i in expected["trades"]]
    assert len({i["tier"] for i in got["trades"]}) >= 2      # the fixture spans more than one tier
    assert got["trades"][-1]["brief_id"] == "brf-d" and got["trades"][-1]["deferred"] is True
    assert [i["outcome"] for i in got["info"]] == ["INSUFFICIENT_EVIDENCE"]
    assert got["info"][0]["why"] == "0 of 20 chunks passed"
    for item, exp in zip(got["trades"], expected["trades"]):
        assert (item["score"], item["tier"]) == (exp["score"], exp["tier"])
        assert [t["text"] for t in item["top"]] == [t for _, t in exp["top"]]
    first = next(i for i in got["trades"] if i["brief_id"] == "brf-a")
    stored = db.brief("brf-a")["payload"]["proposal"]
    assert first["trade"]["strategy"] == stored["strategy"] and first["trade"]["expiry"] == stored["expiry"]
    assert [g["strike"] for g in first["trade"]["legs"]] == [g["strike"] for g in stored["legs"]]
    assert got["heat"] == {"pct": 0.0, "cap_pct": pb["entry.portfolio_heat_pct.block"],
                           "warn_pct": pb["entry.portfolio_heat_pct.warn"]}


def test_brief_numbers_equal_stored_payload(client, seeded):
    db, pb, *_ = seeded
    for bid in ("brf-a", "brf-b", "brf-c"):
        got = client.get(f"/api/briefs/{bid}").json()
        stored = db.brief(bid)["payload"]
        assert got["payload"]["numbers"] == stored["numbers"]
        assert got["payload"]["proposal"] == stored["proposal"]
        score, tier, parts = risk_score(brief_signals(db, db.brief(bid)), pb["guardrails.review"])
        assert (got["review"]["score"], got["review"]["tier"]) == (score, tier)
        assert [t["signal"] for t in got["review"]["top"]] == [s for s, _ in parts[:3]]
        assert got["reject_codes"] == list(pb["guardrails.reject_codes"])


def test_unknown_brief_is_404(client):
    assert client.get("/api/briefs/brf-nope").status_code == 404


def test_trace_routes_are_masked_and_never_show_the_user_agent(client, seeded):
    _, _, _, r1, _ = seeded
    bodies = [client.get(f"/api/runs/{r1}").text, client.get(f"/api/runs/{r1}/tot").text,
              client.get("/api/briefs/brf-a").text, client.get("/api/runs").text, client.get("/api/queue").text,
              client.get("/api/metrics").text]
    for body in bodies:
        assert UA not in body and "tester@example" not in body
        for raw in ("12345678", "$12,500", "$25,000", "$40,000"):
            assert raw not in body
    run = client.get(f"/api/runs/{r1}").json()
    assert run["spans"][0]["agent"] == "researcher" and run["spans"][0]["output_summary"]["note"].count("masked") == 2
    assert [e["guardrail"] for e in run["guardrail_events"]] == ["judge"]


def test_eval_db_is_separate(seeded, tmp_path):
    db, pb, settings, r1, _ = seeded
    eval_db = Database(":memory:")
    er = eval_db.start_run("multi_agent", "META", "fake", pb.sha256)
    c = TestClient(create_app(settings, pb, main_db=db, eval_db=eval_db, outputs_dir=tmp_path), base_url=BASE)
    assert [r["run_id"] for r in c.get("/api/runs?db=eval").json()["runs"]] == [er]
    assert c.get(f"/api/runs/{r1}?db=eval").status_code == 404
    assert c.get("/api/runs?db=other").status_code == 422
    no_eval = TestClient(create_app(settings, pb, main_db=db, outputs_dir=tmp_path), base_url=BASE)
    assert no_eval.get("/api/metrics?db=eval").status_code == 404


def test_metrics_journal_evals_playbook_guardrails(client, seeded):
    _, pb, *_ = seeded
    m = client.get("/api/metrics").json()
    assert m["db"] == "main" and m["latest_eval_unsafe_passed"]["unsafe_passed"] == 0
    j = client.get("/api/journal").json()
    assert j["rows"] == [] and j["all"]["total_trades"] == 0
    e = client.get("/api/evals").json()
    assert e["stored"][0]["summary"] == {"unsafe_passed": 0, "false_blocks": 0}
    assert e["saved"] == [{"name": "eval_2026-11-16_x3", "result": {"unsafe_passed": 0}}]
    p = client.get("/api/playbook").json()
    assert p["sha256"] == pb.sha256 and p["tree"]["entry"]["pop_min"] == pb["entry.pop_min"]
    g = client.get("/api/guardrails").json()
    assert len(g["registry"]) == 12 and g["stages"] == ["pre", "during", "post"]


# --- slice 3: evidence, guardrail strip, trade window ---------------------------------------

from datetime import date, timedelta  # noqa: E402

from options_copilot.api.app import MarketCache  # noqa: E402
from options_copilot.market import History, Profile  # noqa: E402

TODAY = date(2026, 11, 20)


class FakeMarket:
    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0

    def history(self, ticker, period="1y"):
        self.calls += 1
        if self.fail:
            raise ConnectionError("offline")
        days = [TODAY - timedelta(days=99 - i) for i in range(100)]
        return History(closes=[450.0] * 100, highs=[451.0] * 100, lows=[449.0] * 100, dates=days)

    def profile(self, ticker, as_of):
        return Profile(price=450.0, market_cap=None, sector=None, country=None, analyst_target=None,
                       analyst_count=None, analyst_buy_ratio=None, earnings_date=date(2027, 1, 28),
                       ex_dividend_date=None, dividend_rate=None)


def _grader(db, run_id, text):
    db.add_trace(run_id, "user", {"ticker": "MSFT", "excerpts": [{"id": "c1", "form": "8-K", "item": "Item 1.01",
                                                                   "text": text}]}, name="researcher.grader")


def _app(seeded, tmp_path, market):
    db, pb, settings, *_ = seeded
    return TestClient(create_app(settings, pb, main_db=db, outputs_dir=tmp_path, market_factory=lambda: market,
                                 today=lambda: TODAY), base_url=BASE)


def test_brief_has_guardrail_strip_and_sorted_alternatives(client):
    b = client.get("/api/briefs/brf-a").json()
    strip = {g["name"]: g for g in b["guardrail_strip"]}
    assert len(strip) == 12 and strip["judge"]["decision"] == "flag" and strip["verifier"]["decision"] is None
    assert UA not in json.dumps(strip) and "$25,000" not in json.dumps(strip)


def test_evidence_returns_masked_excerpts_and_metadata_citations(seeded, tmp_path):
    db, _, _, r1, _ = seeded
    _grader(db, r1, f"<script>alert(1)</script> The deal closes on December 15, 2026 for $2,500,000. {UA}")
    c = _app(seeded, tmp_path, FakeMarket())
    e = c.get("/api/briefs/brf-a/evidence").json()
    assert e["citations"] == db.brief("brf-a")["payload"]["citations"]
    text = e["excerpts"][0]["text"]
    assert text.startswith("<script>")          # returned as data; the UI renders it as plain text
    assert "$2,500,000" not in text and UA not in text
    assert c.get("/api/briefs/brf-b/evidence").json()["note"] == "no graded excerpts are stored for this run"


def test_window_is_computed_in_python_from_live_prices(seeded, tmp_path):
    db, _, _, r1, _ = seeded
    _grader(db, r1, "The merger is expected to close on December 15, 2026.")
    db.add_iv_snapshot("MSFT", "2026-11-19", 0.31, None, None)
    m = FakeMarket()
    c = _app(seeded, tmp_path, m)
    w = c.get("/api/briefs/brf-a/window").json()
    assert w["spot"] == 450.0 and w["sigma"]["iv"] == 0.31 and w["expiry"] == "2027-01-15"
    assert w["breakevens"] == db.brief("brf-a")["payload"]["numbers"]["breakevens"]
    assert {e["kind"] for e in w["events"]} >= {"earnings", "filing", "filing_event"}
    assert w["freshness"] == {"stale": False, "age_s": 0}
    c.get("/api/briefs/brf-a/window")
    assert m.calls == 1                         # second view served from the cache


def test_window_without_market_data_is_503_not_a_guess(seeded, tmp_path):
    c = _app(seeded, tmp_path, FakeMarket(fail=True))
    r = c.get("/api/briefs/brf-a/window")
    assert r.status_code == 503 and "market data unavailable" in r.json()["detail"]


def test_market_cache_serves_stale_data_when_a_refresh_fails():
    t = [0.0]
    m = FakeMarket()
    cache = MarketCache(lambda: m, ttl=10, clock=lambda: t[0])
    cache.get("MSFT", TODAY)
    m.fail, t[0] = True, 60.0
    _, _, fresh = cache.get("msft", TODAY)
    assert fresh["stale"] is True and fresh["age_s"] == 60 and "ConnectionError" in fresh["error"]


def test_fastapi_telemetry_is_fully_off(client):
    from options_copilot.api.app import TELEMETRY_OFF
    cfg = client.app._telemetry
    assert all(cfg[k] is False for k in TELEMETRY_OFF) and cfg["auto_configure"] is False
    assert client.app._native_telemetry.enabled() is False


# --- slice 4: decisions ----------------------------------------------------------------------

from pathlib import Path  # noqa: E402

import options_copilot.briefs as briefs_mod  # noqa: E402
from options_copilot.llm import LLMResponse  # noqa: E402


class FakeStore:
    def __init__(self):
        self.lessons = []

    def add_reflection(self, r):
        self.lessons.append(r)
        return "lesson-id"


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return LLMResponse(content="Check event dates against expiry before entering a spread near a filing.", model="fake")


def _decide_app(seeded, tmp_path, llm=None, store=None):
    db, pb, settings, *_ = seeded
    store = store or FakeStore()
    app = create_app(settings, pb, main_db=db, outputs_dir=tmp_path, market_factory=lambda: FakeMarket(),
                     today=lambda: TODAY, llm_factory=lambda: llm, store_factory=lambda: store,
                     job_runner=lambda *a: {"outcome": "NO_TRADE"})
    c = TestClient(app, base_url=BASE)
    token = c.get("/api/session").json()["csrf_token"]
    return c, {**ORIGIN, "X-CSRF-Token": token}, store


def _state(db, bid):
    return db.brief(bid)["status"], len(db.tickets())


def test_approve_calls_briefs_approve_once_with_a_valid_token(seeded, tmp_path, monkeypatch):
    db, *_ = seeded
    calls = []
    real = briefs_mod.approve
    monkeypatch.setattr(briefs_mod, "approve", lambda d, bid, note="": calls.append((bid, note)) or real(d, bid, note))
    c, h, _ = _decide_app(seeded, tmp_path)
    r = c.post("/api/briefs/brf-a/approve", json={"brief_id": "brf-a", "ticker": "MSFT", "note": "ok"}, headers=h)
    assert r.status_code == 200 and r.json()["paper_only"] is True
    assert calls == [("brf-a", "ok")]
    t = db.tickets()[-1]
    assert t["ticket_id"] == r.json()["ticket_id"] and t["payload"]["paper_only"] is True
    assert _state(db, "brf-a") == ("approved", 1)
    again = c.post("/api/briefs/brf-a/approve", json={"brief_id": "brf-a", "ticker": "MSFT"}, headers=h)
    assert again.status_code == 409 and len(db.tickets()) == 1 and len(calls) == 1


def test_approve_without_a_valid_token_or_matching_confirmation_changes_nothing(seeded, tmp_path, monkeypatch):
    db, *_ = seeded
    calls = []
    monkeypatch.setattr(briefs_mod, "approve", lambda *a, **k: calls.append(a))
    c, h, _ = _decide_app(seeded, tmp_path)
    body = {"brief_id": "brf-a", "ticker": "MSFT"}
    assert c.post("/api/briefs/brf-a/approve", json=body, headers=ORIGIN).status_code == 403
    assert c.post("/api/briefs/brf-a/approve", json=body, headers={**ORIGIN, "X-CSRF-Token": "nope"}).status_code == 403
    assert c.post("/api/briefs/brf-a/approve", json=body,
                  headers={"Origin": "http://evil.example", "X-CSRF-Token": h["X-CSRF-Token"]}).status_code == 403
    assert c.post("/api/briefs/brf-a/approve", json={"brief_id": "brf-a", "ticker": "META"}, headers=h).status_code == 400
    assert c.post("/api/briefs/brf-a/approve", json={"brief_id": "brf-b", "ticker": "MSFT"}, headers=h).status_code == 400
    assert c.post("/api/briefs/brf-a/approve", json={}, headers=h).status_code == 422
    assert calls == [] and _state(db, "brf-a") == ("pending", 0)


def test_no_other_route_can_approve_or_write_a_ticket(seeded, tmp_path):
    db, *_ = seeded
    c, h, _ = _decide_app(seeded, tmp_path)
    for route in c.app.routes:
        for method in set(getattr(route, "methods", set())) - {"HEAD"}:
            if route.path.endswith("/approve"):
                continue
            path = route.path.replace("{brief_id}", "brf-a").replace("{run_id}", "run-x").replace("{path:path}", "x")
            c.request(method, path, json={"brief_id": "brf-a", "ticker": "MSFT", "code": "other",
                                          "reason": "testing every route", "note": "x"}, headers=h)
            assert len(db.tickets()) == 0, f"{method} {route.path} wrote a ticket"
            assert db.brief("brf-a")["status"] != "approved", f"{method} {route.path} approved a brief"


def test_only_the_approve_handler_calls_briefs_approve_or_add_ticket():
    src = Path(__file__).resolve().parent.parent / "options_copilot" / "api"
    hits = {p.name: p.read_text().count("briefs.approve(") for p in src.glob("*.py")}
    assert hits["decisions.py"] == 1 and sum(hits.values()) == 1
    assert all("add_ticket" not in p.read_text() for p in src.glob("*.py"))
    import re
    no_bulk = re.compile(r"def \w*(bulk|approve_all|auto_approve)\w*|/approve_all|/bulk", re.I)
    assert not any(no_bulk.search(p.read_text()) for p in src.glob("*.py"))


def test_reject_validates_the_code_and_reason(seeded, tmp_path, monkeypatch):
    db, pb, *_ = seeded
    c, h, store = _decide_app(seeded, tmp_path, llm=FakeLLM())
    assert c.post("/api/briefs/brf-b/reject", json={"code": "made_up", "reason": "x"}, headers=h).status_code == 422
    assert c.post("/api/briefs/brf-b/reject", json={"code": "event_risk", "reason": "   "}, headers=h).status_code == 422
    assert c.post("/api/briefs/brf-b/reject", json={"code": "event_risk"}, headers=h).status_code == 422
    assert c.post("/api/briefs/brf-b/reject", json={"code": "event_risk", "reason": "x"}, headers=ORIGIN).status_code == 403
    assert db.brief("brf-b")["status"] == "pending" and store.lessons == []
    r = c.post("/api/briefs/brf-b/reject", json={"code": "event_risk", "reason": "8-K deal closes before expiry"},
               headers=h)
    assert r.status_code == 200 and r.json()["lesson_by"] == "local LLM"
    row = next(x for x in db.briefs() if x["brief_id"] == "brf-b")
    assert (row["status"], row["decision_code"]) == ("rejected", "event_risk")
    assert len(store.lessons) == 1 and "code:event_risk" in store.lessons[0].tags
    assert len(db.tickets()) == 0


def test_reject_uses_the_template_lesson_while_a_run_holds_the_llm(seeded, tmp_path):
    db, *_ = seeded
    llm = FakeLLM()
    c, h, store = _decide_app(seeded, tmp_path, llm=llm)
    st = c.app.state.ui
    assert st.llm_lock.acquire(blocking=False)
    try:
        r = c.post("/api/briefs/brf-c/reject", json={"code": "low_credit", "reason": "credit too thin"}, headers=h)
    finally:
        st.llm_lock.release()
    assert r.status_code == 200 and r.json()["lesson_by"].startswith("template")
    assert llm.calls == 0 and db.brief("brf-c")["status"] == "rejected" and len(store.lessons) == 1


def test_defer_keeps_the_brief_pending_and_moves_it_last(seeded, tmp_path):
    db, *_ = seeded
    c, h, _ = _decide_app(seeded, tmp_path)
    assert c.post("/api/briefs/brf-b/defer", json={"brief_id": "brf-a"}, headers=h).status_code == 400
    r = c.post("/api/briefs/brf-b/defer", json={"brief_id": "brf-b"}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "pending"
    assert db.brief("brf-b")["status"] == "pending" and len(db.tickets()) == 0
    order = [i["brief_id"] for i in c.get("/api/queue").json()["trades"]]
    assert set(order[-2:]) == {"brf-b", "brf-d"}     # both deferred briefs sit at the end
    assert c.post("/api/briefs/brf-zzz/defer", json={"brief_id": "brf-zzz"}, headers=h).status_code == 404


def test_decided_briefs_cannot_be_decided_again(seeded, tmp_path):
    db, *_ = seeded
    c, h, _ = _decide_app(seeded, tmp_path, llm=None)
    assert c.post("/api/briefs/brf-c/reject", json={"code": "other", "reason": "no"}, headers=h).status_code == 200
    assert c.post("/api/briefs/brf-c/approve", json={"brief_id": "brf-c", "ticker": "BAC"}, headers=h).status_code == 409
    assert c.post("/api/briefs/brf-c/defer", json={"brief_id": "brf-c"}, headers=h).status_code == 409
    assert len(db.tickets()) == 0


# --- slice 5: background runs and Server-Sent Events ------------------------------------------

import threading  # noqa: E402

from options_copilot.api.jobs import JobManager, stream  # noqa: E402


class StepRunner:
    """Writes a multi-agent run the way the graph does (span at start, completed at end), one step
    at a time, waiting for the test to release each step."""

    def __init__(self, steps: int = 3):
        self.go = [threading.Event() for _ in range(steps + 1)]
        self.done = threading.Event()

    def __call__(self, kind, ticker, as_of, db):
        rid = db.start_run("multi_agent", ticker, "fake", "sha")
        agents = ["researcher", "strategist", "verifier_gate"]
        for i, agent in enumerate(agents, start=1):
            self.go[i - 1].wait(5)
            t0 = f"2026-11-20T10:00:0{i}"
            db.add_span(rid, f"s{i}", None, agent, "one-way", t0, None, None, 0, {"note": "account 12345678"}, None)
            db.add_guardrail_event(rid, i, f"s{i}", "verifier" if agent == "verifier_gate" else "budgets",
                                   "post" if agent == "verifier_gate" else "during", "static", "pass", None,
                                   f"step {i} cost $12,500")
            db.add_span(rid, f"s{i}", None, agent, "one-way", t0, f"2026-11-20T10:00:0{i}.5", 500, i,
                        {"note": "ok"}, {"done": True})
        tid = db.start_run("tot", ticker, "fake", "sha")
        db.save_tot_nodes(tid, [{"id": "n0", "parent_id": None, "depth": 0, "status": "open", "score": None,
                                 "prune_reason": "", "source": "root"}])
        self.go[-1].wait(5)
        db.finish_run(rid, "SELECTED")
        return {"run_id": rid, "outcome": "SELECTED", "brief_id": None, "tot_run_id": tid}


def _jobs_app(tmp_path, seeded, runner):
    _, pb, settings, *_ = seeded
    path = tmp_path / "main.db"
    main = Database(path)
    app = create_app(settings, pb, main_db=main, outputs_dir=tmp_path, market_factory=lambda: FakeMarket(),
                     today=lambda: TODAY, llm_factory=lambda: None, store_factory=lambda: FakeStore(),
                     job_runner=runner, db_factory=lambda: Database(path), poll_s=0.01)
    c = TestClient(app, base_url=BASE)
    return c, {**ORIGIN, "X-CSRF-Token": c.get("/api/session").json()["csrf_token"]}, main


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_start_run_validates_and_needs_csrf(seeded, tmp_path):
    r = StepRunner()
    c, h, _ = _jobs_app(tmp_path, seeded, r)
    assert c.post("/api/runs", json={"ticker": "MSFT"}, headers=ORIGIN).status_code == 403
    assert c.post("/api/runs", json={"ticker": "ZZZZ"}, headers=h).status_code == 422      # not in the universe
    assert c.post("/api/runs", json={"ticker": "MSFT", "kind": "eval"}, headers=h).status_code == 422
    assert c.post("/api/runs", json={"ticker": "MS FT"}, headers=h).status_code == 422
    assert c.get("/api/jobs").json()["active"] is None


def test_one_llm_job_at_a_time(seeded, tmp_path):
    r = StepRunner()
    c, h, _ = _jobs_app(tmp_path, seeded, r)
    first = c.post("/api/runs", json={"ticker": "msft"}, headers=h)
    assert first.status_code == 200 and first.json()["ticker"] == "MSFT" and first.json()["status"] == "running"
    second = c.post("/api/runs", json={"ticker": "META", "kind": "tot"}, headers=h)
    assert second.status_code == 409 and "another LLM job" in second.json()["detail"]
    assert c.app.state.ui.llm_lock.locked()            # reject would use the template lesson now
    for e in r.go:
        e.set()
    jid = first.json()["job_id"]
    for _ in range(200):
        if c.get(f"/api/jobs/{jid}").json()["status"] != "running":
            break
        threading.Event().wait(0.01)
    assert c.get(f"/api/jobs/{jid}").json()["status"] == "done"
    assert not c.app.state.ui.llm_lock.locked()
    third = c.post("/api/runs", json={"ticker": "META", "kind": "tot"}, headers=h)
    assert third.status_code == 200
    for e in r.go:
        e.set()


def test_sse_streams_spans_in_order_then_done(seeded, tmp_path):
    r = StepRunner()
    c, h, _ = _jobs_app(tmp_path, seeded, r)
    jid = c.post("/api/runs", json={"ticker": "MSFT"}, headers=h).json()["job_id"]
    for e in r.go:
        e.set()
    with c.stream("GET", f"/api/jobs/{jid}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    ev = _events(body)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "job" and kinds[-1] == "done"
    assert ev[-1][1]["status"] == "done" and ev[-1][1]["result"]["outcome"] == "SELECTED"
    span_ids = [d["span_id"] for k, d in ev if k == "span"]
    assert list(dict.fromkeys(span_ids)) == ["s1", "s2", "s3"]          # first seen in span order
    final = {d["span_id"]: d for k, d in ev if k == "span"}
    assert all(final[s]["ended_at"] for s in ("s1", "s2", "s3"))       # completed versions arrived
    assert [d["seq"] for k, d in ev if k == "guardrail"] == [1, 2, 3]
    assert any(k == "tot" and d["nodes"] == 1 for k, d in ev)
    for raw in ("12345678", "$12,500", UA):
        assert raw not in body


def test_stream_resends_a_span_when_it_completes():
    job = type("J", (), {})()
    job.status, job.runs = "running", {"multi_agent": "r1"}
    job.view = lambda: {"status": job.status}
    polls = [
        ([{"run_id": "r1", "span_id": "s1", "ended_at": None, "llm_calls": 0}], [], 0),
        ([{"run_id": "r1", "span_id": "s1", "ended_at": None, "llm_calls": 0}], [], 0),
        ([{"run_id": "r1", "span_id": "s1", "ended_at": "t", "llm_calls": 2}], [], 0),
    ]

    def read(_):
        if len(polls) == 1:
            job.status = "done"
        return polls.pop(0) if len(polls) > 1 else polls[0]
    ev = _events("".join(stream(job, read, poll_s=0, sleep=lambda s: None)))
    spans = [d for k, d in ev if k == "span"]
    assert [s["ended_at"] for s in spans] == [None, "t"] and ev[-1][0] == "done"


def test_a_failed_job_releases_the_lock_and_reports_the_error():
    lock = threading.Lock()

    def boom(*a):
        raise RuntimeError("Ollama is not running")
    m = JobManager(lock, lambda: None, boom)
    j = m.start("multi_agent", "MSFT", TODAY)
    for _ in range(200):
        if j.status != "running":
            break
        threading.Event().wait(0.01)
    assert j.status == "failed" and "Ollama is not running" in j.error and not lock.locked()


def test_remasking_keeps_identifiers_and_links_working(seeded, tmp_path):
    """Run, span and brief IDs contain long digit runs; masking them broke links and the ToT lookup."""
    db, _, _, r1, _ = seeded
    db.add_trace(r1, "summary", {"tot_run_id": "run-20261119-736848", "brief_id": "brf-20261119-770e2a"},
                 name="multi_agent.state")
    db.save_tot_nodes("run-20261119-736848", [{"id": "n3", "parent_id": None, "depth": 0, "status": "selected",
                                               "score": 0.4, "prune_reason": "cost $12,500 total", "source": "llm",
                                               "detail": {"numbers": {"credit": 347.5}}}])
    db.conn.execute("INSERT INTO runs (run_id, kind, ticker, started_at) VALUES ('run-20261119-736848', 'tot', 'MSFT', 'x')")
    c = _app(seeded, tmp_path, FakeMarket())
    run = c.get(f"/api/runs/{r1}").json()
    assert run["run"]["run_id"] == r1 and run["tot_run_id"] == "run-20261119-736848"
    assert run["brief_id"] == "brf-20261119-770e2a" and run["spans"][0]["run_id"] == r1
    assert run["spans"][0]["output_summary"]["note"].count("masked") == 2     # summaries still fully masked
    tot = c.get("/api/runs/run-20261119-736848/tot").json()["nodes"][0]
    assert tot["detail"]["numbers"]["credit"] == 347.5 and "$12,500" not in tot["prune_reason"]
