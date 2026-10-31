"""Phase 4: PaperBroker, journal, IV history, monitor decisions, simulated clock, e2e demo.
Synthetic data, no network, no LLM."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta

import pytest

from agent_fakes import AS_OF, EXP, TICKER, FakeSources
from options_copilot.agent_tools import ToolBox
from options_copilot.briefs import approve, make_brief
from options_copilot.broker import PaperBroker, open_positions
from options_copilot.calculators import bs_price, intrinsic_close_cost
from options_copilot.db import Database
from options_copilot.indicators import iv_rank_from_history
from options_copilot.journal import close_fields, metrics
from options_copilot.llm import LLMError, LLMResponse
from options_copilot.management import Action, Evaluation, Trigger
from options_copilot.mask import Masker
from options_copilot.models import (OptionLeg, OptionType, Position, Reflection, Side, Strategy, position_from_dict,
                                    to_jsonable)
from options_copilot.monitor import decide, run_monitor, snapshot
from options_copilot.react import AgentResult
from options_copilot.sim import SimulatedSources
from options_copilot.tot import alternatives, explore, summary_text

PV = 87_654.0


class Store:
    def __init__(self):
        self.added: list[Reflection] = []

    def add_reflection(self, r):
        self.added.append(r)


class OneShotLLM:
    name = "fake"

    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def chat(self, messages, *, json_schema=None, tools=None):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        return LLMResponse(json.dumps(self.reply), self.name, [], 1)


def entry_ticket(pb, db, sources=None):
    sources = sources or FakeSources()
    box = ToolBox(TICKER, sources, pb, AS_OF, PV)
    tot = explore(box, pb, None, db, Masker(PV))
    brief_id = make_brief(AgentResult(tot.run_id, "accepted", summary_text(tot)), box, db, AS_OF,
                          extra={"alternatives": alternatives(tot)})
    return brief_id, approve(db, brief_id)


@pytest.fixture
def opened(pb):
    db = Database(":memory:")
    sources = FakeSources()
    _, tid = entry_ticket(pb, db, sources)
    f = PaperBroker(db, sources, AS_OF).process(db.ticket(tid))
    assert f.status == "filled"
    return db, sources, f.position_id


# --- calculators and models ------------------------------------------------------------------

def test_bs_price_put_call_parity_and_expiry():
    c = bs_price(100, 105, 0.3, 0.25, OptionType.CALL)
    p = bs_price(100, 105, 0.3, 0.25, OptionType.PUT)
    assert c - p == pytest.approx(100 - 105, abs=1e-9)
    assert bs_price(100, 95, 0.3, 0, OptionType.PUT) == 0 and bs_price(100, 95, 0.3, 0, OptionType.CALL) == 5


def test_intrinsic_close_cost_for_a_breached_spread():
    legs = (OptionLeg(OptionType.PUT, Side.SHORT, 95, EXP, 1, 1), OptionLeg(OptionType.PUT, Side.LONG, 90, EXP, 1, 1))
    assert intrinsic_close_cost(legs, 92.0) == 300.0
    assert intrinsic_close_cost(legs, 80.0) == 500.0 and intrinsic_close_cost(legs, 99.0) == 0.0


# --- PaperBroker ------------------------------------------------------------------------------

def test_entry_ticket_fills_at_limit_and_opens_journal_row(pb, opened):
    db, _, pos_id = opened
    pos = position_from_dict(db.position(pos_id)["payload"])
    t = next(t for t in db.tickets() if t["payload"].get("position_id") == pos_id)
    assert t["status"] == "filled" and pos.entry_credit == round(t["payload"]["limit_credit_per_share"] * 100, 2)
    assert pos.synthetic and pos.entry_spot == 100.0 and pos.entry_iv == pytest.approx(0.30)
    row = db.journal()[0]
    assert row["position_id"] == pos_id and row["date_closed"] is None and row["synthetic"] == 1
    assert row["earnings_note"].startswith("No (") and row["dte"] == (EXP - AS_OF).days


def test_entry_ticket_stays_working_when_mid_is_below_limit(pb):
    db = Database(":memory:")
    _, tid = entry_ticket(pb, db)
    t = db.ticket(tid)
    db.update_ticket(tid, "approved", limit_credit_per_share=t["payload"]["limit_credit_per_share"] + 0.50)
    f = PaperBroker(db, FakeSources(), AS_OF).process(db.ticket(tid))
    assert f.status == "working" and "not filled at limit" in f.message
    assert db.positions() == [] and db.ticket(tid)["status"] == "approved"


def test_pre_phase4_ticket_without_brief_numbers_still_fills(pb):
    db = Database(":memory:")
    _, tid = entry_ticket(pb, db)
    t = db.ticket(tid)
    old = {k: v for k, v in t["payload"].items() if k not in ("kind", "credit_at_brief", "max_loss_at_brief")}
    db.update_ticket(tid, "approved")
    with db.conn:
        db.conn.execute("UPDATE tickets SET payload = ? WHERE ticket_id = ?", (json.dumps(old), tid))
    assert PaperBroker(db, FakeSources(), AS_OF).process(db.ticket(tid)).status == "filled"


def test_exit_closes_with_pnl_of_credit_and_writes_a_lesson(pb, opened):
    db, sources, pos_id = opened
    later = AS_OF + timedelta(days=20)
    sim = SimulatedSources(sources, AS_OF, later)
    run_id, items, _ = run_monitor(db, sim, pb, later, llm=None, masker=Masker(PV), portfolio_value=PV)
    pos = position_from_dict(db.position(pos_id)["payload"])
    close = items[0].close_cost
    assert close is not None and close < pos.entry_credit   # theta decay on a flat stock
    # an owner-approved exit brief, written by hand here to exercise the broker
    from options_copilot.db import new_id
    bid = new_id("brf")
    db.save_brief(bid, run_id, "exit", TICKER, "pending", "EXIT",
                  {"position_id": pos_id, "limit_debit_per_share": round(close / 100, 2), "exit_rule": "r1_profit_target"})
    tid = approve(db, bid)
    store = Store()
    broker = PaperBroker(db, sim, later, store=store, masker=Masker(PV))
    f = broker.process(db.ticket(tid))
    assert f.status == "filled" and db.position(pos_id)["status"] == "closed" and open_positions(db) == []
    row = db.journal()[0]
    expected = close_fields(pos, round(close / 100, 2) * 100, later, "r1_profit_target")
    assert row["pnl"] == expected["pnl"] and row["pnl_pct"] == expected["pnl_pct"]
    assert row["pnl_pct"] == pytest.approx((pos.entry_credit - row["close_cost"]) / pos.entry_credit * 100, abs=0.01)
    assert row["days_held"] == 20 and row["annualized_pct"] == pytest.approx(row["pnl_pct"] * 365 / 20, abs=0.2)
    assert store.added and store.added[0].trigger.value == "trade_close" and store.added[0].synthetic


def test_exit_stays_working_when_debit_is_above_limit(pb, opened):
    db, sources, pos_id = opened
    from options_copilot.db import new_id
    bid = new_id("brf")
    db.save_brief(bid, None, "exit", TICKER, "pending", "EXIT",
                  {"position_id": pos_id, "limit_debit_per_share": 0.01, "exit_rule": "r1_profit_target"})
    f = PaperBroker(db, sources, AS_OF).process(db.ticket(approve(db, bid)))
    assert f.status == "working" and db.position(pos_id)["status"] == "open"


def test_roll_is_all_or_nothing(pb, opened):
    db, sources, pos_id = opened
    entry = next(t for t in db.tickets() if t["payload"].get("position_id") == pos_id)["payload"]
    from options_copilot.db import new_id
    bid = new_id("brf")
    db.save_brief(bid, None, "roll", TICKER, "pending", "CLEAR", {
        "position_id": pos_id, "limit_debit_per_share": 50.0, "exit_rule": "r2_roll",
        "new": {"proposal": {"ticker": TICKER, "strategy": entry["strategy"], "contracts": 1, "legs": entry["legs"]},
                "numbers": {"credit": entry["credit_at_brief"] + 500, "max_loss": entry["max_loss_at_brief"]}}})
    f = PaperBroker(db, sources, AS_OF).process(db.ticket(approve(db, bid)))
    assert f.status == "working" and "roll not filled" in f.message
    assert db.position(pos_id)["status"] == "open"   # the close side did not run alone


def test_expired_positions_settle_at_intrinsic(pb, opened):
    db, sources, pos_id = opened
    after = EXP + timedelta(days=1)
    broker = PaperBroker(db, SimulatedSources(sources, AS_OF, after, move_pct=-20.0), after)
    settled = broker.settle_expired()
    assert len(settled) == 1 and settled[0]["exit_rule"] == "expiry"
    pos = position_from_dict(db.position(pos_id)["payload"])
    assert settled[0]["close_cost"] == intrinsic_close_cost(pos.legs, 80.0)


# --- journal ----------------------------------------------------------------------------------

def _pos(credit=100.0, max_loss=200.0):
    legs = (OptionLeg(OptionType.PUT, Side.SHORT, 95, EXP, 1, 1.1), OptionLeg(OptionType.PUT, Side.LONG, 92, EXP, 0.5, 0.6))
    return Position("pos-x", TICKER, Strategy.PUT_CREDIT_SPREAD, legs, AS_OF, credit, max_loss)


def test_close_fields_match_the_playbook_example_basis():
    f = close_fields(_pos(220.0, 2280.0), 110.0, AS_OF + timedelta(days=14), "r1_profit_target")
    assert f["pnl"] == 110.0 and f["pnl_pct"] == 50.0 and f["days_held"] == 14
    assert f["annualized_pct"] == round(50 * 365 / 14, 1) and f["return_on_risk_pct"] == round(110 / 2280 * 100, 2)


def test_metrics_summary_rows():
    rows = [{"date_opened": "2026-10-01", "date_closed": "2026-10-15", "pnl": 50.0, "pnl_pct": 50.0, "days_held": 14},
            {"date_opened": "2026-10-03", "date_closed": "2026-10-13", "pnl": -100.0, "pnl_pct": -100.0, "days_held": 10},
            {"date_opened": "2026-10-20", "date_closed": None, "pnl": None, "pnl_pct": None, "days_held": None},
            {"date_opened": "2026-11-02", "date_closed": None, "pnl": None, "pnl_pct": None, "days_held": None}]
    m = metrics(rows, "2026-10")
    assert (m["total_trades"], m["closed_trades"], m["active_trades"]) == (3, 2, 1)
    assert m["total_pnl"] == -50.0 and m["win_rate_pct"] == 50.0 and m["avg_days_held"] == 12.0
    assert m["avg_return_pct"] == -25.0 and m["annualized_return_pct"] == round(-25 * 365 / 12, 1)
    assert metrics([], None)["win_rate_pct"] is None


# --- IV history ---------------------------------------------------------------------------------

def test_iv_rank_modes(pb):
    assert iv_rank_from_history(0.30, [], 0.25, pb)[1].startswith("proxy")
    assert iv_rank_from_history(0.30, [], None, pb) is None
    hist = [0.20 + i * 0.01 for i in range(20)]           # 0.20 .. 0.39
    v, label = iv_rank_from_history(0.30, hist, 0.25, pb)
    assert label.startswith("IV percentile") and v == 50.0
    year = [0.20] * 251 + [0.40]
    v, label = iv_rank_from_history(0.30, year, 0.25, pb)
    assert label.startswith("IV rank") and v == pytest.approx(50.0)


def test_toolbox_uses_iv_history_once_available(pb):
    hist = tuple(0.20 + i * 0.005 for i in range(30))
    box = ToolBox(TICKER, FakeSources(), pb, AS_OF, PV, iv_history=hist)
    box._load_chain(EXP)
    assert box._iv_rank()["iv_rank_label"].startswith("IV percentile over 30")


def test_r6_needs_iv_history_and_evaluates_with_it(pb, opened):
    db, sources, pos_id = opened
    pos = position_from_dict(db.position(pos_id)["payload"])
    snap, _ = snapshot(pos, sources, AS_OF, db, pb)
    assert snap.iv_rank_now is None and snap.iv_rank_window_ago is None
    for i in range(30):   # IV climbed for weeks, then fell sharply inside the 5-day window
        db.add_iv_snapshot(TICKER, (AS_OF - timedelta(days=30 - i)).isoformat(), 0.40 + i * 0.01 if i < 27 else 0.25,
                           None, None)
    snap, info = snapshot(pos, sources, AS_OF, db, pb)
    assert snap.iv_rank_now is not None and snap.iv_rank_window_ago is not None
    assert snap.iv_rank_window_ago - snap.iv_rank_now >= 20 and info["iv_rank_label"].startswith("IV percentile")


# --- monitor decisions -------------------------------------------------------------------------

def ev(*triggers, pp=55.0):
    return Evaluation("pos-x", pp, tuple(triggers), ())


def test_mandatory_rule_closes_without_asking_the_llm():
    llm = OneShotLLM({"action": "hold_or_roll_out", "rationale": "ride it"})
    d = decide(ev(Trigger("r3_stop_loss", Action.CLOSE, "loss 1.2x credit"),
                  Trigger("r6_iv_crush", Action.HOLD_OR_ROLL_OUT, "IV fell")), llm)
    assert d.action is Action.CLOSE and d.chosen_by == "rules" and llm.calls == 0


def test_partial_action_is_escalated_to_full_close():
    d = decide(ev(Trigger("r4_big_move", Action.BUY_BACK_CALLS, "up 2 sigma")))
    assert d.action is Action.CLOSE and "escalated to a full close" in d.notes[0]


def test_conflict_llm_chooses_only_among_triggered_actions():
    conflict = (Trigger("r1_profit_target", Action.CLOSE, "55%"), Trigger("r2_roll", Action.ROLL, "20 DTE"))
    d = decide(ev(*conflict), OneShotLLM({"action": "roll", "rationale": "keep the income going"}), Masker(PV))
    assert d.action is Action.ROLL and d.chosen_by == "llm" and set(d.options) == {"close", "roll"}
    d = decide(ev(*conflict), OneShotLLM({"action": "hold_or_roll", "rationale": "hold"}))
    assert d.action is Action.CLOSE and d.chosen_by == "fallback"
    d = decide(ev(*conflict), OneShotLLM(fail=True))
    assert d.action is Action.CLOSE and d.chosen_by == "fallback"
    assert decide(ev(*conflict)).action is Action.CLOSE


def test_no_trigger_holds_and_single_trigger_needs_no_llm():
    llm = OneShotLLM({"action": "close", "rationale": "x"})
    assert decide(ev(), llm).action is None
    assert decide(ev(Trigger("r1_profit_target", Action.CLOSE, "55%")), llm).chosen_by == "rules" and llm.calls == 0


def test_monitor_writes_one_exit_brief_and_never_approves(pb, opened):
    db, sources, pos_id = opened
    day = next(d for d in range(1, 45) if run_monitor(db, SimulatedSources(sources, AS_OF, AS_OF + timedelta(days=d)),
                                                         pb, AS_OF + timedelta(days=d), llm=None, masker=Masker(PV),
                                                         portfolio_value=PV)[1][0].brief_id)
    pending = db.briefs("pending")
    assert len(pending) == 1 and pending[0]["kind"] in ("exit", "roll")
    b = db.brief(pending[0]["brief_id"])
    assert b["payload"]["position_id"] == pos_id and b["payload"]["simulated"] is None
    assert db.position(pos_id)["status"] == "open"          # nothing closes without the owner
    # a second pass does not write a duplicate brief
    later = AS_OF + timedelta(days=day + 1)
    _, items, _ = run_monitor(db, SimulatedSources(sources, AS_OF, later), pb, later, llm=None,
                              masker=Masker(PV), portfolio_value=PV)
    assert items[0].brief_id is None and "already pending" in items[0].note
    assert len(db.briefs("pending")) == 1


def test_monitor_lists_unevaluated_rules(pb, opened):
    db, sources, _ = opened
    _, items, summary = run_monitor(db, sources, pb, AS_OF, llm=None, masker=Masker(PV), portfolio_value=PV)
    assert "r6_iv_crush" in items[0].evaluation.not_evaluated
    assert summary["open_positions"] == 1 and summary["heat_pct"] > 0


# --- simulated clock -------------------------------------------------------------------------

def test_simulated_day_zero_equals_live_and_time_decays(pb):
    live = FakeSources()
    q0 = {(q.option_type, q.strike): q for q in live.chain(TICKER, EXP)[0]}
    same = SimulatedSources(live, AS_OF, AS_OF).chain(TICKER, EXP)[0]
    assert all(abs((q.bid + q.ask) / 2 - (q0[(q.option_type, q.strike)].bid + q0[(q.option_type, q.strike)].ask) / 2) < 0.011
               for q in same)
    later = {(q.option_type, q.strike): q for q in SimulatedSources(live, AS_OF, AS_OF + timedelta(days=20)).chain(TICKER, EXP)[0]}
    otm_put = (OptionType.PUT, 86)
    assert later[otm_put].ask < q0[otm_put].ask
    down = SimulatedSources(live, AS_OF, AS_OF + timedelta(days=1), move_pct=-5)
    assert down.ticker_data(TICKER).profile.price == 95.0
    assert {(q.option_type, q.strike): q for q in down.chain(TICKER, EXP)[0]}[otm_put].bid > q0[otm_put].bid
    with pytest.raises(ValueError):
        SimulatedSources(live, AS_OF, AS_OF - timedelta(days=1))


# --- end-to-end demo ----------------------------------------------------------------------------

def _demo(pb, approve_fn):
    from options_copilot.cli import _render_brief
    from options_copilot.demo import run_demo
    store = Store()
    res = run_demo(TICKER, sources=FakeSources(), pb=pb, as_of=AS_OF, llm=None, masker=Masker(PV), store=store,
                   portfolio_value=PV, approve_fn=approve_fn, render_brief=_render_brief, say=lambda s: None)
    return res, store


def test_demo_runs_end_to_end_with_owner_approval(pb):
    asked = []
    res, store = _demo(pb, lambda bid, text: asked.append(bid) or True)
    assert res.stopped_at == "done", res.text
    assert len(asked) == 2                     # entry and exit, each asked of the owner
    names = [s["step"] for s in res.steps]
    assert any("Reflexion" in n for n in names) and names[-1] == "journal and metrics"
    assert res.steps[-1]["metrics"]["closed_trades"] == 1 and store.added
    assert "SIMULATED" in res.text
    assert "copilot approve" not in res.text   # demo briefs live in its in-memory database


def test_demo_stops_without_owner_approval(pb):
    res, store = _demo(pb, lambda bid, text: False)
    assert res.stopped_at == "entry approval" and not store.added


def test_position_round_trips_through_json():
    p = _pos()
    assert position_from_dict(json.loads(json.dumps(to_jsonable(p)))) == p
    assert replace(p, contracts=2) != p


def test_limit_prices_round_toward_a_fill():
    from options_copilot.calculators import limit_credit_per_share, limit_debit_per_share
    assert limit_credit_per_share(232.5) == 2.32 and limit_credit_per_share(233.0) == 2.33
    assert limit_debit_per_share(61.2) == 0.62 and limit_debit_per_share(62.0) == 0.62
