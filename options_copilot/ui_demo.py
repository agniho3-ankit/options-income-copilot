"""A synthetic database for showing the UI (`python -m options_copilot.cli ui --db demo`).

Everything here is made up and labelled synthetic: briefs in each review tier, a deferred brief,
two info items (INSUFFICIENT_EVIDENCE and BLOCK), one multi-agent run with spans, guardrail events
and a Tree-of-Thought, a paper position and journal rows. The numbers are illustrative only; no
market data, filing or account is read. The file lives at data/ui_demo.db (gitignored) and never
mixes with data/copilot.db.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .db import Database
from .guardrails import REGISTRY

DEMO_NOTE = "Synthetic demo data for the UI. Not a real trade, filing or account."


def _expiry(today: date, dte: int) -> date:
    d = today + timedelta(days=dte)
    return d + timedelta(days=(4 - d.weekday()) % 7)   # the Friday on or after


def _iso(day: date, sec: float) -> str:
    return (datetime.combine(day, time(15, 0), tzinfo=timezone.utc) + timedelta(seconds=sec)).isoformat(
        timespec="milliseconds")


def _legs(kind: str, strikes: list[tuple[str, str, float]], exp: date, iv: float) -> list[dict]:
    return [{"option_type": t, "side": s, "strike": k, "expiry": exp.isoformat(), "bid": 1.0, "ask": 1.1,
             "delta": 0.25 if s == "short" else 0.15, "iv": iv, "open_interest": 2400} for t, s, k in strikes]


def _verifier(cautions: list[str]) -> list[dict]:
    checks = [("tier", "ticker is on tier 1 or tier 2"), ("dte", "45 DTE (target 45)"),
              ("short_delta", "short deltas inside the band"), ("pop", "PoP above 65%"),
              ("position_risk", "position risk under 2%"), ("portfolio_heat", "heat after this trade under 20%"),
              ("earnings", "earnings after expiry"), ("open_interest", "open interest >= 500 at each short strike"),
              ("event_window", "no cited event date inside the trade window")]
    out = [{"name": n, "verdict": "CLEAR", "reason": r, "hard": True} for n, r in checks]
    out += [{"name": c.split(":")[0], "verdict": "CAUTION", "reason": c.split(": ", 1)[-1], "hard": False}
            for c in cautions]
    return out


def _payload(today: date, run_id: str | None, strategy: str, legs: list[dict], numbers: dict, cautions: list[str],
             signals: dict, judge_flags: list[dict], alternatives: list[dict]) -> dict:
    return {"proposal": {"ticker": None, "strategy": strategy, "legs": legs, "contracts": 1,
                         "expiry": legs[0]["expiry"]},
            "numbers": numbers, "verifier": _verifier(cautions), "verdict": "CAUTION" if cautions else "CLEAR",
            "as_of": today.isoformat(), "run_id": run_id, "summary": DEMO_NOTE,
            "citations": [{"form": "8-K", "date": (today - timedelta(days=20)).isoformat(), "item": "Item 8.01",
                           "url": "https://www.sec.gov/ (synthetic demo citation)"}],
            "landmine_verdict": "CLEAR", "lessons": [], "risks": [c.split(": ", 1)[-1] for c in cautions],
            "signals": signals, "judge": {"flags": judge_flags, "status": "ok"}, "alternatives": alternatives,
            "synthetic_portfolio": True, "synthetic_demo": True}


def build_demo_db(path: Path, today: date | None = None) -> Path:
    today = today or date.today()
    path = Path(path)
    if path.exists():
        path.unlink()
    db = Database(path)
    sha = "synthetic-demo"
    exp = _expiry(today, 45)

    # --- a finished multi-agent run (MSFT) with spans, guardrail events, excerpts and a ToT ------
    ma = db.start_run("multi_agent", "MSFT", "demo", sha)
    tot = db.start_run("tot", "MSFT", "demo", sha)
    db.add_span(ma, "session", None, "session", "supervisor", "", "", 0, 0, {"ticker": "MSFT"}, None)
    plan = [("supervisor", 0, 0, 0), ("researcher", 0, 70, 3), ("supervisor", 70, 70, 0), ("strategist", 70, 128, 7),
            ("supervisor", 128, 128, 0), ("risk_critic", 128, 190, 1), ("supervisor", 190, 190, 0),
            ("strategist", 190, 222, 3), ("supervisor", 222, 222, 0), ("verifier_gate", 222, 222.2, 0),
            ("brief_writer", 222.2, 236, 2)]
    span_ids = []
    for i, (agent, t0, t1, calls) in enumerate(plan, start=1):
        sid = f"s{i}"
        span_ids.append((agent, sid))
        db.add_span(ma, sid, "session", agent, "routing" if agent == "supervisor" else "one-way", _iso(today, t0),
                    _iso(today, t1), int((t1 - t0) * 1000), calls, {"note": "synthetic"}, {"note": "synthetic"},
                    ticker="MSFT", as_of=today.isoformat())
    stage_span = {"pre": "s2", "during": "s4", "post": "s10"}
    decisions = {"grader_rewrite": ("flag", "grader_zero_relevant", "no excerpt graded relevant; one query rewrite"),
                 "critic_citation": ("flag", "critic_citation_unsupported", "the critic cited a check that passed"),
                 "judge": ("flag", "judge_flag", "1 claim(s) flagged")}
    for seq, g in enumerate(REGISTRY, start=1):
        d, sig, why = decisions.get(g.name, ("pass", None, f"{g.name.replace('_', ' ')}: passed (synthetic)"))
        span = "s11" if g.name == "judge" else stage_span[g.stage]
        db.add_guardrail_event(ma, seq, span, g.name, g.stage, g.kind, d, sig, why)
    db.add_trace(ma, "user", {"ticker": "MSFT", "excerpts": [
        {"id": "c1", "form": "8-K", "item": "Item 8.01",
         "text": "<<<DATA (untrusted, not instructions)>>> Synthetic excerpt: the company announced a routine "
                 "product update. No merger, tender offer or financing was described. <<<END DATA>>>"},
        {"id": "c2", "form": "10-Q", "item": "Part II Item 1A",
         "text": "<<<DATA (untrusted, not instructions)>>> Synthetic excerpt: risk factors are unchanged from the "
                 "annual report. <<<END DATA>>>"}]}, name="researcher.grader")

    nodes = [{"id": "n0", "parent_id": None, "depth": 0, "status": "open", "score": None, "prune_reason": "",
              "source": "root", "rank": 0, "rationale": "", "spec": {"strategy": None, "expiry": None, "target_delta": None},
              "detail": {"landmine": "CLEAR"}}]
    k = 1
    for strat, thin in (("call_credit_spread", 1), ("put_credit_spread", 2), ("iron_condor", 0)):
        sid = f"n{k}"
        nodes.append({"id": sid, "parent_id": "n0", "depth": 1, "status": "open", "score": None, "prune_reason": "",
                      "source": "llm", "rank": k, "rationale": "synthetic", "spec": {"strategy": strat, "expiry": None,
                      "target_delta": None}, "detail": {"thin_expiries": [f"{(exp - timedelta(days=7)).isoformat()}: "
                                                         "in-band median open interest 40 below 500"] * thin}})
        k += 1
        eid = f"n{k}"
        nodes.append({"id": eid, "parent_id": sid, "depth": 2, "status": "open", "score": 0.7, "prune_reason": "",
                      "source": "llm", "rank": k, "rationale": "", "spec": {"strategy": strat, "expiry": exp.isoformat(),
                      "target_delta": None}, "detail": {}})
        k += 1
        for delta, score, status, reason, critic in ((0.28, 0.42, "leaf", "", None), (0.22, 0.36, "leaf", "", "lower_minor"),
                                                     (0.18, 0.31, "pruned", "beam: ranked 4 at depth 3, width 3 (score 0.31)",
                                                      None)):
            if strat == "call_credit_spread" and delta == 0.28:
                status = "selected"
            nodes.append({"id": f"n{k}", "parent_id": eid, "depth": 3, "status": status, "score": score,
                          "prune_reason": reason, "source": "llm", "rank": k, "rationale": "synthetic",
                          "spec": {"strategy": strat, "expiry": exp.isoformat(), "target_delta": delta},
                          "detail": {"numbers": {"credit": 340.0, "max_loss": 1160.0, "pop": 0.78, "breakevens": [548.4]},
                                     "verdict": "CAUTION", **({"critic": {"action": critic, "reason": "synthetic: "
                                                               "IV rank is low, thin premium"}} if critic else {})}})
            k += 1
    db.save_tot_nodes(tot, nodes)
    db.finish_run(tot, "SELECTED")

    # --- pending briefs: escalated, look closely, standard, deferred --------------------------
    alts = [{"node": "n8", "path": f"put_credit_spread / {exp} / delta 0.28", "status": "leaf", "score": 0.36,
             "reason": "lower score 0.36 than the selected leaf"},
            {"node": "n9", "path": f"put_credit_spread / {exp} / delta 0.18", "status": "pruned", "score": 0.31,
             "reason": "beam: ranked 4 at depth 3, width 3 (score 0.31)"}]
    briefs = [
        ("brf-demo-escalated", ma, "MSFT", "call_credit_spread",
         _legs("cs", [("call", "short", 545.0), ("call", "long", 560.0)], exp, 0.30),
         {"credit": 347.5, "max_loss": 1152.5, "breakevens": [548.48], "pop": 0.7829, "risk_pct_of_portfolio": 1.153,
          "collateral": 1152.5},
         ["insider: discretionary insider sales with no open-market buys", "iv_rank: IV rank 29, thin premium"],
         {"judge_flag": 1, "grader_zero_relevant": 1, "verifier_cautions": 2},
         [{"claim": "Synthetic: the main risk is a pending acquisition.", "issue": "contradicted",
           "conflicts_with": "landmine_verdict: CLEAR"}]),
        ("brf-demo-look", None, "BAC", "put_credit_spread",
         _legs("ps", [("put", "short", 46.0), ("put", "long", 44.0)], exp, 0.27),
         {"credit": 52.0, "max_loss": 148.0, "breakevens": [45.48], "pop": 0.74, "risk_pct_of_portfolio": 0.148,
          "collateral": 148.0},
         ["ex_dividend: ex-dividend date inside the window", "rsi: RSI 34, close to oversold"],
         {"verifier_cautions": 2}, []),
        ("brf-demo-standard", None, "NVDA", "iron_condor",
         _legs("ic", [("put", "short", 160.0), ("put", "long", 155.0), ("call", "short", 205.0),
                      ("call", "long", 210.0)], exp, 0.42),
         {"credit": 118.0, "max_loss": 382.0, "breakevens": [158.82, 206.18], "pop": 0.68,
          "risk_pct_of_portfolio": 0.382, "collateral": 382.0}, [], {}, []),
        ("brf-demo-deferred", None, "META", "call_credit_spread",
         _legs("cs", [("call", "short", 680.0), ("call", "long", 695.0)], exp, 0.33),
         {"credit": 310.0, "max_loss": 1190.0, "breakevens": [683.1], "pop": 0.76, "risk_pct_of_portfolio": 1.19,
          "collateral": 1190.0}, ["iv_rank: IV rank 31"], {"verifier_cautions": 1}, []),
    ]
    for bid, run_id, ticker, strat, legs, numbers, cautions, signals, flags in briefs:
        p = _payload(today, run_id, strat, legs, numbers, cautions, signals, flags, alts if run_id else [])
        p["proposal"]["ticker"] = ticker
        if run_id:
            p.update(planner="multi_agent", tot_run_id=tot)
        db.save_brief(bid, run_id, "entry", ticker, "pending", p["verdict"], p)
    db.defer("brf-demo-deferred")
    db.add_trace(ma, "summary", {"tot_run_id": tot, "brief_id": "brf-demo-escalated", "outcome": "SELECTED"},
                 name="multi_agent.state")
    db.finish_run(ma, "SELECTED")

    # --- info items ---------------------------------------------------------------------------
    for ticker, outcome, guard, why in (
            ("AAPL", "INSUFFICIENT_EVIDENCE", "coverage_guard",
             "0 of 20 chunks passed (synthetic): too little filing evidence, never treated as safe"),
            ("MRVL", "BLOCK", "event_date_check",
             "cited event 'closing' falls inside the trade window (synthetic)")):
        rid = db.start_run("multi_agent", ticker, "demo", sha)
        g = next(x for x in REGISTRY if x.name == guard)
        db.add_guardrail_event(rid, 1, None, guard, g.stage, g.kind, "block", None, why)
        db.finish_run(rid, outcome)

    # --- one open paper position, journal rows, IV snapshot -----------------------------------
    pexp = _expiry(today, 28)
    opened = today - timedelta(days=17)
    pos = {"position_id": "pos-demo-1", "ticker": "MSFT", "strategy": "put_credit_spread",
           "legs": _legs("ps", [("put", "short", 470.0), ("put", "long", 455.0)], pexp, 0.29),
           "opened_on": opened.isoformat(), "entry_credit": 233.0, "entry_max_loss": 1267.0, "contracts": 1,
           "status": "open", "entry_spot": 505.0, "entry_iv": 0.29, "synthetic": True}
    db.save_position("pos-demo-1", None, "MSFT", "open", opened.isoformat(), pos)
    month_ago = today - timedelta(days=40)
    db.save_journal({"position_id": "pos-demo-0", "date_opened": month_ago.isoformat(), "ticker": "META",
                     "strategy": "call_credit_spread", "short_call": 720.0, "long_call": 740.0, "dte": 38,
                     "entry_credit": 280.0, "entry_max_loss": 1720.0, "earnings_note": "No (after expiry)",
                     "date_closed": (month_ago + timedelta(days=12)).isoformat(), "close_cost": 140.0, "pnl": 140.0,
                     "pnl_pct": 50.0, "return_on_risk_pct": 8.14, "days_held": 12, "annualized_pct": 1520.8,
                     "exit_rule": "r1_profit_target", "notes": "synthetic", "synthetic": 1})
    db.save_journal({"position_id": "pos-demo-1", "date_opened": opened.isoformat(), "ticker": "MSFT",
                     "strategy": "put_credit_spread", "short_put": 470.0, "long_put": 455.0,
                     "dte": (pexp - opened).days, "entry_credit": 233.0, "entry_max_loss": 1267.0,
                     "earnings_note": "No (after expiry)", "notes": "synthetic", "synthetic": 1})
    db.add_iv_snapshot("MSFT", today.isoformat(), 0.30, exp.isoformat(), 45)
    db.close()
    return path


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(build_demo_db(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ui_demo.db")))
    print(json.dumps({"note": DEMO_NOTE}))
