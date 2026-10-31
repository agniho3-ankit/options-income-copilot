"""End-to-end demo: screen -> brief -> approve -> paper position -> monitor -> exit brief ->
approve -> close -> reflection -> journal metrics.

Live public data up to the fill; after that a simulated clock (sim.py) moves forward one day
at a time until a management rule writes an exit or roll brief. Approvals are the human's:
`approve_fn` asks the owner and the demo stops on anything but yes. The demo keeps its own
in-memory database, so it never touches the real paper journal. Lessons it writes go to the
reflections collection as synthetic (trigger trade_close).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from .agent_tools import ToolBox
from .briefs import approve, make_brief
from .broker import PaperBroker
from .db import Database
from .journal import metrics, render as render_journal
from .monitor import render as render_monitor, run_monitor
from .react import AgentResult
from .sim import SimulatedSources
from .tot import alternatives, explore, render_tree, summary_text


@dataclass
class DemoResult:
    stopped_at: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""


def run_demo(ticker: str, *, sources: Any, pb, as_of: date, llm, masker, store, portfolio_value: float,
             approve_fn: Callable[[str, str], bool], render_brief: Callable[[dict], str],
             screen_fn: Callable[[str], Any] | None = None, move_pct: float = 0.0,
             db: Database | None = None, say: Callable[[str], None] = print) -> DemoResult:
    db = db or Database(":memory:")
    res = DemoResult("start")
    out: list[str] = []

    def step(name: str, text: str, **data) -> None:
        block = f"\n### {len(res.steps) + 1}. {name}\n{text}"
        out.append(block)
        say(block)
        res.steps.append({"step": name, **data})

    def stop(where: str) -> DemoResult:
        res.stopped_at, res.text = where, "\n".join(out)
        return res

    ticker = ticker.upper()
    if screen_fn is not None:
        sr, _ = screen_fn(ticker)
        score = "-" if sr.quality_score is None else f"{sr.quality_score:.2f}"
        step("screen (live)", f"{ticker}: quality score {score}, passed {sr.passed_screen}"
             + (f"; {'; '.join(sr.reasons)}" if sr.reasons else ""), passed=sr.passed_screen)
        if not sr.passed_screen:
            return stop("screen")

    box = ToolBox(ticker, sources, pb, as_of, portfolio_value)
    t0 = time.monotonic()
    tot = explore(box, pb, llm, db, masker)
    if tot.selected is None:
        step("entry brief (live, Tree-of-Thought)", render_tree(tot.tree, tot.stats) + "\n" + summary_text(tot)
             + "\nNo trade today, so the demo stops here (a trade is never forced).", outcome=tot.outcome)
        return stop("entry")
    brief_id = make_brief(AgentResult(tot.run_id, "accepted", summary_text(tot)), box, db, as_of,
                          extra={"planner": "tot", "alternatives": alternatives(tot)})
    entry_text = render_brief(db.brief(brief_id))
    step("entry brief (live, Tree-of-Thought)", entry_text, brief_id=brief_id,
         seconds=round(time.monotonic() - t0, 1))

    if not approve_fn(brief_id, entry_text):
        step("owner approval", f"{brief_id} not approved; demo stops.")
        return stop("entry approval")
    ticket_id = approve(db, brief_id, note="approved by the owner in the e2e demo")
    broker = PaperBroker(db, sources, as_of, llm=llm, store=store, masker=masker)
    fill = broker.process(db.ticket(ticket_id))
    step("owner approval -> paper ticket -> PaperBroker fill (live mid)",
         f"approved {brief_id}; ticket {ticket_id}: {fill.status.upper()} {fill.message}", ticket_id=ticket_id,
         fill=fill.status)
    if fill.status != "filled":
        return stop("fill")
    pos_id = fill.position_id
    expiry = date.fromisoformat(db.position(pos_id)["payload"]["legs"][0]["expiry"])

    day, exit_brief, sim = 0, None, None
    quiet = []
    while as_of + timedelta(days=day + 1) <= expiry:
        day += 1
        sim = SimulatedSources(sources, as_of, as_of + timedelta(days=day), move_pct)
        run_id, items, summary = run_monitor(db, sim, pb, sim.as_of, llm=llm, masker=masker,
                                             portfolio_value=portfolio_value, store=store, tot_llm=llm,
                                             sim_label=sim.label)
        mine = [i for i in items if i.position_id == pos_id]
        if mine and mine[0].brief_id:
            exit_brief = mine[0].brief_id
            step(f"monitor (simulated day +{day})",
                 (f"days +1..+{day - 1}: no rule fired ({', '.join(quiet)})\n" if quiet else "")
                 + render_monitor(items, summary), run_id=run_id, day=day)
            break
        quiet.append(f"+{day}: {mine[0].evaluation.profit_pct}%" if mine else f"+{day}: -")
    if exit_brief is None:
        step("monitor (simulated)", f"no rule fired through expiry; the position settles at expiry.")
        return stop("monitor")

    exit_text = render_brief(db.brief(exit_brief))
    step("exit/roll brief", exit_text, brief_id=exit_brief)
    if not approve_fn(exit_brief, exit_text):
        step("owner approval", f"{exit_brief} not approved; the position stays open.")
        return stop("exit approval")
    t2 = approve(db, exit_brief, note="approved by the owner in the e2e demo")
    broker = PaperBroker(db, sim, sim.as_of, llm=llm, store=store, masker=masker)
    fill = broker.process(db.ticket(t2))
    step("owner approval -> close ticket -> PaperBroker fill (simulated mid)",
         f"approved {exit_brief}; ticket {t2}: {fill.status.upper()} {fill.message}", fill=fill.status)
    if fill.status != "filled":
        return stop("close")
    lesson = broker.last_lesson
    step("Reflexion on close", lesson.lesson if lesson else "(no lesson written)")
    rows = db.journal()
    m = metrics(rows)
    step("journal and metrics", render_journal(rows, m), metrics=m)
    return stop("done")
