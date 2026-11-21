"""The synthetic demo database fills every screen (UI slice 8). No network."""
from __future__ import annotations

from datetime import date

from options_copilot.config import load_settings
from options_copilot.db import Database
from options_copilot.journal import metrics
from options_copilot.playbook import load_playbook
from options_copilot.review import queue
from options_copilot.ui_demo import build_demo_db


def test_demo_db_has_every_tier_info_items_a_run_and_a_position(tmp_path):
    path = build_demo_db(tmp_path / "ui_demo.db", today=date(2026, 11, 20))
    db = Database(path)
    pb = load_playbook(load_settings(env={}).playbook_path)
    q = queue(db, pb)
    tiers = [(i["brief_id"], i["tier"], i["deferred"]) for i in q["trades"]]
    assert tiers == [("brf-demo-escalated", "escalated", False), ("brf-demo-look", "look_closely", False),
                     ("brf-demo-standard", "standard", False), ("brf-demo-deferred", "standard", True)]
    assert sorted(i["outcome"] for i in q["info"]) == ["BLOCK", "INSUFFICIENT_EVIDENCE"]
    ma = next(r for r in db.conn.execute("SELECT run_id FROM runs WHERE kind = 'multi_agent' AND outcome = 'SELECTED'"))
    assert len(db.spans(ma[0])) == 12 and len(db.guardrail_events(ma[0])) == 12
    tot = next(r for r in db.conn.execute("SELECT run_id FROM runs WHERE kind = 'tot'"))
    nodes = db.tot_nodes(tot[0])
    assert sum(n["status"] == "selected" for n in nodes) == 1 and any(n["status"] == "pruned" for n in nodes)
    assert len(db.positions("open")) == 1 and metrics(db.journal())["closed_trades"] == 1
    assert all(db.brief(b)["payload"]["synthetic_demo"] for b, *_ in tiers)
    assert len(db.tickets()) == 0                     # the demo never contains an approval


def test_rebuilding_replaces_the_file(tmp_path):
    p = build_demo_db(tmp_path / "d.db", today=date(2026, 11, 20))
    build_demo_db(p, today=date(2026, 11, 20))
    assert len(Database(p).briefs("pending")) == 4
