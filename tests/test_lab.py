"""Lab evidence (UI slice 7): saved eval and ablation results, nothing is run. No network."""
from __future__ import annotations

import json

from options_copilot.db import Database
from options_copilot.lab import ablation_table, build, compare


def test_compare_is_direction_aware():
    assert compare(1, 3, False) == {"value": 1, "delta": -2, "direction": "better", "level": 3}
    assert compare(0.8, 0.6, True)["direction"] == "better"
    assert compare(10.2, 7.2, False) == {"value": 10.2, "delta": 3.0, "direction": "worse", "level": 2}
    assert compare(7.5, 7.2, False)["level"] == 1
    assert compare(1, 0, False) == {"value": 1, "delta": 1, "direction": "worse", "level": 3}   # any change from zero
    assert compare(0, 0, False)["direction"] == "same"
    assert compare(None, 1, True)["direction"] == "n/a"


def test_ablation_table_marks_the_baseline_and_compares_every_other_row():
    t = ablation_table([{"config": "full", "unsafe_passed": 0, "false_blocks": 3},
                        {"config": "-judge", "unsafe_passed": 1, "false_blocks": 3}])
    assert t["baseline"] == "full"
    full, judge = t["rows"]
    assert full["cells"]["false_blocks"]["direction"] == "baseline"
    assert judge["cells"]["unsafe_passed"] == {"value": 1, "delta": 1, "direction": "worse", "level": 3}
    assert judge["cells"]["false_blocks"]["direction"] == "same"


def _write(dir_, name, doc):
    (dir_ / name).write_text(json.dumps(doc))


def test_build_reads_saved_files_and_prefers_rescored_before_floor(tmp_path):
    agg = {"unsafe_passed": 0, "false_blocks": 0, "outcome_accuracy": 1.0}
    calib = {"confidence": {"n": 2, "ece": 0.1, "brier": 0.1, "reliability": [{"bin": "0.8-1.0", "n": 2}]}}
    _write(tmp_path, "eval_2026-11-16_x3.json", {"mode": "eval", "repeats": 3, "summary": {
        "aggregate": agg, "calibration": calib, "cases": {"c1": {"runs": 3, "unsafe_passed": 0, "extra": 1}}}})
    _write(tmp_path, "eval_2026-11-16_x3_before_floor.json", {"summary": {"aggregate": {"false_blocks": 4}},
           "summary_rescored": {"aggregate": {"false_blocks": 9}}, "rescored_note": "relabel"})
    _write(tmp_path, "eval_ablation_2026-11-16.json", {"rows": [{"config": "full", "unsafe_passed": 0, "records": [1]},
                                                                {"config": "-verifier", "unsafe_passed": 0}]})
    b = build(Database(":memory:"), tmp_path)
    assert b["after_floor"] == agg and b["before_floor"] == {"false_blocks": 9} and b["before_note"] == "relabel"
    assert b["calibration"] == calib and b["cases"] == [{"case": "c1", "runs": 3, "unsafe_passed": 0}]
    assert [r["config"] for r in b["ablation"]["rows"]] == ["full", "-verifier"] and b["ablation"]["note"]
    assert b["sources"]["eval"] == "eval_2026-11-16_x3.json" and "eval --ablation" in b["run"]["ablation_command"]


def test_build_falls_back_to_the_evals_table_without_files(tmp_path):
    db = Database(":memory:")
    db.add_eval("eval", {"unsafe_passed": 0, "false_blocks": 2})
    db.add_eval("ablation", {"rows": [{"config": "full", "false_blocks": 2}]})
    b = build(db, tmp_path / "missing")
    assert b["after_floor"] == {"unsafe_passed": 0, "false_blocks": 2} and b["calibration"] is None
    assert b["ablation"]["rows"][0]["config"] == "full" and b["ablation"]["note"] is None
    assert [s["mode"] for s in b["sources"]["stored_evals"]] == ["eval", "ablation"]
