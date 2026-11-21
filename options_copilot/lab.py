"""The Lab screen's evidence (UI phase): saved evaluation and ablation results, assembled for display.

Nothing here runs an evaluation. It reads what `eval` and `eval --ablation` already stored: the
`evals` table in the main database and the JSON files in demo_outputs/ (which also hold the
calibration bins and per-case results). The ablation table gets a direction-aware comparison with
the full configuration, so the UI can shade cells without computing anything itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# For each metric: is a higher value better? (None: informational, never shaded)
ABLATION_METRICS: tuple[tuple[str, str, bool | None], ...] = (
    ("unsafe_passed", "Unsafe trades passed", False),
    ("unsafe_standard_tier", "Unsafe, standard tier", False),
    ("false_blocks", "False blocks", False),
    ("verdict_accuracy", "Verdict accuracy", True),
    ("outcome_accuracy", "Outcome accuracy", True),
    ("llm_calls_mean", "LLM calls per run", False),
    ("latency_p50_s", "Latency p50 (s)", False),
)

ABLATION_NOTE = ("Read '-verifier' with care: it shows fewer false blocks only because, without the verifier, "
                 "trades on illiquid expiries (open interest near 3) were selected, and the case labels allowed "
                 "a trade, so they did not count as unsafe. The verifier is never switched off outside the sandbox.")
RUN_EVAL = "python -m options_copilot.cli eval --repeats 3 --save"
RUN_ABLATION = "python -m options_copilot.cli eval --ablation --repeats 1 --save"
LONG_RUN_WARNING = ("An evaluation runs every labelled case several times through the full multi-agent "
                    "pipeline on the local model: about 2-5 minutes per run, so hours for the whole set, "
                    "and it holds the LLM the whole time. Run it from a terminal, in the evaluation sandbox.")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _latest(paths: list[Path]) -> Path | None:
    return sorted(paths, key=lambda p: p.name)[-1] if paths else None


def compare(value: Any, base: Any, higher_is_better: bool | None) -> dict:
    """How one ablation cell compares with the full configuration.

    `direction` is better / worse / same / n/a; `level` 0-3 grows with the relative change
    (0 same, 1 under 10%, 2 under 50%, 3 at least 50%, or any change from zero)."""
    if higher_is_better is None or not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
        return {"value": value, "delta": None, "direction": "n/a", "level": 0}
    delta = round(value - base, 4)
    if delta == 0:
        return {"value": value, "delta": 0.0, "direction": "same", "level": 0}
    rel = math.inf if base == 0 else abs(delta) / abs(base)
    level = 1 if rel < 0.10 else 2 if rel < 0.50 else 3
    better = (delta > 0) == higher_is_better
    return {"value": value, "delta": delta, "direction": "better" if better else "worse", "level": level}


def ablation_table(rows: list[dict]) -> dict:
    """Rows: one per configuration ("full", "-verifier", ...). Columns: the metrics above."""
    base = next((r for r in rows if r.get("config") == "full"), None)
    cols = [{"key": k, "label": label, "higher_is_better": hib} for k, label, hib in ABLATION_METRICS]
    out = []
    for r in rows:
        cells = {k: (compare(r.get(k), base.get(k), hib) if base is not None and r is not base
                     else {"value": r.get(k), "delta": None, "direction": "baseline" if r is base else "n/a",
                           "level": 0})
                 for k, _, hib in ABLATION_METRICS}
        out.append({"config": r.get("config"), "cells": cells})
    return {"columns": cols, "rows": out, "baseline": "full" if base is not None else None}


def build(db: Any, outputs_dir: Path) -> dict:
    files = sorted(outputs_dir.glob("eval_*.json")) if outputs_dir.is_dir() else []
    evals = [p for p in files if "ablation" not in p.name and "before_floor" not in p.name]
    befores = [p for p in files if "before_floor" in p.name]
    ablations = [p for p in files if "ablation" in p.name]

    latest_path, before_path, abl_path = _latest(evals), _latest(befores), _latest(ablations)
    latest, before, abl = (_load(p) if p else None for p in (latest_path, before_path, abl_path))
    rows =db.conn.execute("SELECT eval_id, created_at, mode, summary FROM evals ORDER BY created_at, rowid").fetchall()
    stored = [{"eval_id": r["eval_id"], "created_at": r["created_at"], "mode": r["mode"],
               "summary": json.loads(r["summary"])} for r in rows]

    def aggregate(doc: dict | None, fallback_mode: str) -> dict | None:
        if doc and isinstance(doc.get("summary"), dict):
            return doc["summary"].get("aggregate") or doc["summary"]
        last = next((s for s in reversed(stored) if s["mode"] == fallback_mode), None)
        return last["summary"] if last else None

    after = aggregate(latest, "eval")
    before_agg = None
    before_note = None
    if before:
        rescored = before.get("summary_rescored")
        before_agg = (rescored.get("aggregate") or rescored) if isinstance(rescored, dict) else aggregate(before, "eval")
        before_note = before.get("rescored_note")
    abl_rows = (abl or {}).get("rows") or next((s["summary"].get("rows") for s in reversed(stored)
                                                if s["mode"] == "ablation"), None) or []
    calib = ((latest or {}).get("summary") or {}).get("calibration")
    cases = ((latest or {}).get("summary") or {}).get("cases") or {}
    return {
        "after_floor": after,
        "before_floor": before_agg,
        "before_note": before_note,
        "calibration": calib,
        "cases": [{"case": k, **{kk: v[kk] for kk in ("runs", "verdict_accuracy", "outcome_accuracy", "unsafe_passed",
                                                       "false_blocks", "verdicts", "outcomes") if kk in v}}
                  for k, v in cases.items()],
        "ablation": {**ablation_table([{k: v for k, v in r.items() if k != "records"} for r in abl_rows]),
                     "note": ABLATION_NOTE if any(r.get("config") == "-verifier" for r in abl_rows) else None},
        "repeats": (latest or {}).get("repeats"),
        "sources": {"eval": latest_path.name if latest_path else None,
                    "before_floor": before_path.name if before_path else None,
                    "ablation": abl_path.name if abl_path else None,
                    "stored_evals": [{k: s[k] for k in ("eval_id", "created_at", "mode")} for s in stored]},
        "run": {"warning": LONG_RUN_WARNING, "eval_command": RUN_EVAL, "ablation_command": RUN_ABLATION},
    }
