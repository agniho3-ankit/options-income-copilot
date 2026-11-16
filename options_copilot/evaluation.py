"""HELM-style evaluation metrics (Phase 6). Pure functions, unit-tested on synthetic cases.

A *record* is one replay of one labeled case:
    {"case": id, "repeat": k, "verdict": str, "outcome": str, "strategy": str | None,
     "confidence": float | None, "p_true": float | None, "latency_ms": int, "llm_calls": int,
     "builds": int, "fallbacks": int, "paraphrase_verdict": str | None, "tier": str | None}
A *label* is the owner's:
    {"expected_verdict": str, "expected_outcome": "trade" | "NO_TRADE" | "BLOCK" | "INSUFFICIENT_EVIDENCE",
     "acceptable_strategies": [str] (optional)}

Safety terms:
  * unsafe trade passed: a trade was SELECTED where the label says the outcome is not a trade;
  * false block: no trade was selected where the label says a trade is acceptable.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

NO_TRADE_OUTCOMES = ("NO_TRADE", "BLOCK", "INSUFFICIENT_EVIDENCE")


def outcome_ok(record: dict, label: dict) -> bool:
    exp = label["expected_outcome"]
    if exp == "trade":
        if record["outcome"] != "SELECTED":
            return False
        ok = label.get("acceptable_strategies")
        return not ok or record.get("strategy") in ok
    if exp == "NO_TRADE":
        return record["outcome"] in NO_TRADE_OUTCOMES
    return record["outcome"] == exp


def unsafe_pass(record: dict, label: dict) -> bool:
    return record["outcome"] == "SELECTED" and label["expected_outcome"] != "trade"


def false_block(record: dict, label: dict) -> bool:
    return record["outcome"] != "SELECTED" and label["expected_outcome"] == "trade"


def verdict_ok(record: dict, label: dict) -> bool:
    return record.get("verdict") == label["expected_verdict"]


def ece(confidences: list[float], correct: list[bool], bins: int = 5) -> float | None:
    """Expected calibration error with equal-width bins over [0, 1]."""
    pairs = [(c, k) for c, k in zip(confidences, correct) if c is not None]
    if not pairs:
        return None
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, k) for c, k in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if inside:
            conf = sum(c for c, _ in inside) / len(inside)
            acc = sum(1 for _, k in inside if k) / len(inside)
            err += len(inside) / total * abs(conf - acc)
    return round(err, 4)


def brier(confidences: list[float], correct: list[bool]) -> float | None:
    pairs = [(c, k) for c, k in zip(confidences, correct) if c is not None]
    if not pairs:
        return None
    return round(sum((c - (1.0 if k else 0.0)) ** 2 for c, k in pairs) / len(pairs), 4)


def reliability(confidences: list[float], correct: list[bool], bins: int = 5) -> list[dict]:
    """Per bin: range, count, mean confidence, accuracy (None when empty). Counts are shown so a
    small sample is visible, not hidden."""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, k) for c, k in zip(confidences, correct)
                  if c is not None and ((lo <= c < hi) or (b == bins - 1 and c == 1.0))]
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(inside),
                     "mean_conf": round(sum(c for c, _ in inside) / len(inside), 3) if inside else None,
                     "accuracy": round(sum(1 for _, k in inside if k) / len(inside), 3) if inside else None})
    return rows


def consistency(values: Iterable) -> float:
    """Share of runs that agree with the most common value (1.0 = always the same)."""
    vals = list(values)
    if not vals:
        return 1.0
    return round(Counter(vals).most_common(1)[0][1] / len(vals), 4)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(1, math.ceil(q / 100 * len(vals)))
    return vals[k - 1]


def summarize(records: list[dict], labels: dict[str, dict]) -> dict:
    """Per-case and aggregate metrics for one configuration."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_case[r["case"]].append(r)
    cases = {}
    for cid, rs in by_case.items():
        lab = labels[cid]
        cases[cid] = {
            "runs": len(rs),
            "verdict_accuracy": round(sum(verdict_ok(r, lab) for r in rs) / len(rs), 4),
            "outcome_accuracy": round(sum(outcome_ok(r, lab) for r in rs) / len(rs), 4),
            "unsafe_passed": sum(unsafe_pass(r, lab) for r in rs),
            "false_blocks": sum(false_block(r, lab) for r in rs),
            "verdict_consistency": consistency(r.get("verdict") for r in rs),
            "outcome_consistency": consistency(r["outcome"] for r in rs),
            "paraphrase_agreement": _paraphrase(rs),
            "verdicts": dict(Counter(r.get("verdict") for r in rs)),
            "outcomes": dict(Counter(r["outcome"] for r in rs)),
        }
    n = len(records)
    correct_v = [verdict_ok(r, labels[r["case"]]) for r in records]
    agg = {
        "cases": len(by_case), "runs": n,
        "verdict_accuracy": round(sum(correct_v) / n, 4) if n else None,
        "outcome_accuracy": round(sum(outcome_ok(r, labels[r["case"]]) for r in records) / n, 4) if n else None,
        "unsafe_passed": sum(unsafe_pass(r, labels[r["case"]]) for r in records),
        "false_blocks": sum(false_block(r, labels[r["case"]]) for r in records),
        "false_block_rate": _rate([false_block(r, labels[r["case"]]) for r in records
                                   if labels[r["case"]]["expected_outcome"] == "trade"]),
        "verdict_consistency": _mean(c["verdict_consistency"] for c in cases.values()),
        "outcome_consistency": _mean(c["outcome_consistency"] for c in cases.values()),
        "paraphrase_agreement": _mean(c["paraphrase_agreement"] for c in cases.values()
                                      if c["paraphrase_agreement"] is not None),
        "latency_p50_s": _s(percentile([r["latency_ms"] for r in records], 50)),
        "latency_p95_s": _s(percentile([r["latency_ms"] for r in records], 95)),
        "llm_calls_mean": _mean(r["llm_calls"] for r in records),
        "builds_mean": _mean(r["builds"] for r in records),
        "fallbacks": sum(r.get("fallbacks", 0) for r in records),
    }
    calib = {}
    for key in ("confidence", "p_true"):
        conf = [r.get(key) for r in records]
        n_conf = sum(c is not None for c in conf)
        calib[key] = {"n": n_conf, "ece": ece(conf, correct_v), "brier": brier(conf, correct_v),
                      "reliability": reliability(conf, correct_v)}
    return {"aggregate": agg, "calibration": calib, "cases": cases}


def judge_accuracy(results: list[dict]) -> dict:
    """`results`: [{"faithful": bool (owner label), "flagged": bool (judge)}]. A flag on an
    unfaithful brief is a true positive."""
    tp = sum(1 for r in results if r["flagged"] and not r["faithful"])
    fp = sum(1 for r in results if r["flagged"] and r["faithful"])
    fn = sum(1 for r in results if not r["flagged"] and not r["faithful"])
    tn = sum(1 for r in results if not r["flagged"] and r["faithful"])
    n = len(results)
    return {"n": n, "accuracy": round((tp + tn) / n, 4) if n else None, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def _paraphrase(rs: list[dict]) -> float | None:
    pairs = [(r.get("verdict"), r.get("paraphrase_verdict")) for r in rs if r.get("paraphrase_verdict") is not None]
    if not pairs:
        return None
    return round(sum(a == b for a, b in pairs) / len(pairs), 4)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def _s(ms) -> float | None:
    return None if ms is None else round(ms / 1000, 1)
