"""Load and validate playbook.yaml (procedural memory).

The playbook is read-only at runtime: values are exposed through a frozen object with
read-only mappings, and its SHA-256 is recorded so every brief and trace can name the
exact rule version it was checked against.

Validation is strict on purpose: an unknown key is almost always a typo, and a typo in a
risk threshold must fail loudly rather than fall back to a default.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

NUM = (int, float)


@dataclass(frozen=True)
class Rule:
    """Expected type and inclusive range (or allowed values) for one leaf key."""
    kind: type | tuple[type, ...]
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] | None = None


def _pct(hi: float = 100) -> Rule:
    return Rule(NUM, 0, hi)


def _days(lo: int = 0, hi: int = 400) -> Rule:
    return Rule(int, lo, hi)


SPEC: dict[str, Rule] = {
    "version": Rule(int, 1),
    "universe.tier_1": Rule(list),
    "universe.tier_2": Rule(list),
    "universe.watch": Rule(list),
    "screener.quality_score_min": Rule(NUM, 0, 10),
    "screener.quality_weights.analyst": Rule(NUM, 0, 10),
    "screener.quality_weights.fundamentals": Rule(NUM, 0, 10),
    "screener.quality_weights.trend": Rule(NUM, 0, 10),
    "screener.quality_weights.insider": Rule(NUM, 0, 10),
    "screener.analyst_buy_ratio_min": Rule(NUM, 0, 1),
    "screener.analyst_count_min": Rule(int, 0, 100),
    "screener.analyst_upside_max_pct": _pct(500),
    "screener.market_cap_min_usd": Rule(NUM, 0),
    "screener.price_min_usd": Rule(NUM, 0),
    "screener.sectors": Rule(list),
    "screener.country": Rule(str),
    "screener.exclusions.pending_mna": Rule(bool),
    "screener.exclusions.fda_events": Rule(bool),
    "screener.exclusions.reverse_split_lookback_months": Rule(int, 0, 60),
    "screener.exclusions.insider_sell_buy_ratio_max": Rule(NUM, 0),
    "screener.exclusions.insider_lookback_days": _days(1),
    "entry.dte_min": _days(1),
    "entry.dte_max": _days(1),
    "entry.dte_target": _days(1),
    "entry.cash_secured_put_price_max_usd": Rule(NUM, 0),
    "entry.earnings.block_within_days": _days(),
    "entry.earnings.spreads_only_within_days": _days(),
    "entry.earnings.expiry_must_precede": Rule(bool),
    "entry.pop_min": Rule(NUM, 0, 1),
    "entry.pop_method": Rule(str, choices=("breakeven_lognormal",)),
    "entry.open_interest_min": Rule(int, 0),
    "entry.position_risk_pct.warn": _pct(),
    "entry.position_risk_pct.block": _pct(),
    "entry.portfolio_heat_pct.warn": _pct(),
    "entry.portfolio_heat_pct.block": _pct(),
    "entry.support_resistance.placement": Rule(str, choices=("beyond_levels",)),
    "entry.support_resistance.lookback_trading_days": _days(10, 1000),
    "entry.support_resistance.pivot_window_bars": Rule(int, 1, 50),
    "entry.advisory.dividend_exdate_within_days": _days(),
    "entry.advisory.rsi_oversold": _pct(),
    "entry.advisory.rsi_overbought": _pct(),
    "entry.advisory.iv_rank_low": _pct(),
    "entry.advisory.iv_rank_target": _pct(),
    "entry.advisory.iv_rank_high": _pct(),
    "entry.advisory.unusual_flow_vol_oi_ratio": Rule(NUM, 0),
    "builder.short_delta_min": Rule(NUM, 0, 1),
    "builder.short_delta_max": Rule(NUM, 0, 1),
    "builder.strike_selection": Rule(str, choices=("delta",)),
    "builder.wing_width_pct": Rule(NUM, 0.1, 50),
    "builder.short_call_max": Rule(str, choices=("analyst_target",)),
    "builder.credit_price": Rule(str, choices=("mid",)),
    "management.r1_profit_target_pct": _pct(),
    "management.r2_roll.dte_below": _days(),
    "management.r2_roll.roll_to_dte": _days(1),
    "management.r3_stop_loss.loss_credit_multiple": Rule(NUM, 0.01, 20),
    "management.r3_stop_loss.reentry_cooldown_days": _days(),
    "management.r4_big_move.sigma": Rule(NUM, 0.1, 10),
    "management.r4_big_move.close_all_below_200dma": Rule(bool),
    "management.r5_earnings.close_all_within_days": _days(),
    "management.r5_earnings.close_calls_within_days": _days(),
    "management.r6_iv_crush.iv_rank_drop_points": _pct(),
    "management.r6_iv_crush.window_trading_days": _days(1, 60),
    "management.r6_iv_crush.roll_out_days": _days(1),
    "management.r7_ex_dividend.within_days": _days(),
    "management.r7_ex_dividend.leg": Rule(str, choices=("short_call_itm",)),
    "management.r7_ex_dividend.extrinsic_below_dividend": Rule(bool),
    "management.r8_heat.max_pct": _pct(),
    "management.r8_heat.review_weekday": Rule(
        str, choices=("monday", "tuesday", "wednesday", "thursday", "friday")),
    "management.emergency.sector_drop_pct": _pct(),
    "management.emergency.insider_stake_sale_pct": _pct(),
    "iv_rank.full_history_days": _days(20, 2000),
    "iv_rank.percentile_min_snapshots": _days(2, 2000),
    "iv_rank.realized_vol_window_days": _days(5, 400),
}


class PlaybookError(ValueError):
    """The playbook file is missing, malformed, or violates the schema."""


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(node, dict):
        return {prefix: node}
    flat: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flat.update(_flatten(value, path) if isinstance(value, dict) else {path: value})
    return flat


def _check_leaf(path: str, value: Any, rule: Rule) -> str | None:
    # bool is a subclass of int; never accept True/False where a number is expected.
    if isinstance(value, bool) and rule.kind is not bool:
        return f"{path}: expected a number, got a boolean"
    if not isinstance(value, rule.kind):
        return f"{path}: expected {getattr(rule.kind, '__name__', 'number')}, got {type(value).__name__}"
    if rule.choices is not None and value not in rule.choices:
        return f"{path}: {value!r} is not one of {list(rule.choices)}"
    if isinstance(value, NUM) and not isinstance(value, bool):
        if rule.lo is not None and value < rule.lo:
            return f"{path}: {value} is below the minimum {rule.lo}"
        if rule.hi is not None and value > rule.hi:
            return f"{path}: {value} is above the maximum {rule.hi}"
    if isinstance(value, list) and not all(isinstance(v, str) and v for v in value):
        return f"{path}: every list item must be a non-empty string"
    return None


def _check_consistency(v: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def less(a: str, b: str, strict: bool = True) -> None:
        if (v[a] >= v[b]) if strict else (v[a] > v[b]):
            errors.append(f"{a} ({v[a]}) must be {'<' if strict else '<='} {b} ({v[b]})")

    less("entry.position_risk_pct.warn", "entry.position_risk_pct.block")
    less("entry.portfolio_heat_pct.warn", "entry.portfolio_heat_pct.block")
    less("entry.dte_min", "entry.dte_target", strict=False)
    less("entry.dte_target", "entry.dte_max", strict=False)
    less("management.r2_roll.dte_below", "entry.dte_min")
    less("builder.short_delta_min", "builder.short_delta_max")
    less("entry.earnings.block_within_days", "entry.earnings.spreads_only_within_days")
    less("management.r5_earnings.close_all_within_days", "management.r5_earnings.close_calls_within_days")
    less("entry.advisory.rsi_oversold", "entry.advisory.rsi_overbought")
    less("entry.advisory.iv_rank_low", "entry.advisory.iv_rank_high")
    less("iv_rank.percentile_min_snapshots", "iv_rank.full_history_days")

    weights = sum(v[k] for k in v if k.startswith("screener.quality_weights."))
    if abs(weights - 10) > 1e-9:
        errors.append(f"screener.quality_weights must sum to 10, got {weights}")

    seen: dict[str, str] = {}
    for group in ("universe.tier_1", "universe.tier_2", "universe.watch"):
        for ticker in v[group]:
            if ticker != ticker.upper():
                errors.append(f"{group}: ticker {ticker!r} must be upper case")
            if ticker in seen:
                errors.append(f"{group}: {ticker} is already listed in {seen[ticker]}")
            seen[ticker] = group
    return errors


def validate(raw: Any) -> dict[str, Any]:
    """Return the flattened, validated values or raise PlaybookError listing every problem."""
    if not isinstance(raw, dict):
        raise PlaybookError("playbook must be a YAML mapping at the top level")
    flat = _flatten(raw)
    errors = [f"{k}: unknown key" for k in sorted(flat.keys() - SPEC.keys())]
    errors += [f"{k}: missing" for k in sorted(SPEC.keys() - flat.keys())]
    errors += [e for k in sorted(flat.keys() & SPEC.keys()) if (e := _check_leaf(k, flat[k], SPEC[k]))]
    if not errors:
        errors = _check_consistency(flat)
    if errors:
        raise PlaybookError("invalid playbook:\n  " + "\n  ".join(errors))
    return flat


def _freeze(node: Any) -> Any:
    if isinstance(node, dict):
        return MappingProxyType({k: _freeze(v) for k, v in node.items()})
    if isinstance(node, list):
        return tuple(_freeze(v) for v in node)
    return node


@dataclass(frozen=True)
class Playbook:
    """Read-only, validated playbook. Look values up by dotted path: pb["entry.pop_min"]."""
    version: int
    sha256: str
    source: str
    tree: Mapping[str, Any]

    def __getitem__(self, dotted: str) -> Any:
        node: Any = self.tree
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(dotted)
            node = node[part]
        return node

    def tier_of(self, ticker: str) -> str | None:
        """'tier_1', 'tier_2', 'watch', or None if the ticker is outside the universe."""
        ticker = ticker.upper()
        for group in ("tier_1", "tier_2", "watch"):
            if ticker in self.tree["universe"][group]:
                return group
        return None

    def is_enterable(self, ticker: str) -> bool:
        return self.tier_of(ticker) in ("tier_1", "tier_2")


def load_playbook(path: Path) -> Playbook:
    try:
        text = Path(path).read_bytes()
    except OSError as exc:
        raise PlaybookError(f"cannot read playbook at {Path(path).name}: {exc.strerror}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PlaybookError(f"playbook is not valid YAML: {exc}") from exc
    validate(raw)
    return Playbook(
        version=raw["version"],
        sha256=hashlib.sha256(text).hexdigest(),
        source=Path(path).name,
        tree=_freeze(raw),
    )
