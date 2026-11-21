"""Trade management (Skill #4): which of the 8 rules fire for an open position.

The engine returns every triggered rule with its prescribed action. When rules conflict
(say, take profit and roll) it returns all of them; the agent resolves the conflict later,
choosing only among these rule-compliant actions. Rules whose inputs are missing are
listed as not evaluated, never silently passed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from .models import OptionType, Position, Side
from .playbook import Playbook


class Action(str, Enum):
    CLOSE = "close"                       # buy to close the whole position
    ROLL = "roll"                         # close and reopen ~45 DTE out
    BUY_BACK_CALLS = "buy_back_calls"
    BUY_BACK_PUTS = "buy_back_puts"
    CLOSE_CALLS = "close_calls"
    HOLD_OR_ROLL_OUT = "hold_or_roll_out"
    HOLD_OR_ROLL = "hold_or_roll"
    REVIEW_WEAKEST = "review_weakest"


@dataclass(frozen=True)
class Trigger:
    rule: str
    action: Action
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Snapshot:
    """Current facts for one position. None means unknown."""
    as_of: date
    spot: float
    close_cost: float | None = None        # debit to close now, per contract (calculators)
    sma200: float | None = None
    earnings_date: date | None = None
    earnings_known_none: bool = False
    ex_dividend_date: date | None = None
    dividend: float | None = None          # per share
    short_call_extrinsic: float | None = None  # per share
    iv_rank_now: float | None = None
    iv_rank_window_ago: float | None = None    # IV rank `window_trading_days` ago
    sector_change_pct: float | None = None     # sector ETF move this week
    insider_stake_sold_pct: float | None = None


@dataclass(frozen=True)
class Evaluation:
    position_id: str
    profit_pct: float | None
    triggers: tuple[Trigger, ...]
    not_evaluated: tuple[str, ...]

    @property
    def actions(self) -> list[Action]:
        return list(dict.fromkeys(t.action for t in self.triggers))


def profit_pct(entry_credit: float, close_cost: float) -> float:
    """% of max profit captured: 100 means the position can be closed for free."""
    return (entry_credit - close_cost) / entry_credit * 100


def one_sigma(entry_spot: float, entry_iv: float, days_held: int) -> float:
    """Rule 4 band: entry price x entry IV x sqrt(days held / 365), at least one day."""
    return entry_spot * entry_iv * math.sqrt(max(days_held, 1) / 365)


def _has(pos: Position, option_type: OptionType) -> bool:
    return any(g.option_type is option_type and g.side is Side.SHORT for g in pos.legs)


def evaluate(pos: Position, snap: Snapshot, pb: Playbook) -> Evaluation:
    m = lambda key: pb[f"management.{key}"]  # noqa: E731
    triggers: list[Trigger] = []
    missing: list[str] = []
    dte = (pos.expiry - snap.as_of).days
    days_held = (snap.as_of - pos.opened_on).days
    has_call, has_put = _has(pos, OptionType.CALL), _has(pos, OptionType.PUT)

    pp = None
    if snap.close_cost is None:
        missing += ["r1_profit_target", "r2_roll", "r3_stop_loss", "r6_iv_crush"]
    else:
        pp = profit_pct(pos.entry_credit, snap.close_cost)
        # R1: lock winners
        if pp >= m("r1_profit_target_pct"):
            triggers.append(Trigger("r1_profit_target", Action.CLOSE, f"{pp:.0f}% of max profit captured"))
        # R2: roll when short on time and short of the target
        if dte < m("r2_roll.dte_below") and pp < m("r1_profit_target_pct"):
            triggers.append(Trigger("r2_roll", Action.ROLL, f"{dte} DTE with {pp:.0f}% of max profit",
                                    {"roll_to_dte": m("r2_roll.roll_to_dte")}))
        # R3: stop loss at loss >= multiple x credit
        loss = snap.close_cost - pos.entry_credit
        limit = m("r3_stop_loss.loss_credit_multiple") * pos.entry_credit
        if loss >= limit:
            triggers.append(Trigger("r3_stop_loss", Action.CLOSE,
                                    f"unrealized loss is {loss / pos.entry_credit:.2f}x the credit",
                                    {"reentry_cooldown_days": m("r3_stop_loss.reentry_cooldown_days")}))

    # R4: big move versus a 1-sigma band that widens with time held
    if pos.entry_spot is None or pos.entry_iv is None:
        missing.append("r4_big_move")
    else:
        band = m("r4_big_move.sigma") * one_sigma(pos.entry_spot, pos.entry_iv, days_held)
        move = snap.spot - pos.entry_spot
        if abs(move) > band:
            detail = {"move": round(move, 2), "band": round(band, 2)}
            if move > 0 and has_call:
                triggers.append(Trigger("r4_big_move", Action.BUY_BACK_CALLS,
                                        f"up {move:.2f} vs a 1-sigma band of {band:.2f}", detail))
            elif move < 0:
                if has_put:
                    triggers.append(Trigger("r4_big_move", Action.BUY_BACK_PUTS,
                                            f"down {-move:.2f} vs a 1-sigma band of {band:.2f}", detail))
                if m("r4_big_move.close_all_below_200dma"):
                    if snap.sma200 is None:
                        missing.append("r4_below_200dma")
                    elif snap.spot < snap.sma200:
                        triggers.append(Trigger("r4_big_move", Action.CLOSE,
                                                f"down hard and below the 200-day average {snap.sma200:.2f}",
                                                detail))

    # R5: earnings
    if snap.earnings_date is None:
        if not snap.earnings_known_none:
            missing.append("r5_earnings")
    else:
        days = (snap.earnings_date - snap.as_of).days
        if 0 <= days <= m("r5_earnings.close_all_within_days"):
            triggers.append(Trigger("r5_earnings", Action.CLOSE, f"earnings in {days} days"))
        elif 0 <= days <= m("r5_earnings.close_calls_within_days") and has_call:
            triggers.append(Trigger("r5_earnings", Action.CLOSE_CALLS, f"earnings in {days} days"))

    # R6: IV crush
    if snap.iv_rank_now is None or snap.iv_rank_window_ago is None or pp is None:
        if "r6_iv_crush" not in missing:
            missing.append("r6_iv_crush")
    else:
        drop = snap.iv_rank_window_ago - snap.iv_rank_now
        if drop >= m("r6_iv_crush.iv_rank_drop_points"):
            if pp >= m("r1_profit_target_pct"):
                triggers.append(Trigger("r6_iv_crush", Action.CLOSE, f"IV rank fell {drop:.0f} points"))
            else:
                triggers.append(Trigger("r6_iv_crush", Action.HOLD_OR_ROLL_OUT, f"IV rank fell {drop:.0f} points",
                                        {"roll_out_days": m("r6_iv_crush.roll_out_days")}))

    # R7: ex-dividend early-assignment risk on ITM short calls
    sc = next((g for g in pos.legs if g.option_type is OptionType.CALL and g.side is Side.SHORT), None)
    if sc is not None and snap.ex_dividend_date is not None:
        days = (snap.ex_dividend_date - snap.as_of).days
        if 0 <= days <= m("r7_ex_dividend.within_days") and snap.spot > sc.strike:
            if snap.dividend is None or snap.short_call_extrinsic is None:
                missing.append("r7_ex_dividend")
            elif snap.short_call_extrinsic < snap.dividend:
                profitable = pp is not None and pp > 0
                triggers.append(Trigger("r7_ex_dividend", Action.CLOSE_CALLS if profitable else Action.HOLD_OR_ROLL,
                                        f"ex-dividend in {days} days, short call ITM with extrinsic "
                                        f"{snap.short_call_extrinsic:.2f} < dividend {snap.dividend:.2f}"))

    # Emergency exits
    if snap.sector_change_pct is not None and snap.sector_change_pct <= -m("emergency.sector_drop_pct"):
        triggers.append(Trigger("emergency_sector", Action.CLOSE, f"sector down {-snap.sector_change_pct:.1f}%"))
    if snap.insider_stake_sold_pct is not None and snap.insider_stake_sold_pct >= m("emergency.insider_stake_sale_pct"):
        triggers.append(Trigger("emergency_insider", Action.CLOSE,
                                f"an insider sold {snap.insider_stake_sold_pct:.0f}% of their stake"))

    return Evaluation(pos.position_id, None if pp is None else round(pp, 1), tuple(triggers), tuple(missing))


def heat_review(heat_pct: float, as_of: date, pb: Playbook) -> Trigger | None:
    """R8: on the review weekday, flag the weakest position if heat is above the max."""
    weekday = pb["management.r8_heat.review_weekday"]
    if as_of.strftime("%A").lower() != weekday:
        return None
    if heat_pct > pb["management.r8_heat.max_pct"]:
        return Trigger("r8_heat", Action.REVIEW_WEAKEST, f"portfolio heat {heat_pct:.1f}% on {weekday}")
    return None


# --- rule tracks for the UI's Positions screen (UI phase) ---------------------------------

def _progress(value: float, target: float) -> float:
    """Percent of the way to `target`, clamped to 0..100 (for a progress bar)."""
    if target <= 0:
        return 100.0
    return round(min(max(value / target, 0.0), 1.0) * 100, 1)


def rule_tracks(pos: Position, snap: Snapshot, ev: Evaluation, pb: Playbook) -> list[dict]:
    """One row per management rule: how far the position is toward the rule's trigger.

    Every value is computed here from the snapshot and the playbook; the UI only draws it.
    `state` is "ok" (not triggered), "due" (the rule says act), "breach" (a stop or emergency
    exit), or "unknown" (the data the rule needs is missing; never treated as safe)."""
    m = lambda key: pb[f"management.{key}"]  # noqa: E731
    fired = {t.rule: t for t in ev.triggers}
    missing = set(ev.not_evaluated)
    rows: list[dict] = []

    def row(rule: str, label: str, *, value: str, target: str, progress: float | None, breach: bool = False,
            note: str = "") -> None:
        t = fired.get(rule)
        if rule in missing:
            state = "unknown"
        elif t is not None:
            state = "breach" if breach else "due"
        else:
            state = "ok"
        rows.append({"rule": rule, "label": label, "value": value, "target": target,
                     "progress_pct": None if rule in missing else progress, "state": state,
                     "action": t.action.value if t else None, "reason": t.reason if t else note})

    pp = ev.profit_pct
    goal = m("r1_profit_target_pct")
    row("r1_profit_target", "Profit taken", value="–" if pp is None else f"{pp:.1f}%", target=f"{goal}%",
        progress=None if pp is None else _progress(pp, goal))

    dte = (pos.expiry - snap.as_of).days
    dte_open = (pos.expiry - pos.opened_on).days
    roll_at = m("r2_roll.dte_below")
    row("r2_roll", "Days to expiry", value=f"{dte} DTE", target=f"roll below {roll_at}",
        progress=_progress(dte_open - dte, dte_open - roll_at) if dte_open > roll_at else 100.0)

    limit = m("r3_stop_loss.loss_credit_multiple")
    if snap.close_cost is None:
        row("r3_stop_loss", "Loss vs credit", value="–", target=f"stop at {limit:g}x", progress=None)
    else:
        mult = (snap.close_cost - pos.entry_credit) / pos.entry_credit
        row("r3_stop_loss", "Loss vs credit", value=f"{max(mult, 0.0):.2f}x", target=f"stop at {limit:g}x",
            progress=_progress(mult, limit), breach=True)

    if pos.entry_spot is None or pos.entry_iv is None:
        row("r4_big_move", "Move vs 1σ", value="–", target="1σ band", progress=None)
    else:
        band = m("r4_big_move.sigma") * one_sigma(pos.entry_spot, pos.entry_iv, (snap.as_of - pos.opened_on).days)
        move = snap.spot - pos.entry_spot
        row("r4_big_move", "Move vs 1σ", value=f"{move:+.2f}", target=f"±{band:.2f}", progress=_progress(abs(move), band))

    if snap.earnings_date is None:
        row("r5_earnings", "Earnings", value="none reported" if snap.earnings_known_none else "unknown",
            target=f"close within {m('r5_earnings.close_all_within_days')} days", progress=None)
    else:
        days = (snap.earnings_date - snap.as_of).days
        row("r5_earnings", "Earnings", value=f"in {days} days ({snap.earnings_date.isoformat()})",
            target=f"close within {m('r5_earnings.close_all_within_days')} days", progress=None)

    for rule, label in (("r6_iv_crush", "IV crush"), ("r7_ex_dividend", "Ex-dividend"),
                        ("emergency_sector", "Sector drop"), ("emergency_insider", "Insider sale")):
        t = fired.get(rule)
        if t is not None or rule in missing or rule in ("r6_iv_crush", "r7_ex_dividend"):
            row(rule, label, value="triggered" if t else ("–" if rule in missing else "clear"), target="",
                progress=None, breach=rule.startswith("emergency"))
    return rows


def heat_gauge(heat_pct: float, pb: Playbook) -> dict:
    """Portfolio heat against the R8 cap, with the fill level for a gauge (0..100)."""
    cap = pb["management.r8_heat.max_pct"]
    return {"pct": heat_pct, "cap_pct": cap, "warn_pct": pb["entry.portfolio_heat_pct.warn"],
            "gauge_pct": _progress(heat_pct, cap), "over_cap": heat_pct > cap}
