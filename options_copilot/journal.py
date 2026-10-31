"""The trade journal (playbook Skill #7): one row per synthetic position, plus summary metrics.

P&L % is a percentage of the entry credit (as in the playbook's example row);
return on risk (P&L / max loss) is kept in its own column. Annualized % = P&L % x 365 / days held.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from .models import OptionType, Position, Side


def open_row(pos: Position, earnings: date | None, notes: str = "") -> dict[str, Any]:
    def strike(t: OptionType, s: Side):
        return next((g.strike for g in pos.legs if g.option_type is t and g.side is s), None)
    if earnings is None:
        note = "unknown"
    else:
        note = f"No ({earnings.isoformat()})" if earnings > pos.expiry else f"Yes ({earnings.isoformat()})"
    return {"position_id": pos.position_id, "date_opened": pos.opened_on.isoformat(), "ticker": pos.ticker,
            "strategy": pos.strategy.value, "short_put": strike(OptionType.PUT, Side.SHORT),
            "short_call": strike(OptionType.CALL, Side.SHORT), "long_put": strike(OptionType.PUT, Side.LONG),
            "long_call": strike(OptionType.CALL, Side.LONG), "dte": (pos.expiry - pos.opened_on).days,
            "entry_credit": round(pos.entry_credit * pos.contracts, 2),
            "entry_max_loss": round(pos.entry_max_loss * pos.contracts, 2), "earnings_note": note,
            "notes": notes, "synthetic": 1}


def close_fields(pos: Position, close_cost: float, closed_on: date, exit_rule: str) -> dict[str, Any]:
    """Realized figures. `close_cost` is the debit per contract (negative = net credit received)."""
    pnl = (pos.entry_credit - close_cost) * pos.contracts
    days = max((closed_on - pos.opened_on).days, 0)
    pnl_pct = (pos.entry_credit - close_cost) / pos.entry_credit * 100
    return {"date_closed": closed_on.isoformat(), "close_cost": round(close_cost * pos.contracts, 2),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "return_on_risk_pct": round(pnl / (pos.entry_max_loss * pos.contracts) * 100, 2),
            "days_held": days, "annualized_pct": round(pnl_pct * 365 / max(days, 1), 1), "exit_rule": exit_rule}


def metrics(rows: list[dict[str, Any]], month: str | None = None) -> dict[str, Any]:
    """Skill #7 summary rows. `month` is YYYY-MM (by date opened); None means all rows."""
    rows = [r for r in rows if month is None or r["date_opened"].startswith(month)]
    closed = [r for r in rows if r["date_closed"]]
    out: dict[str, Any] = {"month": month or "all", "total_trades": len(rows), "closed_trades": len(closed),
                           "active_trades": len(rows) - len(closed), "total_pnl": round(sum(r["pnl"] for r in closed), 2),
                           "win_rate_pct": None, "avg_days_held": None, "avg_return_pct": None,
                           "annualized_return_pct": None, "synthetic": True}
    if closed:
        days = sum(r["days_held"] for r in closed) / len(closed)
        avg = sum(r["pnl_pct"] for r in closed) / len(closed)
        out.update(win_rate_pct=round(sum(r["pnl"] > 0 for r in closed) / len(closed) * 100, 1),
                   avg_days_held=round(days, 1), avg_return_pct=round(avg, 2),
                   annualized_return_pct=round(avg * 365 / max(days, 1), 1))
    return out


def render(rows: list[dict[str, Any]], m: dict[str, Any]) -> str:
    def f(v, spec=""):
        return "-" if v is None else format(v, spec) if spec else str(v)
    head = (f"{'opened':<11}{'ticker':<7}{'strategy':<19}{'SP':>7}{'SC':>7}{'LP':>7}{'LC':>7}{'DTE':>5}"
            f"{'credit':>9}{'max loss':>10}  {'earnings':<18}{'closed':<11}{'close':>8}{'P&L':>9}{'P&L%':>8}"
            f"{'days':>5}{'ann.%':>8}  exit rule / notes")
    lines = ["TRADES LOG (synthetic paper positions)", head]
    for r in rows:
        lines.append(f"{r['date_opened']:<11}{r['ticker']:<7}{r['strategy']:<19}{f(r['short_put'], 'g'):>7}"
                     f"{f(r['short_call'], 'g'):>7}{f(r['long_put'], 'g'):>7}{f(r['long_call'], 'g'):>7}{f(r['dte']):>5}"
                     f"{f(r['entry_credit'], '.2f'):>9}{f(r['entry_max_loss'], '.2f'):>10}  {f(r['earnings_note']):<18}"
                     f"{f(r['date_closed']):<11}{f(r['close_cost'], '.2f'):>8}{f(r['pnl'], '.2f'):>9}"
                     f"{f(r['pnl_pct'], '.1f'):>8}{f(r['days_held']):>5}{f(r['annualized_pct'], '.0f'):>8}  "
                     f"{r['exit_rule'] or ''} {r['notes'] or ''}".rstrip())
    lines += ["", f"TRACKING METRICS ({m['month']}): total {m['total_trades']}, closed {m['closed_trades']}, "
              f"active {m['active_trades']}, total P&L {m['total_pnl']:.2f}, win rate {f(m['win_rate_pct'])}%, "
              f"avg holding {f(m['avg_days_held'])} days, avg return {f(m['avg_return_pct'])}% of credit, "
              f"annualized {f(m['annualized_return_pct'])}%"]
    return "\n".join(lines)
