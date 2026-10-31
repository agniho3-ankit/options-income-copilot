"""PaperBroker: turns approved order tickets into synthetic positions, and closes them.

Nothing here talks to a broker or places an order. Fills are simulated against the current
public chain with limit-order semantics: an opening ticket fills at its limit credit when the
live mid credit is at least the limit; a closing ticket fills at its limit debit when the live
mid debit is at most the limit. Otherwise the ticket stays working. A roll fills only if both
its close and its open would fill, so a roll is never half done. Expired positions settle at
intrinsic value. Every position and journal row is synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from .calculators import MULTIPLIER, CalcError, close_cost_per_contract, intrinsic_close_cost
from .db import Database, new_id
from .journal import close_fields, open_row
from .models import (OptionLeg, Position, PositionStatus, ReflectionTrigger, Side, Strategy, leg_from_dict,
                     position_from_dict, to_jsonable)

EPS = 1e-9


@dataclass
class Fill:
    ticket_id: str
    status: str            # filled, working, or error
    message: str
    position_id: str | None = None


class PaperBroker:
    def __init__(self, db: Database, sources: Any, as_of: date, *, llm=None, store=None, masker=None):
        self.db, self.sources, self.as_of = db, sources, as_of
        self.llm, self.store, self.masker = llm, store, masker
        self.last_lesson = None

    # quotes
    def _quotes(self, ticker: str, legs: tuple[OptionLeg, ...]) -> dict[tuple[str, str, float], Any]:
        out = {}
        for expiry in sorted({g.expiry for g in legs}):
            quotes, _ = self.sources.chain(ticker, expiry)
            by = {(q.option_type.value, q.strike): q for q in quotes}
            for g in legs:
                if g.expiry != expiry:
                    continue
                q = by.get((g.option_type.value, g.strike))
                if q is None:
                    raise CalcError(f"no current quote for {g.option_type.value} {g.strike:g} {expiry}")
                out[(g.option_type.value, g.side.value, g.strike)] = q
        return out

    def _mids(self, ticker, legs):
        return {k: (q.bid + q.ask) / 2 for k, q in self._quotes(ticker, legs).items()}

    @staticmethod
    def _credit_ps(legs, mids) -> float:
        return sum(mids[(g.option_type.value, g.side.value, g.strike)] * (1 if g.side is Side.SHORT else -1)
                   for g in legs)

    # tickets
    def process_all(self) -> list[Fill]:
        return [self.process(t) for t in self.db.tickets("approved")]

    def process(self, ticket: dict) -> Fill:
        kind = ticket["payload"].get("kind", "entry")
        try:
            if kind == "entry":
                return self._entry(ticket)
            if kind == "exit":
                return self._exit(ticket)
            if kind == "roll":
                return self._roll(ticket)
            return Fill(ticket["ticket_id"], "error", f"unknown ticket kind {kind}")
        except (CalcError, ValueError) as exc:
            self.db.update_ticket(ticket["ticket_id"], "approved", last_check=self.as_of.isoformat(),
                                  last_message=str(exc))
            return Fill(ticket["ticket_id"], "working", f"cannot price now: {exc}")

    def _entry_check(self, t: dict) -> tuple[bool, float, tuple[OptionLeg, ...], dict]:
        legs = tuple(leg_from_dict(g) for g in t["legs"])
        quotes = self._quotes(t["ticker"], legs)
        mids = {k: (q.bid + q.ask) / 2 for k, q in quotes.items()}
        return self._credit_ps(legs, mids) + EPS >= t["limit_credit_per_share"], self._credit_ps(legs, mids), legs, quotes

    def _open(self, ticket_id: str, t: dict, legs, quotes, note: str = "") -> Position:
        credit = round(t["limit_credit_per_share"] * MULTIPLIER, 2)
        if "max_loss_at_brief" in t:
            brief_credit, brief_loss = t["credit_at_brief"], t["max_loss_at_brief"]
        else:  # tickets written before Phase 4 carry only the brief id
            n = self.db.brief(t["brief_id"])["payload"]["numbers"]
            brief_credit, brief_loss = n["credit"], n["max_loss"]
        # max loss = width x 100 - credit, so it moves one-for-one with the fill credit
        max_loss = round(brief_loss + brief_credit - credit, 2)
        fresh = tuple(replace(g, bid=quotes[(g.option_type.value, g.side.value, g.strike)].bid,
                              ask=quotes[(g.option_type.value, g.side.value, g.strike)].ask) for g in legs)
        ivs = [quotes[(g.option_type.value, g.side.value, g.strike)].iv for g in legs if g.side is Side.SHORT]
        ivs = [v for v in ivs if v]
        data = self.sources.ticker_data(t["ticker"])
        pos = Position(new_id("pos"), t["ticker"], Strategy(t["strategy"]), fresh, self.as_of, credit, max_loss,
                       int(t.get("contracts", 1)), entry_spot=data.profile.price,
                       entry_iv=sum(ivs) / len(ivs) if ivs else None)
        self.db.save_position(pos.position_id, ticket_id, pos.ticker, "open", self.as_of.isoformat(), to_jsonable(pos))
        self.db.save_journal(open_row(pos, data.profile.earnings_date, note))
        return pos

    def _entry(self, ticket: dict) -> Fill:
        t = ticket["payload"]
        ok, mid, legs, quotes = self._entry_check(t)
        if not ok:
            msg = f"not filled at limit: mid credit {mid:.3f} < limit {t['limit_credit_per_share']:.2f}"
            self.db.update_ticket(ticket["ticket_id"], "approved", last_check=self.as_of.isoformat(), last_message=msg)
            return Fill(ticket["ticket_id"], "working", msg)
        pos = self._open(ticket["ticket_id"], t, legs, quotes, note=f"paper fill of {ticket['ticket_id']}")
        self.db.update_ticket(ticket["ticket_id"], "filled", position_id=pos.position_id, filled_on=self.as_of.isoformat(),
                              fill_credit_per_share=t["limit_credit_per_share"])
        return Fill(ticket["ticket_id"], "filled", f"opened {pos.position_id} at {t['limit_credit_per_share']:.2f} credit",
                    pos.position_id)

    def _close_check(self, c: dict) -> tuple[bool, float, Position]:
        row = self.db.position(c["position_id"])
        if row is None or row["status"] != "open":
            raise ValueError(f"position {c['position_id']} is not open")
        pos = position_from_dict(row["payload"])
        debit_ps = close_cost_per_contract(pos.legs, self._mids(pos.ticker, pos.legs)) / MULTIPLIER
        return debit_ps <= c["limit_debit_per_share"] + EPS, debit_ps, pos

    def _exit(self, ticket: dict) -> Fill:
        c = ticket["payload"]
        ok, debit, pos = self._close_check(c)
        if not ok:
            msg = f"not filled at limit: mid debit {debit:.3f} > limit {c['limit_debit_per_share']:.2f}"
            self.db.update_ticket(ticket["ticket_id"], "approved", last_check=self.as_of.isoformat(), last_message=msg)
            return Fill(ticket["ticket_id"], "working", msg)
        row = self.close_position(pos, c["limit_debit_per_share"] * MULTIPLIER, c.get("exit_rule", "exit"))
        self.db.update_ticket(ticket["ticket_id"], "filled", filled_on=self.as_of.isoformat())
        return Fill(ticket["ticket_id"], "filled", f"closed {pos.position_id}: P&L {row['pnl']:.2f} "
                    f"({row['pnl_pct']:.1f}% of credit)", pos.position_id)

    def _roll(self, ticket: dict) -> Fill:
        p = ticket["payload"]
        ok_close, debit, pos = self._close_check(p["close"])
        ok_open, credit, legs, quotes = self._entry_check(p["open"])
        if not (ok_close and ok_open):
            msg = (f"roll not filled: close mid debit {debit:.3f} vs limit {p['close']['limit_debit_per_share']:.2f}, "
                   f"open mid credit {credit:.3f} vs limit {p['open']['limit_credit_per_share']:.2f}")
            self.db.update_ticket(ticket["ticket_id"], "approved", last_check=self.as_of.isoformat(), last_message=msg)
            return Fill(ticket["ticket_id"], "working", msg)
        self.close_position(pos, p["close"]["limit_debit_per_share"] * MULTIPLIER, p["close"].get("exit_rule", "roll"))
        new = self._open(ticket["ticket_id"], p["open"], legs, quotes, note=f"rolled from {pos.position_id}")
        self.db.update_ticket(ticket["ticket_id"], "filled", position_id=new.position_id, filled_on=self.as_of.isoformat())
        return Fill(ticket["ticket_id"], "filled", f"rolled {pos.position_id} into {new.position_id}", new.position_id)

    # closing and expiry
    def close_position(self, pos: Position, close_cost: float, exit_rule: str) -> dict:
        """Close at `close_cost` per contract, update the journal, and write a Reflexion lesson."""
        fields = close_fields(pos, close_cost, self.as_of, exit_rule)
        closed = replace(pos, status=PositionStatus.CLOSED)
        row = self.db.position(pos.position_id)
        self.db.save_position(pos.position_id, row["ticket_id"] if row else None, pos.ticker, "closed",
                              pos.opened_on.isoformat(), to_jsonable(closed), closed_on=self.as_of.isoformat())
        jr = next((r for r in self.db.journal() if r["position_id"] == pos.position_id), None)
        self.db.save_journal({**(jr or {}), **fields})
        if self.masker is not None:
            from .reflexion import write_lesson
            self.last_lesson = write_lesson(ReflectionTrigger.TRADE_CLOSE, pos.ticker,
                         {"strategy": pos.strategy.value, "exit_rule": exit_rule, "pnl_pct_of_credit": fields["pnl_pct"],
                          "days_held": fields["days_held"], "outcome": "win" if fields["pnl"] > 0 else "loss"},
                         llm=self.llm, store=self.store, masker=self.masker, tags=["trade_close", exit_rule])
        return fields

    def settle_expired(self) -> list[dict]:
        out = []
        for row in self.db.positions("open"):
            pos = position_from_dict(row["payload"])
            if pos.expiry <= self.as_of:
                spot = self.sources.ticker_data(pos.ticker).profile.price
                out.append({"position_id": pos.position_id,
                            **self.close_position(pos, intrinsic_close_cost(pos.legs, spot), "expiry")})
        return out


def open_positions(db: Database) -> list[Position]:
    return [position_from_dict(r["payload"]) for r in db.positions("open")]
