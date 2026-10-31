"""Domain models shared by the tools, verifier, agent, and journal.

Numbers on these objects always come from deterministic calculators, never from the
LLM. Records that could describe a holding (Position, JournalEntry, Reflection) carry a
`synthetic` flag, which defaults to True because this build has no real account data.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Outcome of a check, landmine search, or the whole verifier.

    Missing evidence is never safe: INSUFFICIENT_EVIDENCE ranks above CAUTION and
    escalates to the owner. Only BLOCK is more severe.
    """
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BLOCK = "BLOCK"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    @classmethod
    def most_severe(cls, verdicts: list[Verdict]) -> Verdict:
        """Combine verdicts. No verdicts at all means no evidence, not CLEAR."""
        if not verdicts:
            return cls.INSUFFICIENT_EVIDENCE
        return max(verdicts, key=lambda v: v.severity)


_SEVERITY = {Verdict.CLEAR: 0, Verdict.CAUTION: 1, Verdict.INSUFFICIENT_EVIDENCE: 2, Verdict.BLOCK: 3}


class Strategy(str, Enum):
    PUT_CREDIT_SPREAD = "put_credit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    IRON_CONDOR = "iron_condor"
    CASH_SECURED_PUT = "cash_secured_put"


class OptionType(str, Enum):
    PUT = "put"
    CALL = "call"


class Side(str, Enum):
    SHORT = "short"
    LONG = "long"


class BriefKind(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    ROLL = "roll"


class BriefStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class ReflectionTrigger(str, Enum):
    REJECTION = "rejection"
    TRADE_CLOSE = "trade_close"
    VERIFIER = "verifier"


# Expected legs per strategy: (option type, side) pairs, order-independent.
_LEG_SHAPES: dict[Strategy, list[tuple[OptionType, Side]]] = {
    Strategy.PUT_CREDIT_SPREAD: [(OptionType.PUT, Side.SHORT), (OptionType.PUT, Side.LONG)],
    Strategy.CALL_CREDIT_SPREAD: [(OptionType.CALL, Side.SHORT), (OptionType.CALL, Side.LONG)],
    Strategy.IRON_CONDOR: [(OptionType.PUT, Side.SHORT), (OptionType.PUT, Side.LONG),
                           (OptionType.CALL, Side.SHORT), (OptionType.CALL, Side.LONG)],
    Strategy.CASH_SECURED_PUT: [(OptionType.PUT, Side.SHORT)],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def yyyymmdd(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def from_yyyymmdd(value: int) -> date:
    _require(10_000_000 <= value <= 99_991_231, f"filed_date must be yyyymmdd, got {value}")
    return date(value // 10000, value // 100 % 100, value % 100)  # raises on impossible dates


@dataclass(frozen=True)
class OptionLeg:
    option_type: OptionType
    side: Side
    strike: float
    expiry: date
    bid: float
    ask: float
    delta: float | None = None
    iv: float | None = None
    open_interest: int | None = None

    def __post_init__(self) -> None:
        _require(self.strike > 0, f"strike must be positive, got {self.strike}")
        _require(0 <= self.bid <= self.ask, f"need 0 <= bid <= ask, got {self.bid}/{self.ask}")
        if self.delta is not None:
            _require(-1 <= self.delta <= 1, f"delta must be in [-1, 1], got {self.delta}")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class TradeNumbers:
    """Calculator output for a proposal, per contract unless noted."""
    credit: float
    max_loss: float
    breakevens: tuple[float, ...]
    pop: float
    risk_pct_of_portfolio: float
    collateral: float


@dataclass(frozen=True)
class TradeProposal:
    ticker: str
    strategy: Strategy
    legs: tuple[OptionLeg, ...]
    contracts: int = 1
    rationale: str = ""
    numbers: TradeNumbers | None = None

    def __post_init__(self) -> None:
        _require(self.contracts >= 1, "contracts must be at least 1")
        shape = sorted((leg.option_type.value, leg.side.value) for leg in self.legs)
        expected = sorted((t.value, s.value) for t, s in _LEG_SHAPES[self.strategy])
        _require(shape == expected, f"{self.strategy.value} needs legs {expected}, got {shape}")
        _require(len({leg.expiry for leg in self.legs}) == 1, "all legs must share one expiry")

    @property
    def expiry(self) -> date:
        return self.legs[0].expiry

    def leg(self, option_type: OptionType, side: Side) -> OptionLeg | None:
        return next((g for g in self.legs if g.option_type is option_type and g.side is side), None)


@dataclass(frozen=True)
class Candidate:
    ticker: str
    price: float
    tier: str | None
    quality_score: float | None = None
    score_components: dict[str, float] = field(default_factory=dict)
    passed_screen: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckResult:
    """One verifier check. `hard` checks BLOCK on failure; advisory ones add CAUTION."""
    name: str
    verdict: Verdict
    reason: str
    hard: bool = True


@dataclass(frozen=True)
class VerifierResult:
    checks: tuple[CheckResult, ...]
    playbook_sha256: str = ""

    @property
    def verdict(self) -> Verdict:
        return Verdict.most_severe([c.verdict for c in self.checks])

    @property
    def passed(self) -> bool:
        """A trade passes only on positive evidence: no BLOCK and no missing evidence."""
        return self.verdict in (Verdict.CLEAR, Verdict.CAUTION)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.verdict in (Verdict.BLOCK, Verdict.INSUFFICIENT_EVIDENCE)]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    ticker: str
    form: str
    filed_date: int  # yyyymmdd, so Chroma range filters work
    item: str
    source_url: str
    chunk_idx: int
    score: float | None = None

    def __post_init__(self) -> None:
        from_yyyymmdd(self.filed_date)

    def citation(self) -> dict[str, Any]:
        return {"form": self.form, "date": from_yyyymmdd(self.filed_date).isoformat(),
                "item": self.item, "url": self.source_url}


@dataclass
class Brief:
    brief_id: str
    kind: BriefKind
    ticker: str
    summary: str
    proposal: TradeProposal | None = None
    verifier: VerifierResult | None = None
    citations: list[RetrievedChunk] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    status: BriefStatus = BriefStatus.PENDING
    decision_reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    position_id: str
    ticker: str
    strategy: Strategy
    legs: tuple[OptionLeg, ...]
    opened_on: date
    entry_credit: float
    entry_max_loss: float
    contracts: int = 1
    status: PositionStatus = PositionStatus.OPEN
    entry_spot: float | None = None   # underlying price at entry (Rule 4)
    entry_iv: float | None = None     # IV used at entry (Rule 4 sigma)
    synthetic: bool = True

    @property
    def expiry(self) -> date:
        return self.legs[0].expiry


@dataclass
class JournalEntry:
    """One row of the trade log (playbook Skill #7). Dollar fields are per position."""
    date_opened: date
    ticker: str
    strategy: Strategy
    dte: int
    entry_credit: float
    entry_max_loss: float
    short_put: float | None = None
    short_call: float | None = None
    long_put: float | None = None
    long_call: float | None = None
    earnings_note: str = ""
    date_closed: date | None = None
    close_cost: float | None = None  # debit paid to close (the playbook's "Close Credit" column)
    pnl: float | None = None
    pnl_pct: float | None = None
    days_held: int | None = None
    annualized_pct: float | None = None
    notes: str = ""
    synthetic: bool = True


@dataclass
class Reflection:
    lesson: str
    trigger: ReflectionTrigger
    ticker: str = "ANY"
    tags: list[str] = field(default_factory=list)
    created_on: date = field(default_factory=date.today)
    synthetic: bool = True


def leg_from_dict(d: dict[str, Any]) -> OptionLeg:
    return OptionLeg(OptionType(d["option_type"]), Side(d["side"]), float(d["strike"]),
                     date.fromisoformat(d["expiry"]), float(d["bid"]), float(d["ask"]), d.get("delta"),
                     d.get("iv"), d.get("open_interest"))


def position_from_dict(d: dict[str, Any]) -> Position:
    return Position(d["position_id"], d["ticker"], Strategy(d["strategy"]),
                    tuple(leg_from_dict(g) for g in d["legs"]), date.fromisoformat(d["opened_on"]),
                    float(d["entry_credit"]), float(d["entry_max_loss"]), int(d.get("contracts", 1)),
                    PositionStatus(d.get("status", "open")), d.get("entry_spot"), d.get("entry_iv"),
                    bool(d.get("synthetic", True)))


def to_jsonable(obj: Any) -> Any:
    """Convert models (and nested enums, dates, tuples) into plain JSON-ready values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

