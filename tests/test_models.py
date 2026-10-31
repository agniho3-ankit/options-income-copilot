"""Model invariants. Fixtures use made-up tickers (TICKER_A) and synthetic values."""
import json
from datetime import date

import pytest

from options_copilot.models import (
    CheckResult, JournalEntry, OptionLeg, OptionType, Position, Reflection, ReflectionTrigger,
    RetrievedChunk, Side, Strategy, TradeProposal, Verdict, VerifierResult, from_yyyymmdd,
    to_jsonable, yyyymmdd,
)

EXPIRY = date(2026, 12, 18)


def leg(option_type, side, strike, bid=1.0, ask=1.1, expiry=EXPIRY, delta=None):
    return OptionLeg(OptionType(option_type), Side(side), strike, expiry, bid, ask, delta=delta)


def condor_legs():
    return (leg("put", "short", 95, delta=-0.22), leg("put", "long", 92),
            leg("call", "short", 105, delta=0.21), leg("call", "long", 108))


# --- Verdict: missing evidence is never safe ---

def test_no_verdicts_means_insufficient_evidence_not_clear():
    assert Verdict.most_severe([]) is Verdict.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("verdicts,expected", [
    ([Verdict.CLEAR, Verdict.CLEAR], Verdict.CLEAR),
    ([Verdict.CLEAR, Verdict.CAUTION], Verdict.CAUTION),
    ([Verdict.CAUTION, Verdict.INSUFFICIENT_EVIDENCE], Verdict.INSUFFICIENT_EVIDENCE),
    ([Verdict.INSUFFICIENT_EVIDENCE, Verdict.BLOCK, Verdict.CLEAR], Verdict.BLOCK),
])
def test_most_severe_verdict_wins(verdicts, expected):
    assert Verdict.most_severe(verdicts) is expected


def test_verifier_passes_only_on_positive_evidence():
    ok = CheckResult("pop", Verdict.CLEAR, "PoP 70% >= 65%")
    warn = CheckResult("rsi", Verdict.CAUTION, "RSI 72", hard=False)
    missing = CheckResult("landmine", Verdict.INSUFFICIENT_EVIDENCE, "no filings in window")
    blocked = CheckResult("heat", Verdict.BLOCK, "heat 31% > 30%")

    assert VerifierResult((ok, warn)).passed
    assert not VerifierResult(()).passed
    assert not VerifierResult((ok, missing)).passed
    result = VerifierResult((ok, warn, missing, blocked))
    assert result.verdict is Verdict.BLOCK and not result.passed
    assert [c.name for c in result.failures] == ["landmine", "heat"]


# --- OptionLeg ---

def test_leg_mid_is_average_of_bid_and_ask():
    assert leg("put", "short", 95, bid=1.00, ask=1.10).mid == pytest.approx(1.05)


@pytest.mark.parametrize("kwargs", [
    dict(strike=0), dict(bid=1.2, ask=1.1), dict(bid=-0.1, ask=0.1), dict(delta=-1.5),
])
def test_leg_rejects_impossible_values(kwargs):
    base = dict(option_type="put", side="short", strike=95, bid=1.0, ask=1.1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        leg(**base)


# --- TradeProposal shape ---

def test_iron_condor_with_four_matching_legs_is_valid():
    p = TradeProposal("TICKER_A", Strategy.IRON_CONDOR, condor_legs())
    assert p.expiry == EXPIRY
    assert p.leg(OptionType.CALL, Side.SHORT).strike == 105


@pytest.mark.parametrize("strategy,legs", [
    (Strategy.IRON_CONDOR, condor_legs()[:3]),
    (Strategy.PUT_CREDIT_SPREAD, (leg("put", "short", 95), leg("put", "short", 92))),
    (Strategy.CALL_CREDIT_SPREAD, (leg("put", "short", 95), leg("put", "long", 92))),
    (Strategy.CASH_SECURED_PUT, (leg("put", "short", 95), leg("put", "long", 92))),
])
def test_proposal_rejects_wrong_leg_shape(strategy, legs):
    with pytest.raises(ValueError, match="needs legs"):
        TradeProposal("TICKER_A", strategy, legs)


def test_proposal_rejects_mixed_expiries_and_zero_contracts():
    mixed = (leg("put", "short", 95), leg("put", "long", 92, expiry=date(2027, 1, 15)))
    with pytest.raises(ValueError, match="one expiry"):
        TradeProposal("TICKER_A", Strategy.PUT_CREDIT_SPREAD, mixed)
    with pytest.raises(ValueError, match="contracts"):
        TradeProposal("TICKER_A", Strategy.PUT_CREDIT_SPREAD, mixed[:1] + (leg("put", "long", 92),), contracts=0)


# --- RetrievedChunk dates and citations ---

def test_yyyymmdd_round_trip():
    assert yyyymmdd(date(2026, 10, 9)) == 20261009
    assert from_yyyymmdd(20261009) == date(2026, 10, 9)


@pytest.mark.parametrize("bad", [2026109, 20261332, 20260230, 202610090])
def test_bad_filed_dates_are_rejected(bad):
    with pytest.raises(ValueError):
        RetrievedChunk("text", "TICKER_A", "8-K", bad, "1.01", "https://www.sec.gov/x", 0)


def test_chunk_citation_has_form_date_item_url():
    chunk = RetrievedChunk("text", "TICKER_A", "8-K", 20261020, "1.01", "https://www.sec.gov/x", 3)
    assert chunk.citation() == {"form": "8-K", "date": "2026-10-20", "item": "1.01",
                                "url": "https://www.sec.gov/x"}


# --- Synthetic labelling and serialization ---

def test_holding_records_default_to_synthetic():
    pos = Position("pos-1", "TICKER_A", Strategy.IRON_CONDOR, condor_legs(), date(2026, 11, 2), 1.8, 1.2)
    row = JournalEntry(date(2026, 11, 2), "TICKER_A", Strategy.IRON_CONDOR, 45, 1.8, 1.2)
    lesson = Reflection("Check the earnings date before building strikes.", ReflectionTrigger.VERIFIER)
    assert pos.synthetic and row.synthetic and lesson.synthetic
    assert lesson.ticker == "ANY"


def test_to_jsonable_produces_plain_json():
    proposal = TradeProposal("TICKER_A", Strategy.IRON_CONDOR, condor_legs(), rationale="synthetic")
    data = to_jsonable(proposal)
    json.dumps(data)  # must not raise
    assert data["strategy"] == "iron_condor"
    assert data["legs"][0]["expiry"] == "2026-12-18"
    assert data["legs"][0]["option_type"] == "put"
