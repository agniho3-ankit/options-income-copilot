"""The mask layer must not let raw account values or IDs through."""
import json

from options_copilot.mask import Masker, find_leaks
from options_copilot.models import to_jsonable

PV = 123_456.78


def sample():
    return {"position_id": "SYN-778899", "brief_id": "B-1", "account_number": "5550001234",
            "ticker": "TICKER_A", "strike": 95.0, "spot": 101.25, "delta": -0.22,
            "credit": 260.0, "max_loss": 1240.0, "collateral": 1240.0, "pnl": -3456.7, "contracts": 3,
            "portfolio_value": PV, "open_max_losses": [9876.5, 4321.0],
            "notes": "Account 5550001234 is down $3,456.70 today; premium was $2.60 at strike 95.",
            "nested": [{"close_cost": 1999.99, "order_id": 42}]}


def test_account_values_become_percentages_and_ids_become_tokens():
    m = Masker(PV)
    out = m.value(sample())
    assert out["credit"] == "0.21% of portfolio" and out["portfolio_value"] == "100.00% of portfolio"
    assert out["contracts"] == "masked"
    assert out["position_id"].startswith("position_") and out["position_id"] == m.token("SYN-778899", "position")
    assert out["nested"][0]["order_id"].startswith("order_")
    assert out["open_max_losses"][0].endswith("% of portfolio")


def test_public_market_data_passes_through():
    out = Masker(PV).value(sample())
    assert (out["ticker"], out["strike"], out["spot"], out["delta"]) == ("TICKER_A", 95.0, 101.25, -0.22)
    assert "$2.60" in out["notes"] and "strike 95" in out["notes"]


def test_no_raw_dollar_amount_or_id_survives():
    raw = sample()
    text = json.dumps(Masker(PV).value(raw))
    secrets = [PV, 260.0, 1240.0, 3456.7, 9876.5, 4321.0, 1999.99, "SYN-778899", "5550001234", "B-1"]
    assert find_leaks(text, secrets) == []
    assert find_leaks(json.dumps(raw), secrets) != []   # the checker itself does detect leaks


def test_free_text_scrubbing():
    m = Masker(PV)
    assert m.text("lost $12,500 and $1.2M on acct 99887766") == "lost [amount masked] and [amount masked] on acct [id masked]"
    assert m.text("bid $7.10, strike 485, 45 DTE") == "bid $7.10, strike 485, 45 DTE"
    assert m.text("about $50k or $3 million") == "about [amount masked] or [amount masked]"


def test_tokens_are_stable_salted_and_repr_is_blank():
    assert Masker(PV).token("SYN-1") == Masker(1.0).token("SYN-1")
    assert Masker(PV, salt=b"other").token("SYN-1") != Masker(PV).token("SYN-1")
    assert str(PV) not in repr(Masker(PV))


def test_masks_dataclasses_via_to_jsonable():
    from options_copilot.models import TradeNumbers
    n = TradeNumbers(credit=500, max_loss=1000, breakevens=(90.0,), pop=0.7, risk_pct_of_portfolio=1.0,
                     collateral=1000)
    out = Masker(100_000).value(to_jsonable(n))
    assert out["credit"] == "0.50% of portfolio" and out["breakevens"] == [90.0]
