import pytest

from options_copilot.screener import ScreenInputs, fundamentals_from_facts, screen
from options_copilot.verifier import InsiderActivity


def inputs(**kw):
    base = dict(ticker="MRVL", price=100.0, market_cap=2e11, sector="Technology", country="United States",
                analyst_buy_ratio=0.9, analyst_count=30, analyst_target=120.0, sma200=90.0,
                net_income_positive=True, revenue_growth=0.1, insider=InsiderActivity(0, 0, 5))
    base.update(kw)
    return ScreenInputs(**base)


def test_perfect_score_passes(pb):
    c = screen(inputs(), pb)
    assert c.quality_score == 10.0 and c.passed_screen and c.tier == "tier_1"


@pytest.mark.parametrize("kw, component, value", [
    ({"analyst_buy_ratio": 0.4}, "analyst", 1.25), ({"analyst_count": 9}, "analyst", 0.0),
    ({"revenue_growth": -0.05}, "fundamentals", 1.25), ({"net_income_positive": None}, "fundamentals", 1.25),
    ({"price": 88.0}, "trend", 1.25), ({"price": 80.0}, "trend", 0.0),
    ({"insider": InsiderActivity(0, 10)}, "insider", 1.25), ({"insider": InsiderActivity(10, 21)}, "insider", 0.0),
    ({"insider": None}, "insider", 0.0),
])
def test_components(pb, kw, component, value):
    assert screen(inputs(**kw), pb).score_components[component] == value


def test_gates_and_missing_data_fail(pb):
    assert screen(inputs(sector="Communication Services"), pb).passed_screen   # added for META
    c = screen(inputs(sector="Energy"), pb)
    assert not c.passed_screen and any("sector" in r for r in c.reasons)
    assert not screen(inputs(market_cap=None), pb).passed_screen
    assert not screen(inputs(ticker="TICKER_A"), pb).passed_screen
    assert not screen(inputs(reverse_split_recent=True), pb).passed_screen
    low = screen(inputs(analyst_buy_ratio=None, sma200=None), pb)
    assert low.quality_score == 5.0 and not low.passed_screen


def test_fundamentals_from_xbrl_facts():
    def series(*pairs):
        return {"units": {"USD": [{"frame": f, "val": v} for f, v in pairs]}}
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": series(("CY2024", 5.0), ("CY2025", 7.0), ("CY2025Q4", -1.0)),
        "Revenues": series(("CY2015", 10.0), ("CY2016", 20.0)),            # stale tag
        "RevenueFromContractWithCustomerExcludingAssessedTax": series(("CY2024", 100.0), ("CY2025", 110.0)),
    }}}
    ok, growth = fundamentals_from_facts(facts)
    assert ok is True and growth == pytest.approx(0.10)
    assert fundamentals_from_facts({}) == (None, None)
