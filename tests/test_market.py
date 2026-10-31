from datetime import date

from options_copilot.market import Market, pick_expiry
from options_copilot.models import OptionType

AS_OF = date(2026, 10, 30)


def test_pick_expiry_prefers_target_and_before_earnings():
    exps = [date(2026, 11, 20), date(2026, 12, 11), date(2026, 12, 18), date(2027, 1, 15), date(2027, 2, 19)]
    assert pick_expiry(exps, AS_OF, 30, 60, 45, None) == date(2026, 12, 11)
    assert pick_expiry(exps, AS_OF, 30, 60, 45, date(2026, 12, 15)) == date(2026, 12, 11)
    assert pick_expiry(exps, AS_OF, 30, 60, 45, date(2026, 12, 1)) == date(2026, 12, 11)  # none before: still pick
    assert pick_expiry([date(2026, 11, 6)], AS_OF, 30, 60, 45, None) is None


class _DF:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        return self.rows


class _Chain:
    puts = _DF([{"strike": 90.0, "bid": 1.0, "ask": 1.2, "impliedVolatility": 0.3, "openInterest": 600, "volume": 1200},
                {"strike": 85.0, "bid": 0.5, "ask": 0.4, "impliedVolatility": 0.3, "openInterest": 10, "volume": 1}])
    calls = _DF([{"strike": 110.0, "bid": 0.0, "ask": 0.1, "impliedVolatility": 0.00001, "openInterest": float("nan"),
                  "volume": None}])


class _Ticker:
    def __init__(self, t):
        pass

    def option_chain(self, exp):
        return _Chain()


class _YF:
    Ticker = _Ticker


def test_chain_parsing_skips_crossed_quotes_and_bad_iv():
    quotes, flow = Market(_YF).chain("TICKER_A", date(2026, 12, 18))
    assert [(q.option_type, q.strike) for q in quotes] == [(OptionType.PUT, 90.0), (OptionType.CALL, 110.0)]
    assert quotes[0].iv == 0.3 and quotes[0].open_interest == 600 and flow == 2.0
    assert quotes[1].iv is None and quotes[1].open_interest is None
