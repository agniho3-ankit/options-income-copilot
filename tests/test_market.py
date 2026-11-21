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


def test_history_dates_are_plain_dates_even_from_timestamps():
    """yfinance indexes by pandas Timestamps (datetime subclasses); History.dates must hold dates."""
    from datetime import datetime

    class _Hist(dict):
        index = [datetime(2026, 11, 19, 0, 0), datetime(2026, 11, 20, 0, 0)]

    class _T:
        def __init__(self, t):
            pass

        def history(self, period, auto_adjust):
            return _Hist(Close=[1.0, 2.0], High=[1.5, 2.5], Low=[0.5, 1.5])

    class _Y:
        Ticker = _T

    h = Market(_Y).history("TICKER_A")
    assert h.dates == [date(2026, 11, 19), date(2026, 11, 20)]
    assert all(type(d) is date for d in h.dates) and h.dates[-1] <= date(2026, 11, 20)
