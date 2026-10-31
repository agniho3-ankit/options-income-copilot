import pytest

from options_copilot.indicators import iv_rank_proxy, realized_vol, rsi, sma, support_resistance, swing_pivots


def test_sma_and_short_input():
    assert sma([1, 2, 3, 4], 2) == 3.5
    assert sma([1], 2) is None


def test_rsi_extremes_and_flat():
    assert rsi(list(range(1, 30))) == 100.0
    assert rsi(list(range(30, 1, -1))) == pytest.approx(0.0)
    assert rsi([5.0] * 20) == 50.0
    assert rsi([1, 2]) is None


def test_rsi_known_mixed_series_is_between():
    closes = [44, 44.3, 44.1, 43.6, 44.3, 44.8, 45.1, 45.4, 45.8, 46.1, 45.9, 46.2, 45.6, 46.3, 46.3, 46.0]
    assert 50 < rsi(closes) < 80


def test_realized_vol_zero_for_constant_growth():
    closes = [100 * 1.01 ** i for i in range(40)]
    assert realized_vol(closes, 30) == pytest.approx(0.0, abs=1e-9)
    assert realized_vol(closes[:5], 30) is None


def test_swing_pivots_and_nearest_levels():
    highs = [10, 11, 12, 15, 12, 11, 10, 11, 13, 18, 13, 12, 11]
    lows = [9, 8, 7, 9, 10, 9, 5, 8, 9, 12, 10, 9, 8]
    hi, lo = swing_pivots(highs, lows, 2)
    assert hi == [15, 18] and lo == [7, 5]
    lv = support_resistance(highs, lows, price=14, lookback=100, window=2)
    assert lv.support == 7 and lv.resistance == 15
    assert support_resistance(highs, lows, price=20, lookback=100, window=2).resistance is None
    with pytest.raises(ValueError):
        support_resistance([1], [1, 2], 1, 10, 1)


def test_iv_rank_proxy_mapping():
    assert iv_rank_proxy(0.30, 0.40) == 0.0
    assert iv_rank_proxy(0.30, 0.20) == pytest.approx(100.0)
    assert iv_rank_proxy(0.60, 0.20) == 100.0
    assert iv_rank_proxy(0.225, 0.20) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        iv_rank_proxy(0.3, 0)
