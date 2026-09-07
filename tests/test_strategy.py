import pandas as pd

from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.signals import build_signal_pipeline
from vol_regime_trend_bot.strategy import generate_target_positions


def test_generate_target_positions_applies_trend_and_vol_filters():
    prices = pd.Series([100, 101, 102, 103, 104, 110, 95, 96, 97, 98], dtype=float)
    cfg = BacktestConfig(trend_window=3, vol_window=2, max_volatility=0.04)

    positions = generate_target_positions(prices, cfg)

    assert positions.iloc[0] == 0.0
    assert positions.iloc[4] == 1.0
    assert positions.iloc[6] == 0.0


def test_signal_pipeline_is_modular_and_no_lookahead():
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 120], dtype=float)
    cfg = BacktestConfig(trend_window=3, vol_window=3, max_volatility=0.2)

    base = build_signal_pipeline(prices, cfg)
    shocked = prices.copy()
    shocked.iloc[-1] = 200
    shocked_pipeline = build_signal_pipeline(shocked, cfg)

    assert list(base.columns) == ["trend_signal", "regime_signal", "realized_vol", "raw_signal"]
    pd.testing.assert_series_equal(base["raw_signal"].iloc[:-1], shocked_pipeline["raw_signal"].iloc[:-1])
