import pandas as pd

from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.strategy import generate_target_positions


def test_generate_target_positions_applies_trend_and_vol_filters():
    prices = pd.Series([100, 101, 102, 103, 104, 110, 95, 96, 97, 98], dtype=float)
    cfg = BacktestConfig(trend_window=3, vol_window=2, max_volatility=0.04)

    positions = generate_target_positions(prices, cfg)

    assert positions.iloc[0] == 0.0
    assert positions.iloc[4] == 1.0
    assert positions.iloc[6] == 0.0
