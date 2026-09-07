from vol_regime_trend_bot.config import StrategyConfig
from vol_regime_trend_bot.strategy import generate_raw_signals


def test_generate_raw_signals_in_risk_on_trend():
    prices = [100, 101, 102, 103, 104, 105, 106]
    returns = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
    config = StrategyConfig(vol_window=3, trend_window=3, vol_threshold=0.02)

    signals = generate_raw_signals(prices, returns, config)

    assert len(signals) == len(prices) - 1
    assert signals[-1] == 1.0


def test_generate_raw_signals_uses_prior_returns_for_vol_regime():
    prices = [100, 100, 100]
    returns = [0.0, 1.0]
    config = StrategyConfig(vol_window=2, trend_window=1, vol_threshold=0.1)

    signals = generate_raw_signals(prices, returns, config)

    assert signals == [1.0, 1.0]
