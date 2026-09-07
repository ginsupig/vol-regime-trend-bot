from vol_regime_trend_bot.strategy import StrategyConfig, generate_signals
import pytest


def test_generate_signals_identifies_low_vol_uptrend() -> None:
    closes = [100.0 + (i * 0.1) for i in range(220)]
    config = StrategyConfig(trend_window=20, vol_window=10, regime_window=30)

    signals = generate_signals(closes, config)

    assert len(signals) == len(closes)
    assert all(sig in (-1, 0, 1) for sig in signals)
    assert signals[-1] == 1


def test_generate_signals_rejects_non_positive_close() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_signals([100.0, 0.0, 101.0], StrategyConfig(trend_window=2, vol_window=2, regime_window=2))
