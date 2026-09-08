import pandas as pd

from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.signals import (
    apply_signal_confirmation,
    build_signal_pipeline,
    compute_realized_volatility,
    compute_volatility_change_signal,
)
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

    assert list(base.columns) == [
        "trend_signal",
        "regime_signal",
        "tradable_regime_signal",
        "volatility_change_signal",
        "realized_vol",
        "raw_signal",
    ]
    pd.testing.assert_series_equal(base["raw_signal"].iloc[:-1], shocked_pipeline["raw_signal"].iloc[:-1])


def test_hysteresis_keeps_tradable_regime_on_through_small_overshoots():
    prices = pd.Series([100, 101, 100, 101, 100, 101, 100, 101, 100, 103], dtype=float)
    cfg = BacktestConfig(vol_window=2, max_volatility=0.02, vol_hysteresis_buffer=0.5)

    frame = build_signal_pipeline(prices, cfg)

    assert frame["regime_signal"].iloc[8] == 1.0
    assert frame["regime_signal"].iloc[9] == 0.0
    assert frame["tradable_regime_signal"].iloc[9] == 1.0
    assert frame["raw_signal"].iloc[9] == 1.0


def test_signal_confirmation_delays_regime_state_flips():
    signal = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=float)

    got = apply_signal_confirmation(signal, confirmation_bars=2)

    assert got.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_falling_volatility_gate_is_opt_in_and_reported():
    prices = pd.Series([100, 103, 100, 104, 100, 105, 100, 106], dtype=float)
    base_cfg = BacktestConfig(vol_window=2, max_volatility=1.0)
    gated_cfg = BacktestConfig(
        vol_window=2,
        max_volatility=1.0,
        require_falling_volatility=True,
        volatility_change_window=1,
    )

    base = build_signal_pipeline(prices, base_cfg)
    gated = build_signal_pipeline(prices, gated_cfg)
    realized_vol = compute_realized_volatility(prices, 2)
    expected_change = compute_volatility_change_signal(realized_vol, 1).astype(float)

    pd.testing.assert_series_equal(gated["volatility_change_signal"], expected_change)
    assert (gated["raw_signal"] <= base["raw_signal"]).all()
    assert gated["raw_signal"].sum() < base["raw_signal"].sum()
