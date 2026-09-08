"""The strategy had no benchmark, so no result could be read as good or bad.

Measured on the survivorship-free PIT S&P panel (628 names incl. delisted,
2016-2026, exposure-matched per cell), the shipped strategy beats a constant
book holding its own mean |position| in only 28.0% of name-eras on max
drawdown, 4.2% on time-averaged drawdown, 21.3% on CAGR and 18.6% on Sharpe.
Removing the trend filter improves all four (54.4 / 12.2 / 22.8 / 28.7),
reproducing the same direction found on 8 assets over 2000-2026.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_regime_trend_bot.backtest import run_backtest
from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.signals import build_signal_pipeline


def series(n: int = 400, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))))


# --------------------------------------------------------------------------
# Trend filter is opt-in
# --------------------------------------------------------------------------


def test_trend_filter_is_off_by_default():
    prices = series()
    frame = build_signal_pipeline(prices, BacktestConfig())

    assert (frame["raw_signal"] == frame["regime_signal"]).all()


def test_trend_filter_when_enabled_intersects_with_the_regime():
    prices = series()
    cfg = BacktestConfig(use_trend_filter=True)
    frame = build_signal_pipeline(prices, cfg)

    expected = (frame["trend_signal"] > 0) & (frame["regime_signal"] > 0)
    assert (frame["raw_signal"] == expected.astype(float)).all()
    assert frame["raw_signal"].sum() < frame["regime_signal"].sum(), "fixture must exercise the filter"


def test_trend_signal_is_still_reported_as_a_diagnostic_when_disabled():
    prices = series()
    frame = build_signal_pipeline(prices, BacktestConfig(use_trend_filter=False))

    assert frame["trend_signal"].isin([0.0, 1.0]).all()
    assert 0 < frame["trend_signal"].sum() < len(frame)


def test_trend_filter_round_trips_through_env_and_config_file(tmp_path, monkeypatch):
    assert BacktestConfig.from_dict({"use_trend_filter": True}).use_trend_filter is True

    path = tmp_path / "cfg.json"
    path.write_text('{"use_trend_filter": true}', encoding="utf-8")
    assert BacktestConfig.from_json_file(path)["use_trend_filter"] is True

    monkeypatch.setenv("VRTB_USE_TREND_FILTER", "false")
    assert BacktestConfig.from_env()["use_trend_filter"] is False
    monkeypatch.setenv("VRTB_USE_TREND_FILTER", "1")
    assert BacktestConfig.from_env()["use_trend_filter"] is True
    monkeypatch.setenv("VRTB_USE_TREND_FILTER", "banana")
    with pytest.raises(ValueError, match="VRTB_USE_TREND_FILTER"):
        BacktestConfig.from_env()


# --------------------------------------------------------------------------
# Matched-exposure benchmark
# --------------------------------------------------------------------------


def test_metrics_report_a_matched_exposure_benchmark():
    data = pd.DataFrame({"close": series().to_numpy()})

    metrics = run_backtest(data, BacktestConfig())["metrics"]

    assert "benchmark" in metrics
    for key in ("annual_return", "sharpe", "max_drawdown", "average_drawdown", "exposure"):
        assert key in metrics["benchmark"]


def test_benchmark_exposure_equals_the_strategys_own_mean_absolute_position():
    """Anything else compares risk levels rather than skill."""
    data = pd.DataFrame({"close": series().to_numpy()})

    out = run_backtest(data, BacktestConfig())

    assert out["metrics"]["benchmark"]["exposure"] == pytest.approx(
        out["metrics"]["average_abs_position"]
    )


def test_excess_metrics_are_strategy_minus_benchmark():
    data = pd.DataFrame({"close": series().to_numpy()})

    m = run_backtest(data, BacktestConfig())["metrics"]

    assert m["excess_annual_return"] == pytest.approx(m["annual_return"] - m["benchmark"]["annual_return"])
    assert m["excess_sharpe"] == pytest.approx(m["sharpe"] - m["benchmark"]["sharpe"])
    assert m["drawdown_reduction"] == pytest.approx(m["max_drawdown"] - m["benchmark"]["max_drawdown"])


def test_a_permanently_invested_strategy_shows_almost_no_excess():
    """Identity check: a book that is always long must not manufacture excess.

    Not exactly zero -- the strategy is flat for the `vol_window` warmup bars
    while the benchmark holds its average throughout, so a small residual is
    expected and is itself worth pinning down. A benchmark bug (wrong exposure,
    misaligned returns, an off-by-one) misses by far more than these bounds.
    """
    data = pd.DataFrame({"close": series(3000).to_numpy()})
    cfg = BacktestConfig(
        use_trend_filter=False, vol_window=2, max_volatility=10.0, min_abs_position=1.0,
        max_abs_position=1.0, max_leverage=1.0, max_position_change=1.0,
        target_annual_volatility=100.0, fee_bps=0.0, slippage_bps=0.0,
        drawdown_kill_switch=-0.999, drawdown_reentry=-0.998,
    )

    out = run_backtest(data, cfg)
    m = out["metrics"]

    assert (out["results"]["position"].iloc[cfg.vol_window:] == 1.0).all()
    assert m["benchmark"]["exposure"] == pytest.approx(1.0, abs=1e-3)
    assert abs(m["excess_annual_return"]) < 0.01
    assert abs(m["excess_sharpe"]) < 0.05
    assert abs(m["drawdown_reduction"]) < 0.005


def test_average_drawdown_is_reported_alongside_max_drawdown():
    """Max drawdown is one episode; it flipped the verdict on the real panel."""
    data = pd.DataFrame({"close": series().to_numpy()})

    m = run_backtest(data, BacktestConfig())["metrics"]

    assert m["average_drawdown"] <= 0.0
    assert m["average_drawdown"] >= m["max_drawdown"]
