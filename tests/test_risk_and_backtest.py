import pandas as pd
import pytest

from vol_regime_trend_bot.backtest import run_backtest
from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.risk import apply_exposure_limits


def test_apply_exposure_limits_caps_rate_of_change():
    target = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0])

    got = apply_exposure_limits(target, max_abs_position=0.6, max_position_change=0.25)

    assert got.tolist() == [0.0, 0.25, 0.5, 0.25, 0.0]


def test_run_backtest_returns_metrics_and_result_columns():
    data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105, 106, 107]})
    cfg = BacktestConfig(trend_window=2, vol_window=2, max_volatility=0.1, fee_bps=1.0)

    out = run_backtest(data, cfg)

    assert set(["position", "returns", "net_return", "equity_curve"]).issubset(out["results"].columns)
    assert out["metrics"]["trades"] >= 0
    assert out["metrics"]["max_drawdown"] <= 0


def test_run_backtest_metrics_are_consistent_with_results():
    data = pd.DataFrame({"close": [100, 101, 102, 101, 103, 104, 103, 105]})
    cfg = BacktestConfig(
        trend_window=2, vol_window=2, max_volatility=0.2, fee_bps=0.0, periods_per_year=252
    )

    out = run_backtest(data, cfg)
    results = out["results"]
    metrics = out["metrics"]

    observed_periods = max(len(results["net_return"]) - 1, 1)
    expected_annual = (1.0 + metrics["total_return"]) ** (252 / observed_periods) - 1.0
    assert metrics["annual_return"] == pytest.approx(expected_annual)
    expected_sharpe = (
        results["net_return"].mean() / results["net_return"].std(ddof=0) * (cfg.periods_per_year**0.5)
    )
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)

    active = (results["position"].shift(1).fillna(0.0) != 0.0) | (
        (results["position"] - results["position"].shift(1).fillna(0.0)).abs() > 0.0
    )
    expected_win_rate = float((results.loc[active, "net_return"] > 0).mean())
    assert metrics["win_rate"] == pytest.approx(expected_win_rate)


def test_run_backtest_validates_missing_price_column():
    data = pd.DataFrame({"open": [1, 2, 3]})
    cfg = BacktestConfig(trend_window=2, vol_window=2)

    with pytest.raises(ValueError, match="missing required price column"):
        run_backtest(data, cfg)
