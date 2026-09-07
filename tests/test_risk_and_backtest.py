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


def test_run_backtest_validates_missing_price_column():
    data = pd.DataFrame({"open": [1, 2, 3]})
    cfg = BacktestConfig(trend_window=2, vol_window=2)

    with pytest.raises(ValueError, match="missing required price column"):
        run_backtest(data, cfg)
