from __future__ import annotations

import json

import pandas as pd
import pytest

from vol_regime_trend_bot.backtest import run_backtest
from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.execution import PaperExecutionAdapter
from vol_regime_trend_bot.risk import apply_exposure_limits


def test_apply_exposure_limits_caps_rate_of_change():
    target = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0])

    got = apply_exposure_limits(target, max_abs_position=0.6, max_position_change=0.25)

    assert got.tolist() == [0.0, 0.25, 0.5, 0.25, 0.0]


def test_run_backtest_returns_metrics_and_result_columns():
    data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105, 106, 107]})
    cfg = BacktestConfig(trend_window=2, vol_window=2, max_volatility=0.1, fee_bps=1.0)

    out = run_backtest(data, cfg)

    assert set(
        [
            "position",
            "returns",
            "trend_signal",
            "regime_signal",
            "realized_vol",
            "raw_target",
            "sized_target",
            "gross_return",
            "turnover",
            "fees",
            "slippage",
            "net_return",
            "equity_curve",
            "drawdown",
            "kill_switch_triggered",
        ]
    ).issubset(out["results"].columns)
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
    expected_annual = (1.0 + metrics["total_return"]) ** (metrics["periods_per_year"] / observed_periods) - 1.0
    assert metrics["annual_return"] == pytest.approx(expected_annual)
    expected_sharpe = (
        results["net_return"].mean() / results["net_return"].std(ddof=0) * (metrics["periods_per_year"]**0.5)
    )
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)
    assert metrics["final_equity"] == pytest.approx(results["equity_curve"].iloc[-1])
    assert metrics["average_turnover"] == pytest.approx(results["turnover"].iloc[1:].mean())
    assert metrics["fee_drag"] == pytest.approx(results["fees"].sum())
    assert metrics["slippage_drag"] == pytest.approx(results["slippage"].sum())
    assert metrics["average_abs_position"] == pytest.approx(results["position"].abs().mean())

    shifted = results["position"].shift(1).fillna(0.0)
    active = (shifted != 0.0) | ((results["position"] - shifted).abs() > 0.0)
    expected_win_rate = float((results.loc[active, "net_return"] > 0).mean())
    assert metrics["win_rate"] == pytest.approx(expected_win_rate)


def test_run_backtest_validates_missing_price_column():
    data = pd.DataFrame({"open": [1, 2, 3]})
    cfg = BacktestConfig(trend_window=2, vol_window=2)

    with pytest.raises(ValueError, match="missing required price column"):
        run_backtest(data, cfg)


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ([100.0, 0.0, 101.0], "only positive values"),
        ([100.0, -1.0, 101.0], "only positive values"),
        ([100.0, float("nan"), 101.0], "finite numeric values"),
        ([100.0, float("inf"), 101.0], "finite numeric values"),
    ],
)
def test_run_backtest_validates_price_values(prices, message):
    data = pd.DataFrame({"close": prices})
    cfg = BacktestConfig(trend_window=2, vol_window=2)

    with pytest.raises(ValueError, match=message):
        run_backtest(data, cfg)


def test_frequency_inference_uses_datetime_index_when_regular():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="B"),
            "close": [100 + i for i in range(10)],
        }
    )
    cfg = BacktestConfig(trend_window=2, vol_window=2, periods_per_year=999)

    out = run_backtest(data, cfg, timestamp_column="timestamp")

    assert out["metrics"]["periods_per_year"] == 252
    assert out["metrics"]["annualization_source"].startswith("inferred")


def test_irregular_timestamps_fail_or_fallback():
    ts = pd.to_datetime([
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-06",
        "2024-01-07",
    ])
    data = pd.DataFrame({"timestamp": ts, "close": [100, 101, 102, 103, 104]})

    with pytest.raises(ValueError, match="irregular timestamps"):
        run_backtest(data, BacktestConfig(trend_window=2, vol_window=2), timestamp_column="timestamp")

    out = run_backtest(
        data,
        BacktestConfig(trend_window=2, vol_window=2, periods_per_year=365, irregular_timestamp_policy="fallback"),
        timestamp_column="timestamp",
    )
    assert out["metrics"]["periods_per_year"] == 365
    assert out["metrics"]["annualization_source"] == "configured_fallback"


def test_ohlcv_timestamp_and_volume_validation():
    data = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "close": [100, 101, 102],
            "volume": [1, 2, 3],
        }
    )
    cfg = BacktestConfig(trend_window=2, vol_window=2)

    with pytest.raises(ValueError, match="duplicate timestamp"):
        run_backtest(data, cfg, timestamp_column="timestamp")

    bad_volume = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "close": [100, 101, 102],
            "volume": [1, -1, 3],
        }
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        run_backtest(bad_volume, cfg, timestamp_column="timestamp")


def test_drawdown_kill_switch_and_cooldown_reduce_exposure():
    data = pd.DataFrame({"close": [100, 101, 102, 70, 69, 68, 80, 90, 95]})
    cfg = BacktestConfig(
        trend_window=2,
        vol_window=2,
        max_volatility=1.0,
        max_position_change=1.0,
        drawdown_kill_switch=-0.05,
        drawdown_reentry=-0.02,
        cooldown_bars=2,
    )

    out = run_backtest(data, cfg)
    res = out["results"]

    assert out["metrics"]["kill_switch_events"] >= 1
    trigger_idx = res.index[res["kill_switch_triggered"]].tolist()[0]
    pos_idx = res.index.get_loc(trigger_idx)
    if pos_idx + 2 < len(res):
        assert res["position"].iloc[pos_idx + 1] == 0.0


def test_execution_idempotency_and_state_restore(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="B"),
            "close": [100, 101, 102, 103, 102, 104],
        }
    )
    cfg = BacktestConfig(trend_window=2, vol_window=2, max_position_change=1.0)
    state_path = tmp_path / "state.json"

    adapter1 = PaperExecutionAdapter()
    run_backtest(
        data,
        cfg,
        timestamp_column="timestamp",
        execution_adapter=adapter1,
        execution_state_path=str(state_path),
        symbol="TEST",
    )

    assert state_path.exists()
    first_keys = {x.idempotency_key for x in adapter1.submitted}
    assert len(first_keys) == len(adapter1.submitted)

    adapter2 = PaperExecutionAdapter()
    run_backtest(
        data,
        cfg,
        timestamp_column="timestamp",
        execution_adapter=adapter2,
        execution_state_path=str(state_path),
        symbol="TEST",
    )

    assert adapter2.submitted == []
    payload = json.loads(state_path.read_text())
    assert set(payload["seen_intent_keys"]) == first_keys


def test_smoke_integration_backtest_fixture_runs_end_to_end(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="B"),
            "open": [100 + 0.2 * i for i in range(30)],
            "high": [100.5 + 0.2 * i for i in range(30)],
            "low": [99.5 + 0.2 * i for i in range(30)],
            "close": [100 + 0.2 * i for i in range(30)],
            "volume": [10_000 for _ in range(30)],
        }
    ).to_csv(csv_path, index=False)

    data = pd.read_csv(csv_path)
    cfg = BacktestConfig(trend_window=3, vol_window=3, max_volatility=1.0)
    out = run_backtest(data, cfg, timestamp_column="timestamp")

    assert "metrics" in out
    assert "results" in out
    assert out["metrics"]["periods_per_year"] == 252
