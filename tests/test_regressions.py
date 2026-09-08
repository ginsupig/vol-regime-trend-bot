"""Regression tests for defects found by running the bot on real market data.

Every test here failed before the accompanying fix. The existing suite passed
throughout, because it only ever builds perfectly regular nanosecond
``pd.date_range`` indices with no market holidays, and only ever asserts that
the kill switch *reduces* exposure -- never that it gives exposure back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from vol_regime_trend_bot.backtest import run_backtest
from vol_regime_trend_bot.config import BacktestConfig
from vol_regime_trend_bot.data import infer_periods_per_year


def trading_calendar(periods: int = 504, start: str = "2020-01-02") -> pd.DatetimeIndex:
    """A real-shaped trading calendar: weekdays with market holidays removed."""
    return pd.date_range(start, periods=periods, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))


def price_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A seeded random walk at a realistic ~1%/day volatility.

    A noiseless ramp would carry a legitimately huge Sharpe and could not
    distinguish a good number from the 441 the annualization bug produced.
    """
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(index))))
    return pd.DataFrame({"timestamp": index, "close": closes})


# --------------------------------------------------------------------------
# Frequency inference
# --------------------------------------------------------------------------


def test_trading_calendar_with_holidays_is_inferred_not_rejected():
    """Daily equity data is weekday-regular but wall-clock irregular.

    Holidays defeat ``pd.infer_freq``, so the default ``error`` policy used to
    reject every real daily price series outright.
    """
    index = trading_calendar()

    resolved = infer_periods_per_year(
        index.tz_localize("UTC"), fallback_periods_per_year=999, irregular_timestamp_policy="error"
    )

    assert resolved.periods_per_year == 252
    assert resolved.source.startswith("inferred")


def test_trading_calendar_survives_the_2001_market_closure_gap():
    """The longest real US closure (2001-09-10 -> 2001-09-17) is 7 calendar days."""
    index = trading_calendar(periods=300, start="2001-06-01")
    index = index[(index < "2001-09-11") | (index > "2001-09-16")]

    resolved = infer_periods_per_year(
        index.tz_localize("UTC"), fallback_periods_per_year=999, irregular_timestamp_policy="error"
    )

    assert resolved.periods_per_year == 252


def test_data_hole_is_still_rejected_as_irregular():
    """A real gap in the data must not be silently smoothed into a density estimate."""
    index = trading_calendar(periods=300)
    index = index[(index < "2020-06-01") | (index > "2020-09-01")]

    with pytest.raises(ValueError, match="irregular timestamps"):
        infer_periods_per_year(
            index.tz_localize("UTC"), fallback_periods_per_year=999, irregular_timestamp_policy="error"
        )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_frequency_inference_is_independent_of_datetime_resolution(unit):
    """``.view("int64")`` yields the index's own unit, not nanoseconds.

    pandas 2/3 keep non-nanosecond resolutions (``datetime64[ms]`` is what a
    yfinance -> parquet round trip produces), so the hardcoded ``/1e9`` read one
    day as 0.0864 seconds and returned 365_250_000 periods per year.
    """
    index = pd.DatetimeIndex(trading_calendar().astype(f"datetime64[{unit}]")).tz_localize("UTC")

    resolved = infer_periods_per_year(
        index, fallback_periods_per_year=999, irregular_timestamp_policy="error"
    )

    assert resolved.periods_per_year == 252


def test_millisecond_timestamps_do_not_produce_absurd_annualization():
    """End-to-end guard on the metric that exposed the unit bug (Sharpe 441)."""
    index = trading_calendar()
    data = price_frame(index)
    data["timestamp"] = data["timestamp"].astype("datetime64[ms]")
    cfg = BacktestConfig(trend_window=20, vol_window=20)

    metrics = run_backtest(data, cfg, timestamp_column="timestamp")["metrics"]

    assert metrics["periods_per_year"] == 252
    assert abs(metrics["sharpe"]) < 20
    assert abs(metrics["annual_return"]) < 10


def test_regular_intraday_spacing_still_infers_from_the_median_gap():
    index = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")

    resolved = infer_periods_per_year(
        index, fallback_periods_per_year=999, irregular_timestamp_policy="error"
    )

    assert resolved.periods_per_year == 24 * 365


# --------------------------------------------------------------------------
# Drawdown kill switch
# --------------------------------------------------------------------------


def killswitch_config(**overrides) -> BacktestConfig:
    base = dict(
        trend_window=2,
        vol_window=2,
        max_volatility=1.0,
        min_abs_position=1.0,
        max_abs_position=1.0,
        max_leverage=1.0,
        max_position_change=1.0,
        target_annual_volatility=10.0,
        fee_bps=0.0,
        drawdown_kill_switch=-0.15,
        drawdown_reentry=-0.05,
        cooldown_bars=1,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def test_kill_switch_releases_after_cooldown_and_recovery():
    """Flat exposure freezes equity, so realized drawdown can never clear re-entry.

    The bot went permanently flat the first time the kill switch fired -- on real
    SPY data that was 2002-03-26, silencing 91.7% of the sample.
    """
    closes = [100, 101, 102, 103, 104, 80, 79, 78, 85, 95, 105, 118, 132, 148, 165, 185]
    out = run_backtest(pd.DataFrame({"close": closes}), killswitch_config())
    res = out["results"]

    assert out["metrics"]["kill_switch_events"] >= 1
    trigger = res.index.get_loc(res.index[res["kill_switch_triggered"]][0])
    after = res.iloc[trigger + 1 :]

    assert (after["raw_target"] > 0).any(), "fixture must ask to trade again after the kill"
    assert (after["position"].abs() > 0).any(), "kill switch never released exposure"


def test_kill_switch_rearms_for_a_second_drawdown():
    """A one-shot fuse cannot protect the rest of the run."""
    closes = [100, 102, 104, 106, 80, 79, 78, 88, 100, 115, 130, 145, 160, 120, 118, 116, 130, 150]
    out = run_backtest(pd.DataFrame({"close": closes}), killswitch_config())

    assert out["metrics"]["kill_switch_events"] >= 2


def test_kill_switch_stays_flat_while_the_strategy_is_still_underwater():
    """Re-entry must require recovery, not merely the passage of the cooldown."""
    closes = [100, 101, 102, 103, 104, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70]
    out = run_backtest(pd.DataFrame({"close": closes}), killswitch_config(cooldown_bars=1))
    res = out["results"]

    trigger = res.index.get_loc(res.index[res["kill_switch_triggered"]][0])
    assert (res["position"].iloc[trigger + 1 :].abs() == 0).all()


def test_kill_switch_state_survives_a_resume(tmp_path):
    index = trading_calendar(periods=16)
    closes = [100, 101, 102, 103, 104, 80, 79, 78, 85, 95, 105, 118, 132, 148, 165, 185]
    data = pd.DataFrame({"timestamp": index, "close": closes})
    cfg = killswitch_config()
    state_path = str(tmp_path / "state.json")

    whole = run_backtest(data, cfg, timestamp_column="timestamp")["results"]

    run_backtest(data.iloc[:8], cfg, timestamp_column="timestamp", execution_state_path=state_path)
    resumed = run_backtest(
        data, cfg, timestamp_column="timestamp", execution_state_path=state_path
    )["results"]

    tail = whole["position"].iloc[8:]
    assert np.allclose(resumed["position"].to_numpy(), tail.to_numpy())
