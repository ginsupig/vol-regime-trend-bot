import math

from vol_regime_trend_bot.backtest import run_backtest
from vol_regime_trend_bot.config import BacktestConfig, RiskConfig, StrategyConfig
from vol_regime_trend_bot.strategy import scale_positions_with_risk


def test_scale_positions_applies_exposure_and_turnover_limits():
    positions = scale_positions_with_risk(
        raw_signals=[1.0, 1.0, 1.0, 1.0],
        returns=[0.0, 0.01, -0.02, 0.01],
        periods_per_year=252,
        target_vol=0.50,
        max_exposure=0.60,
        max_position_change=0.20,
        min_vol_floor=1e-6,
    )

    assert all(0.0 <= p <= 0.60 for p in positions)
    assert all(abs(b - a) <= 0.2000001 for a, b in zip(positions[:-1], positions[1:]))


def test_backtest_respects_periods_per_year_annualization():
    prices = [100, 101, 102, 103, 104, 105]
    strategy_config = StrategyConfig(vol_window=2, trend_window=2, vol_threshold=1.0)
    risk_config = RiskConfig(target_vol=0.2, max_exposure=1.0, max_position_change=1.0)

    result_252 = run_backtest(
        prices,
        strategy_config=strategy_config,
        risk_config=risk_config,
        backtest_config=BacktestConfig(periods_per_year=252, fee_bps=0.0),
    )
    result_12 = run_backtest(
        prices,
        strategy_config=strategy_config,
        risk_config=risk_config,
        backtest_config=BacktestConfig(periods_per_year=12, fee_bps=0.0),
    )

    assert result_252.periods_per_year == 252
    assert result_12.periods_per_year == 12
    assert not math.isclose(result_252.annual_return, result_12.annual_return)

