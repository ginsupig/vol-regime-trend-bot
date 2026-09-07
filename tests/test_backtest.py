from vol_regime_trend_bot.backtest import run_backtest
from vol_regime_trend_bot.strategy import StrategyConfig


def test_backtest_returns_coherent_metrics() -> None:
    closes = [100.0 + (i * 0.2) for i in range(260)]
    result = run_backtest(
        closes,
        strategy_config=StrategyConfig(trend_window=20, vol_window=10, regime_window=30),
        risk_budget=0.12,
        max_leverage=1.0,
        max_gross_exposure=0.75,
    )

    assert len(result.equity_curve) == len(closes)
    assert len(result.positions) == len(closes)
    assert len(result.signals) == len(closes)
    assert result.total_return > -1.0
    assert result.max_drawdown >= 0.0
    assert 0.0 <= result.win_rate <= 1.0
