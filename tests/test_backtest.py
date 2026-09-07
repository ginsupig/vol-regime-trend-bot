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


def test_backtest_win_rate_tracks_closed_trades(monkeypatch) -> None:
    closes = [100.0, 100.0, 90.0, 81.0, 72.9, 80.19]

    def fake_signals(_: list[float], __: StrategyConfig) -> list[int]:
        return [0, 1, 0, -1, 0, 0]

    monkeypatch.setattr("vol_regime_trend_bot.backtest.generate_signals", fake_signals)
    result = run_backtest(
        closes,
        strategy_config=StrategyConfig(trend_window=2, vol_window=2, regime_window=2),
        risk_budget=1.0,
        max_leverage=1.0,
        max_gross_exposure=1.0,
        vol_lookback=2,
    )

    assert result.trades == 2
    assert result.win_rate == 0.5


def test_backtest_handles_non_positive_ending_equity_annualization(monkeypatch) -> None:
    closes = [100.0, 100.0, 80.0]

    def fake_signals(_: list[float], __: StrategyConfig) -> list[int]:
        return [0, 1, 1]

    monkeypatch.setattr("vol_regime_trend_bot.backtest.generate_signals", fake_signals)
    result = run_backtest(
        closes,
        strategy_config=StrategyConfig(trend_window=2, vol_window=2, regime_window=2),
        risk_budget=100.0,
        max_leverage=10.0,
        max_gross_exposure=10.0,
        vol_lookback=2,
    )

    assert result.total_return < -1.0
    assert result.annualized_return == -1.0
