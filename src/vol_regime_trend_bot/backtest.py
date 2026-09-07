from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .config import BacktestConfig, RiskConfig, StrategyConfig
from .strategy import generate_raw_signals, scale_positions_with_risk


@dataclass(frozen=True)
class BacktestResult:
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_exposure: float
    trades: int
    periods_per_year: int


def _compute_returns(prices: list[float]) -> list[float]:
    if len(prices) < 2:
        raise ValueError("Need at least 2 price points")
    returns = [0.0]
    for prev, current in zip(prices[:-1], prices[1:]):
        if prev <= 0:
            raise ValueError("Prices must be positive")
        returns.append((current / prev) - 1.0)
    return returns


def run_backtest(
    prices: list[float],
    strategy_config: StrategyConfig | None = None,
    risk_config: RiskConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> BacktestResult:
    strategy_config = strategy_config or StrategyConfig()
    risk_config = risk_config or RiskConfig()
    backtest_config = backtest_config or BacktestConfig()
    if backtest_config.periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    returns = _compute_returns(prices)
    raw_signals = generate_raw_signals(prices, returns, strategy_config)
    positions = scale_positions_with_risk(
        raw_signals=raw_signals,
        returns=returns,
        periods_per_year=backtest_config.periods_per_year,
        target_vol=risk_config.target_vol,
        max_exposure=risk_config.max_exposure,
        max_position_change=risk_config.max_position_change,
        min_vol_floor=risk_config.min_vol_floor,
    )

    fee_rate = backtest_config.fee_bps / 10000.0
    pnl: list[float] = []
    prev_position = 0.0
    trades = 0
    for period_return, position in zip(returns, positions):
        turnover = abs(position - prev_position)
        if turnover > 1e-12:
            trades += 1
        pnl.append((position * period_return) - (turnover * fee_rate))
        prev_position = position

    equity = [1.0]
    for r in pnl:
        equity.append(equity[-1] * (1.0 + r))

    total_return = equity[-1] - 1.0
    n_periods = len(pnl)
    if equity[-1] <= 0:
        annual_return = -1.0
    else:
        annual_return = equity[-1] ** (backtest_config.periods_per_year / max(n_periods, 1)) - 1.0
    period_vol = statistics.pstdev(pnl) if len(pnl) > 1 else 0.0
    annual_volatility = period_vol * math.sqrt(backtest_config.periods_per_year)
    sharpe = 0.0 if annual_volatility == 0 else (statistics.mean(pnl) * backtest_config.periods_per_year) / annual_volatility

    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = (value / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    wins = sum(1 for r in pnl if r > 0)
    win_rate = wins / len(pnl) if pnl else 0.0
    avg_exposure = statistics.mean(positions) if positions else 0.0

    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_exposure=avg_exposure,
        trades=trades,
        periods_per_year=backtest_config.periods_per_year,
    )
