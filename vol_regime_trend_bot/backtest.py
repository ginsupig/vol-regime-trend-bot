from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt

from .risk import position_size_from_vol
from .strategy import StrategyConfig, generate_signals


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[float]
    returns: list[float]
    positions: list[float]
    signals: list[int]
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return sqrt(var)


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 1.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def run_backtest(
    closes: list[float],
    strategy_config: StrategyConfig | None = None,
    risk_budget: float = 0.10,
    max_leverage: float = 1.0,
    max_gross_exposure: float = 1.0,
    vol_lookback: int = 20,
    periods_per_year: int = 252,
) -> BacktestResult:
    if strategy_config is None:
        strategy_config = StrategyConfig()
    if len(closes) < 2:
        raise ValueError("Need at least 2 close prices")
    if vol_lookback < 2:
        raise ValueError("vol_lookback must be >= 2")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")

    signals = generate_signals(closes, strategy_config)
    equity_curve = [1.0]
    pnl_returns: list[float] = []
    positions: list[float] = [0.0]
    rolling_returns: deque[float] = deque(maxlen=vol_lookback)

    trade_count = 0
    winning_trade_count = 0
    active_trade_pnl = 0.0
    prior_position = 0.0

    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev <= 0 or curr <= 0:
            raise ValueError("close prices must be positive")

        instrument_ret = (curr / prev) - 1.0
        rolling_returns.append(instrument_ret)

        ann_vol = _stdev(list(rolling_returns)) * sqrt(periods_per_year)
        size = position_size_from_vol(
            ann_vol,
            risk_budget=risk_budget,
            max_leverage=max_leverage,
            max_gross_exposure=max_gross_exposure,
        )

        position = float(signals[i - 1]) * size
        pnl = position * instrument_ret

        if prior_position != position:
            if prior_position != 0.0:
                trade_count += 1
                if active_trade_pnl > 0:
                    winning_trade_count += 1
            active_trade_pnl = 0.0

        if position != 0.0:
            active_trade_pnl += pnl

        pnl_returns.append(pnl)
        equity_curve.append(equity_curve[-1] * (1.0 + pnl))
        positions.append(position)
        prior_position = position

    if prior_position != 0.0:
        trade_count += 1
        if active_trade_pnl > 0:
            winning_trade_count += 1

    total_return = equity_curve[-1] - 1.0
    n = len(pnl_returns)
    mean_ret = sum(pnl_returns) / n if n else 0.0
    realized_vol = _stdev(pnl_returns)
    annualized_vol = realized_vol * sqrt(periods_per_year)
    ending_equity = equity_curve[-1]
    if n == 0:
        annualized_return = 0.0
    elif ending_equity <= 0:
        annualized_return = -1.0
    else:
        annualized_return = (ending_equity ** (periods_per_year / n)) - 1.0
    sharpe = (mean_ret / realized_vol * sqrt(periods_per_year)) if realized_vol > 0 else 0.0
    max_drawdown = _max_drawdown(equity_curve)
    win_rate = (winning_trade_count / trade_count) if trade_count > 0 else 0.0

    return BacktestResult(
        equity_curve=equity_curve,
        returns=pnl_returns,
        positions=positions,
        signals=signals,
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        trades=trade_count,
        win_rate=win_rate,
    )
