from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .risk import apply_exposure_limits
from .strategy import generate_target_positions


def _validate_input_data(data: pd.DataFrame, price_column: str) -> None:
    if data.empty:
        raise ValueError("input data is empty")
    if price_column not in data.columns:
        raise ValueError(f"missing required price column: {price_column}")


def run_backtest(
    data: pd.DataFrame,
    config: BacktestConfig,
    price_column: str = "close",
) -> dict[str, Any]:
    config.validate()
    _validate_input_data(data, price_column)

    prices = data[price_column].astype(float)
    raw_target = generate_target_positions(prices, config)
    position = apply_exposure_limits(
        raw_target,
        max_abs_position=config.max_abs_position,
        max_position_change=config.max_position_change,
    )

    returns = prices.pct_change().fillna(0.0)
    shifted_position = position.shift(1).fillna(0.0)
    gross_return = shifted_position * returns
    turnover = (position - shifted_position).abs()
    fees = turnover * (config.fee_bps / 10_000.0)
    net_return = gross_return - fees

    equity_curve = (1.0 + net_return).cumprod()

    periods_per_year = config.periods_per_year
    observed_periods = max(len(net_return) - 1, 1)
    terminal_equity = float(equity_curve.iloc[-1])
    total_return = terminal_equity - 1.0
    avg = float(net_return.mean())
    vol = float(net_return.std(ddof=0))
    if terminal_equity > 0:
        annual_return = terminal_equity ** (periods_per_year / observed_periods) - 1.0
    else:
        annual_return = -1.0
    annual_vol = vol * np.sqrt(periods_per_year)
    sharpe = (avg / vol) * np.sqrt(periods_per_year) if vol > 0 else 0.0

    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0
    active_periods = (shifted_position != 0.0) | (turnover > 0.0)
    if bool(active_periods.any()):
        win_rate = float((net_return[active_periods] > 0).mean())
    else:
        win_rate = 0.0

    metrics = {
        "total_return": total_return,
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": win_rate,
        "trades": int((turnover.iloc[1:] > 0).sum()),
    }

    result = data.copy()
    result["position"] = position
    result["returns"] = returns
    result["net_return"] = net_return
    result["equity_curve"] = equity_curve

    return {"metrics": metrics, "results": result}
