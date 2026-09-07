from __future__ import annotations

import math
import statistics

from .config import StrategyConfig


def _rolling_mean(values: list[float], end_idx: int, window: int) -> float:
    start = max(0, end_idx - window + 1)
    window_slice = values[start : end_idx + 1]
    return sum(window_slice) / len(window_slice)


def _rolling_pstd(values: list[float], end_idx: int, window: int) -> float:
    start = max(0, end_idx - window + 1)
    window_slice = values[start : end_idx + 1]
    if len(window_slice) <= 1:
        return 0.0
    return statistics.pstdev(window_slice)


def generate_raw_signals(prices: list[float], returns: list[float], config: StrategyConfig) -> list[float]:
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to generate signals")
    if len(returns) != len(prices):
        raise ValueError("returns length must match prices length")

    signals: list[float] = []
    for i, price in enumerate(prices):
        trend = 1.0 if price >= _rolling_mean(prices, i, config.trend_window) else 0.0
        realized_vol = _rolling_pstd(returns, i, config.vol_window)
        regime = 1.0 if realized_vol <= config.vol_threshold else 0.0
        signals.append(1.0 if trend > 0 and regime > 0 else 0.0)
    return signals


def scale_positions_with_risk(
    raw_signals: list[float],
    returns: list[float],
    vol_window: int,
    periods_per_year: int,
    target_vol: float,
    max_exposure: float,
    max_position_change: float,
    min_vol_floor: float,
) -> list[float]:
    if len(raw_signals) != len(returns):
        raise ValueError("raw_signals length must match returns length")
    if vol_window <= 0:
        raise ValueError("vol_window must be positive")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    annualizer = math.sqrt(periods_per_year)
    positions: list[float] = []
    prev = 0.0
    for i, signal in enumerate(raw_signals):
        start = max(0, i - vol_window + 1)
        vol_sample = returns[start : i + 1]
        realized_vol = statistics.pstdev(vol_sample) * annualizer if len(vol_sample) > 1 else 0.0
        vol_denom = max(realized_vol, min_vol_floor)
        desired = signal * min(max_exposure, target_vol / vol_denom)
        delta = max(-max_position_change, min(max_position_change, desired - prev))
        next_position = prev + delta
        next_position = max(0.0, min(max_exposure, next_position))
        positions.append(next_position)
        prev = next_position
    return positions
