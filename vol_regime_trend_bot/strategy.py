from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from statistics import median


@dataclass(frozen=True)
class StrategyConfig:
    trend_window: int = 50
    vol_window: int = 20
    regime_window: int = 100

    def validate(self) -> None:
        if self.trend_window < 2:
            raise ValueError("trend_window must be >= 2")
        if self.vol_window < 2:
            raise ValueError("vol_window must be >= 2")
        if self.regime_window < 2:
            raise ValueError("regime_window must be >= 2")


def _stdev(values: deque[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return sqrt(var)


def generate_signals(closes: list[float], config: StrategyConfig) -> list[int]:
    config.validate()
    if len(closes) < 2:
        return [0] * len(closes)

    returns: deque[float] = deque(maxlen=config.vol_window)
    vol_history: deque[float] = deque(maxlen=config.regime_window)
    trend_prices: deque[float] = deque(maxlen=config.trend_window)

    signals: list[int] = [0]
    trend_prices.append(closes[0])

    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev <= 0 or curr <= 0:
            raise ValueError("close prices must be positive")

        returns.append((curr / prev) - 1.0)
        trend_prices.append(curr)

        if len(returns) < config.vol_window or len(trend_prices) < config.trend_window:
            signals.append(0)
            continue

        current_vol = _stdev(returns)
        if len(vol_history) < config.regime_window - 1:
            vol_history.append(current_vol)
            signals.append(0)
            continue

        low_vol_regime = current_vol <= median(vol_history)
        vol_history.append(current_vol)
        trend_ma = sum(trend_prices) / len(trend_prices)

        if low_vol_regime and curr > trend_ma:
            signals.append(1)
        elif low_vol_regime and curr < trend_ma:
            signals.append(-1)
        else:
            signals.append(0)

    return signals
