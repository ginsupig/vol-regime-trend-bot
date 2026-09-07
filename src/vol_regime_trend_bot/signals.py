from __future__ import annotations

import pandas as pd

from .config import BacktestConfig


def compute_trend_signal(prices: pd.Series, window: int) -> pd.Series:
    trend = prices.rolling(window, min_periods=window).mean()
    return (prices > trend).fillna(False)


def compute_realized_volatility(prices: pd.Series, window: int) -> pd.Series:
    returns = prices.pct_change()
    return returns.rolling(window, min_periods=window).std(ddof=0)


def compute_volatility_regime_signal(
    prices: pd.Series,
    window: int,
    max_volatility: float,
    realized_vol: pd.Series | None = None,
) -> pd.Series:
    realized_vol = realized_vol if realized_vol is not None else compute_realized_volatility(prices, window)
    return (realized_vol <= max_volatility).fillna(False)


def compose_signal(trend_signal: pd.Series, regime_signal: pd.Series) -> pd.Series:
    return (trend_signal & regime_signal).astype(float)


def build_signal_pipeline(prices: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    config.validate()
    trend_signal = compute_trend_signal(prices, config.trend_window)
    realized_vol = compute_realized_volatility(prices, config.vol_window)
    regime_signal = compute_volatility_regime_signal(
        prices,
        window=config.vol_window,
        max_volatility=config.max_volatility,
        realized_vol=realized_vol,
    )
    raw_signal = compose_signal(trend_signal, regime_signal)
    return pd.DataFrame(
        {
            "trend_signal": trend_signal.astype(float),
            "regime_signal": regime_signal.astype(float),
            "realized_vol": realized_vol.astype(float),
            "raw_signal": raw_signal.astype(float),
        },
        index=prices.index,
    )
