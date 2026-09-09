from __future__ import annotations

import numpy as np
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


def apply_regime_hysteresis(
    realized_vol: pd.Series,
    max_volatility: float,
    buffer: float,
) -> pd.Series:
    enter_threshold = float(max_volatility)
    exit_threshold = float(max_volatility) * (1.0 + float(buffer))
    state = False
    values: list[float] = []
    for vol in realized_vol.astype(float):
        if not np.isfinite(vol):
            state = False
        elif state:
            state = vol <= exit_threshold
        else:
            state = vol <= enter_threshold
        values.append(float(state))
    return pd.Series(values, index=realized_vol.index, dtype=float)


def compute_volatility_change_signal(realized_vol: pd.Series, window: int) -> pd.Series:
    change = realized_vol.diff(window)
    return (change <= 0.0).fillna(False)


def apply_signal_confirmation(signal: pd.Series, confirmation_bars: int) -> pd.Series:
    if confirmation_bars <= 1:
        return signal.astype(float)

    desired = signal.astype(float).fillna(0.0)
    state = 0.0
    pending_state = 0.0
    streak = 0
    values: list[float] = []

    for value in desired:
        if value == state:
            pending_state = state
            streak = 0
            values.append(state)
            continue
        if value != pending_state:
            pending_state = value
            streak = 1
        else:
            streak += 1
        if streak >= confirmation_bars:
            state = pending_state
            pending_state = state
            streak = 0
        values.append(state)
    return pd.Series(values, index=signal.index, dtype=float)


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
    tradable_regime_signal = apply_regime_hysteresis(
        realized_vol,
        max_volatility=config.max_volatility,
        buffer=config.vol_hysteresis_buffer,
    )
    volatility_change_signal = compute_volatility_change_signal(
        realized_vol, window=config.volatility_change_window
    )
    if config.require_falling_volatility:
        tradable_regime_signal = (tradable_regime_signal > 0.0) & (volatility_change_signal > 0.0)
        tradable_regime_signal = tradable_regime_signal.astype(float)
    tradable_regime_signal = apply_signal_confirmation(
        tradable_regime_signal,
        confirmation_bars=config.signal_confirmation_bars,
    )
    # The trend signal is always computed and reported, but only gates the book
    # when explicitly enabled -- it is measured subtractive on both universes
    # tested (see README).
    raw_signal = (
        compose_signal(trend_signal, tradable_regime_signal)
        if config.use_trend_filter
        else tradable_regime_signal.astype(float)
    )
    return pd.DataFrame(
        {
            "trend_signal": trend_signal.astype(float),
            "regime_signal": regime_signal.astype(float),
            "tradable_regime_signal": tradable_regime_signal.astype(float),
            "volatility_change_signal": volatility_change_signal.astype(float),
            "realized_vol": realized_vol.astype(float),
            "raw_signal": raw_signal.astype(float),
        },
        index=prices.index,
    )
