from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DrawdownProtectionState:
    kill_active: bool = False
    cooldown_remaining: int = 0


def apply_volatility_targeting(
    raw_target_positions: pd.Series,
    realized_volatility: pd.Series,
    target_annual_volatility: float,
    periods_per_year: int,
    min_abs_position: float,
    max_leverage: float,
) -> pd.Series:
    if target_annual_volatility <= 0:
        raise ValueError("target_annual_volatility must be > 0")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    if min_abs_position < 0:
        raise ValueError("min_abs_position must be >= 0")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")
    if min_abs_position > max_leverage:
        raise ValueError("min_abs_position must be <= max_leverage")

    target_period_vol = target_annual_volatility / np.sqrt(periods_per_year)
    safe_vol = realized_volatility.replace(0.0, np.nan)
    scale = (target_period_vol / safe_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = raw_target_positions.astype(float)
    sized = raw * scale

    abs_sized = sized.abs().clip(lower=0.0, upper=max_leverage)
    enforced = np.where(
        raw.abs() > 0,
        np.maximum(abs_sized, min_abs_position),
        0.0,
    )
    return pd.Series(np.sign(sized) * enforced, index=raw_target_positions.index, dtype=float)


def apply_exposure_limits(
    target_positions: pd.Series,
    max_abs_position: float,
    max_position_change: float,
) -> pd.Series:
    if max_abs_position <= 0:
        raise ValueError("max_abs_position must be > 0")
    if max_position_change <= 0:
        raise ValueError("max_position_change must be > 0")

    clipped_target = target_positions.astype(float).clip(-max_abs_position, max_abs_position)
    limited = []
    prev = 0.0
    for raw in clipped_target.fillna(0.0):
        delta = raw - prev
        if delta > max_position_change:
            current = prev + max_position_change
        elif delta < -max_position_change:
            current = prev - max_position_change
        else:
            current = raw
        limited.append(current)
        prev = current

    return pd.Series(limited, index=target_positions.index, dtype=float)


def apply_drawdown_protection(
    desired_position: float,
    current_drawdown: float,
    state: DrawdownProtectionState,
    kill_switch: float,
    reentry_drawdown: float,
    cooldown_bars: int,
) -> tuple[float, DrawdownProtectionState, bool]:
    if state.kill_active:
        if state.cooldown_remaining > 0:
            return 0.0, DrawdownProtectionState(True, state.cooldown_remaining - 1), False
        if current_drawdown <= reentry_drawdown:
            return 0.0, DrawdownProtectionState(True, 0), False
        return desired_position, DrawdownProtectionState(False, 0), False

    if current_drawdown <= kill_switch:
        return 0.0, DrawdownProtectionState(True, cooldown_bars), True

    return desired_position, DrawdownProtectionState(False, 0), False
