import pandas as pd


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
