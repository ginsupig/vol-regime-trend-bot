from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrequencyResolution:
    periods_per_year: int
    source: str


def _coerce_datetime_index(data: pd.DataFrame, timestamp_column: str | None) -> pd.DataFrame:
    result = data.copy()
    if timestamp_column and timestamp_column in result.columns:
        ts = pd.to_datetime(result[timestamp_column], errors="coerce", utc=True)
        if ts.isna().any():
            bad_row = int(ts[ts.isna()].index[0])
            raise ValueError(f"invalid timestamp at row {bad_row} in column '{timestamp_column}'")
        result = result.drop(columns=[timestamp_column])
        result.index = pd.DatetimeIndex(ts)
    elif isinstance(result.index, pd.DatetimeIndex):
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        else:
            result.index = result.index.tz_convert("UTC")

    if isinstance(result.index, pd.DatetimeIndex):
        if result.index.has_duplicates:
            dup_ts = result.index[result.index.duplicated()][0]
            raise ValueError(f"duplicate timestamp detected: {dup_ts.isoformat()}")
        if not result.index.is_monotonic_increasing:
            raise ValueError("timestamps must be strictly increasing")
    return result


def validate_ohlcv_input(
    data: pd.DataFrame,
    price_column: str,
    timestamp_column: str | None = None,
    volume_column: str | None = "volume",
) -> tuple[pd.DataFrame, pd.Series]:
    if data.empty:
        raise ValueError("input data is empty")
    if price_column not in data.columns:
        raise ValueError(f"missing required price column: {price_column}")

    prepared = _coerce_datetime_index(data, timestamp_column)

    prices = pd.to_numeric(prepared[price_column], errors="coerce")
    if prices.isna().any() or not np.isfinite(prices).all():
        raise ValueError(f"price column '{price_column}' must contain only finite numeric values")
    if (prices <= 0).any():
        raise ValueError(f"price column '{price_column}' must contain only positive values")

    for optional_price_col in ("open", "high", "low", "close"):
        if optional_price_col in prepared.columns:
            values = pd.to_numeric(prepared[optional_price_col], errors="coerce")
            if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"column '{optional_price_col}' must contain only positive finite values")

    if all(col in prepared.columns for col in ("open", "high", "low", "close")):
        bad_high = prepared["high"] < prepared[["open", "low", "close"]].max(axis=1)
        bad_low = prepared["low"] > prepared[["open", "high", "close"]].min(axis=1)
        if bool(bad_high.any()) or bool(bad_low.any()):
            raise ValueError("OHLC columns violate high/low consistency")

    if volume_column and volume_column in prepared.columns:
        volume = pd.to_numeric(prepared[volume_column], errors="coerce")
        if volume.isna().any() or not np.isfinite(volume).all():
            raise ValueError(f"volume column '{volume_column}' must contain only finite numeric values")
        if (volume < 0).any():
            raise ValueError(f"volume column '{volume_column}' must be non-negative")

    return prepared, prices.astype(float)


def infer_periods_per_year(
    index: pd.Index,
    fallback_periods_per_year: int,
    irregular_timestamp_policy: str,
) -> FrequencyResolution:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return FrequencyResolution(periods_per_year=fallback_periods_per_year, source="configured")

    inferred = pd.infer_freq(index)
    freq_map = {
        "B": 252,
        "D": 365,
        "W": 52,
        "ME": 12,
        "MS": 12,
        "M": 12,
        "QE": 4,
        "QS": 4,
        "Q": 4,
        "YE": 1,
        "YS": 1,
        "Y": 1,
        "H": 24 * 365,
        "h": 24 * 365,
        "T": 60 * 24 * 365,
        "min": 60 * 24 * 365,
        "S": 365 * 24 * 60 * 60,
        "s": 365 * 24 * 60 * 60,
    }
    if inferred in freq_map:
        return FrequencyResolution(periods_per_year=freq_map[inferred], source=f"inferred:{inferred}")
    if inferred:
        match = re.fullmatch(r"(\d+)([A-Za-z]+)", inferred)
        if match:
            step = int(match.group(1))
            base = match.group(2)
            if base in freq_map and step > 0:
                scaled = max(int(round(freq_map[base] / step)), 1)
                return FrequencyResolution(periods_per_year=scaled, source=f"inferred:{inferred}")
    business_days = index.tz_localize(None).normalize().to_numpy(dtype="datetime64[D]")
    if len(business_days) >= 2:
        business_deltas = np.array(
            [
                np.busday_count(business_days[i], business_days[i + 1])
                for i in range(len(business_days) - 1)
            ],
            dtype=int,
        )
        if (business_deltas > 0).all() and len(np.unique(business_deltas)) == 1:
            step = int(business_deltas[0])
            scaled = max(int(round(252 / step)), 1)
            return FrequencyResolution(periods_per_year=scaled, source=f"inferred:{step}B")

    deltas = np.diff(index.view("int64"))
    if len(deltas) == 0:
        return FrequencyResolution(periods_per_year=fallback_periods_per_year, source="configured")

    median_delta = float(np.median(deltas))
    if median_delta <= 0:
        raise ValueError("timestamps must be strictly increasing")

    irregular = np.abs(deltas - median_delta) > max(median_delta * 1e-6, 1_000_000_000)
    if bool(irregular.any()):
        if irregular_timestamp_policy == "error":
            first_bad = int(np.where(irregular)[0][0])
            raise ValueError(
                "irregular timestamps detected; either clean data or set irregular_timestamp_policy to 'fallback'"
            )
        return FrequencyResolution(periods_per_year=fallback_periods_per_year, source="configured_fallback")

    seconds_per_period = median_delta / 1_000_000_000.0
    seconds_per_year = 365.25 * 24 * 60 * 60
    inferred_periods = max(int(round(seconds_per_year / seconds_per_period)), 1)
    return FrequencyResolution(periods_per_year=inferred_periods, source="inferred:timedelta")
