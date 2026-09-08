from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

# Measured, not chosen: SPY/GLD/TLT daily bars over 2000-2026 average 251.5
# trading days per calendar year, which rounds to the conventional 252.
TRADING_DAYS_PER_YEAR = 252

# The longest closure in modern US market history is the week of 2001-09-11
# (2001-09-10 -> 2001-09-17, 7 calendar days). Anything longer is a hole in the
# data rather than a market holiday, and must not be smoothed over.
MAX_TRADING_CALENDAR_GAP_DAYS = 10


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


def _gap_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Spacing between consecutive timestamps, in seconds.

    Uses timedelta arithmetic rather than ``index.view("int64")``. The raw
    integer view is expressed in the index's own resolution, and pandas 2/3
    preserve ``s``/``ms``/``us`` indices instead of coercing to nanoseconds
    (a yfinance -> parquet round trip yields ``datetime64[ms]``). Dividing that
    view by 1e9 rescaled every estimate: one day read as 0.0864 seconds, giving
    365_250_000 periods per year and a Sharpe of 441 on real SPY data.
    """
    if len(index) < 2:
        return np.empty(0, dtype=float)
    return (index[1:] - index[:-1]).total_seconds().to_numpy().astype(float)


def _infer_trading_calendar(index: pd.DatetimeIndex) -> FrequencyResolution | None:
    """Recognise a daily trading calendar: weekday-regular, wall-clock irregular.

    Real daily equity data is never evenly spaced -- weekends and market
    holidays see to that -- so it can only be classified by calendar shape, not
    by constant spacing. Returns ``None`` when the index is anything else, so
    the caller still applies ``irregular_timestamp_policy``.
    """
    naive = index.tz_convert(None) if index.tz is not None else index
    days = naive.normalize()
    if days.has_duplicates:
        return None  # multiple bars per day: intraday, not a daily calendar
    if not bool((days.day_of_week < 5).all()):
        return None  # weekend timestamps: a 7-day calendar, not a trading one

    calendar_gaps = (days[1:] - days[:-1]).days.to_numpy()
    if int(calendar_gaps.max()) > MAX_TRADING_CALENDAR_GAP_DAYS:
        return None  # a hole in the data, not a market holiday

    as_dates = days.to_numpy().astype("datetime64[D]")
    business_gaps = np.busday_count(as_dates[:-1], as_dates[1:])
    step = int(np.median(business_gaps))
    if step < 1:
        return None
    if int(business_gaps.min()) < step:
        return None  # spacing tighter than the step: not a fixed-stride calendar

    periods = max(int(round(TRADING_DAYS_PER_YEAR / step)), 1)
    source = "inferred:trading_calendar" if step == 1 else f"inferred:{step}B"
    return FrequencyResolution(periods_per_year=periods, source=source)


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
    gaps = _gap_seconds(index)
    if len(gaps) == 0:
        return FrequencyResolution(periods_per_year=fallback_periods_per_year, source="configured")

    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        raise ValueError("timestamps must be strictly increasing")

    # Absolute tolerance in seconds, with a one-second floor. Expressing this in
    # seconds rather than raw index integers keeps it independent of resolution.
    tolerance = max(median_gap * 1e-6, 1.0)
    if not bool((np.abs(gaps - median_gap) > tolerance).any()):
        inferred_periods = max(int(round(SECONDS_PER_YEAR / median_gap)), 1)
        return FrequencyResolution(periods_per_year=inferred_periods, source="inferred:timedelta")

    calendar = _infer_trading_calendar(index)
    if calendar is not None:
        return calendar

    if irregular_timestamp_policy == "error":
        raise ValueError(
            "irregular timestamps detected; either clean data or set irregular_timestamp_policy to 'fallback'"
        )
    return FrequencyResolution(periods_per_year=fallback_periods_per_year, source="configured_fallback")
