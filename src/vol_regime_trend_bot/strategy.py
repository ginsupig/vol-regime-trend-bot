import pandas as pd

from .config import BacktestConfig


def generate_target_positions(prices: pd.Series, config: BacktestConfig) -> pd.Series:
    """Generate long-only target positions from trend and volatility filters.

    Rule set:
    - Trend filter passes when `price > rolling_mean(price, trend_window)`.
    - Volatility filter passes when rolling return std (window=`vol_window`, ddof=0)
      is less than or equal to `max_volatility`.
    - Target position is `1.0` only when both filters pass, else `0.0`.

    Returns a float series aligned to `prices` with values in {0.0, 1.0}.
    """
    config.validate()
    if prices.isna().all():
        raise ValueError("price series must include non-null values")

    prices = prices.astype(float)
    returns = prices.pct_change()
    trend = prices.rolling(config.trend_window, min_periods=config.trend_window).mean()
    realized_vol = returns.rolling(config.vol_window, min_periods=config.vol_window).std(ddof=0)

    trend_ok = prices > trend
    regime_ok = realized_vol <= config.max_volatility
    signal = (trend_ok & regime_ok).fillna(False)

    return signal.astype(float)
