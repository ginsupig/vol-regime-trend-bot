import pandas as pd

from .config import BacktestConfig


def generate_target_positions(prices: pd.Series, config: BacktestConfig) -> pd.Series:
    config.validate()
    if prices.isna().all():
        raise ValueError("price series must include non-null values")

    prices = prices.astype(float)
    returns = prices.pct_change()
    trend = prices.rolling(config.trend_window, min_periods=config.trend_window).mean()
    realized_vol = returns.rolling(config.vol_window, min_periods=config.vol_window).std()

    trend_ok = prices > trend
    regime_ok = realized_vol <= config.max_volatility
    signal = (trend_ok & regime_ok).fillna(False)

    return signal.astype(float)
