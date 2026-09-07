import pandas as pd

from .config import BacktestConfig
from .signals import build_signal_pipeline


def generate_target_positions(prices: pd.Series, config: BacktestConfig) -> pd.Series:
    """Generate long-only target positions from trend and volatility filters."""
    config.validate()
    if prices.isna().all():
        raise ValueError("price series must include non-null values")

    prices = prices.astype(float)
    return build_signal_pipeline(prices, config)["raw_signal"].astype(float)
