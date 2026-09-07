from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    """Signal parameters; windows are in bars and vol_threshold is per-bar volatility."""

    vol_window: int = 20
    trend_window: int = 50
    vol_threshold: float = 0.02


@dataclass(frozen=True)
class RiskConfig:
    """Risk controls; target_vol is annualized and other values constrain sizing dynamics."""

    target_vol: float = 0.12
    max_exposure: float = 1.0
    max_position_change: float = 0.25
    min_vol_floor: float = 1e-6


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest controls; fee_bps is basis points per unit turnover."""

    periods_per_year: int = 252
    fee_bps: float = 1.0
