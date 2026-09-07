from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    trend_window: int = 50
    vol_window: int = 20
    max_volatility: float = 0.03
    max_abs_position: float = 1.0
    max_position_change: float = 0.25
    fee_bps: float = 2.0
    periods_per_year: int = 252

    def validate(self) -> None:
        if self.trend_window <= 1:
            raise ValueError("trend_window must be > 1")
        if self.vol_window <= 1:
            raise ValueError("vol_window must be > 1")
        if self.max_volatility <= 0:
            raise ValueError("max_volatility must be > 0")
        if self.max_abs_position <= 0:
            raise ValueError("max_abs_position must be > 0")
        if self.max_position_change <= 0:
            raise ValueError("max_position_change must be > 0")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be >= 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
