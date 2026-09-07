from __future__ import annotations


def position_size_from_vol(
    annualized_vol: float,
    risk_budget: float = 0.10,
    max_leverage: float = 1.0,
    max_gross_exposure: float = 1.0,
    min_vol: float = 1e-6,
) -> float:
    if risk_budget <= 0:
        raise ValueError("risk_budget must be > 0")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")
    if max_gross_exposure <= 0:
        raise ValueError("max_gross_exposure must be > 0")
    if min_vol <= 0:
        raise ValueError("min_vol must be > 0")

    vol = max(abs(annualized_vol), min_vol)
    raw_size = risk_budget / vol
    return min(raw_size, max_leverage, max_gross_exposure)
