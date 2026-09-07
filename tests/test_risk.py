import pytest

from vol_regime_trend_bot.risk import position_size_from_vol


def test_position_size_respects_exposure_caps() -> None:
    size = position_size_from_vol(
        annualized_vol=0.05,
        risk_budget=0.15,
        max_leverage=0.8,
        max_gross_exposure=0.6,
    )
    assert size == pytest.approx(0.6)


def test_position_size_uses_min_vol_floor() -> None:
    size = position_size_from_vol(
        annualized_vol=0.0,
        risk_budget=0.10,
        max_leverage=0.4,
        max_gross_exposure=1.0,
    )
    assert size == pytest.approx(0.4)
