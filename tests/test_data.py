from pathlib import Path

import pytest

from vol_regime_trend_bot.data import load_closes_from_csv


def test_load_closes_from_csv_requires_positive_close(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("close\n100\n0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive"):
        load_closes_from_csv(csv_path)


def test_load_closes_from_csv_parses_valid_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("timestamp,close\n2026-01-01,100\n2026-01-02,101\n", encoding="utf-8")

    closes = load_closes_from_csv(csv_path)
    assert closes == [100.0, 101.0]
