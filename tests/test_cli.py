import json
import sys
from pathlib import Path

import pytest

from vol_regime_trend_bot.cli import main


def test_cli_runs_and_prints_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = tmp_path / "prices.csv"
    rows = [f"2026-01-{day:02d},{100 + day}" for day in range(1, 61)]
    csv_path.write_text("timestamp,close\n" + "\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vol-regime-backtest",
            str(csv_path),
            "--trend-window",
            "20",
            "--vol-window",
            "10",
            "--regime-window",
            "30",
            "--risk-budget",
            "0.1",
        ],
    )

    main()
    payload = json.loads(capsys.readouterr().out)

    assert "total_return" in payload
    assert "trades" in payload


def test_cli_rejects_invalid_strategy_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("close\n100\n101\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["vol-regime-backtest", str(csv_path), "--trend-window", "1"])

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_cli_rejects_missing_csv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["vol-regime-backtest", "/tmp/does-not-exist.csv"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
