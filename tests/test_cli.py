from pathlib import Path

from vol_regime_trend_bot.cli import main


def test_cli_runs_backtest_and_prints_metrics(tmp_path, capsys):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("close\n100\n101\n102\n103\n", encoding="utf-8")

    code = main(["--csv", str(csv_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "total_return=" in captured.out


def test_cli_returns_error_code_for_missing_file(capsys):
    code = main(["--csv", str(Path("does-not-exist.csv"))])
    captured = capsys.readouterr()

    assert code == 2
    assert "error:" in captured.err

