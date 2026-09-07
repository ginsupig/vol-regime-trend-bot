import json

import pandas as pd
import pytest

from vol_regime_trend_bot import cli


def test_cli_passes_arguments_to_backtest(monkeypatch, capsys):
    captured = {}

    def fake_read_csv(path):
        captured["csv_path"] = path
        return pd.DataFrame({"adj_close": [100.0, 101.0, 102.0]})

    def fake_run_backtest(data, config, **kwargs):
        captured["kwargs"] = kwargs
        captured["config"] = config
        assert list(data.columns) == ["adj_close"]
        return {"metrics": {"total_return": 0.1}}

    monkeypatch.setattr(cli.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(cli, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vol-regime-backtest",
            "--csv",
            "prices.csv",
            "--price-column",
            "adj_close",
            "--trend-window",
            "10",
            "--vol-window",
            "5",
            "--max-volatility",
            "0.02",
            "--max-abs-position",
            "0.8",
            "--max-position-change",
            "0.2",
            "--fee-bps",
            "3",
            "--periods-per-year",
            "365",
        ],
    )

    cli.main()

    assert captured["csv_path"] == "prices.csv"
    assert captured["kwargs"]["price_column"] == "adj_close"
    assert captured["config"].trend_window == 10
    assert captured["config"].vol_window == 5
    assert captured["config"].max_volatility == pytest.approx(0.02)
    assert captured["config"].max_abs_position == pytest.approx(0.8)
    assert captured["config"].max_position_change == pytest.approx(0.2)
    assert captured["config"].fee_bps == pytest.approx(3.0)
    assert captured["config"].periods_per_year == 365

    payload = json.loads(capsys.readouterr().out)
    assert payload["total_return"] == 0.1
    assert "config" in payload


def test_cli_shows_clean_error_on_invalid_config(monkeypatch, capsys):
    monkeypatch.setattr(cli.pd, "read_csv", lambda _path: pd.DataFrame({"close": [100.0, 101.0]}))
    monkeypatch.setattr(cli, "run_backtest", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid setup")))
    monkeypatch.setattr("sys.argv", ["vol-regime-backtest", "--csv", "prices.csv"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "invalid setup" in capsys.readouterr().err


def test_cli_accepts_config_file_and_paper_mode(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"trend_window": 7, "vol_window": 3}')

    monkeypatch.setattr(cli.pd, "read_csv", lambda _path: pd.DataFrame({"close": [100.0, 101.0, 102.0]}))
    monkeypatch.setenv("VRTB_TREND_WINDOW", "9")
    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda *_args, **_kwargs: {"metrics": {"periods_per_year": 252, "annualization_source": "configured"}},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "vol-regime-backtest",
            "--csv",
            "prices.csv",
            "--config",
            str(config_path),
            "--paper",
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["trend_window"] == 9
    assert payload["paper_intents"] == []
