# vol-regime-trend-bot

Trading bot implementing a volatility-regime + trend-filter strategy with risk-aware position sizing and backtesting.

## Features

- Volatility-regime and trend-filter signal generation
- Risk-based position scaling with exposure and turnover limits
- Backtest metrics with configurable annualization (`periods_per_year`)
- CLI for running a CSV backtest
- Pytest coverage for strategy, risk/backtest, and CLI behavior

## Quickstart

```bash
python -m pip install -e .
python -m pytest
```

## Run backtest from CSV

CSV must include a positive price column (default: `close`):

```bash
vol-regime-backtest --csv ./prices.csv --price-column close --periods-per-year 252
```

Optional controls:

- `--vol-window`, `--trend-window`, `--vol-threshold`
- `--target-vol`, `--max-exposure`, `--max-position-change`
- `--fee-bps`
