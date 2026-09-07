# vol-regime-trend-bot
Trading bot implementing volatility regime + trend filter strategy with backtesting and tests

## Quickstart

```bash
pip install -e .
vol-regime-backtest --csv /path/to/prices.csv
```

CSV input must include a `close` column by default (or pass `--price-column`).
Use `--periods-per-year` to match your data frequency (default `252` for daily bars).

## What is implemented

- Volatility regime + trend-filter signal generation
- Position exposure controls (max absolute position and max position change per step)
- Backtest engine with fees, equity curve, and core performance metrics
- CLI entrypoint: `vol-regime-backtest`
- Pytest coverage for strategy, risk controls, and backtest validation
