# vol-regime-trend-bot

Trading bot implementing volatility regime + trend filter strategy with backtesting and tests.

## Quick start

```bash
python -m pip install -e .
pytest
```

Run a backtest from CSV (requires `close` column):

```bash
vol-regime-backtest /path/to/prices.csv --trend-window 50 --vol-window 20 --regime-window 100
```
