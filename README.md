# vol-regime-trend-bot

Trading bot implementing a volatility-regime + trend edge pipeline with backtesting, risk controls, and paper-execution readiness.

## Quickstart

```bash
pip install -e .

# minimal backtest
vol-regime-backtest --csv /path/to/prices.csv

# timestamp-aware backtest with paper intents + resumable state
vol-regime-backtest \
  --csv /path/to/ohlcv.csv \
  --timestamp-column timestamp \
  --paper \
  --state-path /tmp/vol-regime/state.json \
  --log-level INFO
```

## Runtime configuration

`BacktestConfig` is the single runtime config surface.

Precedence (lowest to highest):
1. built-in safe defaults,
2. `--config` JSON file,
3. env vars (`VRTB_<FIELD_NAME>`),
4. CLI overrides.

Example config file:

```json
{
  "trend_window": 50,
  "vol_window": 20,
  "max_volatility": 0.03,
  "max_abs_position": 1.0,
  "target_annual_volatility": 0.15,
  "drawdown_kill_switch": -0.2,
  "cooldown_bars": 5
}
```

Invalid or unknown config values fail fast with explicit errors.

## Annualization behavior

Metrics use frequency-aware annualization with this precedence:
1. `pandas` offset inference (`inferred:B`, `inferred:2B`, ...),
2. constant timestamp spacing, measured in seconds (`inferred:timedelta`),
3. trading-calendar recognition (`inferred:trading_calendar`) — weekday-only
   timestamps at a fixed business-day stride, which is what real daily equity
   data looks like once weekends and market holidays are taken out,
4. fallback to configured `periods_per_year` only when inference is unavailable or when `--irregular-timestamp-policy fallback` is used.

Spacing is compared in **seconds**, not in the index's raw integer
representation. pandas 2/3 preserve `datetime64[s|ms|us]` indices rather than
coercing everything to nanoseconds, so a resolution-dependent comparison
silently rescales every estimate — a `datetime64[ms]` daily index read one day
as 0.0864 seconds and reported 365,250,000 periods per year.

A daily trading calendar annualizes at 252, which is a measurement rather than a
convention: SPY, GLD and TLT daily bars over 2000–2026 average 251.5 trading days
per calendar year.

`metrics.periods_per_year` and `metrics.annualization_source` are returned so CLI and library outputs are consistent and auditable.

## Risk controls (safe defaults)

- Volatility-targeted sizing (`target_annual_volatility`, `min_abs_position`, `max_leverage`)
- Exposure caps and rate-of-change limiter (`max_abs_position`, `max_position_change`)
- Drawdown kill-switch + cooldown + re-entry threshold (`drawdown_kill_switch`, `cooldown_bars`, `drawdown_reentry`)
  - The switch **arms** on realized equity drawdown and **releases** on the
    drawdown of an unprotected reference curve — the same strategy with the
    switch removed. A flat book earns nothing, so realized equity, its rolling
    peak and its drawdown all freeze the moment the switch fires; gating
    re-entry on that frozen value latches the switch on permanently.
  - On release the drawdown reference is reset to current equity, so the
    strategy resumes with a fresh budget instead of re-tripping against the
    stale pre-kill peak. `metrics.kill_switch_releases` reports how often this
    happened — if it is 0 while `kill_switch_events` is not, the book is flat.
- Transaction costs (`fee_bps`) and slippage (`slippage_bps`) included in PnL path

## Data quality requirements

Input validation includes:
- required price column,
- finite positive prices,
- timestamp parsing to UTC,
- monotonic strictly increasing timestamps,
- duplicate timestamp detection,
- optional OHLC consistency checks,
- non-negative volume checks.

Irregular timestamp spacing fails by default (`error`) and can be explicitly downgraded to `fallback`.

## Troubleshooting

- **`missing required price column`**: pass `--price-column` or fix your CSV headers.
- **`duplicate timestamp detected` / `timestamps must be strictly increasing`**: de-duplicate and sort your data.
- **`irregular timestamps detected`**: weekends and market holidays are recognized automatically, so this now means a genuine hole in the data (any gap over 10 calendar days — longer than the 7-day 2001 market closure) or mixed spacing. Clean the data, or run with `--irregular-timestamp-policy fallback` and set `--periods-per-year`.
- **`invalid JSON config` / `unknown config fields`**: validate config schema against `BacktestConfig`.

## Architecture summary

- `data.py`: OHLCV validation + timestamp/frequency handling
- `signals.py`: modular trend/regime signal pipeline
- `risk.py`: sizing + guardrails + drawdown protection
- `execution.py`: paper adapter, idempotent order intents, checkpoint state
- `backtest.py`: deterministic run loop, metrics, diagnostics
- `cli.py`: operator UX and config wiring

## Tests

Run targeted tests first, then full suite:

```bash
pytest tests/test_strategy.py tests/test_risk_and_backtest.py tests/test_cli.py -q
pytest -q
```
