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
1. infer from validated timestamps when spacing is regular,
2. fallback to configured `periods_per_year` only when inference is unavailable or when `--irregular-timestamp-policy fallback` is used.

`metrics.periods_per_year` and `metrics.annualization_source` are returned so CLI and library outputs are consistent and auditable.

## Risk controls (safe defaults)

- Volatility-targeted sizing (`target_annual_volatility`, `min_abs_position`, `max_leverage`)
- Exposure caps and rate-of-change limiter (`max_abs_position`, `max_position_change`)
- Drawdown kill-switch + cooldown + re-entry threshold (`drawdown_kill_switch`, `cooldown_bars`, `drawdown_reentry`)
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
- **`irregular timestamps detected`**: clean spacing or run with `--irregular-timestamp-policy fallback` and set `--periods-per-year`.
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
