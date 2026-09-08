from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

import pandas as pd

from .backtest import run_backtest
from .config import BacktestConfig
from .execution import PaperExecutionAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run volatility-regime trend-filter backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV with OHLCV/close data")
    parser.add_argument("--config", help="Path to JSON runtime config file")
    parser.add_argument("--price-column", default="close", help="Column used as traded price")
    parser.add_argument("--timestamp-column", default=None, help="Timestamp column to parse as UTC")
    parser.add_argument("--volume-column", default="volume", help="Volume column for validation")

    trend = parser.add_mutually_exclusive_group()
    trend.add_argument("--use-trend-filter", dest="use_trend_filter", action="store_true",
                       default=None, help="Gate the book on the trend signal (measured subtractive)")
    trend.add_argument("--no-trend-filter", dest="use_trend_filter", action="store_false",
                       default=None, help="Size on the volatility regime alone (default)")
    parser.add_argument("--trend-window", type=int, default=None)
    parser.add_argument("--vol-window", type=int, default=None)
    parser.add_argument("--max-volatility", type=float, default=None)
    parser.add_argument("--vol-hysteresis-buffer", type=float, default=None)
    vol_change = parser.add_mutually_exclusive_group()
    vol_change.add_argument(
        "--require-falling-volatility",
        dest="require_falling_volatility",
        action="store_true",
        default=None,
        help="Only trade when realized volatility is flat or falling",
    )
    vol_change.add_argument(
        "--allow-rising-volatility",
        dest="require_falling_volatility",
        action="store_false",
        default=None,
        help="Disable the volatility-change gate (default)",
    )
    parser.add_argument("--volatility-change-window", type=int, default=None)
    parser.add_argument("--signal-confirmation-bars", type=int, default=None)
    parser.add_argument("--max-abs-position", type=float, default=None)
    parser.add_argument("--min-abs-position", type=float, default=None)
    parser.add_argument("--max-position-change", type=float, default=None)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--target-annual-volatility", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--periods-per-year", type=int, default=None)
    parser.add_argument(
        "--irregular-timestamp-policy",
        choices=["error", "fallback"],
        default=None,
        help="How to handle irregular timestamp spacing",
    )
    parser.add_argument("--drawdown-kill-switch", type=float, default=None)
    parser.add_argument("--drawdown-reentry", type=float, default=None)
    parser.add_argument("--cooldown-bars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--paper", action="store_true", help="Generate idempotent paper order intents")
    parser.add_argument("--state-path", default=None, help="State checkpoint path for paper/backtest resumability")
    parser.add_argument("--symbol", default="ASSET", help="Symbol used in order intents")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _build_cli_config(args: argparse.Namespace) -> dict[str, bool | float | int | str]:
    raw = {
        "use_trend_filter": args.use_trend_filter,
        "trend_window": args.trend_window,
        "vol_window": args.vol_window,
        "max_volatility": args.max_volatility,
        "vol_hysteresis_buffer": args.vol_hysteresis_buffer,
        "require_falling_volatility": args.require_falling_volatility,
        "volatility_change_window": args.volatility_change_window,
        "signal_confirmation_bars": args.signal_confirmation_bars,
        "max_abs_position": args.max_abs_position,
        "min_abs_position": args.min_abs_position,
        "max_position_change": args.max_position_change,
        "max_leverage": args.max_leverage,
        "target_annual_volatility": args.target_annual_volatility,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "periods_per_year": args.periods_per_year,
        "irregular_timestamp_policy": args.irregular_timestamp_policy,
        "drawdown_kill_switch": args.drawdown_kill_switch,
        "drawdown_reentry": args.drawdown_reentry,
        "cooldown_bars": args.cooldown_bars,
        "seed": args.seed,
    }
    return {k: v for k, v in raw.items() if v is not None}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        logging.basicConfig(level=getattr(logging, args.log_level))

        file_overrides = BacktestConfig.from_json_file(args.config) if args.config else {}
        env_overrides = BacktestConfig.from_env()
        cli_overrides = _build_cli_config(args)
        cfg = BacktestConfig().apply_overrides(file_overrides).apply_overrides(env_overrides).apply_overrides(
            cli_overrides
        )
        cfg.validate()

        data = pd.read_csv(args.csv)
        adapter = PaperExecutionAdapter() if args.paper else None
        output = run_backtest(
            data,
            cfg,
            price_column=args.price_column,
            timestamp_column=args.timestamp_column,
            volume_column=args.volume_column,
            execution_adapter=adapter,
            execution_state_path=args.state_path,
            symbol=args.symbol,
        )
    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        parser.error(str(exc))

    payload = output["metrics"].copy()
    payload["config"] = asdict(cfg)
    if args.paper and adapter is not None:
        payload["paper_intents"] = [intent.__dict__ for intent in adapter.submitted]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
