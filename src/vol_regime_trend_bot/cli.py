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

    parser.add_argument("--trend-window", type=int, default=50)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--max-volatility", type=float, default=0.03)
    parser.add_argument("--max-abs-position", type=float, default=1.0)
    parser.add_argument("--min-abs-position", type=float, default=0.0)
    parser.add_argument("--max-position-change", type=float, default=0.25)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--target-annual-volatility", type=float, default=0.15)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument(
        "--irregular-timestamp-policy",
        choices=["error", "fallback"],
        default="error",
        help="How to handle irregular timestamp spacing",
    )
    parser.add_argument("--drawdown-kill-switch", type=float, default=-0.2)
    parser.add_argument("--drawdown-reentry", type=float, default=-0.1)
    parser.add_argument("--cooldown-bars", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--paper", action="store_true", help="Generate idempotent paper order intents")
    parser.add_argument("--state-path", default=None, help="State checkpoint path for paper/backtest resumability")
    parser.add_argument("--symbol", default="ASSET", help="Symbol used in order intents")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _build_cli_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        trend_window=args.trend_window,
        vol_window=args.vol_window,
        max_volatility=args.max_volatility,
        max_abs_position=args.max_abs_position,
        min_abs_position=args.min_abs_position,
        max_position_change=args.max_position_change,
        max_leverage=args.max_leverage,
        target_annual_volatility=args.target_annual_volatility,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        periods_per_year=args.periods_per_year,
        irregular_timestamp_policy=args.irregular_timestamp_policy,
        drawdown_kill_switch=args.drawdown_kill_switch,
        drawdown_reentry=args.drawdown_reentry,
        cooldown_bars=args.cooldown_bars,
        seed=args.seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        logging.basicConfig(level=getattr(logging, args.log_level))

        file_config = BacktestConfig.from_json_file(args.config) if args.config else BacktestConfig()
        env_config = BacktestConfig.from_env()
        cli_config = _build_cli_config(args)
        cfg = file_config.merge(env_config).merge(cli_config)
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
