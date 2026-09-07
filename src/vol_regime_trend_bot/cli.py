from __future__ import annotations

import argparse
import csv
import sys

from .backtest import run_backtest
from .config import BacktestConfig, RiskConfig, StrategyConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run volatility-regime trend backtest from a CSV file")
    parser.add_argument("--csv", required=True, help="Path to CSV with price column")
    parser.add_argument("--price-column", default="close", help="Price column name")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--trend-window", type=int, default=50)
    parser.add_argument("--vol-threshold", type=float, default=0.02)
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--max-position-change", type=float, default=0.25)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    return parser


def _read_prices(path: str, price_column: str) -> list[float]:
    prices: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or price_column not in reader.fieldnames:
            raise ValueError(f"CSV missing required column: {price_column}")
        for idx, row in enumerate(reader, start=2):
            raw_value = row.get(price_column, "")
            if raw_value is None or raw_value.strip() == "":
                raise ValueError(f"Empty price at line {idx}")
            try:
                price = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Invalid price '{raw_value}' at line {idx}") from exc
            if price <= 0:
                raise ValueError(f"Price must be positive at line {idx}")
            prices.append(price)

    if len(prices) < 2:
        raise ValueError("CSV must include at least 2 price rows")
    return prices


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        prices = _read_prices(args.csv, args.price_column)
        result = run_backtest(
            prices=prices,
            strategy_config=StrategyConfig(
                vol_window=args.vol_window,
                trend_window=args.trend_window,
                vol_threshold=args.vol_threshold,
            ),
            risk_config=RiskConfig(
                target_vol=args.target_vol,
                max_exposure=args.max_exposure,
                max_position_change=args.max_position_change,
            ),
            backtest_config=BacktestConfig(
                periods_per_year=args.periods_per_year,
                fee_bps=args.fee_bps,
            ),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"total_return={result.total_return:.4f}")
    print(f"annual_return={result.annual_return:.4f}")
    print(f"annual_volatility={result.annual_volatility:.4f}")
    print(f"sharpe={result.sharpe:.4f}")
    print(f"max_drawdown={result.max_drawdown:.4f}")
    print(f"win_rate={result.win_rate:.4f}")
    print(f"avg_exposure={result.avg_exposure:.4f}")
    print(f"trades={result.trades}")
    print(f"periods_per_year={result.periods_per_year}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

