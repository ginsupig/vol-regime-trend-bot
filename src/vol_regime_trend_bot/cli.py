import argparse
import json

import pandas as pd

from .backtest import run_backtest
from .config import BacktestConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run volatility-regime trend-filter backtest")
    parser.add_argument("--csv", required=True, help="Path to input CSV with a close column")
    parser.add_argument("--price-column", default="close")
    parser.add_argument("--trend-window", type=int, default=50)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--max-volatility", type=float, default=0.03)
    parser.add_argument("--max-abs-position", type=float, default=1.0)
    parser.add_argument("--max-position-change", type=float, default=0.25)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BacktestConfig(
        trend_window=args.trend_window,
        vol_window=args.vol_window,
        max_volatility=args.max_volatility,
        max_abs_position=args.max_abs_position,
        max_position_change=args.max_position_change,
        fee_bps=args.fee_bps,
    )

    data = pd.read_csv(args.csv)
    output = run_backtest(data, cfg, price_column=args.price_column)
    print(json.dumps(output["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
