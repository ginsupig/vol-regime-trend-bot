from __future__ import annotations

import argparse
import json

from .backtest import run_backtest
from .data import load_closes_from_csv
from .strategy import StrategyConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run volatility regime trend backtest")
    parser.add_argument("csv", help="Path to CSV containing at least a close column")
    parser.add_argument("--trend-window", type=int, default=50)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--regime-window", type=int, default=100)
    parser.add_argument("--risk-budget", type=float, default=0.10)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--max-gross-exposure", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    closes = load_closes_from_csv(args.csv)
    result = run_backtest(
        closes,
        strategy_config=StrategyConfig(
            trend_window=args.trend_window,
            vol_window=args.vol_window,
            regime_window=args.regime_window,
        ),
        risk_budget=args.risk_budget,
        max_leverage=args.max_leverage,
        max_gross_exposure=args.max_gross_exposure,
    )
    print(
        json.dumps(
            {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "annualized_vol": result.annualized_vol,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "trades": result.trades,
                "win_rate": result.win_rate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
