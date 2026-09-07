from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import infer_periods_per_year, validate_ohlcv_input
from .execution import (
    ExecutionAdapter,
    build_order_intent,
    load_execution_state,
    save_execution_state,
    submit_intent_idempotent,
)
from .risk import DrawdownProtectionState, apply_drawdown_protection, apply_exposure_limits, apply_volatility_targeting
from .signals import build_signal_pipeline

LOGGER = logging.getLogger(__name__)


def _log_event(event: str, **payload: Any) -> None:
    LOGGER.info(json.dumps({"event": event, **payload}, sort_keys=True, default=str))


def run_backtest(
    data: pd.DataFrame,
    config: BacktestConfig,
    price_column: str = "close",
    timestamp_column: str | None = None,
    volume_column: str | None = "volume",
    execution_adapter: ExecutionAdapter | None = None,
    execution_state_path: str | None = None,
    symbol: str = "ASSET",
) -> dict[str, Any]:
    config.validate()
    np.random.seed(config.seed)

    prepared, prices = validate_ohlcv_input(
        data,
        price_column=price_column,
        timestamp_column=timestamp_column,
        volume_column=volume_column,
    )
    if (execution_adapter is not None or execution_state_path is not None) and not isinstance(
        prepared.index, pd.DatetimeIndex
    ):
        raise ValueError(
            "execution intents/state require timestamped data; provide timestamp_column or a DatetimeIndex"
        )
    freq = infer_periods_per_year(
        prepared.index,
        fallback_periods_per_year=config.periods_per_year,
        irregular_timestamp_policy=config.irregular_timestamp_policy,
    )
    periods_per_year = freq.periods_per_year

    _log_event(
        "data_ingest_complete",
        rows=int(len(prepared)),
        datetime_index=isinstance(prepared.index, pd.DatetimeIndex),
        annualization_source=freq.source,
        periods_per_year=periods_per_year,
    )

    signal_frame = build_signal_pipeline(prices, config)
    raw_target = signal_frame["raw_signal"]
    sized_target = apply_volatility_targeting(
        raw_target_positions=raw_target,
        realized_volatility=signal_frame["realized_vol"],
        target_annual_volatility=config.target_annual_volatility,
        periods_per_year=periods_per_year,
        min_abs_position=config.min_abs_position,
        max_leverage=config.max_leverage,
    )
    rate_limited_target = apply_exposure_limits(
        sized_target,
        max_abs_position=min(config.max_abs_position, config.max_leverage),
        max_position_change=config.max_position_change,
    )
    _log_event(
        "signal_generation_complete",
        trend_positives=int((signal_frame["trend_signal"] > 0).sum()),
        regime_positives=int((signal_frame["regime_signal"] > 0).sum()),
        active_targets=int((rate_limited_target != 0.0).sum()),
    )

    returns = prices.pct_change().fillna(0.0)

    position_values: list[float] = []
    gross_return_values: list[float] = []
    turnover_values: list[float] = []
    fees_values: list[float] = []
    slippage_values: list[float] = []
    net_return_values: list[float] = []
    equity_values: list[float] = []
    drawdown_values: list[float] = []
    kill_switch_flags: list[bool] = []

    execution_state = load_execution_state(execution_state_path)
    drawdown_state = DrawdownProtectionState(
        kill_active=execution_state.kill_active,
        cooldown_remaining=execution_state.cooldown_remaining,
    )

    processed_prepared = prepared
    processed_signal = signal_frame
    processed_target = rate_limited_target
    processed_returns = returns
    if execution_state.last_timestamp is not None and isinstance(prepared.index, pd.DatetimeIndex):
        cutoff = pd.Timestamp(execution_state.last_timestamp)
        mask = prepared.index > cutoff
        processed_prepared = prepared.loc[mask]
        processed_signal = signal_frame.loc[mask]
        processed_target = rate_limited_target.loc[mask]
        processed_returns = returns.loc[mask]
    if len(processed_prepared) == 0:
        _log_event("run_no_new_data", last_timestamp=execution_state.last_timestamp)
        empty = processed_prepared.copy()
        for column in (
            "position",
            "returns",
            "trend_signal",
            "regime_signal",
            "realized_vol",
            "raw_target",
            "sized_target",
            "gross_return",
            "turnover",
            "fees",
            "slippage",
            "net_return",
            "equity_curve",
            "drawdown",
            "kill_switch_triggered",
        ):
            empty[column] = pd.Series(dtype=float if column != "kill_switch_triggered" else bool)
        return {
            "metrics": {
                "total_return": 0.0,
                "final_equity": float(execution_state.equity),
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": float(execution_state.current_drawdown),
                "win_rate": 0.0,
                "trades": 0,
                "average_turnover": 0.0,
                "fee_drag": 0.0,
                "slippage_drag": 0.0,
                "average_abs_position": 0.0,
                "periods_per_year": int(periods_per_year),
                "annualization_source": freq.source,
                "kill_switch_events": 0,
                "regime_win_rate": 0.0,
            },
            "results": empty,
        }

    prev_position = float(execution_state.last_position)
    equity = float(execution_state.equity)
    starting_equity = equity
    rolling_peak = float(execution_state.rolling_peak)
    current_drawdown = float(execution_state.current_drawdown)

    for idx, target_position in processed_target.items():
        protected_position, drawdown_state, kill_triggered = apply_drawdown_protection(
            desired_position=float(target_position),
            current_drawdown=current_drawdown,
            state=drawdown_state,
            kill_switch=config.drawdown_kill_switch,
            reentry_drawdown=config.drawdown_reentry,
            cooldown_bars=config.cooldown_bars,
        )

        turnover = abs(protected_position - prev_position)
        gross_return = prev_position * float(processed_returns.loc[idx])
        fees = turnover * (config.fee_bps / 10_000.0)
        slippage = turnover * (config.slippage_bps / 10_000.0)
        net_return = gross_return - fees - slippage

        equity *= 1.0 + net_return
        rolling_peak = max(rolling_peak, equity)
        current_drawdown = equity / rolling_peak - 1.0

        if execution_adapter is not None:
            ts = str(idx)
            intent = build_order_intent(
                symbol=symbol,
                timestamp=ts,
                target_position=protected_position,
                current_position=prev_position,
                reason="target_position_update",
            )
            if intent is not None:
                submitted = submit_intent_idempotent(execution_adapter, intent, execution_state)
                _log_event(
                    "order_intent",
                    timestamp=ts,
                    idempotency_key=intent.idempotency_key,
                    submitted=submitted,
                    side=intent.side,
                    quantity=intent.quantity,
                )
        if kill_triggered:
            _log_event("risk_kill_switch_triggered", timestamp=str(idx), drawdown=current_drawdown)

        position_values.append(protected_position)
        gross_return_values.append(gross_return)
        turnover_values.append(turnover)
        fees_values.append(fees)
        slippage_values.append(slippage)
        net_return_values.append(net_return)
        equity_values.append(equity)
        drawdown_values.append(current_drawdown)
        kill_switch_flags.append(kill_triggered)

        prev_position = protected_position

    execution_state.last_position = prev_position
    execution_state.equity = equity
    execution_state.rolling_peak = rolling_peak
    execution_state.current_drawdown = current_drawdown
    execution_state.kill_active = drawdown_state.kill_active
    execution_state.cooldown_remaining = drawdown_state.cooldown_remaining
    if len(processed_prepared.index) > 0 and isinstance(processed_prepared.index, pd.DatetimeIndex):
        execution_state.last_timestamp = str(processed_prepared.index[-1])
    save_execution_state(execution_state_path, execution_state)

    result_index = processed_prepared.index
    position = pd.Series(position_values, index=result_index, dtype=float)
    gross_return_series = pd.Series(gross_return_values, index=result_index, dtype=float)
    turnover = pd.Series(turnover_values, index=result_index, dtype=float)
    fees = pd.Series(fees_values, index=result_index, dtype=float)
    slippage = pd.Series(slippage_values, index=result_index, dtype=float)
    net_return = pd.Series(net_return_values, index=result_index, dtype=float)
    equity_curve = pd.Series(equity_values, index=result_index, dtype=float)
    drawdown = pd.Series(drawdown_values, index=result_index, dtype=float)
    kill_switch_series = pd.Series(kill_switch_flags, index=result_index, dtype=bool)

    observed_periods = max(len(net_return) - 1, 1)
    terminal_equity = float(equity)
    total_return = terminal_equity / starting_equity - 1.0 if starting_equity > 0 else 0.0
    avg = float(net_return.mean()) if len(net_return) > 0 else 0.0
    vol = float(net_return.std(ddof=0)) if len(net_return) > 0 else 0.0
    if 1.0 + total_return > 0:
        annual_return = (1.0 + total_return) ** (periods_per_year / observed_periods) - 1.0
    else:
        annual_return = -1.0
    annual_vol = vol * np.sqrt(periods_per_year)
    sharpe = (avg / vol) * np.sqrt(periods_per_year) if vol > 0 else 0.0

    shifted_position = position.shift(1).fillna(0.0)
    active_periods = (shifted_position != 0.0) | (turnover > 0.0)
    win_rate = float((net_return[active_periods] > 0).mean()) if bool(active_periods.any()) else 0.0
    avg_turnover = float(turnover.iloc[1:].mean()) if len(turnover) > 1 else 0.0
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    regime_active = processed_signal["regime_signal"] > 0.0
    regime_net_returns = net_return[regime_active]

    metrics = {
        "total_return": total_return,
        "final_equity": terminal_equity,
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": int((turnover.iloc[1:] > 0).sum()),
        "average_turnover": avg_turnover,
        "fee_drag": float(fees.sum()),
        "slippage_drag": float(slippage.sum()),
        "average_abs_position": float(position.abs().mean()),
        "periods_per_year": int(periods_per_year),
        "annualization_source": freq.source,
        "kill_switch_events": int(kill_switch_series.sum()),
        "regime_win_rate": float((regime_net_returns > 0).mean()) if len(regime_net_returns) > 0 else 0.0,
    }

    _log_event(
        "run_summary",
        total_return=metrics["total_return"],
        sharpe=metrics["sharpe"],
        max_drawdown=metrics["max_drawdown"],
        trades=metrics["trades"],
    )

    result = processed_prepared.copy()
    result["position"] = position
    result["returns"] = processed_returns
    result["trend_signal"] = processed_signal["trend_signal"]
    result["regime_signal"] = processed_signal["regime_signal"]
    result["realized_vol"] = processed_signal["realized_vol"]
    result["raw_target"] = raw_target.loc[result_index]
    result["sized_target"] = sized_target.loc[result_index]
    result["gross_return"] = gross_return_series
    result["turnover"] = turnover
    result["fees"] = fees
    result["slippage"] = slippage
    result["net_return"] = net_return
    result["equity_curve"] = equity_curve
    result["drawdown"] = drawdown
    result["kill_switch_triggered"] = kill_switch_series

    return {"metrics": metrics, "results": result}
