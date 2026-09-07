from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints


@dataclass(frozen=True)
class BacktestConfig:
    trend_window: int = 50
    vol_window: int = 20
    max_volatility: float = 0.03
    max_abs_position: float = 1.0
    min_abs_position: float = 0.0
    max_position_change: float = 0.25
    max_leverage: float = 1.0
    target_annual_volatility: float = 0.15
    fee_bps: float = 2.0
    slippage_bps: float = 0.0
    periods_per_year: int = 252
    irregular_timestamp_policy: str = "error"  # error|fallback
    drawdown_kill_switch: float = -0.2
    drawdown_reentry: float = -0.1
    cooldown_bars: int = 5
    seed: int = 7

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BacktestConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json_file(cls, path: str | os.PathLike[str]) -> dict[str, Any]:
        cfg_path = Path(path)
        try:
            payload = json.loads(cfg_path.read_text())
        except OSError as exc:
            raise ValueError(f"unable to read config file: {cfg_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON config: {cfg_path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("config file must contain a JSON object")
        config = cls.from_dict(payload)
        config.validate()
        return payload

    @classmethod
    def from_env(cls, prefix: str = "VRTB_") -> dict[str, Any]:
        type_map: dict[type, Any] = {
            int: int,
            float: float,
            str: str,
        }
        hints = get_type_hints(cls)
        parsed: dict[str, Any] = {}
        for field in fields(cls):
            env_key = f"{prefix}{field.name}".upper()
            if env_key not in os.environ:
                continue
            raw = os.environ[env_key]
            caster = type_map.get(hints.get(field.name, field.type))
            if caster is None:
                continue
            try:
                parsed[field.name] = caster(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value for {env_key}: {raw}") from exc
        config = cls.from_dict(parsed)
        config.validate()
        return parsed

    def apply_overrides(self, override: dict[str, Any]) -> "BacktestConfig":
        base = asdict(self)
        base.update(override)
        merged = BacktestConfig.from_dict(base)
        merged.validate()
        return merged

    def validate(self) -> None:
        if self.trend_window <= 1:
            raise ValueError("trend_window must be > 1")
        if self.vol_window <= 1:
            raise ValueError("vol_window must be > 1")
        if self.max_volatility <= 0:
            raise ValueError("max_volatility must be > 0")
        if self.max_abs_position <= 0:
            raise ValueError("max_abs_position must be > 0")
        if self.min_abs_position < 0:
            raise ValueError("min_abs_position must be >= 0")
        if self.min_abs_position > self.max_abs_position:
            raise ValueError("min_abs_position must be <= max_abs_position")
        if self.max_position_change <= 0:
            raise ValueError("max_position_change must be > 0")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.target_annual_volatility <= 0:
            raise ValueError("target_annual_volatility must be > 0")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if self.irregular_timestamp_policy not in {"error", "fallback"}:
            raise ValueError("irregular_timestamp_policy must be 'error' or 'fallback'")
        if self.drawdown_kill_switch >= 0:
            raise ValueError("drawdown_kill_switch must be < 0")
        if self.drawdown_reentry >= 0:
            raise ValueError("drawdown_reentry must be < 0")
        if self.drawdown_reentry < self.drawdown_kill_switch:
            raise ValueError("drawdown_reentry must be >= drawdown_kill_switch")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
