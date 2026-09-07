from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class OrderIntent:
    idempotency_key: str
    symbol: str
    side: str
    quantity: float
    timestamp: str
    reason: str


class ExecutionAdapter(Protocol):
    def submit_order_intent(self, intent: OrderIntent) -> None: ...


@dataclass
class PaperExecutionAdapter:
    submitted: list[OrderIntent] = field(default_factory=list)

    def submit_order_intent(self, intent: OrderIntent) -> None:
        self.submitted.append(intent)


@dataclass
class ExecutionState:
    last_position: float = 0.0
    last_timestamp: str | None = None
    seen_intent_keys: set[str] = field(default_factory=set)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["seen_intent_keys"] = sorted(self.seen_intent_keys)
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "ExecutionState":
        payload = json.loads(raw)
        return cls(
            last_position=float(payload.get("last_position", 0.0)),
            last_timestamp=payload.get("last_timestamp"),
            seen_intent_keys=set(payload.get("seen_intent_keys", [])),
        )


def load_execution_state(path: str | None) -> ExecutionState:
    if not path:
        return ExecutionState()
    state_path = Path(path)
    if not state_path.exists():
        return ExecutionState()
    return ExecutionState.from_json(state_path.read_text())


def save_execution_state(path: str | None, state: ExecutionState) -> None:
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.to_json())


def build_order_intent(
    symbol: str,
    timestamp: str,
    target_position: float,
    current_position: float,
    reason: str,
) -> OrderIntent | None:
    delta = float(target_position) - float(current_position)
    if abs(delta) < 1e-12:
        return None
    side = "buy" if delta > 0 else "sell"
    quantity = abs(delta)
    key_basis = f"{symbol}|{timestamp}|{round(target_position, 12)}|{round(current_position, 12)}|{reason}"
    key = hashlib.sha256(key_basis.encode("utf-8")).hexdigest()[:20]
    return OrderIntent(
        idempotency_key=key,
        symbol=symbol,
        side=side,
        quantity=quantity,
        timestamp=timestamp,
        reason=reason,
    )


def submit_intent_idempotent(
    adapter: ExecutionAdapter,
    intent: OrderIntent,
    state: ExecutionState,
) -> bool:
    if intent.idempotency_key in state.seen_intent_keys:
        return False
    adapter.submit_order_intent(intent)
    state.seen_intent_keys.add(intent.idempotency_key)
    return True
