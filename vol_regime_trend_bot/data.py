from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {"close"}


def load_closes_from_csv(path: str | Path) -> list[float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    closes: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")

        missing = REQUIRED_COLUMNS - set(name.strip().lower() for name in reader.fieldnames if name)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        close_key = next(name for name in reader.fieldnames if name and name.strip().lower() == "close")
        for idx, row in enumerate(reader, start=2):
            raw_close = row.get(close_key)
            if raw_close is None or raw_close.strip() == "":
                raise ValueError(f"Row {idx}: close is empty")
            try:
                close = float(raw_close)
            except ValueError as exc:
                raise ValueError(f"Row {idx}: close must be numeric, got {raw_close!r}") from exc
            if close <= 0:
                raise ValueError(f"Row {idx}: close must be positive")
            closes.append(close)

    if len(closes) < 2:
        raise ValueError("CSV must contain at least 2 close rows")
    return closes
