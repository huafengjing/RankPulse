from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.modes import SignalMode, TradingMode


@dataclass(frozen=True)
class StatePaths:
    state_path: Path
    event_log_path: Path


def state_paths_for(
    trading_mode: TradingMode,
    signal_mode: SignalMode,
    root: str | Path = "data",
) -> StatePaths:
    base = Path(root) / trading_mode.value / signal_mode.value
    return StatePaths(
        state_path=base / "state.json",
        event_log_path=base / "events.jsonl",
    )
