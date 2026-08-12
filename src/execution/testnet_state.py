from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from src.research.top3_strategy_rules import extreme_weak_exit_time_ms


@dataclass(frozen=True)
class TestnetPosition:
    __test__ = False

    symbol: str
    entry_time_ms: int
    entry_price: float
    qty: float
    leverage: int
    order_id: int
    planned_exit_time_ms: int
    weak_exit_check_time_ms: int
    extreme_weak_exit_check_time_ms: int
    extreme_weak_exit_checked: bool = False
    weak_exit_checked: bool = False


@dataclass(frozen=True)
class TestnetClosedPosition:
    symbol: str
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    qty: float
    leverage: int
    entry_order_id: int
    exit_order_id: int
    realized_pnl: float
    exit_reason: str


@dataclass(frozen=True)
class TestnetState:
    __test__ = False

    open_positions: list[TestnetPosition]
    closed_positions: list[TestnetClosedPosition]
    last_signal_time_ms: int | None = None
    last_exit_check_time_ms: int | None = None
    last_information_time_ms: int | None = None
    last_preflight_time_ms: int | None = None

    def open_position(self, symbol: str) -> TestnetPosition | None:
        return next((position for position in self.open_positions if position.symbol == symbol), None)


class TestnetStateStore:
    __test__ = False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> TestnetState:
        if not self.path.exists():
            return TestnetState(open_positions=[], closed_positions=[])
        raw_state = json.loads(self.path.read_text(encoding="utf-8"))
        return TestnetState(
            open_positions=[TestnetPosition(**_position_with_defaults(item)) for item in raw_state.get("open_positions", [])],
            closed_positions=[TestnetClosedPosition(**item) for item in raw_state.get("closed_positions", [])],
            last_signal_time_ms=raw_state.get("last_signal_time_ms"),
            last_exit_check_time_ms=raw_state.get("last_exit_check_time_ms"),
            last_information_time_ms=raw_state.get("last_information_time_ms"),
            last_preflight_time_ms=raw_state.get("last_preflight_time_ms"),
        )

    def save(self, state: TestnetState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "open_positions": [asdict(position) for position in state.open_positions],
                    "closed_positions": [asdict(position) for position in state.closed_positions],
                    "last_signal_time_ms": state.last_signal_time_ms,
                    "last_exit_check_time_ms": state.last_exit_check_time_ms,
                    "last_information_time_ms": state.last_information_time_ms,
                    "last_preflight_time_ms": state.last_preflight_time_ms,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def save_positions(self, open_positions: list[TestnetPosition]) -> None:
        state = self.load()
        self.save(
            TestnetState(
                open_positions=open_positions,
                closed_positions=state.closed_positions,
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )

    def mark_weak_exit_checked(self, symbol: str) -> None:
        state = self.load()
        self.save(
            TestnetState(
                open_positions=[
                    TestnetPosition(
                        **{
                            **asdict(position),
                            "weak_exit_checked": True,
                        }
                    )
                    if position.symbol == symbol
                    else position
                    for position in state.open_positions
                ],
                closed_positions=state.closed_positions,
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )

    def mark_extreme_weak_exit_checked(self, symbol: str) -> None:
        state = self.load()
        self.save(
            TestnetState(
                open_positions=[
                    TestnetPosition(
                        **{
                            **asdict(position),
                            "extreme_weak_exit_checked": True,
                        }
                    )
                    if position.symbol == symbol
                    else position
                    for position in state.open_positions
                ],
                closed_positions=state.closed_positions,
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )


def _position_with_defaults(raw_position: dict[str, object]) -> dict[str, object]:
    position = dict(raw_position)
    position.setdefault(
        "extreme_weak_exit_check_time_ms",
        extreme_weak_exit_time_ms(int(position["entry_time_ms"])),
    )
    position.setdefault("extreme_weak_exit_checked", False)
    return position
