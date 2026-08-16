from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.research.rankpulse_strategy_rules import extreme_weak_exit_time_ms


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
class BootstrapMetadata:
    bootstrap_completed: bool = False
    activation_time_ms: int | None = None
    strategy_version: str = "rank1_candidate_v1"


@dataclass(frozen=True)
class BootstrapVirtualPosition:
    source: str
    strategy_version: str
    unique_key: str
    symbol: str
    rank: int
    entry_time_ms: int
    entry_price: float
    qty: float
    leverage: int
    planned_exit_time_ms: int
    weak_exit_check_time_ms: int
    extreme_weak_exit_check_time_ms: int
    gain_24h: float
    volume_24h_ratio_7d: float | None
    extreme_weak_exit_checked: bool = False
    weak_exit_checked: bool = False


@dataclass(frozen=True)
class BootstrapClosedVirtualPosition:
    source: str
    strategy_version: str
    unique_key: str
    symbol: str
    rank: int
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    qty: float
    leverage: int
    exit_reason: str


@dataclass(frozen=True)
class TestnetState:
    __test__ = False

    open_positions: list[TestnetPosition]
    closed_positions: list[TestnetClosedPosition]
    bootstrap_metadata: BootstrapMetadata = field(default_factory=BootstrapMetadata)
    bootstrap_virtual_positions: list[BootstrapVirtualPosition] = field(default_factory=list)
    closed_bootstrap_virtual_positions: list[BootstrapClosedVirtualPosition] = field(default_factory=list)
    last_signal_time_ms: int | None = None
    last_exit_check_time_ms: int | None = None
    last_information_time_ms: int | None = None
    last_preflight_time_ms: int | None = None

    def open_position(self, symbol: str) -> TestnetPosition | None:
        return next((position for position in self.open_positions if position.symbol == symbol), None)

    def bootstrap_virtual_position(self, symbol: str) -> BootstrapVirtualPosition | None:
        return next(
            (position for position in self.bootstrap_virtual_positions if position.symbol == symbol),
            None,
        )

    def blocking_position_count(self) -> int:
        return len(self.open_positions) + len(self.bootstrap_virtual_positions)


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
            bootstrap_metadata=BootstrapMetadata(**raw_state.get("bootstrap_metadata", {})),
            bootstrap_virtual_positions=[
                BootstrapVirtualPosition(**item)
                for item in raw_state.get("bootstrap_virtual_positions", [])
            ],
            closed_bootstrap_virtual_positions=[
                BootstrapClosedVirtualPosition(**item)
                for item in raw_state.get("closed_bootstrap_virtual_positions", [])
            ],
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
                    "bootstrap_metadata": asdict(state.bootstrap_metadata),
                    "bootstrap_virtual_positions": [
                        asdict(position)
                        for position in state.bootstrap_virtual_positions
                    ],
                    "closed_bootstrap_virtual_positions": [
                        asdict(position)
                        for position in state.closed_bootstrap_virtual_positions
                    ],
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
                bootstrap_metadata=state.bootstrap_metadata,
                bootstrap_virtual_positions=state.bootstrap_virtual_positions,
                closed_bootstrap_virtual_positions=state.closed_bootstrap_virtual_positions,
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
                bootstrap_metadata=state.bootstrap_metadata,
                bootstrap_virtual_positions=[
                    BootstrapVirtualPosition(
                        **{
                            **asdict(position),
                            "weak_exit_checked": True,
                        }
                    )
                    if position.symbol == symbol
                    else position
                    for position in state.bootstrap_virtual_positions
                ],
                closed_bootstrap_virtual_positions=state.closed_bootstrap_virtual_positions,
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
                bootstrap_metadata=state.bootstrap_metadata,
                bootstrap_virtual_positions=[
                    BootstrapVirtualPosition(
                        **{
                            **asdict(position),
                            "extreme_weak_exit_checked": True,
                        }
                    )
                    if position.symbol == symbol
                    else position
                    for position in state.bootstrap_virtual_positions
                ],
                closed_bootstrap_virtual_positions=state.closed_bootstrap_virtual_positions,
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )

    def close_bootstrap_virtual_position(
        self,
        symbol: str,
        exit_time_ms: int,
        exit_price: float,
        reason: str,
    ) -> BootstrapClosedVirtualPosition | None:
        state = self.load()
        position = state.bootstrap_virtual_position(symbol)
        if position is None:
            return None
        closed = BootstrapClosedVirtualPosition(
            source=position.source,
            strategy_version=position.strategy_version,
            unique_key=position.unique_key,
            symbol=position.symbol,
            rank=position.rank,
            entry_time_ms=position.entry_time_ms,
            exit_time_ms=exit_time_ms,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            leverage=position.leverage,
            exit_reason=reason,
        )
        self.save(
            TestnetState(
                open_positions=state.open_positions,
                closed_positions=state.closed_positions,
                bootstrap_metadata=state.bootstrap_metadata,
                bootstrap_virtual_positions=[
                    item
                    for item in state.bootstrap_virtual_positions
                    if item.symbol != symbol
                ],
                closed_bootstrap_virtual_positions=[
                    *state.closed_bootstrap_virtual_positions,
                    closed,
                ],
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )
        return closed


def _position_with_defaults(raw_position: dict[str, object]) -> dict[str, object]:
    position = dict(raw_position)
    position.setdefault(
        "extreme_weak_exit_check_time_ms",
        extreme_weak_exit_time_ms(int(position["entry_time_ms"])),
    )
    position.setdefault("extreme_weak_exit_checked", False)
    return position
