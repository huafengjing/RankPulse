from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config.modes import SignalMode, TradingMode
from src.config.settings import AppSettings
from src.exchange.binance_testnet import FuturesPosition
from src.execution.bootstrap import BootstrapManager
from src.execution.testnet_runner import TestnetTradingRunner
from src.execution.testnet_state import BootstrapVirtualPosition, TestnetState, TestnetStateStore
from src.market.binance_futures import Kline, Ticker24hrStat
from src.paper.store import PaperEventLogger
from src.paper.trading import PaperSignal
from src.research.rankpulse_strategy_rules import DAY_MS
from tests.test_testnet_execution import FakeTestnetClient


BEIJING = timezone(timedelta(hours=8))
HOUR_MS = 60 * 60 * 1000


def bj_ms(day: int, hour: int = 8) -> int:
    return int(datetime(2026, 8, day, hour, 0, tzinfo=BEIJING).timestamp() * 1000)


def test_bootstrap_creates_live_virtual_rank1_position_without_real_order() -> None:
    workdir = _clean_workdir("bootstrap_rank1_open")
    activation_ms = bj_ms(15, 8)
    signal_ms = activation_ms - 2 * DAY_MS
    execution_client = FakeTestnetClient()
    manager = _manager(
        workdir,
        activation_ms,
        market=StableExitMarket(),
        provider=StaticBootstrapSignals([_signal("ABCUSDT", signal_ms, rank=1)]),
    )

    virtuals = manager.ensure_bootstrap(activation_ms, execution_client.open_positions())

    assert [position.symbol for position in virtuals] == ["ABCUSDT"]
    assert virtuals[0].source == "BOOTSTRAP_VIRTUAL_POSITION"
    assert virtuals[0].leverage == 3
    assert virtuals[0].planned_exit_time_ms == signal_ms + 5 * DAY_MS
    assert execution_client.calls == []


def test_bootstrap_does_not_keep_position_that_already_hit_12h_weak_exit() -> None:
    workdir = _clean_workdir("bootstrap_weak_exit_before_activation")
    activation_ms = bj_ms(15, 8)
    signal_ms = activation_ms - 2 * DAY_MS
    manager = _manager(
        workdir,
        activation_ms,
        market=WeakExitMarket(),
        provider=StaticBootstrapSignals([_signal("WEAKUSDT", signal_ms, rank=1)]),
    )

    virtuals = manager.ensure_bootstrap(activation_ms, {})

    state = TestnetStateStore(workdir / "state.json").load()
    assert virtuals == []
    assert state.bootstrap_virtual_position("WEAKUSDT") is None
    assert state.closed_bootstrap_virtual_positions[0].exit_reason == "bootstrap_weak_12h"


def test_bootstrap_does_not_keep_position_that_already_hit_4h_extreme_weak_exit() -> None:
    workdir = _clean_workdir("bootstrap_extreme_exit_before_activation")
    activation_ms = bj_ms(15, 8)
    signal_ms = activation_ms - DAY_MS
    manager = _manager(
        workdir,
        activation_ms,
        market=ExtremeWeakExitMarket(),
        provider=StaticBootstrapSignals([_signal("EXTREMEUSDT", signal_ms, rank=1)]),
    )

    virtuals = manager.ensure_bootstrap(activation_ms, {})

    state = TestnetStateStore(workdir / "state.json").load()
    assert virtuals == []
    assert state.bootstrap_virtual_position("EXTREMEUSDT") is None
    assert state.closed_bootstrap_virtual_positions[0].exit_reason == "bootstrap_extreme_weak_4h"


def test_bootstrap_symbol_blocks_later_rank2_signal_without_order() -> None:
    workdir = _clean_workdir("bootstrap_blocks_rank2")
    now_ms = bj_ms(15, 8)
    state_store = TestnetStateStore(workdir / "state.json")
    state_store.save(
        TestnetState(
            open_positions=[],
            closed_positions=[],
            bootstrap_virtual_positions=[_virtual("BLOCKUSDT", now_ms - DAY_MS, rank=1)],
        )
    )
    execution_client = FakeTestnetClient()
    logger = PaperEventLogger(workdir / "events.jsonl")
    runner = TestnetTradingRunner(
        market_client=Rank2BlockMarketClient(now_ms),
        execution_client=execution_client,
        state_store=state_store,
        logger=logger,
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
    )

    opened = runner.run_signal_cycle(now_ms)

    assert opened == []
    assert not any(call[0] == "market_open_long" for call in execution_client.calls)
    events = [json.loads(line) for line in (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    block_event = next(event for event in events if event["event"] == "testnet_signal_blocked_by_bootstrap")
    assert block_event["payload"]["reason"] == "SAME_SYMBOL_LOCK_BOOTSTRAP"
    assert block_event["payload"]["symbol"] == "BLOCKUSDT"
    assert block_event["payload"]["new_rank"] == 2
    assert block_event["payload"]["blocked_by_rank"] == 1


def test_bootstrap_virtual_position_closes_at_theoretical_exit_and_symbol_can_trade_again() -> None:
    workdir = _clean_workdir("bootstrap_virtual_planned_close")
    now_ms = bj_ms(15, 8)
    state_store = TestnetStateStore(workdir / "state.json")
    state_store.save(
        TestnetState(
            open_positions=[],
            closed_positions=[],
            bootstrap_virtual_positions=[_virtual("BLOCKUSDT", now_ms - 5 * DAY_MS, rank=1)],
        )
    )
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=Rank2BlockMarketClient(now_ms),
        execution_client=execution_client,
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
    )

    weak_exits, planned_exits = runner.run_hourly_exit_cycle(now_ms)
    opened = runner.run_signal_cycle(now_ms)

    assert weak_exits == []
    assert [position.symbol for position in planned_exits] == ["BLOCKUSDT"]
    assert state_store.load().bootstrap_virtual_position("BLOCKUSDT") is None
    assert [position.symbol for position in opened] == ["BLOCKUSDT"]
    assert ("market_open_long", "BLOCKUSDT", "1.500") in execution_client.calls


def test_bootstrap_is_idempotent_after_daemon_restart() -> None:
    workdir = _clean_workdir("bootstrap_idempotent")
    activation_ms = bj_ms(15, 8)
    provider = CountingBootstrapSignals([_signal("ABCUSDT", activation_ms - DAY_MS, rank=1)])
    manager = _manager(workdir, activation_ms, market=StableExitMarket(), provider=provider)

    first = manager.ensure_bootstrap(activation_ms, {})
    second = manager.ensure_bootstrap(activation_ms + HOUR_MS, {})

    assert [position.unique_key for position in first] == [position.unique_key for position in second]
    assert provider.calls == 1
    assert len(TestnetStateStore(workdir / "state.json").load().bootstrap_virtual_positions) == 1


def test_bootstrap_does_not_create_virtual_when_exchange_real_position_exists() -> None:
    workdir = _clean_workdir("bootstrap_real_position_priority")
    activation_ms = bj_ms(15, 8)
    manager = _manager(
        workdir,
        activation_ms,
        market=StableExitMarket(),
        provider=StaticBootstrapSignals([_signal("ABCUSDT", activation_ms - DAY_MS, rank=1)]),
    )

    virtuals = manager.ensure_bootstrap(
        activation_ms,
        {"ABCUSDT": FuturesPosition("ABCUSDT", position_amt=1.0, entry_price=10.0)},
    )

    assert virtuals == []
    assert TestnetStateStore(workdir / "state.json").load().bootstrap_virtual_position("ABCUSDT") is None


class StaticBootstrapSignals:
    def __init__(self, signals: list[PaperSignal]) -> None:
        self.signals = signals

    def signals_between(self, start_time_ms: int, end_time_ms: int) -> list[PaperSignal]:
        return [
            signal
            for signal in self.signals
            if start_time_ms <= signal.signal_time_ms < end_time_ms
        ]


class CountingBootstrapSignals(StaticBootstrapSignals):
    def __init__(self, signals: list[PaperSignal]) -> None:
        super().__init__(signals)
        self.calls = 0

    def signals_between(self, start_time_ms: int, end_time_ms: int) -> list[PaperSignal]:
        self.calls += 1
        return super().signals_between(start_time_ms, end_time_ms)


class StableExitMarket:
    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        end = end_time_ms or bj_ms(15, 8)
        return [
            Kline(
                open_time_ms=end - (limit - index) * HOUR_MS,
                open=10.0,
                high=11.0,
                low=9.8,
                close=10.5,
                volume=1.0,
                close_time_ms=end - (limit - index - 1) * HOUR_MS - 1,
            )
            for index in range(limit)
        ]


class WeakExitMarket(StableExitMarket):
    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        end = end_time_ms or bj_ms(15, 8)
        return [
            Kline(
                open_time_ms=end - (limit - index) * HOUR_MS,
                open=10.0,
                high=10.1,
                low=9.9,
                close=9.9,
                volume=1.0,
                close_time_ms=end - (limit - index - 1) * HOUR_MS - 1,
            )
            for index in range(limit)
        ]


class ExtremeWeakExitMarket(StableExitMarket):
    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        end = end_time_ms or bj_ms(15, 8)
        return [
            Kline(
                open_time_ms=end - (limit - index) * HOUR_MS,
                open=10.0,
                high=10.1,
                low=9.1,
                close=9.5,
                volume=1.0,
                close_time_ms=end - (limit - index - 1) * HOUR_MS - 1,
            )
            for index in range(limit)
        ]


class Rank2BlockMarketClient(StableExitMarket):
    def __init__(self, signal_time_ms: int) -> None:
        self.signal_time_ms = signal_time_ms

    def usdt_perpetual_symbols(self) -> list[str]:
        return ["BLOCKUSDT", "SKIPUSDT", "LOWUSDT"]

    def ticker_24hr_stats(self) -> dict[str, Ticker24hrStat]:
        return {
            "SKIPUSDT": Ticker24hrStat(symbol="SKIPUSDT", price_change_percent=65.0),
            "BLOCKUSDT": Ticker24hrStat(symbol="BLOCKUSDT", price_change_percent=25.0),
            "LOWUSDT": Ticker24hrStat(symbol="LOWUSDT", price_change_percent=5.0),
        }

    def four_hour_klines(self, symbol: str, limit: int = 42, end_time_ms: int | None = None) -> list[Kline]:
        start = (end_time_ms or self.signal_time_ms) - 42 * 4 * HOUR_MS
        return [
            Kline(start + index * 4 * HOUR_MS, 1, 1, 1, 1, volume, start + (index + 1) * 4 * HOUR_MS - 1)
            for index, volume in enumerate([1.0] * 36 + [2.0] * 6)
        ]

    def latest_prices(self) -> dict[str, float]:
        return {"BLOCKUSDT": 20.0, "SKIPUSDT": 10.0, "LOWUSDT": 1.0}

    def exchange_symbol(self, symbol: str) -> dict[str, object]:
        return {
            "symbol": symbol,
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }


def _manager(
    workdir: Path,
    activation_ms: int,
    market: StableExitMarket,
    provider: StaticBootstrapSignals,
) -> BootstrapManager:
    return BootstrapManager(
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.LIVE,
            signal_mode=SignalMode.PRODUCTION,
            rank1_activation_time_ms=activation_ms,
        ),
        market_client=market,
        signal_provider=provider,
        event_prefix="live",
    )


def _signal(symbol: str, signal_time_ms: int, rank: int) -> PaperSignal:
    return PaperSignal(
        symbol=symbol,
        rank=rank,
        gain_24h=0.30,
        volume_24h_ratio_7d=2.5,
        snapshot_hour_bj="08:00",
        signal_time_ms=signal_time_ms,
        fill_price=10.0,
    )


def _virtual(symbol: str, entry_time_ms: int, rank: int) -> BootstrapVirtualPosition:
    return BootstrapVirtualPosition(
        source="BOOTSTRAP_VIRTUAL_POSITION",
        strategy_version="rank1_candidate_v1",
        unique_key=f"rank1_candidate_v1-{entry_time_ms}-{symbol}",
        symbol=symbol,
        rank=rank,
        entry_time_ms=entry_time_ms,
        entry_price=10.0,
        qty=0.0,
        leverage=3,
        planned_exit_time_ms=entry_time_ms + 5 * DAY_MS,
        weak_exit_check_time_ms=entry_time_ms + 12 * HOUR_MS,
        extreme_weak_exit_check_time_ms=entry_time_ms + 4 * HOUR_MS,
        gain_24h=0.30,
        volume_24h_ratio_7d=2.5,
    )


def _clean_workdir(name: str) -> Path:
    path = Path("tests/.tmp") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
