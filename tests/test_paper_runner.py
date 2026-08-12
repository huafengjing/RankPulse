from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.market.binance_futures import Kline, Ticker24hrStat
from src.paper.runner import PaperTradingRunner
from src.paper.store import PaperEventLogger, PaperStateStore
from src.paper.trading import PaperTradingEngine
from src.research.top3_strategy_rules import DAY_MS, HOUR_MS


BEIJING = timezone(timedelta(hours=8))
FOUR_HOUR_MS = 4 * HOUR_MS


def bj_time_ms(hour: int) -> int:
    return int(datetime(2026, 6, 17, hour, 0, tzinfo=BEIJING).timestamp() * 1000)


def make_4h_kline(open_time_ms: int, volume: float) -> Kline:
    return Kline(
        open_time_ms=open_time_ms,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=volume,
        close_time_ms=open_time_ms + FOUR_HOUR_MS - 1,
    )


def make_1h_kline(open_time_ms: int, high: float, low: float, close: float) -> Kline:
    return Kline(
        open_time_ms=open_time_ms,
        open=10.0,
        high=high,
        low=low,
        close=close,
        volume=1.0,
        close_time_ms=open_time_ms + HOUR_MS - 1,
    )


def completed_4h_klines(signal_time_ms: int) -> list[Kline]:
    start = signal_time_ms - 42 * FOUR_HOUR_MS
    return [make_4h_kline(start + index * FOUR_HOUR_MS, volume) for index, volume in enumerate([1.0] * 36 + [2.0] * 6)]


class FakeMarketClient:
    def __init__(self, signal_time_ms: int) -> None:
        self.signal_time_ms = signal_time_ms

    def usdt_perpetual_symbols(self) -> list[str]:
        return ["RANK1USDT", "RANK2USDT", "RANK3USDT"]

    def ticker_24hr_stats(self) -> dict[str, Ticker24hrStat]:
        return {
            "RANK1USDT": Ticker24hrStat(symbol="RANK1USDT", price_change_percent=30.0),
            "RANK2USDT": Ticker24hrStat(symbol="RANK2USDT", price_change_percent=25.0),
            "RANK3USDT": Ticker24hrStat(symbol="RANK3USDT", price_change_percent=20.0),
        }

    def four_hour_klines(self, symbol: str, limit: int = 42, end_time_ms: int | None = None) -> list[Kline]:
        return completed_4h_klines(end_time_ms or self.signal_time_ms)

    def one_hour_klines(self, symbol: str, limit: int = 13, end_time_ms: int | None = None) -> list[Kline]:
        position_entry_time = self.signal_time_ms
        entry_price = {"RANK2USDT": 22.0, "RANK3USDT": 33.0}.get(symbol, 11.0)
        return [
            make_1h_kline(
                position_entry_time + index * HOUR_MS,
                high=entry_price * 1.04,
                low=entry_price * 0.94,
                close=entry_price * 0.99,
            )
            for index in range(12)
        ]

    def latest_prices(self) -> dict[str, float]:
        return {
            "RANK1USDT": 11.0,
            "RANK2USDT": 22.0,
            "RANK3USDT": 33.0,
        }


def test_runner_opens_rank2_and_rank3_paper_positions_and_persists_state() -> None:
    workdir = _clean_workdir("runner_signal")
    signal_time_ms = bj_time_ms(0)
    store = PaperStateStore(workdir / "state.json")
    logger = PaperEventLogger(workdir / "events.jsonl")
    engine = PaperTradingEngine()
    runner = PaperTradingRunner(
        market_client=FakeMarketClient(signal_time_ms),
        engine=engine,
        store=store,
        logger=logger,
    )

    opened = runner.run_signal_cycle(signal_time_ms)

    assert [position.symbol for position in opened] == ["RANK2USDT", "RANK3USDT"]
    assert store.load().open_position("RANK2USDT") is not None
    events = [json.loads(line) for line in (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["signal_opened", "signal_opened"]


def test_runner_planned_exit_uses_latest_price_and_persists_state() -> None:
    workdir = _clean_workdir("runner_planned_exit")
    signal_time_ms = bj_time_ms(0)
    store = PaperStateStore(workdir / "state.json")
    logger = PaperEventLogger(workdir / "events.jsonl")
    engine = PaperTradingEngine()
    runner = PaperTradingRunner(
        market_client=FakeMarketClient(signal_time_ms),
        engine=engine,
        store=store,
        logger=logger,
    )
    runner.run_signal_cycle(signal_time_ms)

    exits = runner.run_planned_exits(signal_time_ms + 6 * DAY_MS)

    assert [trade_exit.symbol for trade_exit in exits] == ["RANK2USDT", "RANK3USDT"]
    assert all(trade_exit.exit_reason == "planned_6d" for trade_exit in exits)
    assert store.load().open_position("RANK2USDT") is None


def test_runner_12h_weak_exit_uses_completed_1h_klines() -> None:
    workdir = _clean_workdir("runner_weak_exit")
    signal_time_ms = bj_time_ms(0)
    engine = PaperTradingEngine()
    runner = PaperTradingRunner(
        market_client=FakeMarketClient(signal_time_ms),
        engine=engine,
        store=PaperStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
    )
    runner.run_signal_cycle(signal_time_ms)

    exits = runner.run_weak_exit_checks(signal_time_ms + 12 * HOUR_MS)

    assert [trade_exit.symbol for trade_exit in exits] == ["RANK2USDT", "RANK3USDT"]
    assert all(trade_exit.exit_reason == "weak_12h" for trade_exit in exits)


def test_runner_4h_extreme_weak_exit_uses_completed_1h_klines() -> None:
    workdir = _clean_workdir("runner_extreme_weak_exit")
    signal_time_ms = bj_time_ms(0)
    engine = PaperTradingEngine()
    runner = PaperTradingRunner(
        market_client=ExtremeWeakMarketClient(signal_time_ms),
        engine=engine,
        store=PaperStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
    )
    runner.run_signal_cycle(signal_time_ms)

    exits = runner.run_weak_exit_checks(signal_time_ms + 4 * HOUR_MS)

    assert [trade_exit.symbol for trade_exit in exits] == ["RANK2USDT", "RANK3USDT"]
    assert all(trade_exit.exit_reason == "extreme_weak_4h" for trade_exit in exits)


def _clean_workdir(name: str) -> Path:
    path = Path("tests/.tmp") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


class ExtremeWeakMarketClient(FakeMarketClient):
    def one_hour_klines(self, symbol: str, limit: int = 13, end_time_ms: int | None = None) -> list[Kline]:
        position_entry_time = self.signal_time_ms
        entry_price = {"RANK2USDT": 22.0, "RANK3USDT": 33.0}.get(symbol, 11.0)
        return [
            make_1h_kline(
                position_entry_time + index * HOUR_MS,
                high=entry_price * 1.019,
                low=entry_price * 0.919,
                close=entry_price * 0.93,
            )
            for index in range(4)
        ]
