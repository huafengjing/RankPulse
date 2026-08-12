from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.config.modes import SignalMode, TradingMode
from src.config.settings import AppSettings
from src.exchange.binance_filters import SymbolFilters
from src.exchange.binance_live import BinanceLiveAPIError
from src.exchange.binance_testnet import BinanceTestnetAPIError, FuturesPosition
import src.execution.testnet_engine as testnet_engine
from src.execution.testnet_engine import PositionPrecheckFailedError, TestnetExecutionEngine
from src.execution.testnet_state import TestnetPosition, TestnetStateStore
from src.paper.store import PaperEventLogger
from src.paper.trading import PaperSignal


def test_testnet_open_sets_leverage_places_market_order_and_records_state() -> None:
    workdir = _clean_workdir("testnet_open")
    client = FakeTestnetClient()
    state = TestnetStateStore(workdir / "state.json")
    logger = PaperEventLogger(workdir / "events.jsonl")
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=logger,
        settings=AppSettings(trading_mode=TradingMode.TESTNET, signal_mode=SignalMode.TEST_FAST, position_margin_usdt=10),
    )

    position = engine.open_from_signal(
        PaperSignal(
            symbol="AAAUSDT",
            rank=2,
            gain_24h=0.25,
            volume_24h_ratio_7d=2.5,
            snapshot_hour_bj="00:00",
            signal_time_ms=1_700_000_000_000,
            fill_price=20.0,
        ),
        leverage=3,
    )

    assert position is not None
    assert client.calls[:3] == [("set_isolated_margin", "AAAUSDT"), ("set_leverage", "AAAUSDT", 3), ("market_open_long", "AAAUSDT", "1.500")]
    assert state.load().open_position("AAAUSDT") is not None
    event = json.loads((workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["payload"]["trading_mode"] == "testnet"
    assert event["payload"]["signal_mode"] == "test_fast"


def test_testnet_open_ignores_margin_type_already_isolated_error() -> None:
    workdir = _clean_workdir("testnet_open_already_isolated")
    client = FakeTestnetClient()
    client.margin_type_error = BinanceTestnetAPIError(
        status=400,
        code=-4046,
        message="No need to change margin type.",
        path="/fapi/v1/marginType",
    )
    state = TestnetStateStore(workdir / "state.json")
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.TEST_FAST,
            position_margin_usdt=10,
        ),
    )

    position = engine.open_from_signal(_signal("AAAUSDT"), leverage=3)

    assert position is not None
    assert ("set_leverage", "AAAUSDT", 3) in client.calls
    assert ("market_open_long", "AAAUSDT", "1.500") in client.calls


def test_live_event_prefix_keeps_live_logs_separate() -> None:
    workdir = _clean_workdir("live_open_prefix")
    client = FakeTestnetClient()
    state = TestnetStateStore(workdir / "state.json")
    logger = PaperEventLogger(workdir / "events.jsonl")
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=logger,
        settings=AppSettings(trading_mode=TradingMode.LIVE, signal_mode=SignalMode.PRODUCTION, position_margin_usdt=10),
        event_prefix="live",
    )

    assert engine.open_from_signal(_signal("AAAUSDT"), leverage=3) is not None

    event = json.loads((workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["event"] == "live_opened"
    assert event["payload"]["trading_mode"] == "live"
    assert event["payload"]["signal_mode"] == "production"


def test_open_retries_order_query_when_binance_temporarily_says_order_does_not_exist(monkeypatch) -> None:
    monkeypatch.setattr(testnet_engine.time, "sleep", lambda seconds: None)
    workdir = _clean_workdir("open_order_query_retry")
    client = FakeTestnetClient()
    client.order_errors = [
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
    ]
    state = TestnetStateStore(workdir / "state.json")
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.LIVE, signal_mode=SignalMode.PRODUCTION, position_margin_usdt=10),
        event_prefix="live",
    )

    position = engine.open_from_signal(_signal("AAAUSDT"), leverage=3)

    assert position is not None
    assert position.order_id == 123
    assert position.entry_price == 20.0
    assert client.calls.count(("order", "AAAUSDT", 123)) == 2
    event = json.loads((workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert event["event"] == "live_opened"
    assert event["payload"]["confirmation_source"] == "order"


def test_open_reconciles_position_when_order_query_keeps_returning_not_found(monkeypatch) -> None:
    monkeypatch.setattr(testnet_engine.time, "sleep", lambda seconds: None)
    workdir = _clean_workdir("open_order_query_reconcile")
    client = FakeTestnetClient()
    client.order_errors = [
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
    ]
    state = TestnetStateStore(workdir / "state.json")
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.LIVE, signal_mode=SignalMode.PRODUCTION, position_margin_usdt=10),
        event_prefix="live",
    )

    position = engine.open_from_signal(_signal("AAAUSDT"), leverage=3)

    assert position is not None
    assert position.entry_price == 20.0
    assert position.qty == 1.5
    assert state.load().open_position("AAAUSDT") is not None
    events = [
        json.loads(line)
        for line in (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "live_open_order_reconciled",
        "live_opened",
    ]
    assert events[-1]["payload"]["confirmation_source"] == "positionRisk"


def test_testnet_open_skips_when_local_or_exchange_position_exists() -> None:
    workdir = _clean_workdir("testnet_skip")
    client = FakeTestnetClient()
    state = TestnetStateStore(workdir / "state.json")
    state.save_positions(
        [
            TestnetPosition(
                symbol="AAAUSDT",
                entry_time_ms=1,
                entry_price=20,
                qty=1,
                leverage=3,
                order_id=10,
                planned_exit_time_ms=2,
                extreme_weak_exit_check_time_ms=1,
                weak_exit_check_time_ms=2,
            )
        ]
    )
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.TESTNET, signal_mode=SignalMode.TEST_FAST),
    )

    local_skip = engine.open_from_signal(_signal("AAAUSDT"), leverage=3)
    client.exchange_positions = {"BBBUSDT": FuturesPosition("BBBUSDT", 1.0, 20.0)}
    exchange_skip = engine.open_from_signal(_signal("BBBUSDT"), leverage=3)

    assert local_skip is None
    assert exchange_skip is None
    assert client.calls == []


def test_open_raises_clear_skip_message_when_position_precheck_fails() -> None:
    workdir = _clean_workdir("position_precheck_failed")
    client = FakeTestnetClient()
    client.open_positions_error = RuntimeError("Live API request failed after 5 retries: /fapi/v2/positionRisk")
    state = TestnetStateStore(workdir / "state.json")
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.LIVE, signal_mode=SignalMode.PRODUCTION),
        event_prefix="live",
    )

    try:
        engine.open_from_signal(_signal("AAAUSDT"), leverage=3)
        raise AssertionError("Expected PositionPrecheckFailedError")
    except PositionPrecheckFailedError as exc:
        message = str(exc)

    assert "执行跳过：开仓前读取实盘持仓失败，未下单" in message
    assert "原因：Binance positionRisk 网络连接中断" in message
    assert client.calls == []
    assert state.load().open_position("AAAUSDT") is None


def test_testnet_close_uses_reduce_only_and_marks_position_closed() -> None:
    workdir = _clean_workdir("testnet_close")
    client = FakeTestnetClient()
    state = TestnetStateStore(workdir / "state.json")
    state.save_positions(
        [
            TestnetPosition(
                symbol="AAAUSDT",
                entry_time_ms=1_700_000_000_000,
                entry_price=20.0,
                qty=1.5,
                leverage=3,
                order_id=123,
                planned_exit_time_ms=1_700_000_100_000,
                extreme_weak_exit_check_time_ms=1_700_000_025_000,
                weak_exit_check_time_ms=1_700_000_050_000,
            )
        ]
    )
    client.exchange_positions = {"AAAUSDT": FuturesPosition("AAAUSDT", 1.25, 20.0)}
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.TESTNET, signal_mode=SignalMode.TEST_FAST),
    )

    closed = engine.close_position("AAAUSDT", exit_time_ms=1_700_000_100_000, reason="planned")

    assert closed is not None
    assert client.calls == [
        ("market_close_long", "AAAUSDT", "1.250"),
        ("order", "AAAUSDT", 456),
    ]
    assert closed.qty == 1.25
    assert closed.realized_pnl == 12.5
    assert state.load().open_position("AAAUSDT") is None


def test_close_reconciles_position_when_order_query_keeps_returning_not_found(monkeypatch) -> None:
    monkeypatch.setattr(testnet_engine.time, "sleep", lambda seconds: None)
    workdir = _clean_workdir("close_order_query_reconcile")
    client = FakeTestnetClient()
    client.order_errors = [
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
        BinanceLiveAPIError(400, -2013, "Order does not exist.", "/fapi/v1/order"),
    ]
    state = TestnetStateStore(workdir / "state.json")
    state.save_positions(
        [
            TestnetPosition(
                symbol="AAAUSDT",
                entry_time_ms=1_700_000_000_000,
                entry_price=20.0,
                qty=1.5,
                leverage=3,
                order_id=123,
                planned_exit_time_ms=1_700_000_100_000,
                extreme_weak_exit_check_time_ms=1_700_000_025_000,
                weak_exit_check_time_ms=1_700_000_050_000,
            )
        ]
    )
    client.exchange_positions = {"AAAUSDT": FuturesPosition("AAAUSDT", 1.25, 20.0)}
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.LIVE, signal_mode=SignalMode.PRODUCTION),
        event_prefix="live",
    )

    closed = engine.close_position("AAAUSDT", exit_time_ms=1_700_000_100_000, reason="planned")

    assert closed is not None
    assert closed.exit_price == 30.0
    assert closed.qty == 1.25
    assert state.load().open_position("AAAUSDT") is None
    events = [
        json.loads(line)
        for line in (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "live_close_order_reconciled",
        "live_closed",
    ]
    assert events[-1]["payload"]["confirmation_source"] == "positionRisk"


def test_testnet_close_reconciles_stale_local_position_without_order() -> None:
    workdir = _clean_workdir("testnet_close_reconcile")
    client = FakeTestnetClient()
    state = TestnetStateStore(workdir / "state.json")
    state.save_positions(
        [
            TestnetPosition(
                symbol="AAAUSDT",
                entry_time_ms=1_700_000_000_000,
                entry_price=20.0,
                qty=1.5,
                leverage=3,
                order_id=123,
                planned_exit_time_ms=1_700_000_100_000,
                extreme_weak_exit_check_time_ms=1_700_000_025_000,
                weak_exit_check_time_ms=1_700_000_050_000,
            )
        ]
    )
    engine = TestnetExecutionEngine(
        client=client,
        filter_provider=FakeFilterProvider(),
        state_store=state,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(trading_mode=TradingMode.TESTNET, signal_mode=SignalMode.TEST_FAST),
    )

    closed = engine.close_position("AAAUSDT", exit_time_ms=1_700_000_100_000, reason="planned")

    assert closed is None
    assert client.calls == []
    assert state.load().open_position("AAAUSDT") is None


class FakeTestnetClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.exchange_positions: dict[str, FuturesPosition] = {}
        self._last_qty_by_symbol: dict[str, str] = {}
        self.margin_type_error: Exception | None = None
        self.open_positions_error: Exception | None = None
        self.order_errors: list[Exception] = []

    def open_positions(self) -> dict[str, FuturesPosition]:
        if self.open_positions_error is not None:
            raise self.open_positions_error
        return self.exchange_positions

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, object]:
        self.calls.append(("set_leverage", symbol, leverage))
        return {"leverage": leverage}

    def set_isolated_margin(self, symbol: str) -> dict[str, object]:
        self.calls.append(("set_isolated_margin", symbol))
        if self.margin_type_error is not None:
            raise self.margin_type_error
        return {}  # 实际 API 返回空对象 (HTTP 200)

    def order(self, symbol: str, order_id: int) -> dict[str, object]:
        self.calls.append(("order", symbol, order_id))
        if self.order_errors:
            raise self.order_errors.pop(0)
        qty = self._last_qty_by_symbol.get(symbol, "0")
        avg_price = "30.0" if order_id == 456 else "20.0"
        return {"orderId": order_id, "avgPrice": avg_price, "executedQty": qty}

    def market_open_long(self, symbol: str, quantity: str) -> dict[str, object]:
        self.calls.append(("market_open_long", symbol, quantity))
        self._last_qty_by_symbol[symbol] = quantity
        self.exchange_positions[symbol] = FuturesPosition(
            symbol=symbol,
            position_amt=float(quantity),
            entry_price=20.0,
        )
        return {"orderId": 123, "avgPrice": "20.0", "executedQty": quantity}

    def market_close_long(self, symbol: str, quantity: str) -> dict[str, object]:
        self.calls.append(("market_close_long", symbol, quantity))
        self._last_qty_by_symbol[symbol] = quantity
        self.exchange_positions.pop(symbol, None)
        return {"orderId": 456, "avgPrice": "30.0", "executedQty": quantity}


class FakeFilterProvider:
    def filters_for_symbol(self, symbol: str) -> SymbolFilters:
        return SymbolFilters(step_size=0.001, min_qty=0.001, min_notional=5)


def _signal(symbol: str) -> PaperSignal:
    return PaperSignal(
        symbol=symbol,
        rank=2,
        gain_24h=0.25,
        volume_24h_ratio_7d=2.5,
        snapshot_hour_bj="00:00",
        signal_time_ms=1_700_000_000_000,
        fill_price=20.0,
    )


def _clean_workdir(name: str) -> Path:
    path = Path("tests/.tmp") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
