from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config.modes import SignalMode, TradingMode
from src.config.settings import AppSettings
from src.exchange.binance_testnet import FuturesPosition
from src.execution.testnet_runner import TestnetTradingRunner, _safe_print
from src.execution.testnet_state import TestnetPosition, TestnetState, TestnetStateStore
from src.market.binance_futures import Kline, Ticker24hrStat
from src.paper.store import PaperEventLogger
from src.research.rankpulse_strategy_rules import DAY_MS, Top3RegimeContext
from tests.test_testnet_execution import FakeTestnetClient, _clean_workdir


BEIJING = timezone(timedelta(hours=8))


def bj_ms(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 6, 17, hour, minute, tzinfo=BEIJING).timestamp() * 1000)


def test_testnet_runner_opens_in_test_fast_interval_outside_production_hours() -> None:
    workdir = _clean_workdir("testnet_runner_fast")
    now_ms = bj_ms(8, 5)
    settings = AppSettings(
        trading_mode=TradingMode.TESTNET,
        signal_mode=SignalMode.TEST_FAST,
        position_margin_usdt=10,
    )
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=settings,
    )

    opened = runner.run_signal_cycle(now_ms)

    assert [position.symbol for position in opened] == ["RANK2USDT", "RANK3USDT"]
    assert opened[0].planned_exit_time_ms == now_ms + 60 * 60 * 1000
    assert opened[0].extreme_weak_exit_check_time_ms == now_ms + 5 * 60 * 1000
    assert opened[0].weak_exit_check_time_ms == now_ms + 15 * 60 * 1000


def test_testnet_runner_uses_regime_context_for_order_leverage() -> None:
    workdir = _clean_workdir("testnet_runner_regime_context")
    now_ms = bj_ms(8, 5)
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.TEST_FAST,
            position_margin_usdt=10,
        ),
        regime_context_provider=StaticRegimeProvider(Top3RegimeContext(state="RED")),
    )

    opened = runner.run_signal_cycle(now_ms)

    assert [position.leverage for position in opened] == [2, 1]
    assert ("set_leverage", "RANK2USDT", 2) in execution_client.calls
    assert ("set_leverage", "RANK3USDT", 1) in execution_client.calls


def test_production_signal_cycle_generates_regime_context_before_strategy_with_canonical_window() -> None:
    workdir = _clean_workdir("testnet_runner_auto_regime_context")
    now_ms = bj_ms(8, 0) + 45_000
    execution_client = FakeTestnetClient()
    provider = RecordingRegimeProvider(Top3RegimeContext(state="RED"))
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
            top3_regime_enabled=True,
        ),
        regime_context_provider=provider,
    )

    opened = runner.run_signal_cycle(now_ms)

    assert provider.calls == [bj_ms(8, 0)]
    assert [position.leverage for position in opened] == [2, 1]
    assert ("set_leverage", "RANK3USDT", 1) in execution_client.calls


def test_production_runner_opens_rank1_candidate_with_5d_planned_exit() -> None:
    workdir = _clean_workdir("testnet_runner_rank1_candidate")
    now_ms = bj_ms(8, 0)
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=Rank1CandidateMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
    )

    opened = runner.run_signal_cycle(now_ms)

    rank1 = next(position for position in opened if position.symbol == "RANK1USDT")
    rank2 = next(position for position in opened if position.symbol == "RANK2USDT")
    assert rank1.leverage == 3
    assert rank1.planned_exit_time_ms == now_ms + 5 * DAY_MS
    assert rank2.planned_exit_time_ms == now_ms + 6 * DAY_MS
    assert ("set_leverage", "RANK1USDT", 3) in execution_client.calls


def test_regime_generation_failure_fails_closed_before_signal_or_order() -> None:
    workdir = _clean_workdir("testnet_runner_regime_fail_closed")
    now_ms = bj_ms(8, 0)
    market_client = FakeMarketClient(now_ms)
    execution_client = FakeTestnetClient()
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=execution_client,
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
            top3_regime_enabled=True,
        ),
        regime_context_provider=FailingRegimeProvider(),
    )

    with pytest.raises(RuntimeError, match="regime generator failed"):
        runner.run_signal_cycle(now_ms)

    assert market_client.ticker_calls == 0
    assert execution_client.calls == []
    assert state_store.load().last_signal_time_ms is None


def test_regime_enabled_failure_never_falls_back_to_baseline_rank3_5x() -> None:
    workdir = _clean_workdir("testnet_runner_no_rank3_5x_fallback")
    now_ms = bj_ms(8, 0)
    market_client = FakeMarketClient(now_ms)
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
            top3_regime_enabled=True,
        ),
        regime_context_provider=FailingRegimeProvider(),
    )

    with pytest.raises(RuntimeError, match="regime generator failed"):
        runner.run_signal_cycle(now_ms)

    assert ("set_leverage", "RANK3USDT", 5) not in execution_client.calls
    assert not any(call[0] == "market_open_long" for call in execution_client.calls)
    assert market_client.ticker_calls == 0


def test_sync_open_positions_uses_exchange_positions_and_removes_stale_local_state() -> None:
    workdir = _clean_workdir("testnet_runner_sync_exchange_positions")
    state_store = TestnetStateStore(workdir / "state.json")
    state_store.save(
        TestnetState(
            open_positions=[
                _local_position("TLMUSDT"),
                _local_position("CLOUSDT"),
            ],
            closed_positions=[],
        )
    )
    execution_client = FakeTestnetClient()
    execution_client.exchange_positions = {
        "CLOUSDT": FuturesPosition("CLOUSDT", 10.0, 1.0),
        "NEWUSDT": FuturesPosition("NEWUSDT", 5.0, 2.0),
    }
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(bj_ms(8, 0)),
        execution_client=execution_client,
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.LIVE,
            signal_mode=SignalMode.PRODUCTION,
        ),
        event_prefix="live",
    )

    symbols = runner.sync_open_positions_from_exchange()

    assert symbols == ["CLOUSDT", "NEWUSDT"]
    assert [position.symbol for position in state_store.load().open_positions] == ["CLOUSDT"]


def test_23_information_cycle_does_not_generate_or_advance_regime_context() -> None:
    workdir = _clean_workdir("testnet_runner_23_no_regime_generation")
    now_ms = bj_ms(23, 0)
    notifier = FakeNotifier()
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
            top3_regime_enabled=True,
        ),
        signal_notifier=notifier,
        regime_context_provider=FailingRegimeProvider(),
    )

    sent = runner.run_information_cycle(now_ms)

    assert sent is True
    assert len(notifier.messages) == 1
    assert "INFO ONLY - NO ORDER" in notifier.messages[0]


def test_testnet_runner_does_not_repeat_same_signal_cycle() -> None:
    workdir = _clean_workdir("testnet_runner_no_repeat")
    now_ms = bj_ms(8, 5)
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.TEST_FAST,
            position_margin_usdt=10,
        ),
    )

    first = runner.run_signal_cycle(now_ms)
    second = runner.run_signal_cycle(now_ms)

    assert [position.symbol for position in first] == ["RANK2USDT", "RANK3USDT"]
    assert second == []


def test_testnet_runner_fetches_4h_klines_only_for_ticker_top3() -> None:
    workdir = _clean_workdir("testnet_runner_top3_4h")
    now_ms = bj_ms(8, 5)
    market_client = FakeMarketClient(now_ms)
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=FakeTestnetClient(),
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.TEST_FAST,
            position_margin_usdt=10,
        ),
    )

    runner.run_signal_cycle(now_ms)

    assert market_client.four_hour_symbols == ["RANK1USDT", "RANK2USDT", "RANK3USDT"]


def test_signal_output_includes_current_public_market_price(capsys) -> None:
    workdir = _clean_workdir("testnet_runner_signal_price")
    now_ms = bj_ms(8, 0)
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
    )

    runner.run_signal_cycle(now_ms)

    output = capsys.readouterr().out
    assert "2026-06-17 08:00:00 | 交易信号" in output
    assert "信号      | 排名 | 价格 |  涨幅 | 量比 | 结果" in output
    assert "RANK2USDT |    2 |   22 | 25.0% | 1.75 | ✓ 符合规则 3x" in output
    assert "RANK3USDT |    3 |   33 | 20.0% | 1.75 | ✓ 符合规则 5x" in output
    assert output.count("RANK2USDT") == 1
    assert output.count("RANK3USDT") == 1


def test_signal_table_is_forwarded_to_notifier_once() -> None:
    workdir = _clean_workdir("testnet_runner_telegram")
    now_ms = bj_ms(8, 0)
    notifier = FakeNotifier()
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
        signal_notifier=notifier,
    )

    runner.run_signal_cycle(now_ms)

    assert len(notifier.messages) == 1
    assert "2026-06-17 08:00:00 | TRADE SIGNAL" in notifier.messages[0]
    assert "SYMBOL    | RANK | PRICE |  GAIN |  V/R | STATUS" in notifier.messages[0]
    assert "RANK2USDT |    2 |    22 | 25.0% | 1.75 | PASS 3x" in notifier.messages[0]
    assert "RANK3USDT |    3 |    33 | 20.0% | 1.75 | PASS 5x" in notifier.messages[0]


def test_safe_print_does_not_raise_when_stdout_rejects_unicode(monkeypatch) -> None:
    class GbkLikeStdout:
        encoding = "gbk"

        def __init__(self) -> None:
            self.writes: list[str] = []

        def write(self, text: str) -> int:
            text.encode(self.encoding)
            self.writes.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    fake_stdout = GbkLikeStdout()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    _safe_print("SYNUSDT ✗ 量比不足")

    assert fake_stdout.writes


def test_23_information_cycle_forwards_table_without_placing_orders() -> None:
    workdir = _clean_workdir("testnet_runner_23_info")
    now_ms = bj_ms(23, 0)
    notifier = FakeNotifier()
    execution_client = FakeTestnetClient()
    market_client = FakeMarketClient(now_ms)
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=execution_client,
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
        signal_notifier=notifier,
    )

    first = runner.run_information_cycle(now_ms)
    second = runner.run_information_cycle(now_ms + 45_000)

    assert first is True
    assert second is False
    assert len(notifier.messages) == 1
    assert "2026-06-17 23:00:00 | INFO ONLY - NO ORDER" in notifier.messages[0]
    assert "RANK2USDT" in notifier.messages[0]
    assert "RANK2USDT: 符合规则，仅信息观察，不开仓" in notifier.messages[0]
    assert "RANK3USDT: 符合规则，仅信息观察，不开仓" in notifier.messages[0]
    assert execution_client.calls == []
    assert state_store.load().last_information_time_ms == bj_ms(23, 0)
    assert state_store.load().last_signal_time_ms is None


def test_market_preflight_sends_telegram_30_minutes_before_signal() -> None:
    workdir = _clean_workdir("testnet_runner_preflight_ok")
    now_ms = bj_ms(22, 30)
    notifier = FakeNotifier()
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.LIVE,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
        signal_notifier=notifier,
        event_prefix="live",
    )

    first = runner.run_market_preflight_cycle(now_ms)
    second = runner.run_market_preflight_cycle(now_ms + 45_000)

    assert first is True
    assert second is False
    assert len(notifier.messages) == 1
    assert "2026-06-17 22:30:00 | MARKET PREFLIGHT" in notifier.messages[0]
    assert "STATUS: OK" in notifier.messages[0]
    assert "TARGET: 2026-06-17 23:00:00" in notifier.messages[0]
    assert state_store.load().last_preflight_time_ms == bj_ms(22, 30)


def test_market_preflight_sends_telegram_when_binance_public_api_is_blocked() -> None:
    workdir = _clean_workdir("testnet_runner_preflight_blocked")
    now_ms = bj_ms(23, 30)
    notifier = FakeNotifier()
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=BlockedMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.LIVE,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
        signal_notifier=notifier,
        event_prefix="live",
    )

    sent = runner.run_market_preflight_cycle(now_ms)

    assert sent is True
    assert len(notifier.messages) == 1
    assert "STATUS: FAILED" in notifier.messages[0]
    assert "TARGET: 2026-06-18 00:00:00" in notifier.messages[0]
    assert "HTTP 418" in notifier.messages[0]
    assert state_store.load().last_preflight_time_ms == bj_ms(23, 30)


def test_rank2_execution_failure_does_not_block_rank3() -> None:
    workdir = _clean_workdir("testnet_runner_continue_after_failure")
    now_ms = bj_ms(8, 0)
    execution_client = FailingRank2ExecutionClient()
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=execution_client,
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
            max_open_positions=2,
        ),
    )

    opened = runner.run_signal_cycle(now_ms)

    assert [position.symbol for position in opened] == ["RANK3USDT"]
    assert state_store.load().open_position("RANK3USDT") is not None


def test_position_precheck_failure_prints_clear_skip_message(capsys) -> None:
    workdir = _clean_workdir("testnet_runner_position_precheck_message")
    now_ms = bj_ms(8, 0)
    execution_client = FakeTestnetClient()
    execution_client.open_positions_error = RuntimeError("Live API request failed after 5 retries: /fapi/v2/positionRisk")
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.LIVE,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
        event_prefix="live",
    )

    opened = runner.run_signal_cycle(now_ms)

    output = capsys.readouterr().out
    assert opened == []
    assert "[执行跳过] RANK2USDT | Rank 2" in output
    assert "执行跳过：开仓前读取实盘持仓失败，未下单" in output
    assert "原因：Binance positionRisk 网络连接中断" in output


def test_production_runner_does_not_monitor_or_update_state_outside_00_and_08() -> None:
    workdir = _clean_workdir("testnet_runner_production_off_hour")
    now_ms = bj_ms(20, 0)
    market_client = FakeMarketClient(now_ms)
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=FakeTestnetClient(),
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
    )

    opened = runner.run_signal_cycle(now_ms)

    assert opened == []
    assert market_client.ticker_calls == 0
    assert state_store.load().last_signal_time_ms is None


def test_production_runner_uses_one_canonical_key_for_the_00_signal_minute() -> None:
    workdir = _clean_workdir("testnet_runner_production_window")
    first_ms = bj_ms(0, 0) + 5_000
    second_ms = bj_ms(0, 0) + 45_000
    market_client = FakeMarketClient(first_ms)
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=FakeTestnetClient(),
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
            position_margin_usdt=10,
        ),
    )

    first = runner.run_signal_cycle(first_ms)
    second = runner.run_signal_cycle(second_ms)

    assert [position.symbol for position in first] == ["RANK2USDT", "RANK3USDT"]
    assert second == []
    assert market_client.ticker_calls == 1


def test_exit_cycle_runs_only_once_at_each_beijing_top_of_hour() -> None:
    workdir = _clean_workdir("testnet_runner_hourly_exit")
    state_store = TestnetStateStore(workdir / "state.json")
    runner = TestnetTradingRunner(
        market_client=FakeMarketClient(bj_ms(9, 0)),
        execution_client=FakeTestnetClient(),
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
        ),
    )

    off_hour = runner.run_hourly_exit_cycle(bj_ms(9, 5))
    first = runner.run_hourly_exit_cycle(bj_ms(10, 0) + 5_000)
    second = runner.run_hourly_exit_cycle(bj_ms(10, 0) + 45_000)

    assert off_hour == ([], [])
    assert first == ([], [])
    assert second == ([], [])
    assert state_store.load().last_exit_check_time_ms == bj_ms(10, 0)


def test_testnet_runner_weak_exit_uses_test_fast_15_minute_window() -> None:
    workdir = _clean_workdir("testnet_runner_weak")
    now_ms = bj_ms(8, 5)
    settings = AppSettings(
        trading_mode=TradingMode.TESTNET,
        signal_mode=SignalMode.TEST_FAST,
        position_margin_usdt=10,
    )
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=FastMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=settings,
    )
    runner.run_signal_cycle(now_ms)

    closed = runner.run_weak_exit_checks(now_ms + 15 * 60 * 1000)

    assert [position.symbol for position in closed] == ["RANK2USDT", "RANK3USDT"]
    assert ("market_close_long", "RANK2USDT", "1.363") in execution_client.calls


def test_testnet_runner_extreme_weak_exit_uses_test_fast_5_minute_window() -> None:
    workdir = _clean_workdir("testnet_runner_extreme_weak")
    now_ms = bj_ms(8, 5)
    settings = AppSettings(
        trading_mode=TradingMode.TESTNET,
        signal_mode=SignalMode.TEST_FAST,
        position_margin_usdt=10,
    )
    execution_client = FakeTestnetClient()
    runner = TestnetTradingRunner(
        market_client=ExtremeWeakFastMarketClient(now_ms),
        execution_client=execution_client,
        state_store=TestnetStateStore(workdir / "state.json"),
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=settings,
    )
    runner.run_signal_cycle(now_ms)

    closed = runner.run_weak_exit_checks(now_ms + 5 * 60 * 1000)

    assert [position.symbol for position in closed] == ["RANK2USDT", "RANK3USDT"]
    assert all(position.exit_reason == "extreme_weak_4h" for position in closed)


def test_weak_exit_no_exit_is_marked_checked_once() -> None:
    workdir = _clean_workdir("testnet_runner_weak_no_exit_checked")
    now_ms = bj_ms(8, 5)
    state_store = TestnetStateStore(workdir / "state.json")
    settings = AppSettings(
        trading_mode=TradingMode.TESTNET,
        signal_mode=SignalMode.TEST_FAST,
        position_margin_usdt=10,
    )
    runner = TestnetTradingRunner(
        market_client=NoWeakExitMarketClient(now_ms),
        execution_client=FakeTestnetClient(),
        state_store=state_store,
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=settings,
    )
    runner.run_signal_cycle(now_ms)

    first = runner.run_weak_exit_checks(now_ms + 15 * 60 * 1000)
    second = runner.run_weak_exit_checks(now_ms + 16 * 60 * 1000)

    assert first == []
    assert second == []
    assert state_store.load().open_position("RANK2USDT").weak_exit_checked is True  # type: ignore[union-attr]


class FakeMarketClient:
    def __init__(self, signal_time_ms: int) -> None:
        self.signal_time_ms = signal_time_ms
        self.four_hour_symbols: list[str] = []
        self.ticker_calls = 0

    def usdt_perpetual_symbols(self) -> list[str]:
        return ["RANK1USDT", "RANK2USDT", "RANK3USDT"]

    def ticker_24hr_stats(self) -> dict[str, Ticker24hrStat]:
        self.ticker_calls += 1
        return {
            "RANK1USDT": Ticker24hrStat(symbol="RANK1USDT", price_change_percent=30.0),
            "RANK2USDT": Ticker24hrStat(symbol="RANK2USDT", price_change_percent=25.0),
            "RANK3USDT": Ticker24hrStat(symbol="RANK3USDT", price_change_percent=20.0),
        }

    def four_hour_klines(self, symbol: str, limit: int = 42, end_time_ms: int | None = None) -> list[Kline]:
        self.four_hour_symbols.append(symbol)
        start = (end_time_ms or self.signal_time_ms) - 42 * 4 * 60 * 60 * 1000
        return [
            Kline(start + index * 4 * 60 * 60 * 1000, 1, 1, 1, 1, volume, start + (index + 1) * 4 * 60 * 60 * 1000 - 1)
            for index, volume in enumerate([1.0] * 36 + [2.0] * 6)
        ]

    def latest_prices(self) -> dict[str, float]:
        return {"RANK1USDT": 11.0, "RANK2USDT": 22.0, "RANK3USDT": 33.0}

    def exchange_symbol(self, symbol: str) -> dict[str, object]:
        return {
            "symbol": symbol,
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }


class Rank1CandidateMarketClient(FakeMarketClient):
    def four_hour_klines(self, symbol: str, limit: int = 42, end_time_ms: int | None = None) -> list[Kline]:
        self.four_hour_symbols.append(symbol)
        start = (end_time_ms or self.signal_time_ms) - 42 * 4 * 60 * 60 * 1000
        return [
            Kline(start + index * 4 * 60 * 60 * 1000, 1, 1, 1, 1, volume, start + (index + 1) * 4 * 60 * 60 * 1000 - 1)
            for index, volume in enumerate([1.0] * 36 + [4.0] * 6)
        ]


class BlockedMarketClient(FakeMarketClient):
    def usdt_perpetual_symbols(self) -> list[str]:
        raise RuntimeError("Binance API blocked (HTTP 418) for /fapi/v1/exchangeInfo. Do not retry.")


class FastMarketClient(FakeMarketClient):
    def klines(self, symbol: str, interval: str, limit: int, end_time_ms: int | None = None) -> list[Kline]:
        start = self.signal_time_ms
        if limit == 5:
            return [
                Kline(
                    open_time_ms=start + index * 60 * 1000,
                    open=20.0,
                    high=21.0,
                    low=18.8,
                    close=19.5,
                    volume=1.0,
                    close_time_ms=start + (index + 1) * 60 * 1000 - 1,
                )
                for index in range(5)
            ]
        return [
            Kline(
                open_time_ms=start + index * 60 * 1000,
                open=20.0,
                high=20.4,
                low=18.8,
                close=19.5,
                volume=1.0,
                close_time_ms=start + (index + 1) * 60 * 1000 - 1,
            )
            for index in range(15)
        ]


class ExtremeWeakFastMarketClient(FakeMarketClient):
    def klines(self, symbol: str, interval: str, limit: int, end_time_ms: int | None = None) -> list[Kline]:
        start = self.signal_time_ms
        return [
            Kline(
                open_time_ms=start + index * 60 * 1000,
                open=20.0,
                high=20.3,
                low=18.3,
                close=18.6,
                volume=1.0,
                close_time_ms=start + (index + 1) * 60 * 1000 - 1,
            )
            for index in range(limit)
        ]


class NoWeakExitMarketClient(FakeMarketClient):
    def klines(self, symbol: str, interval: str, limit: int, end_time_ms: int | None = None) -> list[Kline]:
        start = self.signal_time_ms
        return [
            Kline(
                open_time_ms=start + index * 60 * 1000,
                open=20.0,
                high=22.0,
                low=19.5,
                close=21.0,
                volume=1.0,
                close_time_ms=start + (index + 1) * 60 * 1000 - 1,
            )
            for index in range(15)
        ]


class FakeNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_signal_table(self, table: str) -> bool:
        self.messages.append(table)
        return True


class StaticRegimeProvider:
    def __init__(self, context: Top3RegimeContext) -> None:
        self.context = context

    def context_at(self, signal_time_ms: int) -> Top3RegimeContext:
        return self.context


class RecordingRegimeProvider(StaticRegimeProvider):
    def __init__(self, context: Top3RegimeContext) -> None:
        super().__init__(context)
        self.calls: list[int] = []

    def context_at(self, signal_time_ms: int) -> Top3RegimeContext:
        self.calls.append(signal_time_ms)
        return self.context


class FailingRegimeProvider:
    def context_at(self, signal_time_ms: int) -> Top3RegimeContext:
        raise RuntimeError("regime generator failed")


class FailingRank2ExecutionClient(FakeTestnetClient):
    def market_open_long(self, symbol: str, quantity: str) -> dict[str, object]:
        if symbol == "RANK2USDT":
            raise RuntimeError("rank2 order failed")
        return super().market_open_long(symbol, quantity)


def _local_position(symbol: str) -> TestnetPosition:
    return TestnetPosition(
        symbol=symbol,
        entry_time_ms=1_700_000_000_000,
        entry_price=1.0,
        qty=1.0,
        leverage=3,
        order_id=1,
        planned_exit_time_ms=1_700_100_000_000,
        extreme_weak_exit_check_time_ms=1_700_010_000_000,
        weak_exit_check_time_ms=1_700_020_000_000,
    )
