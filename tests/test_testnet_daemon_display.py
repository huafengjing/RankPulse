from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config.modes import SignalMode, TradingMode
from src.config.settings import AppSettings
from src.cli.testnet_daemon import (
    _daemon_cycle_lock,
    _run_runner_cycle,
    _has_meaningful_activity,
    reconcile_startup_state,
    render_cycle_summary,
    render_startup_status,
)
from src.exchange.binance_testnet import FuturesPosition
from src.execution.testnet_state import TestnetPosition, TestnetState, TestnetStateStore
from src.paper.store import PaperEventLogger
from tests.test_testnet_execution import _clean_workdir


BEIJING = timezone(timedelta(hours=8))


def bj_ms(day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 6, day, hour, minute, tzinfo=BEIJING).timestamp() * 1000)


def test_startup_status_shows_no_positions_and_next_production_signal() -> None:
    output = render_startup_status(
        state=TestnetState(open_positions=[], closed_positions=[]),
        latest_prices={},
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
        ),
        now_ms=bj_ms(18, 9, 30),
    )

    assert "Rank2/Rank3 的捕捉系统已启动" in output
    assert "当前持仓列表" in output
    assert "暂无持仓" in output
    assert "下次信号时间: 2026-06-19 00:00:00 北京时间" in output


def test_startup_status_can_show_position_sync_warning_without_crashing() -> None:
    output = render_startup_status(
        state=TestnetState(open_positions=[], closed_positions=[]),
        latest_prices={},
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
        ),
        now_ms=bj_ms(18, 9, 30),
        sync_warning="Testnet 时间同步失败",
    )

    assert "持仓同步警告: Testnet 时间同步失败" in output
    assert "Rank2/Rank3 的捕捉系统已启动" in output


def test_startup_status_shows_position_returns() -> None:
    output = render_startup_status(
        state=TestnetState(
            open_positions=[
                TestnetPosition(
                    symbol="AAAUSDT",
                    entry_time_ms=bj_ms(18, 0),
                    entry_price=10.0,
                    qty=3.0,
                    leverage=5,
                    order_id=1,
                    planned_exit_time_ms=bj_ms(24, 0),
                    extreme_weak_exit_check_time_ms=bj_ms(18, 4),
                    weak_exit_check_time_ms=bj_ms(18, 12),
                )
            ],
            closed_positions=[],
        ),
        latest_prices={"AAAUSDT": 11.0},
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
        ),
        now_ms=bj_ms(18, 7),
    )

    assert "代币" in output
    assert "开仓价格" in output
    assert "当前价格" in output
    assert "杠杆收益率" in output
    assert "AAAUSDT" in output
    assert "+10.00%" in output
    assert "+50.00%" in output
    assert "下次信号时间: 2026-06-18 08:00:00 北京时间" in output


def test_empty_daemon_cycle_is_not_meaningful_activity() -> None:
    assert _has_meaningful_activity(
        {
            "opened": [],
            "weak_exits": [],
            "planned_exits": [],
            "open_positions": [],
        }
    ) is False
    assert _has_meaningful_activity({"opened": ["AAAUSDT"]}) is True
    assert _has_meaningful_activity({"error": "network failed"}) is True


def test_daemon_cycle_lock_uses_mode_and_signal_specific_state_directory() -> None:
    settings = AppSettings(trading_mode=TradingMode.LIVE, signal_mode=SignalMode.PRODUCTION)

    lock = _daemon_cycle_lock(TradingMode.LIVE, settings)

    assert str(lock.lock_path).endswith("data\\live\\production\\.lock") or str(lock.lock_path).endswith("data/live/production/.lock")


def test_cycle_summary_is_rendered_as_readable_chinese_text() -> None:
    output = render_cycle_summary(
        {
            "errors": [],
            "information_sent": False,
            "last_signal_time_ms": bj_ms(26, 8),
            "now_ms": bj_ms(26, 8) + 419,
            "open_positions": ["AINUSDT"],
            "opened": ["AINUSDT"],
            "planned_exits": [],
            "signal_mode": "production",
            "trading_mode": "live",
            "weak_exits": [],
        }
    )

    assert "交易模式: live" in output
    assert "信号模式: production" in output
    assert "运行结果: 正常" in output
    assert "本次新开仓: AINUSDT" in output
    assert "当前持仓: AINUSDT" in output
    assert "提前退出: 无" in output
    assert "6D计划退出: 无" in output
    assert "已处理信号时间: 2026-06-26 08:00:00 北京时间" in output


def test_cycle_summary_renders_errors_without_json_dump() -> None:
    output = render_cycle_summary(
        {
            "trading_mode": "live",
            "signal_mode": "production",
            "now_ms": bj_ms(26, 8),
            "errors": [
                {
                    "phase": "signal",
                    "error_type": "RuntimeError",
                    "error": "network failed",
                }
            ],
            "opened": [],
            "weak_exits": [],
            "planned_exits": [],
            "open_positions": [],
        }
    )

    assert "运行结果: 部分步骤失败" in output
    assert "- 信号检查: RuntimeError - network failed" in output
    assert "本次新开仓: 无" in output


def test_cycle_summary_renders_market_data_error_as_user_friendly_chinese() -> None:
    output = render_cycle_summary(
        {
            "trading_mode": "live",
            "signal_mode": "production",
            "now_ms": bj_ms(26, 8),
            "errors": [
                {
                    "phase": "signal",
                    "error_type": "RuntimeError",
                    "error": "Market data request failed after 5 retries: /fapi/v1/exchangeInfo",
                }
            ],
            "opened": [],
            "weak_exits": [],
            "planned_exits": [],
            "open_positions": [],
            "information_sent": False,
        }
    )

    assert "运行结果: 部分步骤失败" in output
    assert "- 信号检查: 读取 Binance 合约交易对列表失败" in output
    assert "结果: 本轮没有生成 Rank2/Rank3 信号，也没有下单。" in output
    assert "技术细节: Market data request failed after 5 retries: /fapi/v1/exchangeInfo" in output
    assert "- signal: RuntimeError" not in output


def test_cycle_summary_renders_http_418_as_user_friendly_chinese() -> None:
    output = render_cycle_summary(
        {
            "trading_mode": "live",
            "signal_mode": "production",
            "now_ms": bj_ms(26, 23),
            "errors": [
                {
                    "phase": "information_signal",
                    "error_type": "RuntimeError",
                    "error": "Binance API blocked (HTTP 418) for /fapi/v1/exchangeInfo. Do not retry.",
                }
            ],
            "opened": [],
            "weak_exits": [],
            "planned_exits": [],
            "open_positions": [],
            "information_sent": False,
        }
    )

    assert "- 23:00观察信息: Binance 临时拦截了当前 IP 的公共行情请求" in output
    assert "位置: 读取 合约交易对列表 时被拒绝。" in output
    assert "结果: 本轮没有生成 Rank2/Rank3 信号，也没有下单。" in output
    assert "- information_signal: RuntimeError" not in output


def test_exit_failure_does_not_block_signal_cycle() -> None:
    runner = FailingExitRunner()
    state_store = FakeStateStore()

    summary = _run_runner_cycle(
        runner=runner,
        state_store=state_store,
        current_ms=bj_ms(18, 8),
    )

    assert runner.signal_called is True
    assert summary["opened"] == ["AAAUSDT"]
    assert summary["errors"] == [
        {
            "phase": "hourly_exit",
            "error_type": "RuntimeError",
            "error": "HTTP 400 close failed",
        }
    ]


def test_startup_reconciliation_removes_local_position_missing_on_testnet() -> None:
    workdir = _clean_workdir("daemon_startup_reconcile")
    state_store = TestnetStateStore(workdir / "state.json")
    state_store.save(
        TestnetState(
            open_positions=[
                _position("GUAUSDT"),
                _position("ZEREBROUSDT"),
            ],
            closed_positions=[],
        )
    )

    state = reconcile_startup_state(
        state_store=state_store,
        exchange_positions={
            "ZEREBROUSDT": FuturesPosition(
                symbol="ZEREBROUSDT",
                position_amt=2318.0,
                entry_price=0.04316,
            )
        },
        logger=PaperEventLogger(workdir / "events.jsonl"),
        settings=AppSettings(
            trading_mode=TradingMode.TESTNET,
            signal_mode=SignalMode.PRODUCTION,
        ),
    )

    assert [position.symbol for position in state.open_positions] == ["ZEREBROUSDT"]
    assert state_store.load().open_position("GUAUSDT") is None


class FailingExitRunner:
    def __init__(self) -> None:
        self.signal_called = False

    def run_hourly_exit_cycle(self, now_ms: int):
        raise RuntimeError("HTTP 400 close failed")

    def run_signal_cycle(self, now_ms: int):
        self.signal_called = True
        return [type("Position", (), {"symbol": "AAAUSDT"})()]

    def run_information_cycle(self, now_ms: int):
        return False

    def run_market_preflight_cycle(self, now_ms: int):
        return False


class FakeStateStore:
    def load(self) -> TestnetState:
        return TestnetState(open_positions=[], closed_positions=[])


def _position(symbol: str) -> TestnetPosition:
    return TestnetPosition(
        symbol=symbol,
        entry_time_ms=bj_ms(18, 0),
        entry_price=10.0,
        qty=1.0,
        leverage=3,
        order_id=1,
        planned_exit_time_ms=bj_ms(24, 0),
        extreme_weak_exit_check_time_ms=bj_ms(18, 4),
        weak_exit_check_time_ms=bj_ms(18, 12),
    )
