from __future__ import annotations

import argparse
import time
import unicodedata
from datetime import datetime, timedelta, timezone

from src.config.lock import FileLock
from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.exchange.binance_testnet import FuturesPosition, TestnetExecutionClient
from src.execution.testnet_runner import TestnetTradingRunner
from src.execution.testnet_state import TestnetState, TestnetStateStore
from src.market.binance_futures import BinanceFuturesMarketClient
from src.notify.telegram import TelegramNotifier
from src.paper.store import PaperEventLogger
from src.research.top3_regime_context_provider import regime_context_provider_for_runtime


BEIJING_TZ = timezone(timedelta(hours=8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Binance Futures Testnet daemon.")
    parser.add_argument("--interval-minutes", type=int, default=None, help="Signal check interval in minutes (default: from SIGNAL_TEST_INTERVAL_MINUTES env)")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    interval_minutes = args.interval_minutes or settings.signal_test_interval_minutes
    with _daemon_cycle_lock(TradingMode.TESTNET, settings):
        print(_load_startup_status(settings), flush=True)

    cycles = 0
    while True:
        try:
            with _daemon_cycle_lock(TradingMode.TESTNET, settings):
                summary = run_one_cycle()
        except Exception as exc:
            summary = {"error": str(exc), "error_type": type(exc).__name__}
        if _has_meaningful_activity(summary):
            print(render_cycle_summary(summary), flush=True)
        cycles += 1
        if args.once or (args.max_cycles is not None and cycles >= args.max_cycles):
            break
        _sleep_until_next_signal(interval_minutes)


def _load_startup_status(settings: AppSettings) -> str:
    settings.assert_can_run()
    if settings.trading_mode != TradingMode.TESTNET:
        raise SystemExit("testnet_daemon requires TRADING_MODE=testnet.")
    if not settings.binance_testnet_api_key or not settings.binance_testnet_api_secret:
        raise SystemExit("BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET are required.")
    paths = state_paths_for(TradingMode.TESTNET, settings.signal_mode)
    state_store = TestnetStateStore(paths.state_path)
    event_logger = PaperEventLogger(paths.event_log_path)
    state = state_store.load()
    sync_warning: str | None = None
    execution_client = TestnetExecutionClient(
        api_key=settings.binance_testnet_api_key,
        api_secret=settings.binance_testnet_api_secret,
    )
    try:
        state = reconcile_startup_state(
            state_store=state_store,
            exchange_positions=execution_client.open_positions(),
            logger=event_logger,
            settings=settings,
        )
    except Exception as exc:
        sync_warning = str(exc)
    latest_prices: dict[str, float] = {}
    if state.open_positions:
        try:
            latest_prices = BinanceFuturesMarketClient().latest_prices()
        except Exception as exc:
            price_warning = f"行情价格读取失败: {exc}"
            sync_warning = (
                f"{sync_warning}; {price_warning}"
                if sync_warning
                else price_warning
            )
    return render_startup_status(
        state=state,
        latest_prices=latest_prices,
        settings=settings,
        now_ms=int(time.time() * 1000),
        sync_warning=sync_warning,
    )


def reconcile_startup_state(
    state_store: TestnetStateStore,
    exchange_positions: dict[str, FuturesPosition],
    logger: PaperEventLogger,
    settings: AppSettings,
) -> TestnetState:
    state = state_store.load()
    stale_positions = [
        position
        for position in state.open_positions
        if position.symbol not in exchange_positions
        or exchange_positions[position.symbol].position_amt == 0
    ]
    if not stale_positions:
        return state

    stale_symbols = {position.symbol for position in stale_positions}
    reconciled_state = TestnetState(
        open_positions=[
            position
            for position in state.open_positions
            if position.symbol not in stale_symbols
        ],
        closed_positions=state.closed_positions,
        last_signal_time_ms=state.last_signal_time_ms,
        last_exit_check_time_ms=state.last_exit_check_time_ms,
        last_information_time_ms=state.last_information_time_ms,
        last_preflight_time_ms=state.last_preflight_time_ms,
    )
    state_store.save(reconciled_state)
    for position in stale_positions:
        logger.log(
            f"{settings.trading_mode.value}_position_reconciled",
            {
                "symbol": position.symbol,
                "reason": "exchange_position_missing_at_startup",
                "local_qty": position.qty,
                "trading_mode": settings.trading_mode.value,
                "signal_mode": settings.signal_mode.value,
            },
        )
    return reconciled_state


def render_startup_status(
    state: TestnetState,
    latest_prices: dict[str, float],
    settings: AppSettings,
    now_ms: int,
    sync_warning: str | None = None,
) -> str:
    lines = [
        "Rank2/Rank3 的捕捉系统已启动",
        "",
        "当前持仓列表",
    ]
    if sync_warning:
        lines.extend([f"持仓同步警告: {sync_warning}", ""])
    if not state.open_positions:
        lines.append("暂无持仓")
    else:
        rows: list[list[str]] = []
        for position in state.open_positions:
            current_price = latest_prices.get(position.symbol)
            return_pct = (
                ((current_price / position.entry_price) - 1) * 100
                if current_price is not None and position.entry_price
                else None
            )
            leveraged_return_pct = (
                return_pct * position.leverage
                if return_pct is not None
                else None
            )
            rows.append(
                [
                    position.symbol,
                    _format_price(position.entry_price),
                    _format_price(current_price),
                    f"{position.leverage}x",
                    _format_percent(return_pct),
                    _format_percent(leveraged_return_pct),
                ]
            )
        lines.append(
            _render_table(
                ["代币", "开仓价格", "当前价格", "杠杆", "收益率", "杠杆收益率"],
                rows,
                ["left", "right", "right", "right", "right", "right"],
            )
        )

    next_signal = _next_signal_time(now_ms, settings)
    lines.extend(
        [
            "",
            f"下次信号时间: {next_signal.strftime('%Y-%m-%d %H:%M:%S')} 北京时间",
        ]
    )
    return "\n".join(lines)


def _next_signal_time(now_ms: int, settings: AppSettings) -> datetime:
    now = datetime.fromtimestamp(now_ms / 1000, tz=BEIJING_TZ)
    if settings.signal_mode.value == "test_fast":
        interval = settings.signal_test_interval_minutes
        next_minute = ((now.minute // interval) + 1) * interval
        base = now.replace(second=0, microsecond=0)
        if next_minute >= 60:
            return (base.replace(minute=0) + timedelta(hours=1))
        return base.replace(minute=next_minute)

    today_08 = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now < today_08:
        return today_08
    tomorrow_00 = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow_00


def _has_meaningful_activity(summary: dict[str, object]) -> bool:
    if summary.get("error") or summary.get("errors"):
        return True
    return any(summary.get(key) for key in ("opened", "weak_exits", "planned_exits", "preflight_sent"))


def _daemon_cycle_lock(trading_mode: TradingMode, settings: AppSettings) -> FileLock:
    paths = state_paths_for(trading_mode, settings.signal_mode)
    return FileLock(paths.state_path.parent / ".lock")


def render_cycle_summary(summary: dict[str, object]) -> str:
    lines = [
        "",
        "=" * 56,
        f"运行时间: {_format_time_value(summary.get('now_ms'))}",
        f"交易模式: {summary.get('trading_mode', 'unknown')}",
        f"信号模式: {summary.get('signal_mode', 'unknown')}",
        "-" * 56,
    ]

    if summary.get("error"):
        lines.extend(
            [
                "运行结果: 发生错误",
                f"错误类型: {summary.get('error_type', 'unknown')}",
                f"错误内容: {summary.get('error')}",
                "=" * 56,
            ]
        )
        return "\n".join(lines)

    errors = summary.get("errors")
    if isinstance(errors, list) and errors:
        lines.append("运行结果: 部分步骤失败")
        for error in errors:
            if isinstance(error, dict):
                lines.extend(_format_phase_error(error))
    else:
        lines.append("运行结果: 正常")

    lines.extend(
        [
            f"本次新开仓: {_format_symbols(summary.get('opened'))}",
            f"提前退出: {_format_symbols(summary.get('weak_exits'))}",
            f"6D计划退出: {_format_symbols(summary.get('planned_exits'))}",
            f"当前持仓: {_format_symbols(summary.get('open_positions'))}",
            f"行情预检: {'已发送' if summary.get('preflight_sent') else '未触发'}",
            f"23:00观察信息: {'已发送' if summary.get('information_sent') else '未发送'}",
            f"已处理信号时间: {_format_time_value(summary.get('last_signal_time_ms'))}",
            "=" * 56,
        ]
    )
    return "\n".join(lines)


def _format_phase_error(error: dict[str, object]) -> list[str]:
    phase = str(error.get("phase", "unknown"))
    error_type = str(error.get("error_type", "Error"))
    message = str(error.get("error", ""))
    phase_name = {
        "signal": "信号检查",
        "information_signal": "23:00观察信息",
        "market_preflight": "行情/IP预检",
        "hourly_exit": "持仓退出检查",
    }.get(phase, phase)

    if "Market data request failed" in message:
        endpoint = _extract_binance_endpoint(message)
        if endpoint == "/fapi/v1/exchangeInfo":
            action = "读取 Binance 合约交易对列表失败"
        elif endpoint == "/fapi/v1/ticker/24hr":
            action = "读取 Binance 24H 涨幅榜失败"
        elif endpoint == "/fapi/v1/klines":
            action = "读取 Binance K线数据失败"
        else:
            action = "读取 Binance 公共行情失败"
        return [
            f"- {phase_name}: {action}",
            "  结果: 本轮没有生成 Rank2/Rank3 信号，也没有下单。",
            "  建议: 保持程序运行，下一次检查会自动重试；如果连续出现，请检查网络、代理或 Binance 访问稳定性。",
            f"  技术细节: {message}",
        ]

    if "Binance API blocked (HTTP 418)" in message:
        endpoint = _extract_binance_endpoint(message)
        target = {
            "/fapi/v1/exchangeInfo": "合约交易对列表",
            "/fapi/v1/ticker/24hr": "24H 涨幅榜",
            "/fapi/v1/klines": "K线数据",
        }.get(endpoint, "公共行情")
        return [
            f"- {phase_name}: Binance 临时拦截了当前 IP 的公共行情请求",
            f"  位置: 读取 {target} 时被拒绝。",
            "  结果: 本轮没有生成 Rank2/Rank3 信号，也没有下单。",
            "  建议: 先停止重复运行的 daemon/监控脚本，等待 Binance 解除临时拦截后再只启动一个 live_daemon。",
            f"  技术细节: {message}",
        ]

    if "positionRisk" in message:
        return [
            f"- {phase_name}: 读取 Binance 当前持仓失败",
            "  结果: 为避免重复开仓或误平仓，本轮相关交易动作已跳过。",
            f"  技术细节: {error_type} - {message}",
        ]

    if "/fapi/v1/order" in message:
        return [
            f"- {phase_name}: Binance 订单接口返回失败",
            "  结果: 请先用 live_diag 或 Binance App 确认真实仓位，再决定是否手动处理。",
            f"  技术细节: {error_type} - {message}",
        ]

    return [f"- {phase_name}: {error_type} - {message}"]


def _extract_binance_endpoint(message: str) -> str | None:
    for endpoint in ("/fapi/v1/exchangeInfo", "/fapi/v1/ticker/24hr", "/fapi/v1/klines"):
        if endpoint in message:
            return endpoint
    return None


def _format_symbols(value: object) -> str:
    if not value:
        return "无"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "无"
    return str(value)


def _format_time_value(value: object) -> str:
    if not isinstance(value, int):
        return "无"
    return datetime.fromtimestamp(value / 1000, tz=BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S 北京时间")


def _render_table(headers: list[str], rows: list[list[str]], alignments: list[str]) -> str:
    widths = [
        max(_display_width(header), *(_display_width(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return " | ".join(
            _pad_display(value, widths[index], alignments[index])
            for index, value in enumerate(values)
        )

    return "\n".join(
        [
            render_row(headers),
            "-+-".join("-" * width for width in widths),
            *(render_row(row) for row in rows),
        ]
    )


def _display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        for char in value
    )


def _pad_display(value: str, width: int, alignment: str) -> str:
    padding = max(0, width - _display_width(value))
    return (" " * padding + value) if alignment == "right" else (value + " " * padding)


def _format_price(price: float | None) -> str:
    if price is None:
        return "N/A"
    return f"{price:.12f}".rstrip("0").rstrip(".")


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _sleep_until_next_signal(interval_minutes: int) -> None:
    """Sleep until the next clock-aligned signal time boundary."""
    now = time.time()
    interval_s = interval_minutes * 60
    next_boundary = ((int(now) // interval_s) + 1) * interval_s
    delay = next_boundary - int(now)
    if delay <= 0:
        delay = interval_s
    time.sleep(delay)


def run_one_cycle(now_ms: int | None = None) -> dict[str, object]:
    settings = AppSettings.from_env_file()
    settings.assert_can_run()
    if settings.trading_mode != TradingMode.TESTNET:
        raise SystemExit("testnet_daemon requires TRADING_MODE=testnet.")
    if not settings.binance_testnet_api_key or not settings.binance_testnet_api_secret:
        raise SystemExit("BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET are required.")

    current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    paths = state_paths_for(TradingMode.TESTNET, settings.signal_mode)
    market_client = BinanceFuturesMarketClient()
    execution_client = TestnetExecutionClient(
        api_key=settings.binance_testnet_api_key,
        api_secret=settings.binance_testnet_api_secret,
    )
    runner = TestnetTradingRunner(
        market_client=market_client,
        execution_client=execution_client,
        state_store=TestnetStateStore(paths.state_path),
        logger=PaperEventLogger(paths.event_log_path),
        settings=settings,
        signal_notifier=TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        ),
        regime_context_provider=regime_context_provider_for_runtime(
            settings,
            paths.state_path.parent / "regime_context.json",
        ),
    )

    return {
        "trading_mode": TradingMode.TESTNET.value,
        "signal_mode": settings.signal_mode.value,
        "now_ms": current_ms,
        **_run_runner_cycle(
            runner=runner,
            state_store=TestnetStateStore(paths.state_path),
            current_ms=current_ms,
        ),
    }


def _run_runner_cycle(
    runner: object,
    state_store: TestnetStateStore,
    current_ms: int,
) -> dict[str, object]:
    errors: list[dict[str, str]] = []
    weak_exits: list[object] = []
    planned_exits: list[object] = []
    opened: list[object] = []
    preflight_sent = False
    information_sent = False

    try:
        preflight_sent = runner.run_market_preflight_cycle(current_ms)  # type: ignore[attr-defined]
    except Exception as exc:
        errors.append(_phase_error("market_preflight", exc))

    try:
        weak_exits, planned_exits = runner.run_hourly_exit_cycle(current_ms)  # type: ignore[attr-defined]
    except Exception as exc:
        errors.append(_phase_error("hourly_exit", exc))

    try:
        information_sent = runner.run_information_cycle(current_ms)  # type: ignore[attr-defined]
    except Exception as exc:
        errors.append(_phase_error("information_signal", exc))

    try:
        opened = runner.run_signal_cycle(current_ms)  # type: ignore[attr-defined]
    except Exception as exc:
        errors.append(_phase_error("signal", exc))

    state = state_store.load()
    return {
        "opened": [position.symbol for position in opened],  # type: ignore[attr-defined]
        "weak_exits": [position.symbol for position in weak_exits],  # type: ignore[attr-defined]
        "planned_exits": [position.symbol for position in planned_exits],  # type: ignore[attr-defined]
        "preflight_sent": preflight_sent,
        "information_sent": information_sent,
        "open_positions": [position.symbol for position in state.open_positions],
        "last_signal_time_ms": state.last_signal_time_ms,
        "errors": errors,
    }


def _phase_error(phase: str, exc: Exception) -> dict[str, str]:
    return {
        "phase": phase,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


if __name__ == "__main__":
    main()
