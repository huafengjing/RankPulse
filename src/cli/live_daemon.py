from __future__ import annotations

import argparse
import time

from src.cli.testnet_daemon import (
    _daemon_cycle_lock,
    _has_meaningful_activity,
    _run_runner_cycle,
    _sleep_until_next_signal,
    reconcile_startup_state,
    render_cycle_summary,
    render_startup_status,
)
from src.config.modes import TradingMode
from src.config.settings import AppSettings, LiveTradingDisabledError
from src.config.state_paths import state_paths_for
from src.exchange.binance_live import LiveExecutionClient
from src.execution.testnet_runner import TestnetTradingRunner
from src.execution.testnet_state import TestnetStateStore
from src.market.binance_futures import BinanceFuturesMarketClient
from src.notify.telegram import TelegramNotifier
from src.paper.store import PaperEventLogger
from src.research.rankpulse_regime_context_provider import regime_context_provider_for_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Binance Futures live daemon.")
    parser.add_argument("--interval-minutes", type=int, default=None, help="Signal check interval in minutes (default: from SIGNAL_TEST_INTERVAL_MINUTES env)")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    interval_minutes = args.interval_minutes or settings.signal_test_interval_minutes
    with _daemon_cycle_lock(TradingMode.LIVE, settings):
        print(_load_startup_status(settings), flush=True)

    cycles = 0
    while True:
        try:
            with _daemon_cycle_lock(TradingMode.LIVE, settings):
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
    try:
        settings.assert_can_run()
    except LiveTradingDisabledError as exc:
        raise SystemExit(str(exc)) from exc
    if settings.trading_mode != TradingMode.LIVE:
        raise SystemExit("live_daemon requires TRADING_MODE=live.")
    if not settings.binance_live_api_key or not settings.binance_live_api_secret:
        raise SystemExit("BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET are required.")

    paths = state_paths_for(TradingMode.LIVE, settings.signal_mode)
    state_store = TestnetStateStore(paths.state_path)
    event_logger = PaperEventLogger(paths.event_log_path)
    state = state_store.load()
    sync_warning: str | None = None
    execution_client = LiveExecutionClient(
        api_key=settings.binance_live_api_key,
        api_secret=settings.binance_live_api_secret,
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
            sync_warning = f"{sync_warning}; {price_warning}" if sync_warning else price_warning

    return render_startup_status(
        state=state,
        latest_prices=latest_prices,
        settings=settings,
        now_ms=int(time.time() * 1000),
        sync_warning=sync_warning,
    )


def run_one_cycle(now_ms: int | None = None) -> dict[str, object]:
    settings = AppSettings.from_env_file()
    try:
        settings.assert_can_run()
    except LiveTradingDisabledError as exc:
        raise SystemExit(str(exc)) from exc
    if settings.trading_mode != TradingMode.LIVE:
        raise SystemExit("live_daemon requires TRADING_MODE=live.")
    if not settings.binance_live_api_key or not settings.binance_live_api_secret:
        raise SystemExit("BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET are required.")

    current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    paths = state_paths_for(TradingMode.LIVE, settings.signal_mode)
    market_client = BinanceFuturesMarketClient()
    execution_client = LiveExecutionClient(
        api_key=settings.binance_live_api_key,
        api_secret=settings.binance_live_api_secret,
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
        event_prefix="live",
    )

    return {
        "trading_mode": TradingMode.LIVE.value,
        "signal_mode": settings.signal_mode.value,
        "now_ms": current_ms,
        **_run_runner_cycle(
            runner=runner,
            state_store=TestnetStateStore(paths.state_path),
            current_ms=current_ms,
        ),
    }


if __name__ == "__main__":
    main()
