from __future__ import annotations

import argparse
import time

from src.cli.testnet_daemon import render_cycle_summary
from src.config.lock import FileLock
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
    parser = argparse.ArgumentParser(description="Run one Binance Futures live cycle.")
    parser.add_argument("--now-ms", type=int, default=None)
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    try:
        settings.assert_can_run()
    except LiveTradingDisabledError as exc:
        raise SystemExit(str(exc)) from exc
    if settings.trading_mode != TradingMode.LIVE:
        raise SystemExit("live_once requires TRADING_MODE=live.")
    if not settings.binance_live_api_key or not settings.binance_live_api_secret:
        raise SystemExit("BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET are required.")

    now_ms = args.now_ms if args.now_ms is not None else int(time.time() * 1000)
    if args.now_ms is not None and abs(args.now_ms - int(time.time() * 1000)) > 3600_000:
        raise SystemExit("--now-ms is too far from current time. This parameter is for testing only.")
    paths = state_paths_for(TradingMode.LIVE, settings.signal_mode)

    lock = FileLock(paths.state_path.parent / ".lock")
    with lock:
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
        runner.ensure_bootstrap(now_ms)
        weak_exits, planned_exits = runner.run_hourly_exit_cycle(now_ms)
        information_sent = runner.run_information_cycle(now_ms)
        opened = runner.run_signal_cycle(now_ms)
        errors: list[dict[str, str]] = []
        try:
            open_position_symbols = runner.sync_open_positions_from_exchange()
            open_positions_source = "binance"
        except Exception as exc:
            errors.append(
                {
                    "phase": "position_sync",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            open_position_symbols = [
                position.symbol
                for position in TestnetStateStore(paths.state_path).load().open_positions
            ]
            open_positions_source = "local"
        state = TestnetStateStore(paths.state_path).load()

        print(
            render_cycle_summary(
                {
                    "trading_mode": TradingMode.LIVE.value,
                    "signal_mode": settings.signal_mode.value,
                    "now_ms": now_ms,
                    "opened": [position.symbol for position in opened],
                    "weak_exits": [position.symbol for position in weak_exits],
                    "planned_exits": [position.symbol for position in planned_exits],
                    "information_sent": information_sent,
                    "open_positions": open_position_symbols,
                    "open_positions_source": open_positions_source,
                    "errors": errors,
                    "last_signal_time_ms": state.last_signal_time_ms,
                    "preflight_sent": False,
                }
            )
        )


if __name__ == "__main__":
    main()
