from __future__ import annotations

import argparse
import json
import time

from src.config.lock import FileLock
from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.exchange.binance_testnet import TestnetExecutionClient
from src.execution.testnet_runner import TestnetTradingRunner
from src.execution.testnet_state import TestnetStateStore
from src.market.binance_futures import BinanceFuturesMarketClient
from src.notify.telegram import TelegramNotifier
from src.paper.store import PaperEventLogger
from src.research.top3_regime_context_provider import regime_context_provider_for_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Binance Futures Testnet cycle.")
    parser.add_argument("--now-ms", type=int, default=None)
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    settings.assert_can_run()
    if settings.trading_mode != TradingMode.TESTNET:
        raise SystemExit("testnet_once requires TRADING_MODE=testnet.")
    if not settings.binance_testnet_api_key or not settings.binance_testnet_api_secret:
        raise SystemExit("BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET are required.")

    now_ms = args.now_ms if args.now_ms is not None else int(time.time() * 1000)
    if args.now_ms is not None and abs(args.now_ms - int(time.time() * 1000)) > 3600_000:
        raise SystemExit("--now-ms is too far from current time. This parameter is for testing only.")
    paths = state_paths_for(TradingMode.TESTNET, settings.signal_mode)

    lock = FileLock(paths.state_path.parent / ".lock")
    with lock:
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
        weak_exits, planned_exits = runner.run_hourly_exit_cycle(now_ms)
        information_sent = runner.run_information_cycle(now_ms)
        opened = runner.run_signal_cycle(now_ms)
        state = TestnetStateStore(paths.state_path).load()

        print(
            json.dumps(
                {
                    "trading_mode": TradingMode.TESTNET.value,
                    "signal_mode": settings.signal_mode.value,
                    "now_ms": now_ms,
                    "opened": [position.symbol for position in opened],
                    "weak_exits": [position.symbol for position in weak_exits],
                    "planned_exits": [position.symbol for position in planned_exits],
                    "information_sent": information_sent,
                    "open_positions": [position.symbol for position in state.open_positions],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
