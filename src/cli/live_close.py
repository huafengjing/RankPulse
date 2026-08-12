from __future__ import annotations

import argparse
import json
import time

from src.config.lock import FileLock
from src.config.modes import TradingMode
from src.config.settings import AppSettings, LiveTradingDisabledError
from src.config.state_paths import state_paths_for
from src.exchange.binance_live import LiveExecutionClient
from src.execution.testnet_runner import TestnetTradingRunner
from src.execution.testnet_state import TestnetStateStore
from src.market.binance_futures import BinanceFuturesMarketClient
from src.paper.store import PaperEventLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Close a live Binance Futures position with reduce-only market order.")
    parser.add_argument("symbol", help="Trading pair, e.g. BTCUSDT")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    settings = AppSettings.from_env_file()
    try:
        settings.assert_can_run()
    except LiveTradingDisabledError as exc:
        raise SystemExit(str(exc)) from exc
    if settings.trading_mode != TradingMode.LIVE:
        raise SystemExit("live_close requires TRADING_MODE=live.")
    if not settings.binance_live_api_key or not settings.binance_live_api_secret:
        raise SystemExit("BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET are required.")

    paths = state_paths_for(TradingMode.LIVE, settings.signal_mode)
    market_client = BinanceFuturesMarketClient()
    execution_client = LiveExecutionClient(
        api_key=settings.binance_live_api_key,
        api_secret=settings.binance_live_api_secret,
    )

    lock = FileLock(paths.state_path.parent / ".lock")
    with lock:
        runner = TestnetTradingRunner(
            market_client=market_client,
            execution_client=execution_client,
            state_store=TestnetStateStore(paths.state_path),
            logger=PaperEventLogger(paths.event_log_path),
            settings=settings,
            event_prefix="live",
        )

        now_ms = int(time.time() * 1000)
        position = runner.engine.close_position(symbol, now_ms, "manual_live")
        if position is None:
            print(json.dumps({"error": f"{symbol} position not found or close failed"}, ensure_ascii=False))
        else:
            print(
                json.dumps(
                    {
                        "symbol": position.symbol,
                        "entry_price": position.entry_price,
                        "exit_price": position.exit_price,
                        "realized_pnl": position.realized_pnl,
                        "exit_reason": position.exit_reason,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
