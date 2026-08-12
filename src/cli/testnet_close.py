from __future__ import annotations

import argparse
import json
import time

from src.config.lock import FileLock
from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.exchange.binance_testnet import TestnetExecutionClient
from src.execution.testnet_state import TestnetStateStore
from src.market.binance_futures import BinanceFuturesMarketClient
from src.paper.store import PaperEventLogger
from src.execution.testnet_runner import MarketSymbolFilterProvider, TestnetTradingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Close a testnet position.")
    parser.add_argument("symbol", help="交易对，如 BTCUSDT")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    settings = AppSettings.from_env_file()
    if settings.trading_mode != TradingMode.TESTNET:
        raise SystemExit("需要 TRADING_MODE=testnet")

    paths = state_paths_for(TradingMode.TESTNET, settings.signal_mode)
    market_client = BinanceFuturesMarketClient()
    execution_client = TestnetExecutionClient(
        api_key=settings.binance_testnet_api_key,
        api_secret=settings.binance_testnet_api_secret,
    )

    lock = FileLock(paths.state_path.parent / ".lock")
    with lock:
        runner = TestnetTradingRunner(
            market_client=market_client,
            execution_client=execution_client,
            state_store=TestnetStateStore(paths.state_path),
            logger=PaperEventLogger(paths.event_log_path),
            settings=settings,
        )

        now_ms = int(time.time() * 1000)
        position = runner.engine.close_position(symbol, now_ms, "manual_test")
        if position is None:
            print(json.dumps({"error": f"{symbol} 未找到持仓或平仓失败"}, ensure_ascii=False))
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
