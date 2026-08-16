from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

from src.config.lock import FileLock
from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.market.binance_futures import BinanceFuturesMarketClient
from src.paper.runner import PaperTradingRunner
from src.paper.store import PaperEventLogger, PaperStateStore
from src.research.rankpulse_regime_context_provider import regime_context_provider_from_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RankPulse paper trading cycle.")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--event-log-path", default=None)
    parser.add_argument("--now-ms", type=int, default=None)
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    if settings.trading_mode != TradingMode.PAPER:
        settings = replace(settings, trading_mode=TradingMode.PAPER)
    settings.assert_can_run()
    paths = state_paths_for(TradingMode.PAPER, settings.signal_mode)
    state_path = args.state_path or str(paths.state_path)
    event_log_path = args.event_log_path or str(paths.event_log_path)
    now_ms = args.now_ms if args.now_ms is not None else int(time.time() * 1000)
    if args.now_ms is not None and abs(args.now_ms - int(time.time() * 1000)) > 3600_000:
        raise SystemExit("--now-ms is too far from current time. This parameter is for testing only.")
    store = PaperStateStore(state_path)
    engine = store.load()

    lock = FileLock(paths.state_path.parent / ".lock")
    with lock:
        runner = PaperTradingRunner(
            market_client=BinanceFuturesMarketClient(),
            engine=engine,
            store=store,
            logger=PaperEventLogger(event_log_path),
            regime_context_provider=regime_context_provider_from_path(settings.top3_regime_context_path),
        )

        weak_exits = runner.run_weak_exit_checks(now_ms)
        planned_exits = runner.run_planned_exits(now_ms)
        opened = runner.run_signal_cycle(now_ms)

        print(
            json.dumps(
                {
                    "now_ms": now_ms,
                    "opened": [position.symbol for position in opened],
                    "weak_exits": [trade_exit.symbol for trade_exit in weak_exits],
                    "planned_exits": [trade_exit.symbol for trade_exit in planned_exits],
                    "open_positions": [position.symbol for position in engine.open_positions()],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
