from __future__ import annotations

import json

from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.paper.store import PaperStateStore


def main() -> None:
    settings = AppSettings.from_env_file()
    paths = state_paths_for(TradingMode.PAPER, settings.signal_mode)
    engine = PaperStateStore(paths.state_path).load()
    print(
        json.dumps(
            {
                "trading_mode": TradingMode.PAPER.value,
                "signal_mode": settings.signal_mode.value,
                "state_path": str(paths.state_path),
                "event_log_path": str(paths.event_log_path),
                "open_positions": [position.__dict__ for position in engine.open_positions()],
                "closed_trades": [trade.__dict__ for trade in engine.closed_trades],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
