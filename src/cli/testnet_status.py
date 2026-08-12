from __future__ import annotations

import json

from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.execution.testnet_state import TestnetStateStore


def main() -> None:
    settings = AppSettings.from_env_file()
    paths = state_paths_for(TradingMode.TESTNET, settings.signal_mode)
    state = TestnetStateStore(paths.state_path).load()
    print(
        json.dumps(
            {
                "trading_mode": TradingMode.TESTNET.value,
                "signal_mode": settings.signal_mode.value,
                "state_path": str(paths.state_path),
                "event_log_path": str(paths.event_log_path),
                "last_signal_time_ms": state.last_signal_time_ms,
                "last_exit_check_time_ms": state.last_exit_check_time_ms,
                "open_positions": [position.__dict__ for position in state.open_positions],
                "closed_positions": [position.__dict__ for position in state.closed_positions],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
