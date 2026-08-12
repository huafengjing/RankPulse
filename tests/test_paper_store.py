from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.paper.store import PaperEventLogger, PaperStateStore
from src.paper.trading import PaperSignal, PaperTradingEngine


def test_paper_state_store_saves_and_loads_open_positions() -> None:
    workdir = _clean_workdir("store_state")
    state_path = workdir / "paper_state.json"
    store = PaperStateStore(state_path)
    engine = PaperTradingEngine()
    engine.on_signal(
        PaperSignal(
            symbol="AAAUSDT",
            rank=2,
            gain_24h=0.25,
            volume_24h_ratio_7d=2.5,
            snapshot_hour_bj="00:00",
            signal_time_ms=1_700_000_000_000,
            fill_price=10.0,
        )
    )

    store.save(engine)
    loaded_engine = store.load()

    position = loaded_engine.open_position("AAAUSDT")
    assert position is not None
    assert position.entry_price == 10.0
    assert position.leverage == 3


def test_paper_event_logger_appends_json_lines() -> None:
    workdir = _clean_workdir("store_events")
    log_path = workdir / "paper_events.jsonl"
    logger = PaperEventLogger(log_path)

    logger.log("signal_opened", {"symbol": "AAAUSDT", "rank": 2})
    logger.log("planned_exit", {"symbol": "AAAUSDT", "exit_price": 12.0})

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        {"event": "signal_opened", "payload": {"symbol": "AAAUSDT", "rank": 2}},
        {"event": "planned_exit", "payload": {"symbol": "AAAUSDT", "exit_price": 12.0}},
    ]


def _clean_workdir(name: str) -> Path:
    path = Path("tests/.tmp") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path
