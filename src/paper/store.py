from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.paper.trading import PaperTradingConfig, PaperTradingEngine


class PaperStateStore:
    def __init__(self, path: str | Path, config: PaperTradingConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config

    def load(self) -> PaperTradingEngine:
        if not self.path.exists():
            return PaperTradingEngine(config=self.config)

        snapshot = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict):
            raise ValueError("Paper state snapshot must be a JSON object.")
        return PaperTradingEngine.from_snapshot(snapshot, config=self.config)

    def save(self, engine: PaperTradingEngine) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(engine.snapshot(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


class PaperEventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"event": event, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(f"{line}\n")
