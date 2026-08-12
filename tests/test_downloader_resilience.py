from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.downloader import download_symbols_klines


class MemoryCache:
    def __init__(self) -> None:
        self.frames: dict[Path, pd.DataFrame] = {}

    def kline_path(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> Path:
        return Path(f"{symbol}_{interval}_{start_ms}_{end_ms}.csv")

    def read_klines(self, path: Path) -> pd.DataFrame | None:
        return self.frames.get(path)

    def write_klines(self, path: Path, frame: pd.DataFrame) -> None:
        self.frames[path] = frame.copy()


class FlakyClient:
    def klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1500):
        if symbol == "BADUSDT":
            raise RuntimeError("simulated ssl failure")
        return [
            [
                start_time,
                "1",
                "2",
                "1",
                "2",
                "10",
                start_time + 299_999,
                "20",
                1,
                "5",
                "10",
                "0",
            ]
        ]


def test_single_symbol_failure_does_not_interrupt_download(tmp_path) -> None:
    stats = download_symbols_klines(
        FlakyClient(),
        MemoryCache(),
        ["AAAUSDT", "BADUSDT", "BBBUSDT"],
        "5m",
        0,
        0,
        tmp_path,
        sleep_seconds=0,
    )
    failed_path = tmp_path / "logs" / "failed_downloads.csv"
    failed = pd.read_csv(failed_path)
    assert stats == {"completed": 2, "failed": 1}
    assert failed.iloc[0]["symbol"] == "BADUSDT"
    assert "simulated ssl failure" in failed.iloc[0]["error"]
