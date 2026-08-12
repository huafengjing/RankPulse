from __future__ import annotations

import pandas as pd

from src.data.downloader import load_cached_klines


def test_load_cached_klines_dedupes_overlapping_cache_files(tmp_path) -> None:
    root = tmp_path / "raw" / "klines" / "5m" / "AAAUSDT"
    root.mkdir(parents=True)
    older = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "interval": "5m", "open_time": 0, "open": 1},
            {"symbol": "AAAUSDT", "interval": "5m", "open_time": 300_000, "open": 2},
        ]
    )
    newer = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "interval": "5m", "open_time": 300_000, "open": 20},
            {"symbol": "AAAUSDT", "interval": "5m", "open_time": 600_000, "open": 3},
        ]
    )
    older.to_csv(root / "AAAUSDT_5m_0_300000.csv", index=False)
    newer.to_csv(root / "AAAUSDT_5m_0_600000.csv", index=False)

    loaded = load_cached_klines(tmp_path, "5m")

    assert len(loaded) == 3
    assert loaded.loc[loaded["open_time"] == 300_000, "open"].iloc[0] == 20
