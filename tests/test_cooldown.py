from __future__ import annotations

import pandas as pd

from src.research.signals import identify_first_top_signals


def test_cooldown_requires_no_top10_hit_in_previous_window() -> None:
    rows = []
    for minute in [0, 60, 2 * 24 * 60, 4 * 24 * 60, 8 * 24 * 60]:
        ts = minute * 60_000
        rows.append({"symbol": "AAAUSDT", "open_time": ts, "open_time_utc": pd.Timestamp(ts, unit="ms", tz="UTC"), "rank": 8, "rolling_24h_change_pct": 0.2, "close": 1, "quote_volume": 1, "is_top5": False, "is_top10": True, "is_top20": True})
    signals = identify_first_top_signals(pd.DataFrame(rows), cooldown_days=3, observation_hours=1)
    assert len(signals) == 2
    assert list(signals["signal_time"]) == [0, 8 * 24 * 60 * 60_000]
