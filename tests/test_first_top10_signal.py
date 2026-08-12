from __future__ import annotations

import pandas as pd

from src.research.signals import identify_first_top_signals


def test_first_top10_signal_and_top5_conversion() -> None:
    rankings = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "open_time": 0, "open_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "rank": 10, "rolling_24h_change_pct": 0.2, "close": 1, "quote_volume": 1, "is_top5": False, "is_top10": True, "is_top20": True},
            {"symbol": "AAAUSDT", "open_time": 300_000, "open_time_utc": pd.Timestamp(300_000, unit="ms", tz="UTC"), "rank": 5, "rolling_24h_change_pct": 0.3, "close": 1, "quote_volume": 1, "is_top5": True, "is_top10": True, "is_top20": True},
        ]
    )
    signals = identify_first_top_signals(rankings, cooldown_days=3, observation_hours=1)
    assert len(signals) == 1
    assert bool(signals.iloc[0]["entered_top5_later"]) is True
    assert int(signals.iloc[0]["time_to_top5_minutes"]) == 5
