from __future__ import annotations

import pandas as pd

from src.research.ranking import add_rolling_24h_change


def test_rolling_24h_change_uses_exact_past_close() -> None:
    rows = []
    for i in range(289):
        rows.append({"symbol": "AAAUSDT", "open_time": i * 300_000, "close": 100 + i, "open_time_utc": pd.Timestamp(i * 300_000, unit="ms", tz="UTC"), "quote_volume": 1})
    frame = pd.DataFrame(rows)
    result = add_rolling_24h_change(frame, interval_minutes=5)
    assert result.loc[288, "rolling_24h_change_pct"] == (388 / 100) - 1
    assert bool(result.loc[287, "has_full_24h_history"]) is False
