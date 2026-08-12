from __future__ import annotations

import pandas as pd

from src.research.ranking import build_rankings


def make_rows(symbol: str, base: float, final: float) -> list[dict]:
    rows = []
    for i in range(289):
        close = base if i == 0 else final
        rows.append({"symbol": symbol, "open_time": i * 300_000, "close": close, "open_time_utc": pd.Timestamp(i * 300_000, unit="ms", tz="UTC"), "quote_volume": 1})
    return rows


def test_ranking_orders_by_rolling_change() -> None:
    frame = pd.DataFrame(make_rows("AAAUSDT", 100, 130) + make_rows("BBBUSDT", 100, 110))
    rankings = build_rankings(frame, interval_minutes=5)
    last = rankings[rankings["open_time"] == 288 * 300_000]
    assert last.iloc[0]["symbol"] == "AAAUSDT"
    assert int(last.iloc[0]["rank"]) == 1
