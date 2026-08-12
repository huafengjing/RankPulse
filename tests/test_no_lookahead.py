from __future__ import annotations

import pandas as pd

from src.backtest.engine import run_top10_immediate_backtest


def test_future_top5_is_recorded_but_not_required_for_immediate_entry() -> None:
    signals = pd.DataFrame([{"symbol": "AAAUSDT", "signal_time": 0, "signal_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "rank": 10, "rolling_24h_change_pct": 0.2, "quote_volume": 1, "is_first_top10": True, "entered_top5_later": False, "time_to_top5_minutes": None, "eligible_for_entry": True}])
    klines = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "open_time": 0, "open_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "open": 90, "high": 90, "low": 90, "close": 90},
            {"symbol": "AAAUSDT", "open_time": 300_000, "open_time_utc": pd.Timestamp(300_000, unit="ms", tz="UTC"), "open": 100, "high": 101, "low": 99, "close": 100},
        ]
    )
    rankings = pd.DataFrame([{"open_time": 300_000, "open_time_utc": pd.Timestamp(300_000, unit="ms", tz="UTC"), "symbol": "AAAUSDT", "rank": 9, "rolling_24h_change_pct": 0.2, "close": 100, "quote_volume": 1, "is_top5": False, "is_top10": True, "is_top20": True}])
    trades = run_top10_immediate_backtest(signals, klines, rankings, "ps")
    assert len(trades) == 1
    assert bool(trades.iloc[0]["reached_top5_after_top10"]) is False
