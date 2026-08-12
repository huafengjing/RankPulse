from __future__ import annotations

import pandas as pd

from src.backtest.engine import run_top10_immediate_backtest


def base_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"open_time": 0, "open_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "symbol": "AAAUSDT", "rank": 10, "rolling_24h_change_pct": 0.2, "close": 100, "quote_volume": 1, "is_top5": False, "is_top10": True, "is_top20": True},
            {"open_time": 300_000, "open_time_utc": pd.Timestamp(300_000, unit="ms", tz="UTC"), "symbol": "AAAUSDT", "rank": 9, "rolling_24h_change_pct": 0.21, "close": 100, "quote_volume": 1, "is_top5": False, "is_top10": True, "is_top20": True},
        ]
    )


def test_entry_occurs_on_next_candle_open() -> None:
    signals = pd.DataFrame([{"symbol": "AAAUSDT", "signal_time": 0, "signal_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "rank": 10, "rolling_24h_change_pct": 0.2, "quote_volume": 1, "is_first_top10": True, "entered_top5_later": False, "time_to_top5_minutes": None, "eligible_for_entry": True}])
    klines = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "open_time": 0, "open_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "open": 90, "high": 90, "low": 90, "close": 90},
            {"symbol": "AAAUSDT", "open_time": 300_000, "open_time_utc": pd.Timestamp(300_000, unit="ms", tz="UTC"), "open": 100, "high": 111, "low": 99, "close": 110},
        ]
    )
    trades = run_top10_immediate_backtest(signals, klines, base_rankings(), "ps")
    assert trades.iloc[0]["entry_time_utc"] == pd.Timestamp(300_000, unit="ms", tz="UTC")
    assert trades.iloc[0]["entry_price_raw"] == 100
    assert bool(trades.iloc[0]["tp1_hit"]) is True


def test_same_candle_tp_and_sl_prefers_sl() -> None:
    signals = pd.DataFrame([{"symbol": "AAAUSDT", "signal_time": 0, "signal_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "rank": 10, "rolling_24h_change_pct": 0.2, "quote_volume": 1, "is_first_top10": True, "entered_top5_later": False, "time_to_top5_minutes": None, "eligible_for_entry": True}])
    klines = pd.DataFrame(
        [
            {"symbol": "AAAUSDT", "open_time": 0, "open_time_utc": pd.Timestamp(0, unit="ms", tz="UTC"), "open": 90, "high": 90, "low": 90, "close": 90},
            {"symbol": "AAAUSDT", "open_time": 300_000, "open_time_utc": pd.Timestamp(300_000, unit="ms", tz="UTC"), "open": 100, "high": 111, "low": 94, "close": 100},
        ]
    )
    trades = run_top10_immediate_backtest(signals, klines, base_rankings(), "ps", sl_pct=-0.05)
    assert trades.iloc[0]["exit_reason"] == "sl"
    assert bool(trades.iloc[0]["sl_hit"]) is True
