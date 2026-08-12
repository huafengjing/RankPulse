from __future__ import annotations

import pandas as pd


def add_rolling_24h_change(klines: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    periods = int(24 * 60 / interval_minutes)
    frame = klines.sort_values(["symbol", "open_time"]).copy()
    previous = frame.groupby("symbol")["close"].shift(periods)
    frame["rolling_24h_change_pct"] = frame["close"] / previous - 1.0
    frame["has_full_24h_history"] = previous.notna()
    return frame


def build_rankings(klines: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    frame = add_rolling_24h_change(klines, interval_minutes)
    frame = frame[frame["has_full_24h_history"]].copy()
    frame["rank"] = frame.groupby("open_time")["rolling_24h_change_pct"].rank(method="first", ascending=False)
    frame["rank"] = frame["rank"].astype(int)
    frame["is_top5"] = frame["rank"] <= 5
    frame["is_top10"] = frame["rank"] <= 10
    frame["is_top20"] = frame["rank"] <= 20
    columns = [
        "open_time",
        "open_time_utc",
        "symbol",
        "rank",
        "rolling_24h_change_pct",
        "close",
        "quote_volume",
        "is_top5",
        "is_top10",
        "is_top20",
    ]
    return frame[columns].sort_values(["open_time", "rank"]).reset_index(drop=True)
