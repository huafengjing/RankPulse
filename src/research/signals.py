from __future__ import annotations

import pandas as pd


def identify_first_top_signals(
    rankings: pd.DataFrame,
    top_n: int = 10,
    cooldown_days: int = 3,
    observation_hours: int = 72,
) -> pd.DataFrame:
    cooldown_ms = cooldown_days * 24 * 60 * 60 * 1000
    observation_ms = observation_hours * 60 * 60 * 1000
    rows = []
    signal_id = 1
    top_col = f"is_top{top_n}"
    if top_col not in rankings.columns:
        rankings = rankings.copy()
        rankings[top_col] = rankings["rank"] <= top_n

    for symbol, group in rankings.sort_values("open_time").groupby("symbol"):
        top_hits = group[group[top_col]].copy()
        previous_time: int | None = None
        for _, row in top_hits.iterrows():
            now = int(row["open_time"])
            is_first = previous_time is None or now - previous_time > cooldown_ms
            if not is_first:
                previous_time = now
                continue
            future = group[(group["open_time"] > now) & (group["open_time"] <= now + observation_ms)]
            top5_later = future[future["is_top5"]]
            entered_top5_later = not top5_later.empty
            time_to_top5 = None
            if entered_top5_later:
                time_to_top5 = int((int(top5_later.iloc[0]["open_time"]) - now) / 60_000)
            rows.append(
                {
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "signal_time": now,
                    "signal_time_utc": row["open_time_utc"],
                    "rank": int(row["rank"]),
                    "rolling_24h_change_pct": float(row["rolling_24h_change_pct"]),
                    "close": float(row["close"]),
                    "quote_volume": float(row.get("quote_volume", 0)),
                    "is_first_top10": top_n == 10,
                    "previous_top10_time_utc": pd.NaT if previous_time is None else pd.to_datetime(previous_time, unit="ms", utc=True),
                    "cooldown_days": cooldown_days,
                    "entered_top5_later": entered_top5_later,
                    "time_to_top5_minutes": time_to_top5,
                    "eligible_for_entry": True,
                    "filtered_reason": "",
                }
            )
            signal_id += 1
            previous_time = now
    return pd.DataFrame(rows)
