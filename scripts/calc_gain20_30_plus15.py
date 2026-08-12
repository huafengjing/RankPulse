from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
FEATURES = Path("outputs/robust_filter/signal_features.csv")


def load_klines(symbols: set[str]) -> dict[str, pd.DataFrame]:
    frames = []
    for path in DATA_DIR.glob("*/*.parquet"):
        frame = pd.read_parquet(path, columns=["symbol", "open_time", "open", "high", "low", "close"])
        if frame.empty:
            continue
        frame = frame[frame["symbol"].isin(symbols)]
        if not frame.empty:
            frame["_mtime"] = path.stat().st_mtime
            frames.append(frame)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("_mtime").drop_duplicates(["symbol", "open_time"], keep="last")
    df = df.drop(columns="_mtime").sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol", sort=False)}


def hit_before_minus10(group: pd.DataFrame, signal_time: int, threshold: float) -> bool:
    idx = group.index[group["open_time"] > signal_time]
    if len(idx) == 0:
        return False
    entry_idx = int(idx[0])
    entry = group.loc[entry_idx]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = entry_time + 240 * 60 * 60 * 1000
    target = entry_price * (1 + threshold)
    stop = entry_price * 0.90
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    for _, bar in path.iterrows():
        if float(bar["low"]) <= stop:
            return False
        if float(bar["high"]) >= target:
            return True
    return False


def main() -> None:
    features = pd.read_csv(FEATURES)
    features["signal_time"] = pd.to_datetime(features["signal_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    selected = features[
        (features["rolling_24h_gain_at_signal"] >= 0.20)
        & (features["rolling_24h_gain_at_signal"] < 0.30)
    ].copy()
    klines = load_klines(set(selected["symbol"]))
    hits = 0
    evaluated = 0
    for _, sig in selected.iterrows():
        group = klines.get(sig["symbol"])
        if group is None:
            continue
        evaluated += 1
        hits += int(hit_before_minus10(group, int(sig["signal_time"]), 0.15))
    print(f"selected={len(selected)} evaluated={evaluated} plus15_hits={hits} plus15_rate={hits / evaluated:.6f}")


if __name__ == "__main__":
    main()
