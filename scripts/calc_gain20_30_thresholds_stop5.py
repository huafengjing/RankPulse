from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
FEATURES = Path("outputs/robust_filter/signal_features.csv")
OUT = Path("outputs/gain20_30_stop5_thresholds")

STOP_PCT = -0.05
LOOKAHEAD_HOURS = 240
THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00, 5.00, 10.00]


def load_klines(symbols: set[str]) -> dict[str, pd.DataFrame]:
    frames = []
    for path in DATA_DIR.glob("*/*.parquet"):
        frame = pd.read_parquet(path, columns=["symbol", "open_time", "open", "high", "low", "close"])
        if frame.empty:
            continue
        frame = frame[frame["symbol"].isin(symbols)]
        if frame.empty:
            continue
        frame["_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("_mtime").drop_duplicates(["symbol", "open_time"], keep="last")
    df = df.drop(columns="_mtime").sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol", sort=False)}


def scan_signal(group: pd.DataFrame, signal_time: int) -> dict:
    idx = group.index[group["open_time"] > signal_time]
    if len(idx) == 0:
        return {"evaluated": False}
    entry_idx = int(idx[0])
    entry = group.loc[entry_idx]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = entry_time + LOOKAHEAD_HOURS * 60 * 60 * 1000
    stop_price = entry_price * (1 + STOP_PCT)
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    result = {
        "evaluated": True,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "hit_stop_first": False,
        "max_forward_return": float(path["high"].max() / entry_price - 1) if len(path) else 0.0,
        "min_forward_return": float(path["low"].min() / entry_price - 1) if len(path) else 0.0,
    }
    for threshold in THRESHOLDS:
        result[f"hit_plus_{int(threshold * 100)}_before_stop"] = False

    for _, bar in path.iterrows():
        # Conservative same-candle rule: stop is assumed to trigger before upside target.
        if float(bar["low"]) <= stop_price:
            result["hit_stop_first"] = True
            return result
        high = float(bar["high"])
        for threshold in THRESHOLDS:
            key = f"hit_plus_{int(threshold * 100)}_before_stop"
            if not result[key] and high >= entry_price * (1 + threshold):
                result[key] = True
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES)
    features["signal_time"] = pd.to_datetime(features["signal_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    selected = features[
        (features["rolling_24h_gain_at_signal"] >= 0.20)
        & (features["rolling_24h_gain_at_signal"] < 0.30)
    ].copy()
    klines = load_klines(set(selected["symbol"]))
    rows = []
    for _, sig in selected.sort_values("signal_time_utc").iterrows():
        group = klines.get(sig["symbol"])
        if group is None:
            continue
        scanned = scan_signal(group, int(sig["signal_time"]))
        if not scanned["evaluated"]:
            continue
        row = {
            "signal_id": int(sig["signal_id"]),
            "symbol": sig["symbol"],
            "signal_time_utc": sig["signal_time_utc"],
            "rolling_24h_gain_at_signal": sig["rolling_24h_gain_at_signal"],
            **scanned,
        }
        rows.append(row)
    detail = pd.DataFrame(rows)
    summary_rows = []
    total = len(detail)
    for threshold in THRESHOLDS:
        key = f"hit_plus_{int(threshold * 100)}_before_stop"
        count = int(detail[key].sum())
        summary_rows.append(
            {
                "threshold": f"+{int(threshold * 100)}%",
                "hit_count": count,
                "hit_rate": count / total if total else 0.0,
            }
        )
    stop_count = int(detail["hit_stop_first"].sum())
    summary_rows.append({"threshold": "先到-5%止损", "hit_count": stop_count, "hit_rate": stop_count / total if total else 0.0})
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(OUT / "signal_threshold_detail.csv", index=False)
    summary.to_csv(OUT / "threshold_summary.csv", index=False)
    print(f"selected={len(selected)} evaluated={total}")
    print(summary.to_string(index=False))
    print(f"Wrote {OUT / 'threshold_summary.csv'}")
    print(f"Wrote {OUT / 'signal_threshold_detail.csv'}")


if __name__ == "__main__":
    main()
