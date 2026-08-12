from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.drop_top10_short_filter_research as research
from src.data.downloader import INTERVAL_MS, load_cached_klines
from src.data.quality import check_klines


OUT = ROOT / "outputs" / "drop_top10_short_filter"
DROP_MIN = 0.10
DROP_MAX = 0.15
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def scan_thresholds(group: pd.DataFrame, entry_time: int, entry_price: float) -> dict[str, object]:
    end_time = entry_time + research.LOOKAHEAD_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    result: dict[str, object] = {
        "hit_plus_10_first": False,
        "first_terminal_event": "lookahead_end",
    }
    for threshold in THRESHOLDS:
        result[f"hit_minus_{int(threshold * 100)}_before_plus10"] = False

    for _, bar in path.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        # Conservative for shorts: if +10 and downside threshold print in the
        # same 5m candle, count +10 first.
        if high >= entry_price * 1.10:
            result["hit_plus_10_first"] = True
            result["first_terminal_event"] = "plus10_liquidation"
            return result
        for threshold in THRESHOLDS:
            key = f"hit_minus_{int(threshold * 100)}_before_plus10"
            if not result[key] and low <= entry_price * (1.0 - threshold):
                result[key] = True
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    research.BASE_DROP_MIN = DROP_MIN
    research.BASE_DROP_MAX = DROP_MAX

    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", research.INTERVAL)
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    end_ms = int(klines["open_time"].max())
    start_ms = end_ms - research.LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    klines = klines[(klines["open_time"] >= start_ms) & (klines["open_time"] <= end_ms)].copy()
    klines = klines.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    check_klines(klines, INTERVAL_MS[research.INTERVAL]).to_csv(OUT / "drop10_15_data_quality.csv", index=False)

    print("Building minimal volume features/rankings/signals...", flush=True)
    grouped = klines.groupby("symbol", sort=False)
    klines["quote_volume_1h_sum"] = grouped["quote_volume"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    klines["quote_volume_24h_avg_5m"] = grouped["quote_volume"].transform(lambda s: s.rolling(288, min_periods=20).mean())
    candle_range = (klines["high"] - klines["low"]).replace(0, pd.NA)
    klines["signal_candle_lower_wick_pct"] = (
        (klines[["open", "close"]].min(axis=1) - klines["low"]).clip(lower=0) / candle_range
    )
    klines["volume_1h_vs_24h_avg"] = klines["quote_volume_1h_sum"] / (klines["quote_volume_24h_avg_5m"] * 12)
    rankings = research.build_rankings(klines)
    signals = research.identify_base_signals(rankings)
    kmap = {symbol: group.reset_index(drop=True) for symbol, group in klines.groupby("symbol", sort=False)}

    entries = []
    for _, signal in signals.iterrows():
        entry = research.future_entry(kmap[signal["symbol"]], int(signal["signal_time"]))
        if entry is None:
            continue
        entries.append(
            {
                "signal_id": int(signal["signal_id"]),
                "entry_time": int(entry["open_time"]),
                "entry_time_utc": entry["open_time_utc"],
                "entry_price": float(entry["open"]),
            }
        )
    base = signals.merge(pd.DataFrame(entries), on="signal_id", how="inner")
    combo_rows = []
    for _, signal in base.iterrows():
        group = kmap[signal["symbol"]]
        rows_at_signal = group[group["open_time"] == int(signal["signal_time"])]
        if rows_at_signal.empty:
            continue
        row = rows_at_signal.iloc[0]
        combo_rows.append(
            {
                "signal_id": int(signal["signal_id"]),
                "volume_1h_vs_24h_avg": float(row["volume_1h_vs_24h_avg"]) if pd.notna(row["volume_1h_vs_24h_avg"]) else pd.NA,
                "signal_candle_lower_wick_pct": float(row["signal_candle_lower_wick_pct"]) if pd.notna(row["signal_candle_lower_wick_pct"]) else pd.NA,
            }
        )
    combo_features = pd.DataFrame(combo_rows)
    combo_features.to_csv(OUT / "combo3_drop10_15_features.csv", index=False)
    combo_ids = set(
        combo_features[
            combo_features["volume_1h_vs_24h_avg"].between(1.2, 5)
            & (combo_features["signal_candle_lower_wick_pct"] <= 0.40)
        ]["signal_id"]
    )
    selected = base[base["signal_id"].isin(combo_ids)].copy()

    rows = []
    for _, signal in selected.iterrows():
        labels = scan_thresholds(kmap[signal["symbol"]], int(signal["entry_time"]), float(signal["entry_price"]))
        rows.append({**signal.to_dict(), **labels})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "combo3_drop10_15_threshold_detail.csv", index=False)

    total = len(detail)
    summary_rows = []
    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        key = f"hit_minus_{pct}_before_plus10"
        count = int(detail[key].sum()) if total else 0
        summary_rows.append(
            {
                "trigger_drop_bucket": "10-15%",
                "combo_name": "Combo 3",
                "threshold": f"-{pct}%",
                "hit_count_before_plus10": count,
                "hit_rate_before_plus10": count / total if total else 0.0,
                "signal_count": total,
            }
        )
    plus10_count = int(detail["hit_plus_10_first"].sum()) if total else 0
    summary_rows.append(
        {
            "trigger_drop_bucket": "10-15%",
            "combo_name": "Combo 3",
            "threshold": "first_plus10_liquidation",
            "hit_count_before_plus10": plus10_count,
            "hit_rate_before_plus10": plus10_count / total if total else 0.0,
            "signal_count": total,
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "combo3_drop10_15_threshold_summary.csv", index=False)

    print(f"base_signals_10_15={len(base)} combo3_signals={total}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(OUT / "combo3_drop10_15_threshold_summary.csv", flush=True)


if __name__ == "__main__":
    main()
