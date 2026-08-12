from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import load_cached_klines


OUT = ROOT / "outputs" / "drop_top10_short_filter"
THRESHOLDS = [10, 15, 20, 30, 40, 50, 60, 70]
LOOKAHEAD_MS = 120 * 60 * 60 * 1000


def main() -> None:
    backtest = pd.read_csv(OUT / "backtest_trades.csv")
    events = backtest[
        backtest["rule_name"].eq("base_15_20")
        & backtest["entry_model"].eq("Hold Drop Top10 60m")
        & backtest["exit_rule"].eq("Exit A")
    ].copy()
    events["entry_time"] = pd.to_datetime(events["entry_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))

    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines[klines["symbol"].isin(set(events["symbol"]))].sort_values(["symbol", "open_time"])
    kmap = {symbol: group.reset_index(drop=True) for symbol, group in klines.groupby("symbol", sort=False)}

    detail_rows = []
    for _, event in events.iterrows():
        group = kmap[event["symbol"]]
        entry_time = int(event["entry_time"])
        entry_row = group[group["open_time"].eq(entry_time)]
        if entry_row.empty:
            continue
        entry_price = float(entry_row.iloc[0]["open"])
        path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= entry_time + LOOKAHEAD_MS)]
        hits = {threshold: False for threshold in THRESHOLDS}
        first_liquidation = False

        for _, bar in path.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            # Conservative same-candle rule: adverse +10% wins before downside targets.
            if high >= entry_price * 1.10:
                first_liquidation = not any(hits.values())
                break
            for threshold in THRESHOLDS:
                if not hits[threshold] and low <= entry_price * (1.0 - threshold / 100.0):
                    hits[threshold] = True

        detail_rows.append(
            {
                "signal_id": int(event["signal_id"]),
                "symbol": event["symbol"],
                "entry_time_utc": event["entry_time_utc"],
                "entry_price": entry_price,
                "first_liquidation_up_10": first_liquidation,
                **{f"hit_minus_{threshold}_before_plus10": hits[threshold] for threshold in THRESHOLDS},
            }
        )

    detail = pd.DataFrame(detail_rows)
    total = len(detail)
    summary_rows = []
    for threshold in THRESHOLDS:
        col = f"hit_minus_{threshold}_before_plus10"
        count = int(detail[col].sum()) if total else 0
        summary_rows.append(
            {
                "entry_model": "Hold Drop Top10 60m",
                "threshold": f"-{threshold}%",
                "hit_count_before_plus10": count,
                "hit_rate_before_plus10": count / total if total else 0.0,
                "trade_count": total,
            }
        )
    liquidation_count = int(detail["first_liquidation_up_10"].sum()) if total else 0
    summary_rows.append(
        {
            "entry_model": "Hold Drop Top10 60m",
            "threshold": "first_liquidation_up_10",
            "hit_count_before_plus10": liquidation_count,
            "hit_rate_before_plus10": liquidation_count / total if total else 0.0,
            "trade_count": total,
        }
    )

    summary = pd.DataFrame(summary_rows)
    detail.to_csv(OUT / "hold60_threshold_detail.csv", index=False)
    summary.to_csv(OUT / "hold60_threshold_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(OUT / "hold60_threshold_summary.csv")


if __name__ == "__main__":
    main()
