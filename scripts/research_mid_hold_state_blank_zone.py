from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, HOUR_MS, OUT, load_kline_map, ms_to_bj_string


STRATEGY = "rank1_mixed_3x5d_v23_5x2d_v56_4060v23"
SRC = OUT / "rank1_portfolio_variant_comparison" / "trade_details_all_variants.csv"
OUT_DIR = OUT / "mid_hold_state_blank_zone_research"


def price_at_or_before(frame: pd.DataFrame, target_ms: int) -> tuple[int | None, float]:
    if frame.empty:
        return None, np.nan
    exact = frame[frame["open_time"] == target_ms]
    if not exact.empty:
        row = exact.iloc[-1]
        return int(row["open_time"]), float(row["open"])
    prior = frame[frame["open_time"] < target_ms]
    if prior.empty:
        return None, np.nan
    row = prior.iloc[-1]
    return int(row["open_time"]), float(row["close"])


def path_metrics(frame: pd.DataFrame, entry_price: float, start_ms: int, end_ms: int) -> dict[str, float]:
    path = frame[(frame["open_time"] >= start_ms) & (frame["open_time"] < end_ms)].sort_values("open_time")
    if path.empty:
        return {"mfe": np.nan, "mae": np.nan, "close_return": np.nan}
    return {
        "mfe": float(path["high"].max() / entry_price - 1.0),
        "mae": float(path["low"].min() / entry_price - 1.0),
        "close_return": float(path.iloc[-1]["close"] / entry_price - 1.0),
    }


def load_current_trades() -> pd.DataFrame:
    source = pd.read_csv(SRC, encoding="utf-8-sig")
    trades = source[
        source["strategy"].eq(STRATEGY)
        & source["status"].isin(["completed", "open_mark_to_market"])
        & source["pnl_u"].notna()
        & source["entry_price"].notna()
    ].copy()
    trades["entry_time_ms"] = pd.to_numeric(trades["entry_time_ms"], errors="coerce").astype("int64")
    trades["exit_time_ms"] = pd.to_numeric(trades["exit_time_ms"], errors="coerce").astype("int64")
    trades["final_outcome"] = np.select(
        [
            trades["liquidated"].fillna(False).astype(bool),
            pd.to_numeric(trades["pnl_u"], errors="coerce") > 0,
        ],
        ["liquidation", "winner"],
        default="loser_non_liq",
    )
    return trades.sort_values(["entry_time_ms", "rank", "symbol"]).reset_index(drop=True)


def enrich_mid_hold_states(trades: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checkpoints = {"24h": 24 * HOUR_MS, "48h": 48 * HOUR_MS, "72h": 72 * HOUR_MS, "96h": 96 * HOUR_MS}
    for row in trades.itertuples(index=False):
        symbol = str(row.symbol)
        frame = kline_map.get(symbol, pd.DataFrame())
        entry_time = int(row.entry_time_ms)
        entry_price = float(row.entry_price)
        out = row._asdict()

        for name, offset in checkpoints.items():
            target = entry_time + offset
            price_time, price = price_at_or_before(frame, target)
            out[f"price_time_{name}_ms"] = price_time
            out[f"price_time_{name}_bj"] = ms_to_bj_string(price_time) if price_time is not None else ""
            out[f"return_{name}"] = price / entry_price - 1.0 if np.isfinite(price) else np.nan
            out[f"levered_return_{name}"] = out[f"return_{name}"] * float(row.leverage) if np.isfinite(price) else np.nan
            out[f"survived_to_{name}"] = bool(int(row.exit_time_ms) >= target)

        windows = {
            "24_48h": (entry_time + 24 * HOUR_MS, entry_time + 48 * HOUR_MS),
            "2d_4d": (entry_time + 48 * HOUR_MS, entry_time + 96 * HOUR_MS),
            "24h_4d": (entry_time + 24 * HOUR_MS, entry_time + 96 * HOUR_MS),
        }
        for name, (start, end) in windows.items():
            metrics = path_metrics(frame, entry_price, start, end)
            out[f"mfe_{name}"] = metrics["mfe"]
            out[f"mae_{name}"] = metrics["mae"]
            out[f"close_return_{name}"] = metrics["close_return"]

        out["delta_24_48h"] = out["return_48h"] - out["return_24h"]
        out["delta_2d_4d"] = out["return_96h"] - out["return_48h"]
        rows.append(out)
    return pd.DataFrame(rows)


def pf(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def stats(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["pnl_u"], errors="coerce")
    liq = frame["liquidated"].fillna(False).astype(bool)
    return {
        "bucket": label,
        "trades": int(len(frame)),
        "net_pnl_u": float(pnl.sum()) if len(frame) else 0.0,
        "pf": pf(pnl.dropna()) if len(frame) else 0.0,
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(frame) else np.nan,
        "median_pnl_u": float(pnl.median()) if len(frame) else np.nan,
        "liq_count": int(liq.sum()) if len(frame) else 0,
        "liq_rate_pct": float(liq.mean() * 100) if len(frame) else np.nan,
        "median_return_24h_pct": float(frame["return_24h"].median() * 100) if len(frame) else np.nan,
        "median_return_48h_pct": float(frame["return_48h"].median() * 100) if len(frame) else np.nan,
        "median_return_72h_pct": float(frame["return_72h"].median() * 100) if len(frame) else np.nan,
        "median_return_96h_pct": float(frame["return_96h"].median() * 100) if len(frame) else np.nan,
        "median_mfe_24_48h_pct": float(frame["mfe_24_48h"].median() * 100) if len(frame) else np.nan,
        "median_mae_24_48h_pct": float(frame["mae_24_48h"].median() * 100) if len(frame) else np.nan,
        "median_mfe_2d_4d_pct": float(frame["mfe_2d_4d"].median() * 100) if len(frame) else np.nan,
        "median_mae_2d_4d_pct": float(frame["mae_2d_4d"].median() * 100) if len(frame) else np.nan,
    }


def bucket_series(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True).astype("string").fillna("missing")


def grouped_stats(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        label = key if isinstance(key, tuple) else (key,)
        rows.append({col: value for col, value in zip(group_cols, label)} | stats(group, str(label)))
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_current_trades()
    symbols = sorted(trades["symbol"].astype(str).unique())
    start = int(trades["entry_time_ms"].min()) - DAY_MS
    end = int(trades["exit_time_ms"].max()) + DAY_MS
    kline_map = load_kline_map(symbols, start, end)
    enriched = enrich_mid_hold_states(trades, kline_map)

    # Only positions that were still alive at the checkpoint are valid for that blank-zone state.
    alive_48 = enriched[enriched["survived_to_48h"]].copy()
    alive_96 = enriched[enriched["survived_to_96h"]].copy()

    lever_bins = [-np.inf, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, np.inf]
    lever_labels = ["<-50%", "-50~-25%", "-25~-10%", "-10~0%", "0~10%", "10~25%", "25~50%", ">=50%"]
    alive_48["levered_48h_bucket"] = bucket_series(alive_48["levered_return_48h"], lever_bins, lever_labels)
    alive_96["levered_4d_bucket"] = bucket_series(alive_96["levered_return_96h"], lever_bins, lever_labels)

    near_4d = alive_96[alive_96["levered_return_96h"].between(-0.15, 0.15, inclusive="both")].copy()
    near_48 = alive_48[alive_48["levered_return_48h"].between(-0.15, 0.15, inclusive="both")].copy()

    enriched.to_csv(OUT_DIR / "trade_mid_hold_states.csv", index=False, encoding="utf-8-sig")
    grouped_stats(alive_48, ["final_outcome"]).to_csv(OUT_DIR / "outcome_summary_alive_48h.csv", index=False, encoding="utf-8-sig")
    grouped_stats(alive_96, ["final_outcome"]).to_csv(OUT_DIR / "outcome_summary_alive_4d.csv", index=False, encoding="utf-8-sig")
    grouped_stats(alive_48, ["levered_48h_bucket"]).to_csv(OUT_DIR / "bucket_48h_levered_return.csv", index=False, encoding="utf-8-sig")
    grouped_stats(alive_96, ["levered_4d_bucket"]).to_csv(OUT_DIR / "bucket_4d_levered_return.csv", index=False, encoding="utf-8-sig")
    grouped_stats(alive_96, ["rank", "levered_4d_bucket"]).to_csv(OUT_DIR / "rank_bucket_4d_levered_return.csv", index=False, encoding="utf-8-sig")

    near_rows = [
        stats(near_48, "48H levered return -15%~+15%"),
        stats(near_4d, "4D levered return -15%~+15%"),
    ]
    pd.DataFrame(near_rows).to_csv(OUT_DIR / "near_cost_summary.csv", index=False, encoding="utf-8-sig")

    print("output", OUT_DIR)
    print("\nAlive >=48H by final outcome")
    print(grouped_stats(alive_48, ["final_outcome"]).round(2).to_string(index=False))
    print("\nAlive >=4D by final outcome")
    print(grouped_stats(alive_96, ["final_outcome"]).round(2).to_string(index=False))
    print("\n48H levered-return buckets")
    print(grouped_stats(alive_48, ["levered_48h_bucket"]).round(2).to_string(index=False))
    print("\n4D levered-return buckets")
    print(grouped_stats(alive_96, ["levered_4d_bucket"]).round(2).to_string(index=False))
    print("\nNear-cost summary")
    print(pd.DataFrame(near_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
