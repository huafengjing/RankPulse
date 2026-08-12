from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (
    CACHE_DIR,
    DAY_MS,
    EARLY_REASON,
    OUT,
    add_entry_factors,
    load_kline_map,
    max_drawdown,
    profit_factor,
    simulate_main_trades_with_position_limit,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt


PREFIX = "current_main_strategy_old_half"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def apply_current_main_filters(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].lt(0.80)
        & signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)

    in_2040 = signals["gain_24h"].ge(0.20) & signals["gain_24h"].lt(0.40)
    pass_2040 = signals["volume_24h_ratio_7d"].ge(1.5) & signals["volume_24h_ratio_7d"].lt(5.0)

    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )

    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)

    return signals[
        ((~in_2040) | pass_2040)
        & ((~in_4060) | pass_4060)
        & (~in_6080)
    ].sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "early_12h": int(completed["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in completed else 0,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "gross_profit_u": round(float(pnl[pnl > 0].sum()), 2) if len(pnl) else 0.0,
        "gross_loss_u": round(float(pnl[pnl < 0].sum()), 2) if len(pnl) else 0.0,
        "net_pnl_u": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "pf": round(float(profit_factor(pnl)), 2),
        "win_rate_pct": round(float(len(wins) / len(completed) * 100), 2) if len(completed) else np.nan,
        "avg_return_pct": round(float(ret.mean()), 2) if len(ret) else np.nan,
        "median_return_pct": round(float(ret.median()), 2) if len(ret) else np.nan,
        "max_drawdown_u": round(max_drawdown(pnl), 2),
        "best_trade_u": round(float(pnl.max()), 2) if len(pnl) else np.nan,
        "worst_trade_u": round(float(pnl.min()), 2) if len(pnl) else np.nan,
        "drop_top1_u": round(float(pnl.sum() - pnl.nlargest(1).sum()), 2) if len(pnl) >= 1 else np.nan,
        "drop_top3_u": round(float(pnl.sum() - pnl.nlargest(3).sum()), 2) if len(pnl) >= 3 else np.nan,
        "drop_top5_u": round(float(pnl.sum() - pnl.nlargest(5).sum()), 2) if len(pnl) >= 5 else np.nan,
    }


def add_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gain_24h_bucket"] = pd.cut(
        frame["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%"],
        right=False,
    )
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = int(latest_signal_end_dt().timestamp() * 1000)
    old_start = end - 365 * DAY_MS
    old_end = end - 180 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), old_start - 7 * DAY_MS, old_end + 7 * DAY_MS)
    signals = apply_current_main_filters(generate_signals(old_start, old_end, kline_map), kline_map)
    trades = add_bucket(simulate_main_trades_with_position_limit(signals, kline_map))

    summary = pd.DataFrame([summarize(trades)])
    monthly_rows = []
    for month, group in trades.groupby("month", sort=True):
        monthly_rows.append({"month": month, **summarize(group)})
    monthly = pd.DataFrame(monthly_rows)

    bucket_rows = []
    for bucket, group in trades[trades["status"].eq("completed")].groupby("gain_24h_bucket", observed=False):
        bucket_rows.append({"bucket": str(bucket), **summarize(group)})
    buckets = pd.DataFrame(bucket_rows)

    trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8")
    buckets.to_csv(OUT / f"{PREFIX}_bucket_stats.csv", index=False, encoding="utf-8")

    print("========== Old Half Current Main Strategy ==========")
    print(summary.to_string(index=False))
    print()
    print("========== Monthly ==========")
    print(monthly.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
