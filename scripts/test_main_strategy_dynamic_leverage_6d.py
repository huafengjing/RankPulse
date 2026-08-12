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
    FEE_RATE,
    OUT,
    load_kline_map,
    max_drawdown,
    profit_factor,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals, latest_signal_end_dt
from scripts.test_main_strategy_hold_days_leverage import (
    apply_current_main_filters,
    simulate_with_position_limit,
)


PREFIX = "current_main_strategy_6d_dynamic_leverage"
LIQ_THRESHOLD = {2: -50.0, 3: -33.0, 5: -20.0}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def leverage_for_optimized(row: pd.Series) -> int:
    gain = float(row["gain_24h"])
    volume = float(row["volume_24h_ratio_7d"])
    rank = int(row["rank"])
    if 0.20 <= gain < 0.40 and 1.5 <= volume < 5:
        return 5
    if 0.40 <= gain < 0.60 and rank == 2 and 3 <= volume < 6:
        return 2
    return 3


def leveraged_pnl(row: pd.Series, leverage: int) -> tuple[float, float, bool]:
    mae = float(row["mae_pct"])
    if leverage > 1 and mae <= LIQ_THRESHOLD[leverage]:
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def apply_leverage(trades: pd.DataFrame, version: str) -> pd.DataFrame:
    out = trades[trades["status"].eq("completed")].copy()
    if version == "default_3x":
        out["leverage"] = 3
    elif version == "optimized_bucket_leverage":
        out["leverage"] = out.apply(leverage_for_optimized, axis=1)
    else:
        raise ValueError(version)
    values = out.apply(lambda row: leveraged_pnl(row, int(row["leverage"])), axis=1)
    out["version"] = version
    out["leveraged_pnl_u"] = [value[0] for value in values]
    out["leveraged_return_pct"] = [value[1] for value in values]
    out["liquidated"] = [value[2] for value in values]
    return out


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group.sort_values("entry_time_ms").copy()
    pnl = completed["leveraged_pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed["leveraged_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "trades": int(len(completed)),
        "liquidations": int(completed["liquidated"].sum()) if "liquidated" in completed else 0,
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
    out = frame.copy()
    out["gain_bucket"] = pd.cut(
        out["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%"],
        right=False,
    )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = int(latest_signal_end_dt().timestamp() * 1000)
    start = end - 180 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start - 7 * DAY_MS, end + 8 * DAY_MS)
    signals = apply_current_main_filters(generate_signals(start, end, kline_map), kline_map)
    base_trades = add_bucket(simulate_with_position_limit(signals, kline_map, 6))

    default = apply_leverage(base_trades, "default_3x")
    optimized = apply_leverage(base_trades, "optimized_bucket_leverage")
    all_rows = pd.concat([default, optimized], ignore_index=True)

    summary = pd.DataFrame([{"version": version, **summarize(group)} for version, group in all_rows.groupby("version", sort=True)])

    monthly_rows = []
    for (version, month), group in all_rows.groupby(["version", "month"], sort=True):
        monthly_rows.append({"version": version, "month": month, **summarize(group)})
    monthly = pd.DataFrame(monthly_rows)

    bucket_rows = []
    for (version, bucket), group in all_rows.groupby(["version", "gain_bucket"], observed=False, sort=True):
        bucket_rows.append({"version": version, "bucket": str(bucket), **summarize(group)})
    bucket = pd.DataFrame(bucket_rows)

    lev_dist = (
        optimized.groupby(["gain_bucket", "rank", "leverage"], observed=False)
        .size()
        .reset_index(name="trades")
    )

    base_trades.to_csv(OUT / f"{PREFIX}_base_trades.csv", index=False, encoding="utf-8-sig")
    all_rows.to_csv(OUT / f"{PREFIX}_leveraged_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8")
    bucket.to_csv(OUT / f"{PREFIX}_bucket_stats.csv", index=False, encoding="utf-8")
    lev_dist.to_csv(OUT / f"{PREFIX}_optimized_leverage_distribution.csv", index=False, encoding="utf-8")

    print("========== Dynamic Leverage 6D Compare ==========")
    print(summary.to_string(index=False))
    print()
    print("========== Monthly ==========")
    print(monthly.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
