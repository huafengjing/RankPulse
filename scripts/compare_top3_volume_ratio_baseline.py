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
    FOUR_HOUR_MS,
    OUT,
    aggregate_4h,
    bucket_ma_structure,
    bucket_volume,
    load_kline_map,
    max_drawdown,
    mfe_mae,
    profit_factor,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals, latest_signal_end_dt
from scripts.test_main_strategy_hold_days_leverage import (
    cached_symbols,
    simulate_with_position_limit,
)


PREFIX = "top3_volume_ratio_old_vs_excl_recent24h"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}
LIQ_THRESHOLDS = {2: -50.0, 3: -33.0, 5: -20.0}


def add_entry_factors_with_volume_mode(
    signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    volume_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    h4_cache: dict[str, pd.DataFrame] = {}
    for _, signal in signals.iterrows():
        symbol = str(signal["symbol"])
        entry_time = int(signal["signal_time"])
        h1 = kline_map.get(symbol, pd.DataFrame())
        if symbol not in h4_cache:
            h4_cache[symbol] = aggregate_4h(h1)
        h4 = h4_cache[symbol]
        last_4h_open = entry_time - FOUR_HOUR_MS
        factor = {
            "ma_structure_4h": "missing",
            "distance_to_4h_ma7_pct": np.nan,
            "volume_24h_ratio_7d": np.nan,
            "volume_24h_ratio_7d_mode": volume_mode,
        }
        if not h4.empty:
            up_to = h4[h4["open_time"] <= last_4h_open].copy()
            if len(up_to) >= 21:
                close = float(up_to.iloc[-1]["close"])
                ma7 = float(up_to.tail(7)["close"].mean())
                ma21 = float(up_to.tail(21)["close"].mean())
                factor["ma_structure_4h"] = bucket_ma_structure(close, ma7, ma21)
                factor["distance_to_4h_ma7_pct"] = (close / ma7 - 1.0) * 100 if ma7 > 0 else np.nan

            recent_6 = up_to.tail(6)
            if volume_mode == "including_recent24h":
                baseline_42 = up_to.tail(42)
            elif volume_mode == "excluding_recent24h":
                baseline_42 = up_to.iloc[-48:-6] if len(up_to) >= 48 else pd.DataFrame()
            else:
                raise ValueError(f"unknown volume_mode={volume_mode}")

            if len(recent_6) == 6 and len(baseline_42) == 42:
                volume_24h = float(recent_6["volume"].sum())
                avg_daily_volume_7d = float(baseline_42["volume"].sum()) / 7.0
                factor["volume_24h_ratio_7d"] = volume_24h / avg_daily_volume_7d if avg_daily_volume_7d > 0 else np.nan

        factor["volume_24h_ratio_7d_bucket"] = bucket_volume(factor["volume_24h_ratio_7d"])
        rows.append(factor)
    return pd.concat([signals.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_gain_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gain_bucket"] = pd.cut(
        out["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%", ">=80%"],
        right=False,
    )
    return out


def apply_main_filters(
    all_signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    volume_mode: str,
) -> pd.DataFrame:
    signals = all_signals[
        all_signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & all_signals["rank"].isin([2, 3])
        & all_signals["gain_24h"].lt(0.80)
        & all_signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors_with_volume_mode(signals, kline_map, volume_mode)

    in_2040 = signals["gain_24h"].ge(0.20) & signals["gain_24h"].lt(0.40)
    pass_2040 = signals["volume_24h_ratio_7d"].ge(1.5) & signals["volume_24h_ratio_7d"].lt(5.0)

    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )

    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)

    return add_gain_bucket(
        signals[
            ((~in_2040) | pass_2040)
            & ((~in_4060) | pass_4060)
            & (~in_6080)
        ]
        .sort_values(["signal_time", "rank", "symbol"])
        .reset_index(drop=True)
    )


def leverage_for_row(row: pd.Series) -> int:
    gain = float(row["gain_24h"])
    rank = int(row["rank"])
    if 0.10 <= gain < 0.20:
        return 3
    if 0.20 <= gain < 0.40:
        return 3 if rank == 2 else 5
    if 0.40 <= gain < 0.60:
        return 2
    raise ValueError(f"unexpected traded row: gain={gain}, rank={rank}, symbol={row.get('symbol')}")


def leveraged_pnl(row: pd.Series, leverage: int) -> tuple[float, float, bool]:
    mae = float(row["mae_pct"])
    if mae <= LIQ_THRESHOLDS[leverage]:
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def summarize(group: pd.DataFrame, pnl_col: str, ret_col: str, liq_col: str | None = None) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed[pnl_col].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed[ret_col].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "early_12h": int(completed["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in completed else 0,
        "liquidations": int(completed[liq_col].sum()) if liq_col else 0,
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


def run_variant(volume_mode: str, all_signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = apply_main_filters(all_signals, kline_map, volume_mode)
    trades = simulate_with_position_limit(signals, kline_map, 6)
    trades = add_gain_bucket(trades)
    trades["volume_mode"] = volume_mode
    completed_mask = trades["status"].eq("completed")
    trades["leverage"] = np.nan
    trades.loc[completed_mask, "leverage"] = trades[completed_mask].apply(leverage_for_row, axis=1)
    values = trades[completed_mask].apply(lambda row: leveraged_pnl(row, int(row["leverage"])), axis=1)
    trades["leveraged_pnl_u"] = np.nan
    trades["leveraged_return_pct"] = np.nan
    trades["liquidated"] = False
    trades.loc[completed_mask, "leveraged_pnl_u"] = [v[0] for v in values]
    trades.loc[completed_mask, "leveraged_return_pct"] = [v[1] for v in values]
    trades.loc[completed_mask, "liquidated"] = [v[2] for v in values]
    return signals, trades


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = int(latest_signal_end_dt().timestamp() * 1000)
    start = end - 180 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start - 10 * DAY_MS, end + 8 * DAY_MS)
    all_signals = generate_signals(start, end, kline_map)

    all_signal_frames = []
    all_trade_frames = []
    for mode in ["including_recent24h", "excluding_recent24h"]:
        signals, trades = run_variant(mode, all_signals, kline_map)
        all_signal_frames.append(signals.assign(volume_mode=mode))
        all_trade_frames.append(trades)

    signals_df = pd.concat(all_signal_frames, ignore_index=True)
    trades_df = pd.concat(all_trade_frames, ignore_index=True)

    summary_rows = []
    for mode, group in trades_df.groupby("volume_mode", sort=True):
        summary_rows.append({"volume_mode": mode, **summarize(group, "leveraged_pnl_u", "leveraged_return_pct", "liquidated")})
    summary = pd.DataFrame(summary_rows)

    monthly_rows = []
    for (mode, month), group in trades_df.groupby(["volume_mode", "month"], sort=True):
        monthly_rows.append({"volume_mode": mode, "month": month, **summarize(group, "leveraged_pnl_u", "leveraged_return_pct", "liquidated")})
    monthly = pd.DataFrame(monthly_rows)

    bucket_rows = []
    for (mode, bucket), group in trades_df.groupby(["volume_mode", "gain_bucket"], sort=True, observed=True):
        bucket_rows.append({"volume_mode": mode, "gain_bucket": bucket, **summarize(group, "leveraged_pnl_u", "leveraged_return_pct", "liquidated")})
    bucket = pd.DataFrame(bucket_rows)

    rank_rows = []
    for (mode, rank), group in trades_df.groupby(["volume_mode", "rank"], sort=True):
        rank_rows.append({"volume_mode": mode, "rank": int(rank), **summarize(group, "leveraged_pnl_u", "leveraged_return_pct", "liquidated")})
    rank = pd.DataFrame(rank_rows)

    # Compare final completed trade sets by signal identity.
    completed = trades_df[trades_df["status"].eq("completed")].copy()
    completed["trade_key"] = completed["symbol"].astype(str) + "|" + completed["entry_time_ms"].astype("int64").astype(str) + "|" + completed["rank"].astype("int64").astype(str)
    old_keys = set(completed[completed["volume_mode"].eq("including_recent24h")]["trade_key"])
    new_keys = set(completed[completed["volume_mode"].eq("excluding_recent24h")]["trade_key"])
    change_rows = [
        {"change_type": "old_only_completed_trades", "count": len(old_keys - new_keys)},
        {"change_type": "new_only_completed_trades", "count": len(new_keys - old_keys)},
        {"change_type": "both_completed_trades", "count": len(old_keys & new_keys)},
    ]
    changed = pd.DataFrame(change_rows)

    signals_df.to_csv(OUT / f"{PREFIX}_signals.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(OUT / f"{PREFIX}_gain_bucket.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / f"{PREFIX}_rank.csv", index=False, encoding="utf-8-sig")
    changed.to_csv(OUT / f"{PREFIX}_trade_set_change.csv", index=False, encoding="utf-8-sig")

    print("========== Volume Ratio Baseline Comparison ==========")
    print(summary.to_string(index=False))
    print()
    print("========== Monthly ==========")
    print(monthly[["volume_mode", "month", "trades", "liquidations", "net_pnl_u", "pf", "win_rate_pct", "median_return_pct", "max_drawdown_u"]].to_string(index=False))
    print()
    print("========== Gain Bucket ==========")
    print(bucket[["volume_mode", "gain_bucket", "trades", "liquidations", "net_pnl_u", "pf", "win_rate_pct", "median_return_pct", "max_drawdown_u"]].to_string(index=False))
    print()
    print("========== Rank ==========")
    print(rank[["volume_mode", "rank", "trades", "liquidations", "net_pnl_u", "pf", "win_rate_pct", "median_return_pct", "max_drawdown_u"]].to_string(index=False))
    print()
    print("========== Trade Set Change ==========")
    print(changed.to_string(index=False))
    print()
    print(f"Wrote output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
