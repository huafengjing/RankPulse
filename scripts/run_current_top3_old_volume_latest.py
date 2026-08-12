from __future__ import annotations

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
    ms_to_bj_string,
    profit_factor,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals, latest_signal_end_dt, ms_to_utc
from scripts.test_main_strategy_hold_days_leverage import apply_current_main_filters, simulate_with_position_limit


PREFIX = "current_top3_old_volume_latest"
LIQ_THRESHOLD = {2: -50.0, 3: -33.0, 5: -20.0}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def add_gain_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gain_bucket"] = pd.cut(
        out["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%", ">=80%"],
        right=False,
    )
    return out


def leverage_for_current_main(row: pd.Series) -> int:
    gain = float(row["gain_24h"])
    rank = int(row["rank"])
    if 0.10 <= gain < 0.20:
        return 3
    if 0.20 <= gain < 0.40:
        return 3 if rank == 2 else 5
    if 0.40 <= gain < 0.60:
        return 2
    raise ValueError(f"Unexpected traded row: gain={gain}, rank={rank}, symbol={row.get('symbol')}")


def leveraged_pnl(row: pd.Series, leverage: int) -> tuple[float, float, bool]:
    mae = float(row["mae_pct"])
    if leverage > 1 and mae <= LIQ_THRESHOLD[leverage]:
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["leveraged_pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed["leveraged_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "early_12h": int(completed["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in completed else 0,
        "liquidations": int(completed["liquidated"].sum()) if "liquidated" in completed else 0,
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signal_end_dt = latest_signal_end_dt()
    signal_end = int(signal_end_dt.timestamp() * 1000)
    signal_start = signal_end - 180 * DAY_MS
    symbols = cached_symbols()
    kline_map = load_kline_map(symbols, signal_start - 7 * DAY_MS, signal_end + 8 * DAY_MS)

    signals = apply_current_main_filters(generate_signals(signal_start, signal_end, kline_map), kline_map)
    signals = add_gain_bucket(signals)
    trades = simulate_with_position_limit(signals, kline_map, 6)
    trades = add_gain_bucket(trades)

    completed_mask = trades["status"].eq("completed")
    trades["leverage"] = np.nan
    trades.loc[completed_mask, "leverage"] = trades[completed_mask].apply(leverage_for_current_main, axis=1)
    values = trades[completed_mask].apply(lambda row: leveraged_pnl(row, int(row["leverage"])), axis=1)
    trades["leveraged_pnl_u"] = np.nan
    trades["leveraged_return_pct"] = np.nan
    trades["liquidated"] = False
    trades.loc[completed_mask, "leveraged_pnl_u"] = [value[0] for value in values]
    trades.loc[completed_mask, "leveraged_return_pct"] = [value[1] for value in values]
    trades.loc[completed_mask, "liquidated"] = [value[2] for value in values]
    trades["entry_date_bj"] = pd.to_datetime(trades["entry_time_bj"]).dt.strftime("%Y-%m-%d")
    trades["exit_date_bj"] = pd.to_datetime(trades["exit_time_bj"], errors="coerce").dt.strftime("%Y-%m-%d")

    summary = pd.DataFrame([{"version": "current_top3_old_volume_tuned_leverage", **summarize(trades)}])

    monthly_rows = []
    for month, group in trades.groupby("month", sort=True):
        monthly_rows.append({"month": month, **summarize(group)})
    monthly = pd.DataFrame(monthly_rows)

    bucket_rows = []
    for bucket, group in trades.groupby("gain_bucket", sort=True, observed=True):
        bucket_rows.append({"gain_bucket": bucket, **summarize(group)})
    bucket = pd.DataFrame(bucket_rows)

    daily_rows = []
    for date, group in trades.groupby("entry_date_bj", sort=True):
        daily_rows.append({"entry_date_bj": date, **summarize(group)})
    daily = pd.DataFrame(daily_rows)
    last_10_daily = daily.tail(10).copy()

    signals.to_csv(OUT / f"{PREFIX}_signals.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(OUT / f"{PREFIX}_gain_bucket.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / f"{PREFIX}_daily_by_entry_bj.csv", index=False, encoding="utf-8-sig")
    last_10_daily.to_csv(OUT / f"{PREFIX}_last10_daily_by_entry_bj.csv", index=False, encoding="utf-8-sig")

    print("========== Current Top3 Old Volume Latest ==========")
    print(f"Signal end UTC: {signal_end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Signal end BJ:  {ms_to_bj_string(signal_end)}")
    print(f"Symbols loaded: {len(kline_map)}")
    print(summary.to_string(index=False))
    print()
    print("========== Last 10 Daily PnL by BJ Entry Date ==========")
    print(last_10_daily[[
        "entry_date_bj",
        "signals",
        "trades",
        "skipped",
        "early_12h",
        "liquidations",
        "wins",
        "losses",
        "net_pnl_u",
        "pf",
        "win_rate_pct",
        "median_return_pct",
        "max_drawdown_u",
    ]].to_string(index=False))
    print()
    print(f"Wrote output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
