from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (
    CACHE_DIR,
    DAY_MS,
    OUT,
    add_entry_factors,
    load_kline_map,
    max_drawdown,
    ms_to_utc,
    profit_factor,
    simulate_main_trades_with_position_limit,
)
from scripts.backtest_futures_top2_fixed_time import (
    generate_signals,
    latest_signal_end_dt,
)


SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}
PREFIX = "recent_half_4060_rank2_volume_3_7_6080_drop_rank2"


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def summarize(group: pd.DataFrame) -> dict[str, float | int]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "过滤后信号": int(len(group)),
        "完成交易": int(len(completed)),
        "持仓重复跳过": int((group["status"] == "skipped").sum()) if "status" in group else 0,
        "完成交易symbol数": int(completed["symbol"].nunique()) if not completed.empty else 0,
        "盈利单": int(len(wins)),
        "亏损单": int(len(losses)),
        "盈利总金额U": float(pnl[pnl > 0].sum()) if len(pnl) else 0.0,
        "亏损总金额U": float(pnl[pnl < 0].sum()) if len(pnl) else 0.0,
        "净收益U": float(pnl.sum()) if len(pnl) else 0.0,
        "PF": profit_factor(pnl),
        "胜率": float(len(wins) / len(completed)) if len(completed) else np.nan,
        "平均收益率": float(returns.mean()) if len(returns) else np.nan,
        "中位数收益率": float(returns.median()) if len(returns) else np.nan,
        "最大回撤U": max_drawdown(pnl),
        "最大单笔收益U": float(pnl.max()) if len(pnl) else np.nan,
        "最大单笔亏损U": float(pnl.min()) if len(pnl) else np.nan,
        "去最大1笔后U": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "去最大3笔后U": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "去最大5笔后U": float(pnl.sum() - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
    }


def apply_base_filters(signals: pd.DataFrame) -> pd.DataFrame:
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].lt(0.80)
        & signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)
    return signals[~in_6080].copy()


def apply_old_4060_rank2_only(signals: pd.DataFrame) -> pd.DataFrame:
    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    return signals[(~in_4060) | signals["rank"].eq(2)].copy()


def apply_new_4060_rank2_volume_3_7(signals: pd.DataFrame) -> pd.DataFrame:
    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )
    return signals[(~in_4060) | pass_4060].copy()


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
    recent_start = end - 180 * DAY_MS
    kline_start = recent_start - 7 * DAY_MS
    kline_end = end + 7 * DAY_MS

    symbols = cached_symbols()
    kline_map = load_kline_map(symbols, kline_start, kline_end)
    signals = generate_signals(recent_start, end, kline_map)
    signals = apply_base_filters(signals)
    signals = add_entry_factors(signals, kline_map)

    old_signals = apply_old_4060_rank2_only(signals).sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    new_signals = apply_new_4060_rank2_volume_3_7(signals).sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)

    old_trades = add_bucket(simulate_main_trades_with_position_limit(old_signals, kline_map))
    new_trades = add_bucket(simulate_main_trades_with_position_limit(new_signals, kline_map))

    old_summary = summarize(old_trades)
    new_summary = summarize(new_trades)
    summary = pd.DataFrame(
        [
            {"版本": "之前: 40-60只保留Rank2", **old_summary},
            {"版本": "新增: 40-60保留Rank2且volume 3-7", **new_summary},
        ]
    )
    summary.to_csv(OUT / f"{PREFIX}_compare_summary.csv", index=False, encoding="utf-8-sig")

    old_trades.to_csv(OUT / f"{PREFIX}_old_trades.csv", index=False, encoding="utf-8-sig")
    new_trades.to_csv(OUT / f"{PREFIX}_new_trades.csv", index=False, encoding="utf-8-sig")

    bucket_rows = []
    for version, trades in [
        ("之前: 40-60只保留Rank2", old_trades),
        ("新增: 40-60保留Rank2且volume 3-7", new_trades),
    ]:
        completed = trades[trades["status"].eq("completed")].copy()
        subset = completed[completed["gain_24h_bucket"].astype(str).eq("40%-60%")].copy()
        bucket_rows.append({"版本": version, **summarize(subset)})
    bucket_summary = pd.DataFrame(bucket_rows)
    bucket_summary.to_csv(OUT / f"{PREFIX}_4060_summary.csv", index=False, encoding="utf-8-sig")

    new_4060 = new_trades[
        new_trades["status"].eq("completed") & new_trades["gain_24h_bucket"].astype(str).eq("40%-60%")
    ].copy()
    new_4060.to_csv(OUT / f"{PREFIX}_4060_trades.csv", index=False, encoding="utf-8-sig")

    monthly_rows = []
    for month, group in new_trades.groupby("month", sort=True):
        monthly_rows.append({"月份": month, **summarize(group)})
    pd.DataFrame(monthly_rows).to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")

    print("========== 40-60 Rank2 + Volume 3-7 主策略对比 ==========")
    print(summary.to_string(index=False))
    print()
    print("========== 40-60 子集 ==========")
    print(bucket_summary.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
