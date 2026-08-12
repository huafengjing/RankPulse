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
    OUT,
    add_entry_factors,
    load_kline_map,
    max_drawdown,
    profit_factor,
    simulate_main_trades_with_position_limit,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt


PREFIX = "main_strategy_filter_2040_volume_lt15"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def apply_main_filters(
    signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    filter_2040_lt15: bool,
    filter_2040_5_7: bool = False,
) -> pd.DataFrame:
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].lt(0.80)
        & signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)

    in_2040 = signals["gain_24h"].ge(0.20) & signals["gain_24h"].lt(0.40)
    pass_2040 = signals["volume_24h_ratio_7d"].ge(1.5)
    if filter_2040_5_7:
        pass_2040 = pass_2040 & signals["volume_24h_ratio_7d"].lt(5.0)

    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )

    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)

    mask = ((~in_4060) | pass_4060) & (~in_6080)
    if filter_2040_lt15:
        mask = mask & ((~in_2040) | pass_2040)
    return signals[mask].sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "过滤后信号": int(len(group)),
        "完成交易": int(len(completed)),
        "持仓重复跳过": int((group["status"] != "completed").sum()) if "status" in group else 0,
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
    start = end - 180 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start - 7 * DAY_MS, end + 7 * DAY_MS)
    raw_signals = generate_signals(start, end, kline_map)

    rows = []
    outputs = []
    for label, filter_lt15, filter_5_7 in [
        ("before", False, False),
        ("after_filter_2040_volume_lt15", True, False),
        ("after_filter_2040_volume_1p5_to_5", True, True),
    ]:
        signals = apply_main_filters(raw_signals, kline_map, filter_lt15, filter_5_7)
        trades = add_bucket(simulate_main_trades_with_position_limit(signals, kline_map))
        trades["版本"] = label
        outputs.append(trades)
        rows.append({"版本": label, **summarize(trades)})

    all_trades = pd.concat(outputs, ignore_index=True)
    summary = pd.DataFrame(rows)
    all_trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")

    bucket_rows = []
    for (version, bucket), group in all_trades[all_trades["status"].eq("completed")].groupby(["版本", "gain_24h_bucket"], observed=False):
        bucket_rows.append({"版本": version, "24H涨幅桶": str(bucket), **summarize(group)})
    pd.DataFrame(bucket_rows).to_csv(OUT / f"{PREFIX}_bucket_stats.csv", index=False, encoding="utf-8-sig")

    print("========== Filter 20-40 Volume <1.5 Compare ==========")
    print(summary.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
