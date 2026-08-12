from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_4060_volume_3_7_pretrade_factors import (
    add_pretrade_features,
    build_rank_context,
    categorical_table,
    single_variable_table,
)
from scripts.backfill_old_half_and_run_main_strategy import CACHE_DIR, DAY_MS, OUT, load_kline_map


SOURCE = OUT / "main_strategy_early_exit_7d_cooldown_compare_trades.csv"
PREFIX = "main_strategy_399_gain20_40_rank23_single_factor"


BASE_COLUMNS = {
    "symbol",
    "rank",
    "entry_time_ms",
    "entry_time_utc",
    "entry_time_bj",
    "snapshot_hour_bj",
    "gain_24h",
    "month",
    "volume_24h_ratio_7d",
    "volume_24h_ratio_7d_bucket",
    "ma_structure_4h",
    "distance_to_4h_ma7_pct",
    "status",
    "skip_reason",
    "entry_price",
    "exit_time_ms",
    "exit_time_utc",
    "exit_time_bj",
    "exit_price",
    "exit_reason",
    "holding_days",
    "pnl_u",
    "net_return_pct",
    "mfe_pct",
    "mae_pct",
    "mfe_12h_pct",
    "mae_12h_pct",
    "close_return_12h_pct",
    "max_price_during_trade",
    "min_price_during_trade",
    "is_win",
    "blocking_position_exit_time_ms",
    "blocking_position_exit_time_utc",
    "blocking_position_exit_time_bj",
    "gain_24h_bucket",
    "window",
}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def version_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if col not in BASE_COLUMNS]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot infer version column: {candidates}")
    return candidates[0]


def normalize_single_table(frame: pd.DataFrame, rank: int) -> pd.DataFrame:
    out = single_variable_table(frame).copy()
    out.insert(0, "rank", rank)
    out["因子结论"] = out["分离能力"].map({"YES": "核心前置信号", "WEAK": "仅辅助信号", "NO": "无效因子"})
    return out


def categorical_win_loss_table(frame: pd.DataFrame, rank: int) -> pd.DataFrame:
    rows = []
    for col in ["close_gt_ma7", "ma7_gt_ma21", "fresh_current_rank", "snapshot_hour_bj"]:
        for value, group in frame.groupby(col, dropna=False, sort=True):
            pnl = group["pnl_u"].astype(float)
            rows.append(
                {
                    "rank": rank,
                    "变量": col,
                    "取值": value,
                    "交易数": int(len(group)),
                    "盈利单": int((pnl > 0).sum()),
                    "亏损单": int((pnl < 0).sum()),
                    "胜率": float((pnl > 0).mean()) if len(group) else np.nan,
                    "净收益U": float(pnl.sum()) if len(group) else 0.0,
                    "win_mean_binary": float(group[col].astype(str).isin(["True", "fresh", "00:00"]).mean())
                    if col in {"close_gt_ma7", "ma7_gt_ma21", "fresh_current_rank", "snapshot_hour_bj"}
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    trades = pd.read_csv(SOURCE, encoding="utf-8-sig")
    version_col = version_column(trades)
    trades = trades[
        trades["window"].eq("recent_180d")
        & trades[version_col].eq("before_no_early_exit_cooldown")
        & trades["status"].eq("completed")
        & trades["gain_24h"].ge(0.20)
        & trades["gain_24h"].lt(0.40)
        & trades["rank"].isin([2, 3])
    ].copy()

    start = int(trades["entry_time_ms"].min()) - 14 * DAY_MS
    end = int(trades["entry_time_ms"].max()) + 8 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start, end)
    context = build_rank_context(kline_map, start, end)
    enriched = add_pretrade_features(trades, kline_map, context)
    context_lookup = context.set_index(["signal_time", "symbol"], drop=False)
    fresh_values = []
    for _, row in enriched.iterrows():
        try:
            ctx = context_lookup.loc[(int(row["entry_time_ms"]), row["symbol"])]
            if isinstance(ctx, pd.DataFrame):
                ctx = ctx.iloc[0]
            previous_rank = ctx.get("previous_rank")
            previous_time = ctx.get("previous_signal_time")
            is_fresh = (
                pd.isna(previous_rank)
                or int(previous_rank) != int(row["rank"])
                or int(row["entry_time_ms"]) - int(previous_time) > 12 * 60 * 60 * 1000
            )
        except Exception:
            is_fresh = False
        fresh_values.append("fresh" if is_fresh else "persistent")
    enriched["fresh_current_rank"] = fresh_values
    enriched.to_csv(OUT / f"{PREFIX}_trades_enriched.csv", index=False, encoding="utf-8-sig")

    single_tables = []
    categorical_tables = []
    for rank, group in enriched.groupby("rank", sort=True):
        single_tables.append(normalize_single_table(group.copy(), int(rank)))
        categorical_tables.append(categorical_win_loss_table(group.copy(), int(rank)))

    single = pd.concat(single_tables, ignore_index=True)
    categorical = pd.concat(categorical_tables, ignore_index=True)
    single.to_csv(OUT / f"{PREFIX}_numeric_single_variable.csv", index=False, encoding="utf-8-sig")
    categorical.to_csv(OUT / f"{PREFIX}_categorical_single_variable.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for rank, group in enriched.groupby("rank", sort=True):
        pnl = group["pnl_u"].astype(float)
        summary_rows.append(
            {
                "rank": int(rank),
                "交易数": int(len(group)),
                "盈利单": int((pnl > 0).sum()),
                "亏损单": int((pnl < 0).sum()),
                "净收益U": float(pnl.sum()),
                "胜率": float((pnl > 0).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / f"{PREFIX}_sample_summary.csv", index=False, encoding="utf-8-sig")

    print("========== Sample ==========")
    print(summary.to_string(index=False))
    print()
    print("========== Numeric Single Variables ==========")
    print(single.to_string(index=False))
    print()
    print("========== Categorical Single Variables ==========")
    print(categorical.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
