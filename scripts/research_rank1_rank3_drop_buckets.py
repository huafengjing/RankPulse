from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from research_drop_top3_short_edge import DAY_MS, load_kline_map, ms, utc
    from research_losers_rank10_extension import (
        apply_position_conflict,
        build_rank10_signals,
        complete_months,
        completed,
        load_config,
        monthly_summary,
        precompute_outcomes,
        summarize_trades,
    )
except ModuleNotFoundError:
    from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, ms, utc
    from scripts.research_losers_rank10_extension import (
        apply_position_conflict,
        build_rank10_signals,
        complete_months,
        completed,
        load_config,
        monthly_summary,
        precompute_outcomes,
        summarize_trades,
    )


RANKS = [1, 3]
HOLDING_DAYS = [1, 2, 3]
BUCKETS = ["0~20%", "20~40%", "40~60%"]


def focused_drop_bucket(drop_pct: float) -> str | None:
    if 0 <= drop_pct < 20:
        return "0~20%"
    if 20 <= drop_pct < 40:
        return "20~40%"
    if 40 <= drop_pct < 60:
        return "40~60%"
    return None


def tail_pnl(trades: pd.DataFrame, count: int) -> float:
    done = completed(trades)
    pnl = done["pnl_usdt"].astype(float)
    return float(pnl.sum() - pnl.nlargest(count).sum())


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if pd.isna(value) else ("inf" if np.isinf(value) else f"{value:.3f}")
            )
    rows = [
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    rows.extend("|" + "|".join(map(str, row)) + "|" for row in selected.itertuples(index=False, name=None))
    return "\n".join(rows)


def build_report(
    output: Path,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    pf_matrix = summary.pivot(index=["rank", "drop_bucket"], columns="holding_days", values="profit_factor").reset_index()
    pf_matrix.columns = ["rank", "drop_bucket", "1D PF", "2D PF", "3D PF"]
    pnl_matrix = summary.pivot(index=["rank", "drop_bucket"], columns="holding_days", values="net_pnl_usdt").reset_index()
    pnl_matrix.columns = ["rank", "drop_bucket", "1D Net", "2D Net", "3D Net"]

    robust = summary[
        summary["trades"].ge(50)
        & summary["profit_factor"].gt(1.15)
        & summary["net_pnl_usdt"].gt(0)
        & summary["net_pnl_ex_best_5_usdt"].gt(0)
    ].sort_values(["profit_factor", "net_pnl_usdt"], ascending=False)
    best = robust.iloc[0] if len(robust) else summary.sort_values("profit_factor", ascending=False).iloc[0]
    best_monthly = monthly[
        monthly["rank"].eq(best["rank"])
        & monthly["drop_bucket"].eq(best["drop_bucket"])
        & monthly["holding_days"].eq(best["holding_days"])
    ]
    complete_best = best_monthly[~best_monthly["partial_month"]]
    positive_months = int((complete_best["net_pnl_usdt"] > 0).sum())

    rank1_best = summary[summary["rank"].eq(1) & summary["trades"].ge(50)].sort_values("profit_factor", ascending=False).iloc[0]
    rank3_best = summary[summary["rank"].eq(3) & summary["trades"].ge(50)].sort_values("profit_factor", ascending=False).iloc[0]
    lines = [
        "# Rank1 / Rank3 Drop Bucket Study",
        "",
        "## 1. 研究规则",
        "",
        "- 仅交易跌幅榜 Rank1 与 Rank3，Rank2 完全排除。",
        "- 跌幅桶分别为 0%-20%、20%-40%、40%-60%；每个 Rank×桶独立回测，桶外信号不会占用仓位。",
        "- 固定持仓1D、2D、3D；每笔100 USDT，1X隔离；开/平仓费各0.10%，滑点0，Funding未计。",
        "- 同一 Rank×桶内，同币已有持仓时跳过新信号。没有Volume、MA、OI、TP、SL或提前退出。",
        f"- Kline最新：{cfg['cache_latest_utc']}；统一信号区间：{cfg['signal_start_utc']} 至 {cfg['unified_signal_end_utc']}。",
        "- 2026-07为部分月；历史合约主数据不完整，仍存在已退市合约幸存者偏差限制。",
        "- 样本少于20笔只列示，不参与强弱或最佳组合判断；Rank1与Rank3表格是独立桶策略，不代表两者合并后的组合净值。",
        "",
        "## 2. PF 对比",
        "",
        markdown_table(pf_matrix, ["rank", "drop_bucket", "1D PF", "2D PF", "3D PF"]),
        "",
        "## 3. 净收益对比",
        "",
        markdown_table(pnl_matrix, ["rank", "drop_bucket", "1D Net", "2D Net", "3D Net"]),
        "",
        "## 4. 完整统计",
        "",
        markdown_table(
            summary,
            [
                "rank", "drop_bucket", "holding_days", "trades", "wins", "losses", "liquidations",
                "profit_factor", "net_pnl_usdt", "win_rate_pct", "median_return_pct", "max_drawdown_usdt",
                "net_pnl_ex_best_1_usdt", "net_pnl_ex_best_3_usdt", "net_pnl_ex_best_5_usdt",
            ],
        ),
        "",
        "## 5. 去极值后仍通过的描述性组合",
        "",
        (
            markdown_table(
                robust,
                ["rank", "drop_bucket", "holding_days", "trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "positive_month_ratio"],
            )
            if len(robust)
            else "没有组合同时满足交易数≥50、PF>1.15、净收益>0、去最佳5笔后仍盈利。"
        ),
        "",
        "## 6. 最佳描述性组合月度结果",
        "",
        markdown_table(best_monthly, ["month", "partial_month", "trades", "net_pnl_usdt", "profit_factor", "win_rate_pct"]),
        "",
        "## 7. 结论",
        "",
        f"- Rank1最高PF组合：{rank1_best['drop_bucket']} / {int(rank1_best['holding_days'])}D，PF {rank1_best['profit_factor']:.3f}，净收益 {rank1_best['net_pnl_usdt']:.2f}。",
        f"- Rank3最高PF组合：{rank3_best['drop_bucket']} / {int(rank3_best['holding_days'])}D，PF {rank3_best['profit_factor']:.3f}，净收益 {rank3_best['net_pnl_usdt']:.2f}。",
        f"- 全部组合中最佳描述性结果：Rank{int(best['rank'])} + {best['drop_bucket']} + {int(best['holding_days'])}D；PF {best['profit_factor']:.3f}，净收益 {best['net_pnl_usdt']:.2f}，去最佳5笔 {best['net_pnl_ex_best_5_usdt']:.2f}。",
        f"- 该组合完整月份盈利数为 {positive_months}/{len(complete_best)}。这是样本内分桶发现，只能决定是否预注册OOS，不能作为已确认策略。",
    ]
    (output / "Rank1_Rank3_DropBucket_Study.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / f"rank1_rank3_drop_buckets_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)

    cfg = load_config()
    cfg["holding_days"] = HOLDING_DAYS
    cfg["max_rank"] = 3
    cfg["selected_ranks"] = RANKS
    cfg["focused_drop_buckets"] = BUCKETS

    print("[1/4] Loading local 1H cache", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - max(HOLDING_DAYS) * DAY_MS
    candidates = [
        ms(day + pd.Timedelta(hours=hour))
        for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC")
        for hour in cfg["snapshot_utc_hours"]
        if ms(day + pd.Timedelta(hours=hour)) <= latest_signal
    ]
    signal_end = max(candidates)
    cfg["cache_latest_utc"] = str(utc(cache_end))
    cfg["unified_signal_end_utc"] = str(utc(signal_end))
    cfg["actual_output_directory"] = str(output.resolve())
    (output / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[2/4] Rebuilding historical rankings and outcomes", flush=True)
    signals, snapshot_audit = build_rank10_signals(signal_start, signal_end, kline_map)
    outcomes = precompute_outcomes(signals[signals["rank"].isin(RANKS)], kline_map, cfg)
    outcomes["focused_drop_bucket"] = outcomes["drop_24h_pct"].map(focused_drop_bucket)
    complete_month_set = complete_months(signal_start, signal_end)

    print("[3/4] Simulating independent Rank x bucket portfolios", flush=True)
    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    for rank in RANKS:
        for bucket in BUCKETS:
            bucket_outcomes = outcomes[outcomes["focused_drop_bucket"].eq(bucket)]
            for hold in HOLDING_DAYS:
                strategy_id = f"Rank{rank}_{bucket}_{hold}D"
                trades = apply_position_conflict(
                    bucket_outcomes,
                    {rank},
                    hold,
                    {rank: float(cfg["per_symbol_notional_usdt"])},
                    strategy_id,
                    "fixed_per_symbol",
                )
                trades["focused_drop_bucket"] = bucket
                trade_frames.append(trades)
                keys = {"rank": rank, "drop_bucket": bucket, "holding_days": hold}
                stats = summarize_trades(trades, complete_month_set)
                stats |= {
                    "net_pnl_ex_best_1_usdt": tail_pnl(trades, 1),
                    "net_pnl_ex_best_3_usdt": tail_pnl(trades, 3),
                    "net_pnl_ex_best_5_usdt": tail_pnl(trades, 5),
                }
                stats["sample_status"] = "insufficient_sample" if stats["trades"] < 20 else ("low_sample" if stats["trades"] < 50 else "adequate")
                summary_rows.append(keys | stats)
                monthly_rows.extend(monthly_summary(trades, complete_month_set, keys))

    summary = pd.DataFrame(summary_rows).sort_values(["rank", "drop_bucket", "holding_days"])
    monthly = pd.DataFrame(monthly_rows).sort_values(["rank", "drop_bucket", "holding_days", "month"])
    all_trades = pd.concat(trade_frames, ignore_index=True)
    summary.to_csv(output / "rank1_rank3_bucket_holding_summary.csv", index=False)
    monthly.to_csv(output / "rank1_rank3_bucket_monthly.csv", index=False)
    all_trades.to_csv(output / "rank1_rank3_bucket_all_trades.csv", index=False)
    signals[signals["rank"].isin(RANKS)].to_csv(output / "rank1_rank3_signals.csv", index=False)

    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"],
        "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "summary_rows_expected": 18,
        "summary_rows_actual": len(summary),
        "duplicate_signal_symbol_snapshot": int(signals[signals["rank"].isin(RANKS)].duplicated(["snapshot_time_ms", "symbol"]).sum()),
        "duplicate_signal_rank_snapshot": int(signals[signals["rank"].isin(RANKS)].duplicated(["snapshot_time_ms", "rank"]).sum()),
        "all_exit_windows_within_cache": bool(signal_end + max(HOLDING_DAYS) * DAY_MS <= cache_end),
        "missing_entry_or_exit_outcomes": int(len(signals[signals["rank"].isin(RANKS)]) * len(HOLDING_DAYS) - len(outcomes)),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit["missing_hour_count"].sum()),
        "cache_invalid_rows_removed": int(cache_audit["invalid_rows_removed"].sum()),
    }
    (output / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print("[4/4] Writing report", flush=True)
    build_report(output, summary, monthly, cfg)
    print(json.dumps({"output": str(output.resolve()), "quality": quality}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
