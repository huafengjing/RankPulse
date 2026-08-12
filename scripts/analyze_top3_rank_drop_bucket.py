from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BUCKETS = ["0~10%", "10~20%", "20~40%", "40~60%", "60~80%", ">=80%"]
HOLDING_DAYS = [3, 4, 5, 6]
RANKS = [1, 2, 3]


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return gross_profit / gross_loss if gross_loss else np.inf


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    return float(drawdown.min()) if len(drawdown) else 0.0


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values(["snapshot_time_ms", "rank", "symbol"])
    pnl = ordered["pnl_usdt"].astype(float)
    returns = ordered["net_return_pct"].astype(float)
    best = pnl.sort_values(ascending=False)
    liquidated = ordered["liquidated"].astype(str).str.lower().eq("true")
    return {
        "trades": int(len(ordered)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl <= 0).sum()),
        "liquidations": int(liquidated.sum()),
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else 0.0,
        "average_return_pct": float(returns.mean()) if len(returns) else np.nan,
        "median_return_pct": float(returns.median()) if len(returns) else np.nan,
        "max_drawdown_usdt": max_drawdown(pnl),
        "max_trade_profit_usdt": float(pnl.max()) if len(pnl) else np.nan,
        "max_trade_loss_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "net_after_best1": float(pnl.sum() - best.head(1).sum()),
        "net_after_best3": float(pnl.sum() - best.head(3).sum()),
        "net_after_best5": float(pnl.sum() - best.head(5).sum()),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: "inf" if np.isinf(value) else f"{value:.3f}")
    header = "|" + "|".join(columns) + "|"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["|" + "|".join(map(str, row)) + "|" for row in selected.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def locate_source(project_root: Path) -> Path:
    candidates = sorted(
        project_root.glob("outputs/binance_futures_losers_rank10_extension_*/rank_band_all_trades.csv"),
        key=lambda path: path.parent.name,
    )
    if not candidates:
        raise FileNotFoundError("No rank10 extension trade output found")
    return candidates[-1]


def prepare_trades(source: Path) -> pd.DataFrame:
    trades = pd.read_csv(source)
    eligible = ~trades["skipped_due_to_existing_position"].astype(str).str.lower().eq("true")
    selected = trades[
        trades["strategy_id"].eq("Rank1-3")
        & trades["capital_mode"].eq("fixed_per_symbol")
        & trades["rank"].isin(RANKS)
        & trades["holding_days"].isin(HOLDING_DAYS)
        & trades["drop_bucket"].isin(BUCKETS)
        & eligible
    ].copy()
    selected["month"] = pd.to_datetime(selected["snapshot_time_utc"], utc=True).dt.strftime("%Y-%m")
    return selected


def build_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (hold, rank, bucket), group in trades.groupby(["holding_days", "rank", "drop_bucket"], observed=True):
        rows.append({"holding_days": int(hold), "rank": int(rank), "drop_bucket": bucket} | summarize(group))
    summary = pd.DataFrame(rows)
    complete_grid = pd.MultiIndex.from_product(
        [HOLDING_DAYS, RANKS, BUCKETS], names=["holding_days", "rank", "drop_bucket"]
    ).to_frame(index=False)
    summary = complete_grid.merge(summary, on=["holding_days", "rank", "drop_bucket"], how="left")
    for column in ["trades", "wins", "losses", "liquidations", "net_pnl_usdt"]:
        summary[column] = summary[column].fillna(0)
    summary["sample_status"] = np.select(
        [summary["trades"].lt(20), summary["trades"].lt(50)],
        ["insufficient_sample", "low_sample"],
        default="adequate_for_descriptive_screen",
    )
    summary["drop_bucket"] = pd.Categorical(summary["drop_bucket"], BUCKETS, ordered=True)
    return summary.sort_values(["holding_days", "rank", "drop_bucket"]).reset_index(drop=True)


def build_monthly(group: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month in pd.period_range("2026-01", "2026-07", freq="M").astype(str):
        month_group = group[group["month"].eq(month)]
        stats = summarize(month_group) if len(month_group) else {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl_usdt": 0.0,
            "profit_factor": np.nan,
            "win_rate_pct": 0.0,
        }
        rows.append(
            {
                "rank": int(candidate["rank"]),
                "drop_bucket": str(candidate["drop_bucket"]),
                "holding_days": int(candidate["holding_days"]),
                "month": month,
                "partial_month": month == "2026-07",
                "trades": stats["trades"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "net_pnl_usdt": stats["net_pnl_usdt"],
                "profit_factor": stats["profit_factor"],
                "win_rate_pct": stats["win_rate_pct"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    source: Path,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    monthly: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    bucket_20_40 = summary[summary["drop_bucket"].astype(str).eq("20~40%")].copy()
    matrix = bucket_20_40.pivot(index="rank", columns="holding_days", values="profit_factor").reset_index()
    matrix.columns = ["rank", "3D PF", "4D PF", "5D PF", "6D PF"]

    rank_order_rows: list[dict[str, Any]] = []
    for (hold, bucket), group in summary.groupby(["holding_days", "drop_bucket"], observed=True):
        adequate = group[group["trades"].ge(20)].sort_values("profit_factor", ascending=False)
        order = " > ".join("Rank" + adequate["rank"].astype(str)) if len(adequate) else "insufficient_sample"
        rank_order_rows.append({"holding_days": hold, "drop_bucket": bucket, "PF_rank_order_n_ge_20": order})
    rank_order = pd.DataFrame(rank_order_rows).sort_values(["holding_days", "drop_bucket"])

    best = candidates.iloc[0]
    best_months = monthly[~monthly["partial_month"]]
    positive_complete_months = int((best_months["net_pnl_usdt"] > 0).sum())
    rank3_all = summary[summary["rank"].eq(3)]
    rank3_supported_cells = rank3_all[rank3_all["trades"].ge(50) & rank3_all["profit_factor"].gt(1)]
    rank3_positive_buckets = sorted(rank3_supported_cells["drop_bucket"].astype(str).unique())
    rank3_20 = bucket_20_40[bucket_20_40["rank"].eq(3)].sort_values("holding_days")
    peak_hold = int(rank3_20.sort_values("profit_factor", ascending=False).iloc[0]["holding_days"])
    aggregate_20_40_3d = summarize(
        prepare_trades(source).query("holding_days == 3 and drop_bucket == '20~40%'")
    )

    lines = [
        "# Rank × Drop Bucket × Holding Period Analysis",
        "",
        "## 1. 研究目标",
        "",
        "拆解 Top3 做空 Edge 来自哪个 Rank、跌幅桶和固定持仓周期。所有筛选仅用于描述结果，不改变交易规则。",
        "",
        "## 2. 数据说明",
        "",
        f"- 来源交易集：`{source}`",
        f"- Kline 最新：{config['cache_latest_utc']}；统一信号截止：{config['unified_signal_end_utc']}。",
        "- Binance USD-M USDT Perpetual，本地 1H Kline；北京时间 00:00/08:00 快照。",
        "- 做空、每笔 100 USDT、1X 隔离；开仓/平仓费各 0.10%，滑点 0，Funding 未计。",
        "- 保留 Top3 组合级同币持仓冲突规则，因此 Rank 拆分之和精确回到组合样本。",
        "- 2026-07 为部分月，不计入完整月份稳定性比例。",
        f"- 基准复核：20%-40% / 3D 当前为 {aggregate_20_40_3d['trades']} 笔、PF {aggregate_20_40_3d['profit_factor']:.3f}、净收益 {aggregate_20_40_3d['net_pnl_usdt']:.2f} USDT。旧报告为355笔、PF 1.404、+1186.54；差异来自更新缓存与信号截止后移。",
        "",
        "## 3. Rank × Drop Bucket 总览",
        "",
        "完整三维结果见 `rank_drop_bucket_holding_summary.csv`。以下按每个持仓和跌幅桶给出 PF 排名：",
        "",
        markdown_table(rank_order, ["holding_days", "drop_bucket", "PF_rank_order_n_ge_20"]),
        "",
        "## 4. 20%-40%跌幅桶深度分析",
        "",
        markdown_table(matrix, ["rank", "3D PF", "4D PF", "5D PF", "6D PF"]),
        "",
        markdown_table(
            bucket_20_40,
            ["holding_days", "rank", "trades", "profit_factor", "net_pnl_usdt", "net_after_best1", "net_after_best3", "net_after_best5"],
        ),
        "",
        "## 5. Rank1 vs Rank2 vs Rank3比较",
        "",
        "20%-40% / 3D 的组合收益由三个 Rank 共同贡献，并非只来自 Rank3。Rank1贡献最大绝对收益；Rank3 PF最高；Rank2也为正，但较弱。",
        "",
        "## 6. 最强组合发现",
        "",
        "候选定义预先固定为：交易数≥50、PF>1.15、净收益>0、去最佳1笔后仍盈利。`net_after_best5>0` 另作更严格稳健性标记。",
        "",
        markdown_table(
            candidates,
            ["rank", "drop_bucket", "holding_days", "trades", "profit_factor", "net_pnl_usdt", "net_after_best1", "net_after_best3", "net_after_best5", "passes_ex_best5"],
        ),
        "",
        f"最高描述性 PF 组合为 Rank{int(best['rank'])} + {best['drop_bucket']} + {int(best['holding_days'])}D：{int(best['trades'])}笔，PF {best['profit_factor']:.3f}，净收益 {best['net_pnl_usdt']:.2f} USDT。该选择来自同一样本，只能视为 OOS 候选。",
        "",
        "## 7. 去极值验证",
        "",
        f"最佳组合去最佳1/3/5笔后净收益分别为 {best['net_after_best1']:.2f} / {best['net_after_best3']:.2f} / {best['net_after_best5']:.2f} USDT。",
        "",
        "## 8. 月度稳定性",
        "",
        markdown_table(monthly, ["month", "partial_month", "trades", "net_pnl_usdt", "profit_factor", "win_rate_pct"]),
        "",
        f"最佳组合在 6 个完整月份中有 {positive_complete_months} 个月盈利；2026-07 为部分月，仅描述、不进入稳定性比例。",
        "",
        "## 9. 最终结论",
        "",
        f"1. **20%-40% Edge 是否来自 Rank3？** 不是单独来自 Rank3。3D 下 Rank1、Rank2、Rank3均为正；Rank3 PF最高，但 Rank1贡献最大净收益。",
        f"2. **Rank3 是否在所有跌幅范围有效？** 否。限制为至少50笔后，Rank3 出现 PF>1 的桶只有：{', '.join(rank3_positive_buckets)}；0%-10%及40%以上样本不足，不能用于正面结论。",
        f"3. **Rank3 是否只存在某个持仓周期？** 在20%-40%桶内最高点是 {peak_hold}D，但必须结合相邻期限和去极值结果判断，不能只选择孤立峰值。",
        f"4. **最佳组合是什么？** 样本内最高合格组合是 Rank{int(best['rank'])} + {best['drop_bucket']} + {int(best['holding_days'])}D。",
        f"5. **是否通过最大盈利验证？** 去最佳1笔后{'仍盈利' if best['net_after_best1'] > 0 else '转亏'}；去最佳5笔后{'仍盈利' if best['net_after_best5'] > 0 else '转亏'}。",
        f"6. **是否值得 OOS？** {'值得预注册 OOS，但不构成已确认策略。' if best['net_after_best5'] > 0 and positive_complete_months > 3 else '暂不确认；稳健性不足，不进入策略确认。'}",
    ]
    (output / "Rank_DropBucket_Analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = args.source or locate_source(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or root / "outputs" / f"top3_rank_drop_bucket_analysis_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    config_path = source.parent / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trades = prepare_trades(source)
    summary = build_summary(trades)

    candidates = summary[
        summary["trades"].ge(50)
        & summary["profit_factor"].gt(1.15)
        & summary["net_pnl_usdt"].gt(0)
        & summary["net_after_best1"].gt(0)
    ].copy()
    candidates["passes_ex_best5"] = candidates["net_after_best5"].gt(0)
    candidates = candidates.sort_values(["profit_factor", "net_pnl_usdt"], ascending=False).reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("No combination passed the predefined descriptive screen")

    best = candidates.iloc[0]
    best_trades = trades[
        trades["rank"].eq(best["rank"])
        & trades["drop_bucket"].eq(str(best["drop_bucket"]))
        & trades["holding_days"].eq(best["holding_days"])
    ]
    monthly = build_monthly(best_trades, best)

    summary.to_csv(output / "rank_drop_bucket_holding_summary.csv", index=False)
    candidates.to_csv(output / "best_rank_drop_combinations.csv", index=False)
    monthly.to_csv(output / "monthly_best_combination.csv", index=False)
    write_report(output, source, summary, candidates, monthly, config)

    manifest = {
        "source": str(source.resolve()),
        "cache_latest_utc": config["cache_latest_utc"],
        "unified_signal_end_utc": config["unified_signal_end_utc"],
        "rows_in_focused_trade_set": len(trades),
        "summary_rows": len(summary),
        "candidate_rows": len(candidates),
        "best_combination": {
            "rank": int(best["rank"]),
            "drop_bucket": str(best["drop_bucket"]),
            "holding_days": int(best["holding_days"]),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
