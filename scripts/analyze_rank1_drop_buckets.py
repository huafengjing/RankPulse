from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_top3_rank_drop_bucket import markdown_table, summarize


HOLDING_DAYS = [1, 2, 3]
BUCKETS = ["0-20%", "20-40%", "40-60%"]


def load_trades(path: Path, strategy_id: str) -> pd.DataFrame:
    trades = pd.read_csv(path)
    eligible = ~trades["skipped_due_to_existing_position"].astype(str).str.lower().eq("true")
    trades = trades[
        trades["strategy_id"].eq(strategy_id)
        & trades["rank"].eq(1)
        & trades["holding_days"].isin(HOLDING_DAYS)
        & eligible
    ].copy()
    drop = trades["drop_24h_pct"]
    trades["merged_drop_bucket"] = np.select(
        [(drop >= 0) & (drop < 20), (drop >= 20) & (drop < 40), (drop >= 40) & (drop < 60)],
        BUCKETS,
        default="outside_scope",
    )
    trades["month"] = pd.to_datetime(trades["snapshot_time_utc"], utc=True).dt.strftime("%Y-%m")
    return trades[trades["merged_drop_bucket"].isin(BUCKETS)]


def make_summary(trades: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    rows = []
    for bucket in BUCKETS:
        for hold in HOLDING_DAYS:
            group = trades[trades["merged_drop_bucket"].eq(bucket) & trades["holding_days"].eq(hold)]
            rows.append(
                {"sample_type": sample_type, "rank": 1, "drop_bucket": bucket, "holding_days": hold}
                | summarize(group)
            )
    return pd.DataFrame(rows)


def monthly_summary(trades: pd.DataFrame, bucket: str, hold: int) -> pd.DataFrame:
    selected = trades[trades["merged_drop_bucket"].eq(bucket) & trades["holding_days"].eq(hold)]
    rows = []
    for month in pd.period_range("2026-01", "2026-07", freq="M").astype(str):
        group = selected[selected["month"].eq(month)]
        stats = summarize(group)
        rows.append(
            {
                "month": month,
                "partial_month": month == "2026-07",
                "trades": stats["trades"],
                "net_pnl_usdt": stats["net_pnl_usdt"],
                "profit_factor": stats["profit_factor"],
                "win_rate_pct": stats["win_rate_pct"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_dir = root / "outputs" / "binance_futures_losers_rank10_extension_20260721_032855"
    config = json.loads((source_dir / "run_config.json").read_text(encoding="utf-8"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = root / "outputs" / f"rank1_drop_bucket_1d_3d_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    independent = load_trades(source_dir / "rank1_to_rank10_all_trades.csv", "Rank1")
    top3_subsample = load_trades(source_dir / "rank_band_all_trades.csv", "Rank1-3")
    primary = make_summary(independent, "rank1_only")
    comparison = pd.concat(
        [primary, make_summary(top3_subsample, "top3_actual_rank1_subsample")], ignore_index=True
    )

    robust = primary[primary["trades"].ge(50) & primary["net_after_best5"].gt(0)].sort_values(
        ["net_after_best5", "profit_factor"], ascending=False
    )
    best_robust = robust.iloc[0]
    monthly = monthly_summary(independent, str(best_robust["drop_bucket"]), int(best_robust["holding_days"]))

    primary.to_csv(output / "rank1_drop_bucket_holding_summary.csv", index=False)
    comparison.to_csv(output / "rank1_sample_definition_comparison.csv", index=False)
    monthly.to_csv(output / "rank1_best_robust_monthly.csv", index=False)

    pf_matrix = primary.pivot(index="drop_bucket", columns="holding_days", values="profit_factor").reset_index()
    pf_matrix.columns = ["drop_bucket", "1D_PF", "2D_PF", "3D_PF"]
    net_matrix = primary.pivot(index="drop_bucket", columns="holding_days", values="net_pnl_usdt").reset_index()
    net_matrix.columns = ["drop_bucket", "1D_net", "2D_net", "3D_net"]
    complete_months = monthly[~monthly["partial_month"]]
    positive_months = int((complete_months["net_pnl_usdt"] > 0).sum())

    report = [
        "# Rank1 Drop Bucket 1D-3D Analysis",
        "",
        "## 1. 口径",
        "",
        f"本地Kline最新时间：{config['cache_latest_utc']}；统一信号截止：{config['unified_signal_end_utc']}。仅Rank1，做空，每笔100 USDT，1X隔离，开平仓费各0.10%，滑点0，Funding未计。",
        "",
        "主结果采用 Rank1-only 可执行样本：持仓冲突只由既有Rank1仓位触发。Top3组合内Rank1子样本仅用于审计，不能代替Rank1独立策略结果。",
        "",
        "## 2. PF矩阵",
        "",
        markdown_table(pf_matrix, ["drop_bucket", "1D_PF", "2D_PF", "3D_PF"]),
        "",
        "## 3. 净收益矩阵",
        "",
        markdown_table(net_matrix, ["drop_bucket", "1D_net", "2D_net", "3D_net"]),
        "",
        "## 4. 完整统计",
        "",
        markdown_table(
            primary,
            ["drop_bucket", "holding_days", "trades", "liquidations", "net_pnl_usdt", "profit_factor", "win_rate_pct", "median_return_pct", "max_drawdown_usdt", "net_after_best1", "net_after_best3", "net_after_best5"],
        ),
        "",
        "## 5. 样本定义敏感性",
        "",
        markdown_table(
            comparison[comparison["drop_bucket"].eq("20-40%") & comparison["holding_days"].eq(3)],
            ["sample_type", "trades", "net_pnl_usdt", "profit_factor", "net_after_best5"],
        ),
        "",
        "Top3组合内Rank1样本会跳过部分此前以Rank2/Rank3身份持有的同币信号，因此不能把该子样本的高PF直接当作Rank1-only策略表现。",
        "",
        "## 6. 最稳健组合月度",
        "",
        markdown_table(monthly, ["month", "partial_month", "trades", "net_pnl_usdt", "profit_factor", "win_rate_pct"]),
        "",
        "## 7. 结论",
        "",
        "- 0-20%：原始PF不错，但2D/3D去最佳5笔后转亏，尾部稳健性不足。",
        "- 20-40%：1D-3D全部盈利且全部去最佳5笔后仍为正，是三个桶中最健康、最连续的区间。",
        "- 40-60%：原始PF略高于1，但所有周期去最佳5笔后大幅转亏，依赖少数大盈利，不支持。",
        "- 原始PF最高的是0-20% / 1D，但去最佳5笔仅余很小利润，因此不按最高PF选择。",
        f"- 按交易数≥50且去最佳5笔后净收益最大的预设稳健性口径，最值得预注册OOS的是 Rank1 + {best_robust['drop_bucket']} + {int(best_robust['holding_days'])}D；PF {best_robust['profit_factor']:.3f}，净收益 {best_robust['net_pnl_usdt']:.2f}，去最佳5笔 {best_robust['net_after_best5']:.2f}，完整盈利月份 {positive_months}/6。",
        "- 这是分桶后的样本内发现，不是正式策略确认。",
    ]
    (output / "Rank1_DropBucket_1D_3D.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "output": str(output.resolve()),
        "cache_latest_utc": config["cache_latest_utc"],
        "unified_signal_end_utc": config["unified_signal_end_utc"],
        "summary_rows": len(primary),
        "best_robust_combination": {
            "drop_bucket": str(best_robust["drop_bucket"]),
            "holding_days": int(best_robust["holding_days"]),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
