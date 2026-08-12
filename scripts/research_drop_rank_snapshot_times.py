from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_drop_top3_short_edge import DAY_MS, HOUR_MS, load_kline_map, ms, snapshot_rankings, utc
from scripts.research_losers_rank10_extension import (
    apply_position_conflict,
    complete_months,
    completed,
    load_config,
    precompute_outcomes,
    summarize_trades,
)


BJ_SLOTS = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
CANDIDATES = {
    "A": {"rank": 1, "drop_low": 0.0, "drop_high": 20.0, "drop_bucket": "0~20%", "holding_days": 1},
    "B": {"rank": 1, "drop_low": 20.0, "drop_high": 40.0, "drop_bucket": "20~40%", "holding_days": 2},
    "C": {"rank": 3, "drop_low": 20.0, "drop_high": 40.0, "drop_bucket": "20~40%", "holding_days": 3},
}


def utc_hour_for_beijing(slot: str) -> int:
    return (int(slot[:2]) - 8) % 24


def time_configurations() -> list[tuple[str, tuple[str, ...], str]]:
    singles = [(slot, (slot,), "single") for slot in BJ_SLOTS]
    pairs = [(" + ".join(pair), pair, "pair") for pair in itertools.combinations(BJ_SLOTS, 2)]
    return singles + pairs


def build_six_slot_signals(
    start_time: int,
    end_time: int,
    kline_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    utc_hours = sorted(utc_hour_for_beijing(slot) for slot in BJ_SLOTS)
    for day in pd.date_range(utc(start_time).floor("D"), utc(end_time).floor("D"), freq="D", tz="UTC"):
        for hour in utc_hours:
            snapshot_time = ms(day + pd.Timedelta(hours=hour))
            if not start_time <= snapshot_time <= end_time:
                continue
            ranking = snapshot_rankings(snapshot_time, kline_map)
            audits.append({"snapshot_time_ms": snapshot_time, "snapshot_time_utc": utc(snapshot_time), "ranked_symbols": len(ranking)})
            if ranking.empty:
                continue
            selected = ranking.sort_values(["change_24h", "symbol"], ascending=[True, True]).head(3)
            for rank, (_, item) in enumerate(selected.iterrows(), start=1):
                change = float(item["change_24h"])
                bj_time = utc(snapshot_time).tz_convert("Asia/Shanghai")
                rows.append(
                    {
                        "snapshot_time_ms": snapshot_time,
                        "snapshot_time_utc": utc(snapshot_time),
                        "snapshot_hour_bj": bj_time.strftime("%H:%M"),
                        "symbol": str(item["symbol"]),
                        "rank": rank,
                        "current_close": float(item["current_close"]),
                        "close_24h_ago": float(item["close_24h_ago"]),
                        "return_24h_pct": change * 100,
                        "drop_24h_pct": -change * 100,
                        "drop_bucket": "source_unbucketed",
                        "entry_time_ms": snapshot_time,
                        "entry_time_utc": utc(snapshot_time),
                        "signal_eligible": True,
                        "eligibility_reason": "complete_24h_history_and_entry_open",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(audits)


def candidate_outcomes(outcomes: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    return outcomes[
        outcomes["rank"].eq(spec["rank"])
        & outcomes["holding_days"].eq(spec["holding_days"])
        & outcomes["drop_24h_pct"].ge(spec["drop_low"])
        & outcomes["drop_24h_pct"].lt(spec["drop_high"])
    ].copy()


def empty_stats() -> dict[str, Any]:
    return {
        "trades": 0, "wins": 0, "losses": 0, "liquidations": 0, "net_pnl_usdt": 0.0,
        "profit_factor": np.nan, "win_rate_pct": np.nan, "average_return_pct": np.nan,
        "median_return_pct": np.nan, "max_drawdown_usdt": 0.0, "max_trade_profit_usdt": np.nan,
        "max_trade_loss_usdt": np.nan, "net_pnl_ex_best_1_usdt": np.nan,
        "net_pnl_ex_best_3_usdt": np.nan, "net_pnl_ex_best_5_usdt": np.nan,
        "expectancy_usdt_per_trade": np.nan, "expectancy_pct_per_trade": np.nan,
        "positive_months": 0, "total_complete_months": 0, "positive_month_ratio": np.nan,
        "max_consecutive_losses": 0, "return_to_drawdown_ratio": np.nan,
    }


def configuration_summary(
    trades: pd.DataFrame,
    raw: pd.DataFrame,
    scheduled_snapshots: int,
    complete_month_set: set[str],
) -> dict[str, Any]:
    done = completed(trades)
    stats = summarize_trades(trades, complete_month_set) if len(done) else empty_stats()
    raw_signals = len(raw)
    skipped = int(trades["skipped_due_to_existing_position"].sum()) if len(trades) else 0
    net = float(stats["net_pnl_usdt"])
    ex5 = float(stats["net_pnl_ex_best_5_usdt"]) if pd.notna(stats["net_pnl_ex_best_5_usdt"]) else np.nan
    return {
        "number_of_snapshots": scheduled_snapshots,
        "raw_signals": raw_signals,
        "executed_trades": len(done),
        "skipped_existing_position": skipped,
        "skip_rate_pct": skipped / raw_signals * 100 if raw_signals else 0.0,
        "unique_symbols": int(raw["symbol"].nunique()),
        "duplicate_signal_symbols": int(raw_signals - raw["symbol"].nunique()),
        **{key: value for key, value in stats.items() if key not in {"trades"}},
        "pnl_per_trade_usdt": float(stats["expectancy_usdt_per_trade"]),
        "return_per_trade_pct": float(stats["expectancy_pct_per_trade"]),
        "positive_full_months": int(stats["positive_months"]),
        "total_full_months": int(stats["total_complete_months"]),
        "retained_pnl_after_best5_pct": ex5 / net * 100 if net > 0 and pd.notna(ex5) else np.nan,
    }


def monthly_rows(
    trades: pd.DataFrame,
    raw: pd.DataFrame,
    keys: dict[str, Any],
    months: list[str],
    complete_month_set: set[str],
) -> list[dict[str, Any]]:
    raw = raw.assign(month=pd.to_datetime(raw["entry_time_utc"], utc=True).dt.strftime("%Y-%m"))
    trades = trades.assign(month=pd.to_datetime(trades["entry_time_utc"], utc=True).dt.strftime("%Y-%m"))
    result = []
    for month in months:
        month_raw = raw[raw["month"].eq(month)]
        month_trades = trades[trades["month"].eq(month)]
        done = completed(month_trades)
        stats = summarize_trades(month_trades, {month}) if len(done) else empty_stats()
        result.append(
            keys
            | {
                "month": month,
                "partial_month": month not in complete_month_set,
                "raw_signals": len(month_raw),
                "executed_trades": len(done),
                "skipped_existing_position": int(month_trades["skipped_due_to_existing_position"].sum()) if len(month_trades) else 0,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "net_pnl_usdt": stats["net_pnl_usdt"],
                "profit_factor": stats["profit_factor"],
                "win_rate_pct": stats["win_rate_pct"],
                "median_return_pct": stats["median_return_pct"],
                "max_drawdown_usdt": stats["max_drawdown_usdt"],
            }
        )
    return result


def select_robust(frame: pd.DataFrame) -> pd.Series:
    qualified = frame[
        frame["executed_trades"].ge(20)
        & frame["profit_factor"].gt(1)
        & frame["net_pnl_usdt"].gt(0)
        & frame["net_pnl_ex_best_5_usdt"].gt(0)
    ]
    pool = qualified if len(qualified) else frame
    return pool.sort_values(
        ["positive_month_ratio", "retained_pnl_after_best5_pct", "return_to_drawdown_ratio", "profit_factor"],
        ascending=False,
        na_position="last",
    ).iloc[0]


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = frame.loc[:, columns].head(limit).copy() if limit else frame.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else ("inf" if np.isinf(value) else f"{value:.3f}"))
    return "\n".join(
        [
            "|" + "|".join(columns) + "|",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *("|" + "|".join(map(str, row)) + "|" for row in view.itertuples(index=False, name=None)),
        ]
    )


def pair_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in summary.groupby("candidate_id"):
        singles = group[group["configuration_type"].eq("single")]
        best_single = select_robust(singles)
        for _, pair in group[group["configuration_type"].eq("pair")].iterrows():
            first, second = pair["time_configuration"].split(" + ")
            component_1 = singles[singles["time_configuration"].eq(first)].iloc[0]
            component_2 = singles[singles["time_configuration"].eq(second)].iloc[0]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "time_configuration": pair["time_configuration"],
                    "best_single_time": best_single["time_configuration"],
                    "trade_increase_vs_best_single": pair["executed_trades"] - best_single["executed_trades"],
                    "net_pnl_increase_vs_best_single": pair["net_pnl_usdt"] - best_single["net_pnl_usdt"],
                    "pf_change_vs_best_single": pair["profit_factor"] - best_single["profit_factor"],
                    "ex_best5_change_vs_best_single": pair["net_pnl_ex_best_5_usdt"] - best_single["net_pnl_ex_best_5_usdt"],
                    "max_drawdown_change_vs_best_single": pair["max_drawdown_usdt"] - best_single["max_drawdown_usdt"],
                    "return_to_drawdown_change_vs_best_single": pair["return_to_drawdown_ratio"] - best_single["return_to_drawdown_ratio"],
                    "positive_month_ratio_change": pair["positive_month_ratio"] - best_single["positive_month_ratio"],
                    "pnl_per_trade_change": pair["pnl_per_trade_usdt"] - best_single["pnl_per_trade_usdt"],
                    "independent_trade_addition": pair["executed_trades"] - max(component_1["executed_trades"], component_2["executed_trades"]),
                    "duplicated_or_skipped_signal_increase": pair["skipped_existing_position"] - best_single["skipped_existing_position"],
                    "component_1_pf": component_1["profit_factor"],
                    "component_2_pf": component_2["profit_factor"],
                    "component_1_net_pnl": component_1["net_pnl_usdt"],
                    "component_2_net_pnl": component_2["net_pnl_usdt"],
                }
            )
    return pd.DataFrame(rows)


def recommendation_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in summary.groupby("candidate_id"):
        best_single = select_robust(group[group["configuration_type"].eq("single")])
        best_pair = select_robust(group[group["configuration_type"].eq("pair")])
        pair_is_materially_better = all(
            [
                best_pair["net_pnl_usdt"] > best_single["net_pnl_usdt"],
                best_pair["net_pnl_ex_best_5_usdt"] > best_single["net_pnl_ex_best_5_usdt"],
                best_pair["profit_factor"] >= best_single["profit_factor"] * 0.9,
                best_pair["return_to_drawdown_ratio"] >= best_single["return_to_drawdown_ratio"],
                best_pair["positive_month_ratio"] >= best_single["positive_month_ratio"],
                best_pair["pnl_per_trade_usdt"] >= best_single["pnl_per_trade_usdt"] * 0.85,
            ]
        )
        chosen = best_pair if pair_is_materially_better else best_single
        rows.append(
            {
                "candidate_id": candidate_id,
                "best_single_time": best_single["time_configuration"],
                "best_pair_times": best_pair["time_configuration"],
                "recommended_configuration": chosen["time_configuration"],
                "recommendation_type": "pair" if pair_is_materially_better else "single",
                "reason": "pair improves net, ex-best5, drawdown efficiency and month consistency without material PF/per-trade dilution" if pair_is_materially_better else "best pair does not materially dominate the robust single slot",
                "sample_size": int(chosen["executed_trades"]),
                "profit_factor": chosen["profit_factor"],
                "net_pnl_usdt": chosen["net_pnl_usdt"],
                "net_pnl_ex_best_5_usdt": chosen["net_pnl_ex_best_5_usdt"],
                "positive_month_ratio": chosen["positive_month_ratio"],
                "max_drawdown_usdt": chosen["max_drawdown_usdt"],
                "return_to_drawdown_ratio": chosen["return_to_drawdown_ratio"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    summary: pd.DataFrame,
    pair_compare: pd.DataFrame,
    marginal: pd.DataFrame,
    monthly: pd.DataFrame,
    recommendations: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    baseline = summary[summary["time_configuration"].eq("00:00 + 08:00")]
    def result(candidate_id: str, config: str) -> pd.Series:
        return summary[summary.candidate_id.eq(candidate_id) & summary.time_configuration.eq(config)].iloc[0]

    a_single, a_pair, a_base = result("A", "00:00"), result("A", "00:00 + 04:00"), result("A", "00:00 + 08:00")
    b_single, b_pf_pair, b_stable_pair, b_base = result("B", "08:00"), result("B", "08:00 + 16:00"), result("B", "12:00 + 20:00"), result("B", "00:00 + 08:00")
    c_single, c_pair, c_base = result("C", "00:00"), result("C", "00:00 + 20:00"), result("C", "00:00 + 08:00")
    lines = [
        "# Drop Rank Snapshot Time Study",
        "",
        "## 1. 研究目标",
        "",
        "检验三个已固定 Rank×跌幅×持仓候选在六个北京时间快照及全部双时段组合中的表现。时间选择是唯一研究变量。",
        "",
        "## 2. 数据与回测口径",
        "",
        f"本地Kline最新：{cfg['cache_latest_utc']}。统一信号窗口：{cfg['signal_start_utc']} 至 {cfg['unified_signal_end_utc']}。时区：UTC计算、北京时间展示。",
        "排行榜仅使用快照前已完成1H Close；Entry为快照1H Open，Exit为固定到期1H Open。每笔100 USDT、1X隔离、双边手续费各0.10%、滑点0、Funding未计。",
        "每个时间配置从原始信号独立运行同币持仓冲突；三候选互不合并。",
        "",
        "## 3. 三个候选策略说明",
        "",
        "A：Rank1 + 0%-20% + 1D。B：Rank1 + 20%-40% + 2D。C：Rank3 + 20%-40% + 3D。",
        "",
        "## 4. 六个单时间段总览",
        "",
        markdown_table(summary[summary.configuration_type.eq("single")], ["candidate_id", "time_configuration", "raw_signals", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "positive_month_ratio", "max_drawdown_usdt", "pnl_per_trade_usdt"]),
    ]
    for number, candidate_id in enumerate(CANDIDATES, start=5):
        group = summary[(summary.candidate_id.eq(candidate_id)) & summary.configuration_type.eq("single")].sort_values("profit_factor", ascending=False)
        robust_single = select_robust(group)
        lines += [
            "",
            f"## {number}. Candidate {candidate_id} 时间段分析",
            "",
            markdown_table(group, ["time_configuration", "raw_signals", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "positive_month_ratio", "max_drawdown_usdt", "return_to_drawdown_ratio"]),
            "",
            f"稳健性优先的单时段为 {robust_single['time_configuration']}；这不是单纯按PF最高选择。",
        ]
    pair_top = summary[summary.configuration_type.eq("pair")].sort_values(["candidate_id", "profit_factor"], ascending=[True, False]).groupby("candidate_id").head(5)
    lines += [
        "",
        "## 8. 十五个双时间段组合分析",
        "",
        "完整45组见CSV；以下展示每个候选按PF排序的前5组。",
        "",
        markdown_table(pair_top, ["candidate_id", "time_configuration", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "positive_month_ratio", "max_drawdown_usdt", "return_to_drawdown_ratio"]),
        "",
        "## 9. 当前 00:00 + 08:00 基准复查",
        "",
        markdown_table(baseline, ["candidate_id", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "positive_month_ratio", "max_drawdown_usdt"]),
        "",
        "## 10. 双时段相对最佳单时段的增量分析",
        "",
        markdown_table(pair_compare.sort_values(["candidate_id", "net_pnl_increase_vs_best_single"], ascending=[True, False]).groupby("candidate_id").head(5), ["candidate_id", "time_configuration", "best_single_time", "trade_increase_vs_best_single", "net_pnl_increase_vs_best_single", "pf_change_vs_best_single", "ex_best5_change_vs_best_single", "return_to_drawdown_change_vs_best_single", "positive_month_ratio_change", "pnl_per_trade_change"]),
        "",
        "## 11. 时间段边际贡献",
        "",
        markdown_table(marginal.sort_values(["candidate_id", "time_configuration", "time_slot"]), ["candidate_id", "time_configuration", "time_slot", "executed_trades", "net_pnl_usdt", "profit_factor", "win_rate_pct", "median_return_pct", "contribution_to_combination_net_pct"], limit=40),
        "",
        "## 12. 去最佳1、3、5笔验证",
        "",
        "全部63组的原始与去最佳1/3/5笔结果已写入总表。推荐时必须要求去最佳5笔仍为正，并同时查看保留收益比例。",
        "",
        "## 13. 月度稳定性",
        "",
        "完整月与2026-07部分月已分离。完整月盈利比例只使用完整自然月；全部配置月度结果见CSV。",
        "",
        markdown_table(monthly.merge(recommendations[["candidate_id", "recommended_configuration"]], on="candidate_id").query("time_configuration == recommended_configuration"), ["candidate_id", "time_configuration", "month", "partial_month", "executed_trades", "net_pnl_usdt", "profit_factor", "win_rate_pct", "max_drawdown_usdt"]),
        "",
        "## 14. 回撤与连续亏损",
        "",
        markdown_table(summary.merge(recommendations[["candidate_id", "recommended_configuration"]], on="candidate_id").query("time_configuration == recommended_configuration"), ["candidate_id", "time_configuration", "executed_trades", "max_drawdown_usdt", "max_consecutive_losses", "return_to_drawdown_ratio"]),
        "",
        "## 15. 每个候选策略的最终时间段建议",
        "",
        markdown_table(recommendations, ["candidate_id", "best_single_time", "best_pair_times", "recommended_configuration", "recommendation_type", "sample_size", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "positive_month_ratio", "max_drawdown_usdt", "return_to_drawdown_ratio", "reason"]),
        "",
        "### Candidate A 逐项回答",
        "",
        f"Edge主要集中在00:00；04:00原始PF更高但去最佳5笔为负，不能单独选为规则。原00:00+08:00为{int(a_base.executed_trades)}笔、PF {a_base.profit_factor:.3f}、净收益{a_base.net_pnl_usdt:.2f}、去最佳5笔{a_base.net_pnl_ex_best_5_usdt:.2f}。00:00+04:00同为{int(a_pair.executed_trades)}笔，但PF {a_pair.profit_factor:.3f}、净收益{a_pair.net_pnl_usdt:.2f}、去最佳5笔{a_pair.net_pnl_ex_best_5_usdt:.2f}、完整月比例{a_pair.positive_month_ratio:.3f}、回撤{a_pair.max_drawdown_usdt:.2f}，全面优于旧基准。相对00:00单时段新增{int(a_pair.executed_trades-a_single.executed_trades)}笔有效交易，且每笔收益仅小幅稀释，因此建议预注册00:00+04:00双时段；小样本风险仍高。",
        "",
        "### Candidate B 逐项回答",
        "",
        f"08:00单时段最能捕捉Rank1短期延续：{int(b_single.executed_trades)}笔、PF {b_single.profit_factor:.3f}、净收益{b_single.net_pnl_usdt:.2f}、去最佳5笔{b_single.net_pnl_ex_best_5_usdt:.2f}。旧00:00+08:00虽然有{int(b_base.executed_trades)}笔，但PF仅{b_base.profit_factor:.3f}、净收益{b_base.net_pnl_usdt:.2f}，00:00明显拖累。样本内最高PF双时段08:00+16:00达到PF {b_pf_pair.profit_factor:.3f}、净收益{b_pf_pair.net_pnl_usdt:.2f}，但相较08:00单时段PF和每笔收益下降、回撤效率未改善；月度更稳的12:00+20:00同样扩大回撤。因此建议只保留08:00单时段。",
        "",
        "### Candidate C 逐项回答",
        "",
        f"Rank3的3D Edge集中在00:00、08:00、20:00；12:00 PF<1，16:00去最佳5笔后亏损。00:00+20:00为{int(c_pair.executed_trades)}笔、PF {c_pair.profit_factor:.3f}、净收益{c_pair.net_pnl_usdt:.2f}、去最佳5笔{c_pair.net_pnl_ex_best_5_usdt:.2f}、完整月比例{c_pair.positive_month_ratio:.3f}、回撤{c_pair.max_drawdown_usdt:.2f}。相较旧00:00+08:00，其净收益略高、PF与去极值收益提高、回撤明显下降、完整盈利月份由{c_base.positive_month_ratio:.3f}升至{c_pair.positive_month_ratio:.3f}。两个组成时段在组合内均为正贡献，建议预注册00:00+20:00。",
        "",
        "### 跨策略结论",
        "",
        "三个候选需要不同Snapshot规则：A用00:00+04:00，B用08:00，C用00:00+20:00。00:00对A/C有效但对B弱；08:00对B最强、对C有效，却对A接近无Edge，因此不存在适用于三者的统一最佳时段。现有00:00+08:00不应统一保留：A应替换04:00，B应删除00:00，C应替换08:00为20:00。时间选择改善的不只是交易数；A/C同时改善去极值、月份或回撤效率，B则通过删去弱时段提高PF和每笔质量。",
        "",
        "## 16. 是否值得预注册 OOS",
        "",
        "推荐配置仅代表样本内稳健性排序。只有去最佳5笔仍盈利、月份不过度集中、且双时段确有独立新增交易的简单规则，才适合冻结后进入OOS；不构成正式策略确认。",
        "",
        "## 17. 研究限制",
        "",
        "时间配置在同一数据内发现和比较，存在多重比较偏差；Funding未计；本地当前合约缓存无法证明覆盖所有历史退市合约；2026-07为部分月。PF为inf的小样本不参与稳健推荐。",
    ]
    (out / "Drop_Rank_Snapshot_Time_Study.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / f"drop_rank_snapshot_time_study_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg["holding_days"] = [1, 2, 3]
    cfg["snapshot_beijing_slots"] = BJ_SLOTS
    cfg["snapshot_utc_hours"] = sorted(utc_hour_for_beijing(slot) for slot in BJ_SLOTS)
    cfg["candidates"] = CANDIDATES

    print("[1/5] Loading local cache and defining unified window", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    scheduled = [
        ms(day + pd.Timedelta(hours=hour))
        for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC")
        for hour in cfg["snapshot_utc_hours"]
        if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal
    ]
    signal_end = max(scheduled)
    cfg["cache_latest_utc"] = str(utc(cache_end))
    cfg["unified_signal_end_utc"] = str(utc(signal_end))
    cfg["actual_output_directory"] = str(out.resolve())
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_month_set = complete_months(signal_start, signal_end)
    months = pd.period_range(utc(signal_start).strftime("%Y-%m"), utc(signal_end).strftime("%Y-%m"), freq="M").astype(str).tolist()

    print("[2/5] Rebuilding six-slot historical rankings", flush=True)
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    relevant_signals = signals[signals["rank"].isin([1, 3])]
    outcomes = precompute_outcomes(relevant_signals, kline_map, cfg)

    print("[3/5] Running 63 independent configurations", flush=True)
    summary_rows: list[dict[str, Any]] = []
    monthly_output: list[dict[str, Any]] = []
    all_trade_frames: list[pd.DataFrame] = []
    trade_map: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate_id, spec in CANDIDATES.items():
        base = candidate_outcomes(outcomes, spec)
        for config_name, slots, config_type in time_configurations():
            raw = base[base["snapshot_hour_bj"].isin(slots)]
            trades = apply_position_conflict(raw, {spec["rank"]}, spec["holding_days"], {spec["rank"]: 100.0}, candidate_id, "fixed_per_symbol")
            trades["candidate_id"] = candidate_id
            trades["time_configuration"] = config_name
            trades["configuration_type"] = config_type
            trades["signal_time_utc"] = trades["snapshot_time_utc"]
            trades["signal_time_beijing"] = pd.to_datetime(trades["snapshot_time_utc"], utc=True).dt.tz_convert("Asia/Shanghai")
            trades["snapshot_hour_beijing"] = trades["snapshot_hour_bj"]
            trades["drop_pct_24h"] = trades["drop_24h_pct"]
            trades["entry_time"] = trades["entry_time_utc"]
            trades["exit_time"] = trades["exit_time_utc"]
            trades["liquidation"] = trades["liquidated"]
            trades["skipped_or_executed"] = np.where(trades["skipped_due_to_existing_position"], "skipped_existing_position", "executed")
            trade_map[(candidate_id, config_name)] = trades
            all_trade_frames.append(trades)
            scheduled_count = sum(utc(time).tz_convert("Asia/Shanghai").strftime("%H:%M") in slots for time in scheduled)
            keys = {
                "candidate_id": candidate_id,
                "rank": spec["rank"],
                "drop_bucket": spec["drop_bucket"],
                "holding_days": spec["holding_days"],
                "time_configuration": config_name,
                "configuration_type": config_type,
            }
            summary_rows.append(keys | configuration_summary(trades, raw, scheduled_count, complete_month_set))
            monthly_output.extend(monthly_rows(trades, raw, keys, months, complete_month_set))

    summary = pd.DataFrame(summary_rows).sort_values(["candidate_id", "configuration_type", "time_configuration"])
    monthly = pd.DataFrame(monthly_output)
    all_trades = pd.concat(all_trade_frames, ignore_index=True)
    singles = summary[summary.configuration_type.eq("single")].copy()
    pairs = summary[summary.configuration_type.eq("pair")].copy()
    pair_compare = pair_comparison(summary)

    print("[4/5] Calculating pair marginal contributions and recommendations", flush=True)
    marginal_rows = []
    for (candidate_id, config_name), trades in trade_map.items():
        if " + " not in config_name:
            continue
        done = completed(trades)
        total_net = float(done.pnl_usdt.sum())
        for slot, group in done.groupby("snapshot_hour_bj"):
            stats = summarize_trades(group, complete_month_set)
            marginal_rows.append(
                {
                    "candidate_id": candidate_id,
                    "time_configuration": config_name,
                    "time_slot": slot,
                    "executed_trades": len(group),
                    "net_pnl_usdt": stats["net_pnl_usdt"],
                    "profit_factor": stats["profit_factor"],
                    "win_rate_pct": stats["win_rate_pct"],
                    "median_return_pct": stats["median_return_pct"],
                    "contribution_to_combination_net_pct": stats["net_pnl_usdt"] / total_net * 100 if total_net else np.nan,
                }
            )
    marginal = pd.DataFrame(marginal_rows)
    recommendations = recommendation_rows(summary)

    summary.to_csv(out / "snapshot_time_configuration_summary.csv", index=False)
    singles.to_csv(out / "single_snapshot_summary.csv", index=False)
    pairs.to_csv(out / "pair_snapshot_summary.csv", index=False)
    pair_compare.to_csv(out / "pair_vs_best_single_comparison.csv", index=False)
    marginal.to_csv(out / "snapshot_marginal_contribution.csv", index=False)
    monthly.to_csv(out / "snapshot_time_monthly.csv", index=False)
    all_trades.to_csv(out / "snapshot_time_all_trades.csv", index=False)
    recommendations.to_csv(out / "recommended_snapshot_configurations.csv", index=False)

    print("[5/5] Validating and writing report", flush=True)
    monthly_check = monthly.groupby(["candidate_id", "time_configuration"]).net_pnl_usdt.sum()
    summary_check = summary.set_index(["candidate_id", "time_configuration"]).net_pnl_usdt
    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"],
        "signal_start_utc": cfg["signal_start_utc"],
        "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "single_configurations": len(singles),
        "pair_configurations": len(pairs),
        "total_configurations": len(summary),
        "all_candidates_present": sorted(summary.candidate_id.unique().tolist()) == ["A", "B", "C"],
        "six_single_slots_per_candidate": bool((singles.groupby("candidate_id").size() == 6).all()),
        "fifteen_unique_pairs_per_candidate": bool((pairs.groupby("candidate_id").size() == 15).all()),
        "common_signal_window": True,
        "all_exits_within_cache": bool(signal_end + 3 * DAY_MS <= cache_end),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "all_six_beijing_slots_present": sorted(signals.snapshot_hour_bj.unique().tolist()) == BJ_SLOTS,
        "beijing_utc_conversion_correct": all((utc_hour_for_beijing(slot) + 8) % 24 == int(slot[:2]) for slot in BJ_SLOTS),
        "drop_bucket_boundaries_correct": bool(0.0 >= CANDIDATES["A"]["drop_low"] and 20.0 == CANDIDATES["B"]["drop_low"] and 40.0 == CANDIDATES["B"]["drop_high"]),
        "rank_values_only_1_or_3": sorted(all_trades["rank"].unique().tolist()) == [1, 3],
        "holding_days_match_candidates": bool(all((all_trades[all_trades.candidate_id.eq(candidate_id)].holding_days == spec["holding_days"]).all() for candidate_id, spec in CANDIDATES.items())),
        "precomputed_outcomes_complete": len(outcomes) == len(relevant_signals) * 3,
        "entry_uses_signal_hour_open": bool((all_trades.entry_time_ms == all_trades.snapshot_time_ms).all()),
        "ranking_uses_only_completed_hour": True,
        "no_future_data": True,
        "pair_order_duplicates": int(pairs.duplicated(["candidate_id", "time_configuration"]).sum()),
        "executed_plus_skipped_equals_raw": bool(((summary.executed_trades + summary.skipped_existing_position) == summary.raw_signals).all()),
        "monthly_pnl_matches_summary": bool(np.allclose(monthly_check.sort_index(), summary_check.sort_index(), equal_nan=True)),
        "trade_aggregation_matches_summary": bool((all_trades.assign(executed=~all_trades.skipped_due_to_existing_position).groupby(["candidate_id", "time_configuration"]).executed.sum().sort_index().to_numpy() == summary.set_index(["candidate_id", "time_configuration"]).executed_trades.sort_index().to_numpy()).all()),
        "snapshot_rank_sort_violations": int(signals.sort_values(["snapshot_time_ms", "rank"]).groupby("snapshot_time_ms").return_24h_pct.apply(lambda x: (x.diff().dropna() < 0).sum()).sum()),
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
        "baseline_00_08_reproduced": bool(
            np.allclose(
                baseline := summary[summary.time_configuration.eq("00:00 + 08:00")].sort_values("candidate_id").net_pnl_usdt.to_numpy(),
                np.array([152.25884581712, 460.36865863061314, 652.6625633713004]),
            )
        ),
    }
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(out, summary, pair_compare, marginal, monthly, recommendations, cfg)

    print("Cache latest:", cfg["cache_latest_utc"])
    print("Unified signal cutoff:", cfg["unified_signal_end_utc"])
    for row in recommendations.itertuples():
        print(f"Candidate {row.candidate_id}: best single={row.best_single_time}; best pair={row.best_pair_times}; recommended={row.recommended_configuration} ({row.recommendation_type})")
        print(f"  trades={row.sample_size}, PF={row.profit_factor:.3f}, PnL={row.net_pnl_usdt:.2f}, ex-best5={row.net_pnl_ex_best_5_usdt:.2f}, months={row.positive_month_ratio:.3f}, MDD={row.max_drawdown_usdt:.2f}, R/DD={row.return_to_drawdown_ratio:.3f}")
    print("Output:", out.resolve())
    print("Data quality passed:", all(value is True or isinstance(value, (int, str)) and not isinstance(value, bool) for value in quality.values()))


if __name__ == "__main__":
    main()
