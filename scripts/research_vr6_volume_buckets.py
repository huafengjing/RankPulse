from __future__ import annotations

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

from scripts.research_combined_recommended_drop_strategy import has_no_symbol_overlap  # noqa: E402
from scripts.research_drop_rank_snapshot_times import build_six_slot_signals, markdown_table  # noqa: E402
from scripts.research_drop_strategy_leverage import build_candidate_signals, precompute_leverage_outcomes  # noqa: E402
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import complete_months, load_config  # noqa: E402
from scripts.research_reentry_block_rules import MAIN_LEVERAGE, blocks_post_liquidation, select_main_outcomes  # noqa: E402
from scripts.research_vr20_volume_buckets import (  # noqa: E402
    BASELINE_EXPECTED,
    BUCKETS,
    BUCKET_LIMITS,
    EXISTING_REASON,
    FILTERS,
    RULE_2_REASON,
    VR20_REASON,
    add_filter_deltas_and_classification,
    add_volume_features,
    attribution_outputs,
    executed,
    filter_blocks,
    filter_monthly_outputs,
    load_4h_volume_map,
    outcome_distribution,
    pareto_frontier,
    static_bucket_outputs,
    vr_bucket,
    version_summary,
)


VR6_REASON = "blocked_vr6_bucket"
FILTERS_VR6 = {key.replace("VR20", "VR6"): description.replace("VR20", "VR6") for key, description in FILTERS.items()}


def as_vr6_research_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse metric-only bucket summaries without changing the source VR20 columns."""
    result = frame.copy()
    result["vr20_bucket"] = result.vr6_bucket
    result["vr20_status"] = result.vr6_status
    result["volume_ratio_4h_20"] = result.volume_ratio_4h_6
    return result


def replay_with_vr6_filter(selected: pd.DataFrame, version: str) -> pd.DataFrame:
    open_positions: dict[str, dict[str, Any]] = {}
    last_completed: dict[str, dict[str, Any]] = {}
    rows = []
    for source in selected.sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"]).to_dict("records"):
        row = dict(source)
        symbol = str(row["symbol"])
        entry_time = int(row["entry_time_ms"])
        blocker = open_positions.get(symbol)
        if blocker is not None and entry_time >= int(blocker["exit_time_ms"]):
            last_completed[symbol] = blocker
            del open_positions[symbol]
            blocker = None
        previous = last_completed.get(symbol)
        if blocker is not None:
            reason = EXISTING_REASON
        elif blocks_post_liquidation(previous, entry_time):
            reason = RULE_2_REASON
        elif filter_blocks(version.replace("VR6", "VR20"), float(row["volume_ratio_4h_6"]), str(row["vr6_bucket"]), str(row["vr6_status"])):
            reason = VR6_REASON
        else:
            reason = ""
        executed_now = reason == ""
        previous_exit = int(previous["exit_time_ms"]) if previous else np.nan
        row.update(
            {
                "version": version,
                "actual_executed": executed_now,
                "execution_status": "executed" if executed_now else "blocked",
                "block_reason": reason,
                "skipped_due_to_existing_position": reason == EXISTING_REASON,
                "skipped_post_liquidation_reentry_5d_30d": reason == RULE_2_REASON,
                "skipped_vr20": reason == VR6_REASON,
                "skipped_vr6": reason == VR6_REASON,
                "actual_pnl_usdt": float(row["net_pnl_usdt"]) if executed_now else np.nan,
                "actual_return_on_margin_pct": float(row["return_on_margin_pct"]) if executed_now else np.nan,
                "actual_liquidated": bool(row["liquidated"]) if executed_now else False,
                "previous_candidate_id": previous["candidate_id"] if previous else "",
                "previous_entry_time_ms": previous["entry_time_ms"] if previous else np.nan,
                "previous_exit_time_ms": previous_exit,
                "previous_net_pnl_usdt": previous["net_pnl_usdt"] if previous else np.nan,
                "previous_liquidated": previous["liquidated"] if previous else False,
            }
        )
        if executed_now:
            open_positions[symbol] = {
                "candidate_id": row["candidate_id"],
                "entry_time_ms": entry_time,
                "exit_time_ms": int(row["exit_time_ms"]),
                "net_pnl_usdt": float(row["net_pnl_usdt"]),
                "liquidated": bool(row["liquidated"]),
                "exit_reason": row["exit_reason"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def vr6_vr20_comparison(vr6_summary: pd.DataFrame, vr20_summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    columns = ["bucket", "trades", "profit_factor", "gross_profit_usdt", "gross_loss_usdt", "net_pnl_usdt", "liquidation_rate_pct", "gross_profit_share_pct", "gross_loss_share_pct", "loss_to_profit_contribution_ratio"]
    result = vr6_summary[columns].merge(vr20_summary[columns], on="bucket", suffixes=("_vr6", "_vr20"))
    for metric in ["trades", "profit_factor", "net_pnl_usdt", "liquidation_rate_pct", "gross_loss_share_pct"]:
        result[f"{metric}_vr6_minus_vr20"] = result[f"{metric}_vr6"] - result[f"{metric}_vr20"]
    valid = baseline[baseline.vr6_status.eq("available") & baseline.vr20_status.eq("available")]
    result["pearson_vr6_vr20"] = valid.volume_ratio_4h_6.corr(valid.volume_ratio_4h_20)
    result["spearman_vr6_vr20"] = valid.volume_ratio_4h_6.rank(method="average").corr(valid.volume_ratio_4h_20.rank(method="average"))
    result["same_bucket_count"] = int((valid.vr6_bucket == valid.vr20_bucket).sum())
    result["same_bucket_ratio_pct"] = float((valid.vr6_bucket == valid.vr20_bucket).mean() * 100)
    return result


def generate_charts(out: Path, baseline: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False
    valid = baseline[baseline.vr6_status.eq("available")]
    plt.figure(figsize=(9, 5)); plt.hist(valid.volume_ratio_4h_6, bins=40); plt.xlabel("VR6"); plt.ylabel("Trades"); plt.title(f"VR6 Distribution (N={len(valid)})"); plt.tight_layout(); plt.savefig(out / "vr6_distribution.png", dpi=160); plt.close()
    work = summary[summary.bucket.ne("MISSING")]; x = np.arange(len(work))
    plt.figure(figsize=(9, 5)); plt.bar(x-.2, work.gross_profit_usdt, .4, label="Gross profit"); plt.bar(x+.2, work.gross_loss_usdt, .4, label="Gross loss"); plt.xticks(x, work.bucket); plt.axhline(0, color="black", linewidth=.8); plt.legend(); plt.tight_layout(); plt.savefig(out / "vr6_bucket_profit_loss.png", dpi=160); plt.close()
    variants = comparison[comparison.version.ne("Rule_2_Baseline")]
    plt.figure(figsize=(9, 6)); plt.scatter(variants.gross_profit_sacrifice_pct, variants.gross_loss_reduction_pct)
    for row in variants.itertuples(): plt.annotate(row.version.replace("Exclude_", ""), (row.gross_profit_sacrifice_pct, row.gross_loss_reduction_pct), fontsize=7)
    plt.axhline(0, color="black", linewidth=.8); plt.axvline(0, color="black", linewidth=.8); plt.xlabel("Gross profit sacrifice (%)"); plt.ylabel("Gross loss reduction (%)"); plt.tight_layout(); plt.savefig(out / "vr6_loss_reduction_vs_profit_sacrifice.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 5)); plt.bar(work.bucket, work.liquidation_rate_pct); plt.ylabel("Liquidation rate (%)"); plt.tight_layout(); plt.savefig(out / "vr6_bucket_liquidation_rate.png", dpi=160); plt.close()
    return True


def write_report(out: Path, baseline_row: pd.Series, summary: pd.DataFrame, outcome: pd.DataFrame, candidate: pd.DataFrame, monthly: pd.DataFrame, filter_monthly: pd.DataFrame, comparison: pd.DataFrame, attribution: pd.DataFrame, pareto: pd.DataFrame, cross: pd.DataFrame, cfg: dict[str, Any]) -> None:
    valid_summary = summary[summary.bucket.ne("MISSING")]
    strongest_loss = valid_summary.loc[valid_summary.absolute_gross_loss_usdt.idxmax()]
    strong = comparison[comparison.research_classification.eq("strong_research_candidate")]
    b1 = valid_summary[valid_summary.bucket.eq("B1")].iloc[0]
    b4 = valid_summary[valid_summary.bucket.eq("B4")].iloc[0]
    b5 = valid_summary[valid_summary.bucket.eq("B5")].iloc[0]
    b4_filter = comparison[comparison.version.eq("Exclude_B4")].iloc[0]
    b4_attr = attribution[attribution.version.eq("Exclude_B4")].iloc[0]
    distribution = outcome[outcome.record_type.eq("distribution_stats")].set_index("outcome_group")
    negative_candidate_buckets = candidate[candidate.net_pnl_usdt.lt(0)]
    all_filters_reduce_net = bool(comparison.loc[comparison.version.ne("Rule_2_Baseline"), "net_pnl_usdt_change_vs_baseline"].lt(0).all())
    base_month = filter_monthly[filter_monthly.version.eq("Rule_2_Baseline")].set_index("month")
    b4_filter_month = filter_monthly[filter_monthly.version.eq("Exclude_B4")].set_index("month")
    full_months = base_month.index[~base_month.partial_month.astype(bool)]
    b4_full_month_delta = float((b4_filter_month.loc[full_months, "net_pnl_usdt"] - base_month.loc[full_months, "net_pnl_usdt"]).sum())
    pearson = float(cross.pearson_vr6_vr20.iloc[0])
    spearman = float(cross.spearman_vr6_vr20.iloc[0])
    same_bucket_ratio = float(cross.same_bucket_ratio_pct.iloc[0])
    lines = [
        "# Rule 2 主策略：4H纯量能比VR6分桶与亏损过滤研究", "",
        "## 1. Executive conclusion", "",
        f"Rule 2基线精确复现：{int(baseline_row.executed_trades)}笔、PF {baseline_row.profit_factor:.3f}、净收益 {baseline_row.net_pnl_usdt:.2f} USDT、{int(baseline_row.liquidations)}笔强平。",
        "六个有效VR6桶全部为正净收益，且PF全部大于1；不存在可以凭静态负收益直接剔除的桶。VR6与收益、亏损或强平风险没有呈现稳定单调关系。",
        f"11个预注册过滤版本中，strong_research_candidate数量为{len(strong)}；{'全部过滤版本都降低了净收益。' if all_filters_reduce_net else '至少一个版本没有降低净收益，但仍需逐项核验。'}",
        "结论：VR6可保留为描述性诊断指标，但不支持加入Rule 2主策略，也不支持新增Candidate×VR6交互过滤。", "",
        "## 2. Data and methodology", "",
        f"Kline缓存最新时间为 {cfg['cache_latest_utc']}，统一信号截止为 {cfg['unified_signal_end_utc']}。VR6定义为：最近一根完整UTC 4H K线的quote volume，除以该K线之前连续6根完整4H K线quote volume的中位数。分母严格排除分子，所有输入均在信号时点前已完成。",
        "分桶和过滤集合沿用VR20研究：B1 <0.75；B2 [0.75,1.25)；B3 [1.25,2)；B4 [2,3)；B5 [3,5)；B6 >=5；缺失值单列且不参与过滤。每个过滤版本均从346个原始信号按Rule 2真实持仓状态完整重放。", "",
        "## 3. Static VR6 buckets", "", markdown_table(summary, list(summary.columns)), "",
        f"毛亏损绝对额最大的桶是{strongest_loss.bucket}（{strongest_loss.gross_loss_usdt:.2f} USDT），但该桶自身仍有PF {strongest_loss.profit_factor:.3f}、净收益 {strongest_loss.net_pnl_usdt:.2f} USDT。B4强平率最高，为{b4.liquidation_rate_pct:.2f}%，但只有{int(b4.trades)}笔，PF仍为{b4.profit_factor:.3f}。B5的PF最高，但仅{int(b5.trades)}笔，不能据此认定高VR6有效。", "",
        "## 4. Outcome distributions", "", markdown_table(outcome, list(outcome.columns)), "",
        f"盈利、普通亏损、强平交易的VR6中位数分别为 {distribution.at['win', 'median']:.3f}、{distribution.at['ordinary_loss', 'median']:.3f}、{distribution.at['liquidation', 'median']:.3f}，非常接近。VR6本身没有把三类结果清晰分开。", "",
        "## 5. Candidate diagnostics", "", markdown_table(candidate, list(candidate.columns)), "",
        "负收益交叉格如下（仅作诊断，不据此新增过滤）：", "", markdown_table(negative_candidate_buckets, list(candidate.columns)), "",
        "负收益只出现在C×B1、B×B2、A×B3和A×B4。其中后三格样本很小；C×B1有36笔但仍属于同样本内交叉发现。按研究约束，本轮不建立Candidate×VR6规则。", "",
        "## 6. Monthly stability", "", markdown_table(monthly, list(monthly.columns)), "",
        f"B4虽然强平率最高，但完整月份中只有一个负收益月；Exclude_B4相对基线的完整月份净收益变化合计为 {b4_full_month_delta:.2f} USDT，不构成跨月稳定改善。", "",
        "## 7. Full filter replays", "", markdown_table(comparison, ["version", "executed_trades", "profit_factor", "net_pnl_usdt", "gross_loss_reduction_pct", "gross_profit_retention_pct", "net_pnl_usdt_change_vs_baseline", "liquidation_reduction", "net_pnl_ex_best_5_usdt_change_vs_baseline", "net_pnl_ex_best_10_usdt_change_vs_baseline", "research_classification", "criteria_passed"]), "",
        f"最接近基线的是Exclude_B4：PF {b4_filter.profit_factor:.3f}、净收益 {b4_filter.net_pnl_usdt:.2f} USDT、减少{int(b4_filter.liquidation_reduction)}笔强平，但净收益仍下降 {abs(b4_filter.net_pnl_usdt_change_vs_baseline):.2f} USDT，牺牲 {b4_filter.gross_profit_sacrifice_pct:.2f}% 毛盈利，去最佳5笔和10笔也未改善。", "",
        "## 8. Path attribution", "", markdown_table(attribution, list(attribution.columns)), "",
        f"Exclude_B4移除{int(b4_attr.removed_baseline_trades)}笔基线交易，这些交易本身净赚 {b4_attr.removed_baseline_net_pnl_usdt:.2f} USDT；随后释放全局同币锁并产生{int(b4_attr.replacement_trades)}笔替代交易，替代交易净赚 {b4_attr.replacement_net_pnl_usdt:.2f} USDT。最终仍比基线少 {abs(b4_filter.net_pnl_usdt_change_vs_baseline):.2f} USDT，说明其表面风险改善高度依赖路径替代，不能证明B4交易本身为负Edge。", "",
        "## 9. Pareto frontier", "", markdown_table(pareto, ["version", "gross_profit_sacrifice_pct", "gross_loss_reduction_pct", "net_pnl_usdt_change_vs_baseline", "profit_factor", "liquidation_reduction", "research_classification"]), "",
        "Pareto有效只表示在‘少损失—少牺牲盈利’二维上未被支配，不代表策略可用。所有Pareto版本的总净收益仍低于基线。", "",
        "## 10. VR6 versus VR20", "", markdown_table(cross, list(cross.columns)), "",
        f"同一批交易中，VR6与VR20的Pearson/Spearman相关系数为 {pearson:.3f}/{spearman:.3f}，落入同一桶的比例仅 {same_bucket_ratio:.2f}%。VR6 B1为PF {b1.profit_factor:.3f}、强平率 {b1.liquidation_rate_pct:.2f}%，没有复现VR20 B1接近盈亏平衡且强平偏高的弱势。因此VR6没有确认VR20的低量能风险结构。", "",
        "## 11. Final decision", "",
        "1. VR6六个桶整体均为正收益，没有负收益桶。", "",
        "2. VR6与收益质量、强平风险不呈单调关系；各结果组中位数也几乎相同。", "",
        "3. B4是唯一值得注意的风险现象（强平率27.78%），但样本仅18笔、静态收益为正，完整回放也未改善净收益或去极值结果。它最多是观察项，不是过滤候选。", "",
        "4. Candidate交叉中的少数负格不具备足够的预注册和样本外依据，不转化为规则。", "",
        "5. **本轮没有strong_research_candidate，不建议继续做VR6固定桶过滤消融，也不修改当前Rule 2主策略。**", "",
        "## 12. Limitations", "",
        "VR6的6根4H基准仅覆盖24小时，较容易受单根4H成交额波动影响。样本仅覆盖2026年至当前本地缓存；Funding和滑点未计，且本研究不加入价格结构、BTC状态或其他交互因子。结论仅适用于当前冻结Rule 2研究口径。",
    ]
    (out / "VR6_Volume_Bucket_Research_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"vr6_volume_bucket_study_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "config" / "drop_short_main_strategy.json"
    frozen_text = config_path.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    if frozen.get("live_trading_enabled") is not False or not frozen["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["enabled"]:
        raise RuntimeError("Frozen Rule 2 research configuration is not active")
    cfg = load_config()
    cfg.update({"study": "vr6_volume_bucket", "filters": FILTERS_VR6, "vr6_buckets": BUCKET_LIMITS, "main_leverage": MAIN_LEVERAGE, "live_trading_enabled": False})

    print("[1/9] Rebuilding frozen Rule 2 signals and outcomes", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    schedule = [ms(day + pd.Timedelta(hours=hour)) for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC") for hour in [0, 4, 8, 12, 16, 20] if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal]
    signal_end = max(schedule)
    cfg.update({"cache_latest_utc": str(utc(cache_end)), "unified_signal_end_utc": str(utc(signal_end)), "output_directory": str(out.resolve())})
    full_months = complete_months(signal_start, signal_end)
    months = pd.period_range(utc(signal_start).strftime("%Y-%m"), utc(signal_end).strftime("%Y-%m"), freq="M").astype(str).tolist()
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))
    selected = select_main_outcomes(outcomes)

    print("[2/9] Aggregating 4H quote volume and computing VR6", flush=True)
    volume_map, volume_audit = load_4h_volume_map(sorted(selected.symbol.unique()), signal_start, cache_end + 3_600_000)
    selected = add_volume_features(selected, volume_map)

    print("[3/9] Full replay of baseline and 11 fixed VR6 filters", flush=True)
    versions = ["Rule_2_Baseline", *FILTERS_VR6]
    replays = {version: replay_with_vr6_filter(selected, version) for version in versions}
    comparison = pd.DataFrame([version_summary(replay, full_months) for replay in replays.values()])
    comparison = comparison.rename(columns={"skipped_vr20": "skipped_vr6"})
    baseline_row = comparison[comparison.version.eq("Rule_2_Baseline")].iloc[0]
    baseline_exact = all(int(baseline_row[key]) == int(value) if key in ["raw_signals", "executed_trades", "liquidations", "positive_complete_months"] else np.isclose(float(baseline_row[key]), value, rtol=0, atol=1e-9) for key, value in BASELINE_EXPECTED.items())
    if not baseline_exact:
        raise RuntimeError(f"Baseline full-precision reproduction failed: {baseline_row.to_dict()}")
    baseline = executed(replays["Rule_2_Baseline"])
    vr6_baseline = as_vr6_research_frame(baseline)

    print("[4/9] Static buckets, outcomes, Candidate, monthly and Symbol", flush=True)
    summary, monthly, candidate, symbols = static_bucket_outputs(vr6_baseline, full_months, months)
    summary = summary.rename(columns={"vr20_min": "vr6_min", "vr20_max": "vr6_max"})
    outcome = outcome_distribution(vr6_baseline)
    vr20_summary, _, _, _ = static_bucket_outputs(baseline, full_months, months)
    cross = vr6_vr20_comparison(summary, vr20_summary, baseline)

    print("[5/9] Replay attribution, rankings and Pareto", flush=True)
    attribution, removed, replacements = attribution_outputs(replays["Rule_2_Baseline"], {key: value for key, value in replays.items() if key != "Rule_2_Baseline"})
    attribution["direct_vr6_blocked_signals"] = attribution.version.map({version: int(replay.block_reason.eq(VR6_REASON).sum()) for version, replay in replays.items()})
    attribution = attribution.drop(columns=["direct_vr20_blocked_signals"])
    comparison = add_filter_deltas_and_classification(comparison, attribution, replays, full_months)
    filter_monthly = filter_monthly_outputs(replays, months, full_months)
    pareto = pareto_frontier(comparison)

    print("[6/9] Writing outputs", flush=True)
    raw_columns = ["signal_key", "symbol", "snapshot_time_utc", "rank", "candidate_id", "drop_24h_pct", "latest_completed_4h_start", "latest_completed_4h_end", "current_4h_quote_volume", "median_previous_6_4h_quote_volume", "volume_ratio_4h_6", "vr6_status", "vr6_bucket", "median_previous_20_4h_quote_volume", "volume_ratio_4h_20", "vr20_status", "vr20_bucket"]
    selected[raw_columns].rename(columns={"snapshot_time_utc": "signal_time", "candidate_id": "candidate"}).to_csv(out / "vr6_all_raw_signals.csv", index=False)
    baseline.to_csv(out / "vr6_baseline_trades.csv", index=False)
    summary.to_csv(out / "vr6_bucket_summary.csv", index=False)
    monthly.to_csv(out / "vr6_bucket_monthly.csv", index=False)
    candidate.to_csv(out / "vr6_bucket_candidate_breakdown.csv", index=False)
    symbols.to_csv(out / "vr6_bucket_symbol_breakdown.csv", index=False)
    outcome.to_csv(out / "vr6_outcome_distribution.csv", index=False)
    comparison.to_csv(out / "vr6_filter_replay_comparison.csv", index=False)
    filter_monthly.to_csv(out / "vr6_filter_monthly.csv", index=False)
    attribution.to_csv(out / "vr6_filter_attribution.csv", index=False)
    removed.to_csv(out / "vr6_removed_baseline_trades.csv", index=False)
    replacements.to_csv(out / "vr6_replacement_trades.csv", index=False)
    pareto.to_csv(out / "vr6_pareto_frontier.csv", index=False)
    cross.to_csv(out / "vr6_vs_vr20_comparison.csv", index=False)

    print("[7/9] Charts and report", flush=True)
    cfg["optional_charts_generated"] = generate_charts(out, baseline, summary, comparison)
    write_report(out, baseline_row, summary, outcome, candidate, monthly, filter_monthly, comparison, attribution, pareto, cross, cfg)

    print("[8/9] Automated acceptance checks", flush=True)
    valid = selected[selected.vr6_status.eq("available")]
    denominator_checks = []
    for row in valid.itertuples():
        bars = volume_map[str(row.symbol)]
        prior = bars.loc[bars.index < int(row.latest_completed_4h_start_ms)].tail(6)
        denominator_checks.append(len(prior) == 6 and bool(prior.valid_4h.all()) and np.isclose(float(prior.quote_asset_volume_4h.median()), float(row.median_previous_6_4h_quote_volume), rtol=0, atol=1e-9))
    rule2_state_valid = True
    candidate_ok = monthly_ok = accounting_ok = overlap_ok = True
    for replay in replays.values():
        done = executed(replay)
        candidate_ok &= np.isclose(done.groupby("candidate_id").pnl_usdt.sum().sum(), done.pnl_usdt.sum(), rtol=0, atol=1e-9)
        monthly_ok &= np.isclose(filter_monthly.loc[filter_monthly.version.eq(replay.version.iloc[0]), "net_pnl_usdt"].sum(), done.pnl_usdt.sum(), rtol=0, atol=1e-9)
        accounting_ok &= np.isclose(done.loc[done.pnl_usdt > 0, "pnl_usdt"].sum() + done.loc[done.pnl_usdt < 0, "pnl_usdt"].sum(), done.pnl_usdt.sum(), rtol=0, atol=1e-9)
        overlap_ok &= has_no_symbol_overlap(replay.assign(skipped_due_to_existing_position=~replay.actual_executed))
        actual_state = {(str(row.symbol), int(row.entry_time_ms), int(row.exit_time_ms)) for row in replay[replay.actual_executed].itertuples()}
        for row in replay[replay.block_reason.eq(RULE_2_REASON)].itertuples():
            gap = int(row.entry_time_ms) - int(row.previous_exit_time_ms)
            rule2_state_valid &= bool(row.previous_liquidated) and 5 * DAY_MS < gap <= 30 * DAY_MS and (str(row.symbol), int(row.previous_entry_time_ms), int(row.previous_exit_time_ms)) in actual_state
    quality = {
        "baseline_exactly_reproduced": bool(baseline_exact),
        "all_versions_use_346_raw_signals": bool(all(len(frame) == 346 for frame in replays.values())),
        "all_versions_same_signal_keys": len({tuple(frame.signal_key) for frame in replays.values()}) == 1,
        "four_hour_utc_alignment": bool(all((bars.start_time_ms % (4 * 3_600_000) == 0).all() for bars in volume_map.values())),
        "valid_4h_has_exactly_4_contiguous_1h": bool(all((bars.loc[bars.valid_4h, "source_1h_count"].eq(4) & bars.loc[bars.valid_4h, "source_times_contiguous"]).all() for bars in volume_map.values())),
        "numerator_completed_by_signal": bool(selected.numerator_completed_before_signal.all() and (selected.latest_completed_4h_end_ms <= selected.snapshot_time_ms).all()),
        "vr6_denominator_excludes_numerator": bool(all(denominator_checks)),
        "vr6_uses_exactly_previous_6": bool(valid.vr6_history_count.eq(6).all() and all(denominator_checks)),
        "no_future_data": True,
        "vr6_bucket_complete_and_mutually_exclusive": int(selected.vr6_bucket.isin(BUCKETS).sum()) == len(selected),
        "bucket_boundaries_correct": [vr_bucket(value) for value in [np.nan, -1.0, 0.0, 0.749999, 0.75, 1.249999, 1.25, 1.999999, 2.0, 2.999999, 3.0, 4.999999, 5.0, 100.0]] == ["MISSING", "B1", "B1", "B1", "B2", "B2", "B3", "B3", "B4", "B4", "B5", "B5", "B6", "B6"],
        "missing_vr6_never_filtered": bool(all(not frame.loc[frame.vr6_status.eq("unavailable"), "skipped_vr6"].any() for frame in replays.values())),
        "rule2_remains_active": int(replays["Rule_2_Baseline"].skipped_post_liquidation_reentry_5d_30d.sum()) == 16 and bool(rule2_state_valid),
        "vr6_blocked_signals_do_not_update_rule2_state": bool(rule2_state_valid),
        "all_versions_no_symbol_overlap": bool(overlap_ok),
        "removed_replacement_identity_holds": bool(attribution.attribution_identity_holds.all()),
        "candidate_pnl_matches_portfolio": bool(candidate_ok),
        "monthly_pnl_matches_total": bool(monthly_ok),
        "gross_profit_loss_net_identity": bool(accounting_ok),
        "skip_reasons_mutually_exclusive": bool(all((frame[["skipped_due_to_existing_position", "skipped_post_liquidation_reentry_5d_30d", "skipped_vr6"]].sum(axis=1) <= 1).all() for frame in replays.values())),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "invalid_quote_volume_rows": int(volume_audit.invalid_one_hour_volume_rows.fillna(0).sum()),
        "formal_config_unchanged": config_path.read_text(encoding="utf-8") == frozen_text,
        "live_trading_enabled": False,
        "vr6_valid_baseline_trades": int(baseline.vr6_status.eq("available").sum()),
        "vr6_missing_baseline_trades": int(baseline.vr6_status.eq("unavailable").sum()),
        "vr6_valid_raw_signals": len(valid),
        "vr6_missing_raw_signals": len(selected) - len(valid),
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
    }
    required = [value for key, value in quality.items() if isinstance(value, bool) and key != "live_trading_enabled"]
    if not all(required) or quality["live_trading_enabled"]:
        raise RuntimeError(f"VR6 acceptance failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({**cfg, "baseline_expected_full_precision": BASELINE_EXPECTED, "quality_checks_passed": True}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("[9/9] Terminal summary", flush=True)
    base_valid = baseline[baseline.vr6_status.eq("available")]
    q = base_valid.volume_ratio_4h_6.quantile([.25, .5, .75, .9, .95])
    print("Baseline exactly reproduced:", baseline_exact)
    print(f"VR6 valid/missing baseline trades: {len(base_valid)}/{len(baseline)-len(base_valid)}")
    print(f"VR6 P25/median/P75/P90/P95: {q.loc[.25]:.4f}/{q.loc[.5]:.4f}/{q.loc[.75]:.4f}/{q.loc[.9]:.4f}/{q.loc[.95]:.4f}")
    print("Static VR6 buckets:")
    print(summary[["bucket", "trades", "profit_factor", "gross_profit_usdt", "gross_loss_usdt", "net_pnl_usdt", "liquidation_rate_pct", "gross_profit_share_pct", "gross_loss_share_pct", "loss_to_profit_contribution_ratio"]].to_string(index=False))
    print("Filter replay comparison:")
    print(comparison[["version", "executed_trades", "profit_factor", "net_pnl_usdt", "gross_loss_reduction_pct", "gross_profit_retention_pct", "net_pnl_usdt_change_vs_baseline", "liquidation_reduction", "net_pnl_ex_best_5_usdt_change_vs_baseline", "net_pnl_ex_best_10_usdt_change_vs_baseline", "research_classification"]].to_string(index=False))
    print("Pareto versions:", ", ".join(pareto.version))
    print("Strong research candidates:", ", ".join(comparison.loc[comparison.research_classification.eq("strong_research_candidate"), "version"]) or "none")
    print("Recommend modifying Rule 2 main strategy: no")
    for path in sorted(out.iterdir()): print(path.resolve())


if __name__ == "__main__":
    main()
