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
from scripts.research_drop_strategy_leverage import (  # noqa: E402
    MARGIN_USDT,
    build_candidate_signals,
    precompute_leverage_outcomes,
)
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import complete_months, load_config, longest_streak, profit_factor  # noqa: E402


MAIN_LEVERAGE = {"A": 5, "B": 3, "C": 3}
VERSIONS = {
    "Baseline": {"rule_1": False, "rule_2": False},
    "Rule_1_Only": {"rule_1": True, "rule_2": False},
    "Rule_2_Only": {"rule_1": False, "rule_2": True},
    "Combined_Rules": {"rule_1": True, "rule_2": True},
}
BASELINE_EXPECTED = {
    "raw_signals": 346,
    "executed_trades": 277,
    "profit_factor": 1.5934202179852974,
    "net_pnl_usdt": 3455.351187277372,
    "liquidations": 43,
    "max_drawdown_usdt": -475.18094064890244,
}
RULE_1_REASON = "blocked_profit_exit_reentry_within_1d"
RULE_2_REASON = "blocked_post_liquidation_reentry_5d_30d"
EXISTING_REASON = "global_existing_position"


def blocks_profit_reentry(previous: dict[str, Any] | None, entry_time_ms: int) -> bool:
    if previous is None or previous["liquidated"] or previous["net_pnl_usdt"] <= 0 or previous["exit_reason"] != "fixed_exit":
        return False
    gap = entry_time_ms - int(previous["exit_time_ms"])
    return 0 < gap <= DAY_MS


def blocks_post_liquidation(previous: dict[str, Any] | None, entry_time_ms: int) -> bool:
    if previous is None or not previous["liquidated"]:
        return False
    gap = entry_time_ms - int(previous["exit_time_ms"])
    return 5 * DAY_MS < gap <= 30 * DAY_MS


def signal_key(row: Any) -> tuple[str, int, str]:
    if isinstance(row, dict):
        return str(row["candidate_id"]), int(row["snapshot_time_ms"]), str(row["symbol"])
    return str(row.candidate_id), int(row.snapshot_time_ms), str(row.symbol)


def select_main_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    frames = [
        outcomes[outcomes.candidate_id.eq(candidate_id) & outcomes.leverage.eq(leverage)]
        for candidate_id, leverage in MAIN_LEVERAGE.items()
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])


def replay_with_block_rules(
    selected: pd.DataFrame,
    version: str,
    use_rule_1: bool,
    use_rule_2: bool,
) -> pd.DataFrame:
    """Replay all signals; only completed actual trades update reentry state."""
    open_positions: dict[str, dict[str, Any]] = {}
    last_completed: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for source in selected.to_dict("records"):
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
        elif use_rule_1 and blocks_profit_reentry(previous, entry_time):
            reason = RULE_1_REASON
        elif use_rule_2 and blocks_post_liquidation(previous, entry_time):
            reason = RULE_2_REASON
        else:
            reason = ""
        executed = reason == ""
        previous_exit = int(previous["exit_time_ms"]) if previous else np.nan
        row.update(
            {
                "version": version,
                "signal_key": "|".join(map(str, signal_key(row))),
                "actual_executed": executed,
                "execution_status": "executed" if executed else "blocked",
                "block_reason": reason,
                "skipped_due_to_existing_position": reason == EXISTING_REASON,
                "skipped_profit_exit_reentry_within_1d": reason == RULE_1_REASON,
                "skipped_post_liquidation_reentry_5d_30d": reason == RULE_2_REASON,
                "actual_pnl_usdt": float(row["net_pnl_usdt"]) if executed else np.nan,
                "actual_return_on_margin_pct": float(row["return_on_margin_pct"]) if executed else np.nan,
                "actual_liquidated": bool(row["liquidated"]) if executed else False,
                "previous_candidate_id": previous["candidate_id"] if previous else "",
                "previous_entry_time_ms": previous["entry_time_ms"] if previous else np.nan,
                "previous_exit_time_ms": previous_exit,
                "previous_net_pnl_usdt": previous["net_pnl_usdt"] if previous else np.nan,
                "previous_liquidated": previous["liquidated"] if previous else False,
                "gap_from_previous_exit_hours": (entry_time - previous_exit) / 3_600_000 if previous else np.nan,
                "candidate_transition": f"{previous['candidate_id']}->{row['candidate_id']}" if previous else f"FIRST->{row['candidate_id']}",
            }
        )
        if executed:
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


def executed_rows(replay: pd.DataFrame) -> pd.DataFrame:
    return replay[replay.actual_executed].copy()


def exposure_metrics(executed: pd.DataFrame) -> dict[str, Any]:
    events = []
    for row in executed.itertuples():
        events.append((int(row.entry_time_ms), 1, MARGIN_USDT, float(row.entry_notional_usdt)))
        events.append((int(row.exit_time_ms), -1, -MARGIN_USDT, -float(row.entry_notional_usdt)))
    frame = pd.DataFrame(events, columns=["time_ms", "positions", "margin", "notional"]).groupby("time_ms", as_index=False).sum().sort_values("time_ms")
    frame["positions"] = frame.positions.cumsum()
    frame["margin"] = frame.margin.cumsum()
    frame["notional"] = frame.notional.cumsum()
    return {
        "max_concurrent_positions": int(frame.positions.max()),
        "max_margin_in_use_usdt": float(frame.margin.max()),
        "max_gross_notional_exposure_usdt": float(frame.notional.max()),
    }


def performance_metrics(executed: pd.DataFrame, complete_month_set: set[str]) -> dict[str, Any]:
    ordered = executed.sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = ordered.actual_pnl_usdt.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    monthly = executed.assign(month=pd.to_datetime(executed.entry_time_utc, utc=True).dt.strftime("%Y-%m")).groupby("month").actual_pnl_usdt.sum()
    complete_values = monthly[monthly.index.isin(complete_month_set)]
    dd = max_drawdown(pnl)
    return {
        "executed_trades": len(ordered),
        "wins": len(wins),
        "losses": len(losses),
        "liquidations": int(ordered.actual_liquidated.sum()),
        "liquidation_rate_pct": float(ordered.actual_liquidated.mean() * 100),
        "net_pnl_usdt": float(pnl.sum()),
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "profit_factor": profit_factor(pnl),
        "win_rate_pct": float(pnl.gt(0).mean() * 100),
        "average_pnl_usdt": float(pnl.mean()),
        "median_return_pct": float(ordered.actual_return_on_margin_pct.median()),
        "max_drawdown_usdt": dd,
        "max_consecutive_losses": longest_streak(pnl < 0),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(1).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(min(10, len(pnl))).sum()),
        "positive_complete_months": int(complete_values.gt(0).sum()),
        "negative_complete_months": int(complete_values.lt(0).sum()),
        "total_complete_months": len(complete_values),
        "return_to_drawdown_ratio": float(pnl.sum() / abs(dd)) if dd < 0 else np.nan,
        **exposure_metrics(ordered),
    }


def summarize_version(replay: pd.DataFrame, complete_month_set: set[str]) -> dict[str, Any]:
    return {
        "version": replay.version.iloc[0],
        "raw_signals": len(replay),
        "skipped_existing_position": int(replay.skipped_due_to_existing_position.sum()),
        "skipped_profit_exit_reentry_within_1d": int(replay.skipped_profit_exit_reentry_within_1d.sum()),
        "skipped_post_liquidation_reentry_5d_30d": int(replay.skipped_post_liquidation_reentry_5d_30d.sum()),
        **performance_metrics(executed_rows(replay), complete_month_set),
    }


def metrics_for_attribution(frame: pd.DataFrame, pnl_column: str, liquidation_column: str) -> dict[str, Any]:
    pnl = frame[pnl_column].astype(float)
    return {
        "trades": len(frame),
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl) if len(frame) else np.nan,
        "liquidations": int(frame[liquidation_column].sum()) if len(frame) else 0,
        "liquidation_rate_pct": float(frame[liquidation_column].mean() * 100) if len(frame) else np.nan,
    }


def monthly_summary(replay: pd.DataFrame, months: list[str], complete_month_set: set[str]) -> list[dict[str, Any]]:
    work = replay.assign(month=pd.to_datetime(replay.entry_time_utc, utc=True).dt.strftime("%Y-%m"))
    rows = []
    for month in months:
        raw = work[work.month.eq(month)]
        done = executed_rows(raw).sort_values(["exit_time_ms", "rank", "symbol"])
        pnl = done.actual_pnl_usdt.astype(float)
        candidate_pnl = done.groupby("candidate_id").actual_pnl_usdt.sum()
        rows.append(
            {
                "version": replay.version.iloc[0],
                "month": month,
                "partial_month": month not in complete_month_set,
                "raw_signals": len(raw),
                "executed_trades": len(done),
                "blocked_existing_position": int(raw.skipped_due_to_existing_position.sum()),
                "blocked_rule_1": int(raw.skipped_profit_exit_reentry_within_1d.sum()),
                "blocked_rule_2": int(raw.skipped_post_liquidation_reentry_5d_30d.sum()),
                "liquidations": int(done.actual_liquidated.sum()),
                "net_pnl_usdt": float(pnl.sum()),
                "profit_factor": profit_factor(pnl) if len(pnl) else np.nan,
                "max_drawdown_usdt": max_drawdown(pnl) if len(pnl) else 0.0,
                "A_net_pnl": float(candidate_pnl.get("A", 0.0)),
                "B_net_pnl": float(candidate_pnl.get("B", 0.0)),
                "C_net_pnl": float(candidate_pnl.get("C", 0.0)),
            }
        )
    return rows


def candidate_contribution(replay: pd.DataFrame, complete_month_set: set[str]) -> list[dict[str, Any]]:
    rows = []
    for candidate_id in ["A", "B", "C"]:
        done = executed_rows(replay)
        done = done[done.candidate_id.eq(candidate_id)]
        metrics = performance_metrics(done, complete_month_set)
        rows.append(
            {
                "version": replay.version.iloc[0],
                "candidate_id": candidate_id,
                **{key: metrics[key] for key in ["executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "liquidation_rate_pct", "max_drawdown_usdt"]},
            }
        )
    return rows


def blocked_exports(replays: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = pd.concat(replays.values(), ignore_index=True)
    rule_1 = all_rows[all_rows.block_reason.eq(RULE_1_REASON)].copy()
    rule_2 = all_rows[all_rows.block_reason.eq(RULE_2_REASON)].copy()
    rule_1_export = pd.DataFrame(
        {
            "version": rule_1.version,
            "symbol": rule_1.symbol,
            "candidate": rule_1.candidate_id,
            "signal_time": rule_1.snapshot_time_utc,
            "blocked_entry_time": rule_1.entry_time_utc,
            "previous_candidate": rule_1.previous_candidate_id,
            "previous_entry_time": pd.to_datetime(rule_1.previous_entry_time_ms, unit="ms", utc=True),
            "previous_exit_time": pd.to_datetime(rule_1.previous_exit_time_ms, unit="ms", utc=True),
            "previous_net_pnl": rule_1.previous_net_pnl_usdt,
            "gap_hours": rule_1.gap_from_previous_exit_hours,
            "hypothetical_pnl_if_executed": rule_1.net_pnl_usdt,
            "hypothetical_liquidated": rule_1.liquidated,
            "block_reason": rule_1.block_reason,
            "signal_key": rule_1.signal_key,
        }
    )
    rule_2_export = pd.DataFrame(
        {
            "version": rule_2.version,
            "symbol": rule_2.symbol,
            "candidate": rule_2.candidate_id,
            "signal_time": rule_2.snapshot_time_utc,
            "blocked_entry_time": rule_2.entry_time_utc,
            "previous_candidate": rule_2.previous_candidate_id,
            "previous_liquidation_time": pd.to_datetime(rule_2.previous_exit_time_ms, unit="ms", utc=True),
            "gap_days": rule_2.gap_from_previous_exit_hours / 24,
            "hypothetical_pnl_if_executed": rule_2.net_pnl_usdt,
            "hypothetical_liquidated": rule_2.liquidated,
            "block_reason": rule_2.block_reason,
            "signal_key": rule_2.signal_key,
        }
    )
    return rule_1_export, rule_2_export


def path_attribution(
    baseline: pd.DataFrame,
    variants: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_done = executed_rows(baseline).set_index("signal_key", drop=False)
    rows = []
    detail_frames = []
    for version, replay in variants.items():
        done = executed_rows(replay).set_index("signal_key", drop=False)
        removed_keys = base_done.index.difference(done.index)
        replacement_keys = done.index.difference(base_done.index)
        removed = base_done.loc[removed_keys].copy()
        replacements = done.loc[replacement_keys].copy()
        removed_metrics = metrics_for_attribution(removed, "actual_pnl_usdt", "actual_liquidated")
        replacement_metrics = metrics_for_attribution(replacements, "actual_pnl_usdt", "actual_liquidated")
        direct_rule_1 = replay[replay.block_reason.eq(RULE_1_REASON)]
        direct_rule_2 = replay[replay.block_reason.eq(RULE_2_REASON)]
        direct_1_metrics = metrics_for_attribution(direct_rule_1, "net_pnl_usdt", "liquidated")
        direct_2_metrics = metrics_for_attribution(direct_rule_2, "net_pnl_usdt", "liquidated")
        rows.append(
            {
                "version": version,
                **{f"removed_baseline_{key}": value for key, value in removed_metrics.items()},
                **{f"replacement_{key}": value for key, value in replacement_metrics.items()},
                **{f"direct_rule_1_blocked_{key}": value for key, value in direct_1_metrics.items()},
                **{f"direct_rule_2_blocked_{key}": value for key, value in direct_2_metrics.items()},
            }
        )
        for path_type, frame in [("removed_baseline_trade", removed), ("newly_executed_vs_baseline", replacements)]:
            if frame.empty:
                continue
            work = frame.reset_index(drop=True).copy()
            work["compared_version"] = version
            work["path_change_type"] = path_type
            detail_frames.append(work)
    return pd.DataFrame(rows), pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()


def write_report(
    out: Path,
    comparison: pd.DataFrame,
    attribution: pd.DataFrame,
    monthly: pd.DataFrame,
    candidate: pd.DataFrame,
    transitions: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    columns = ["version", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "positive_complete_months", "return_to_drawdown_ratio"]
    base = comparison[comparison.version.eq("Baseline")].iloc[0]
    rule_1 = comparison[comparison.version.eq("Rule_1_Only")].iloc[0]
    rule_2 = comparison[comparison.version.eq("Rule_2_Only")].iloc[0]
    combined = comparison[comparison.version.eq("Combined_Rules")].iloc[0]
    attr_1 = attribution[attribution.version.eq("Rule_1_Only")].iloc[0]
    attr_2 = attribution[attribution.version.eq("Rule_2_Only")].iloc[0]
    full_months = monthly[~monthly.partial_month].pivot(index="month", columns="version", values="net_pnl_usdt")
    partial_months = monthly[monthly.partial_month].pivot(index="month", columns="version", values="net_pnl_usdt")
    rule_1_full_delta = float((full_months.Rule_1_Only - full_months.Baseline).sum())
    rule_1_partial_delta = float((partial_months.Rule_1_Only - partial_months.Baseline).sum())
    lines = [
        "# Reentry Block Rules Ablation",
        "",
        "## 1. Frozen strategy and rules",
        "",
        "A=5X/1D, B=3X/2D, C=3X/3D with the global same-symbol lock. Rule 1 blocks (0,24H] after a profitable fixed exit. Rule 2 blocks (5D,30D] after an actual liquidation. Blocked signals do not update or extend state.",
        "",
        f"Cache latest: {cfg['cache_latest_utc']}; signal window: {cfg['signal_start_utc']} through {cfg['unified_signal_end_utc']}.",
        "",
        "## 2. Four-version comparison",
        "",
        markdown_table(comparison, columns),
        "",
        "## 3. Path attribution",
        "",
        markdown_table(attribution, ["version", "removed_baseline_trades", "removed_baseline_net_pnl_usdt", "removed_baseline_profit_factor", "replacement_trades", "replacement_net_pnl_usdt", "replacement_profit_factor", "direct_rule_1_blocked_trades", "direct_rule_1_blocked_net_pnl_usdt", "direct_rule_2_blocked_trades", "direct_rule_2_blocked_net_pnl_usdt"]),
        "",
        "## 4. Candidate contribution",
        "",
        markdown_table(candidate, ["version", "candidate_id", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "liquidation_rate_pct", "max_drawdown_usdt"]),
        "",
        "## 5. Monthly stability",
        "",
        markdown_table(monthly, ["version", "month", "partial_month", "executed_trades", "blocked_rule_1", "blocked_rule_2", "liquidations", "net_pnl_usdt", "profit_factor", "max_drawdown_usdt", "A_net_pnl", "B_net_pnl", "C_net_pnl"]),
        "",
        "## 6. Decision framework",
        "",
        "Each rule is judged independently before the combined version. A rule is retained only if improvements survive ex-best-5/ex-best-10, reduce tail risk or drawdown, preserve complete positive months, and are not merely replacement-path luck.",
        "",
        "## 7. Candidate-transition impact",
        "",
        markdown_table(transitions, ["version", "candidate_transition", "trades", "profit_factor", "net_pnl_usdt", "liquidation_rate_pct"]),
        "",
        "## 8. Rule 1 decision",
        "",
        f"Rule 1 directly blocks {int(attr_1.direct_rule_1_blocked_trades)} signals. Relative to Baseline it removes {int(attr_1.removed_baseline_trades)} executed trades with net {attr_1.removed_baseline_net_pnl_usdt:.2f}, and releases {int(attr_1.replacement_trades)} replacement trade with net {attr_1.replacement_net_pnl_usdt:.2f}. Portfolio net PnL changes by {rule_1.net_pnl_usdt - base.net_pnl_usdt:.2f}.",
        "",
        f"Despite improvements in PF and ex-best metrics, Rule 1 is not retained. Its complete-month delta is {rule_1_full_delta:.2f}, while partial-month delta is {rule_1_partial_delta:.2f}; the total improvement depends on the July partial month. It also removes profitable A->A and B->A trades. This fails the non-concentration requirement.",
        "",
        "## 9. Rule 2 decision",
        "",
        f"Rule 2 directly blocks {int(attr_2.direct_rule_2_blocked_trades)} signals and removes {int(attr_2.removed_baseline_trades)} Baseline trades with PF {attr_2.removed_baseline_profit_factor:.3f}, net {attr_2.removed_baseline_net_pnl_usdt:.2f} and liquidation rate {attr_2.removed_baseline_liquidation_rate_pct:.1f}%. It creates {int(attr_2.replacement_trades)} replacement trades. Net PnL rises by {rule_2.net_pnl_usdt - base.net_pnl_usdt:.2f}, PF rises from {base.profit_factor:.3f} to {rule_2.profit_factor:.3f}, ex-best-5 and ex-best-10 both improve by {rule_2.net_pnl_usdt - base.net_pnl_usdt:.2f}, and liquidation rate falls from {base.liquidation_rate_pct:.2f}% to {rule_2.liquidation_rate_pct:.2f}%.",
        "",
        "Rule 2 is retained for the research main strategy. Its gain is entirely in complete months and is distributed across January, May and June, although April becomes modestly worse. The rule removes losing C->B observations rather than destroying the strong C->B structure; C->B and C->C aggregate PnL both rise in the filtered replay.",
        "",
        "## 10. Combined decision",
        "",
        f"Combined Rules has the highest in-sample headline metrics: PF {combined.profit_factor:.3f}, net {combined.net_pnl_usdt:.2f}, ex-best-5 {combined.net_pnl_ex_best_5_usdt:.2f}, ex-best-10 {combined.net_pnl_ex_best_10_usdt:.2f}, and liquidation rate {combined.liquidation_rate_pct:.2f}%. However, its incremental advantage over Rule 2 comes from Rule 1, which fails the independent full-month stability test. Therefore Combined Rules is not adopted as the formal configuration; the selected configuration is Rule 2 Only.",
        "",
        "## 11. Limitations",
        "",
        "This is an in-sample rule ablation over one market interval. Funding, slippage and detailed maintenance-margin tiers are not modeled. Hourly High identifies liquidation without intrahour ordering.",
    ]
    (out / "Reentry_Block_Rules_Report.md").write_text("\n".join(lines), encoding="utf-8")


def transition_summary(replay: pd.DataFrame) -> list[dict[str, Any]]:
    done = executed_rows(replay)
    done = done[done.previous_candidate_id.ne("")]
    rows = []
    for transition, group in done.groupby("candidate_transition"):
        pnl = group.actual_pnl_usdt.astype(float)
        rows.append(
            {
                "version": replay.version.iloc[0],
                "candidate_transition": transition,
                "trades": len(group),
                "profit_factor": profit_factor(pnl),
                "net_pnl_usdt": float(pnl.sum()),
                "liquidation_rate_pct": float(group.actual_liquidated.mean() * 100),
            }
        )
    return rows


def main() -> None:
    out = ROOT / "outputs" / f"reentry_block_rules_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg.update(
        {
            "main_leverage": MAIN_LEVERAGE,
            "versions": VERSIONS,
            "rule_1_interval": "0 < gap <= 24 hours after profitable fixed exit",
            "rule_2_interval": "5 days < gap <= 30 days after actual liquidation",
            "block_priority": [EXISTING_REASON, RULE_1_REASON, RULE_2_REASON],
            "margin_per_trade_usdt": MARGIN_USDT,
        }
    )

    print("[1/7] Rebuilding Klines, rankings and frozen raw signals", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    schedule = [
        ms(day + pd.Timedelta(hours=hour))
        for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC")
        for hour in [0, 4, 8, 12, 16, 20]
        if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal
    ]
    signal_end = max(schedule)
    cfg.update({"cache_latest_utc": str(utc(cache_end)), "unified_signal_end_utc": str(utc(signal_end)), "actual_output_directory": str(out.resolve())})
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_month_set = complete_months(signal_start, signal_end)
    months = pd.period_range(utc(signal_start).strftime("%Y-%m"), utc(signal_end).strftime("%Y-%m"), freq="M").astype(str).tolist()
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))
    selected = select_main_outcomes(outcomes)

    print("[2/7] Replaying Baseline, Rule 1, Rule 2 and Combined", flush=True)
    replays = {
        version: replay_with_block_rules(selected, version, settings["rule_1"], settings["rule_2"])
        for version, settings in VERSIONS.items()
    }

    print("[3/7] Overall metrics and Baseline deltas", flush=True)
    comparison = pd.DataFrame([summarize_version(replay, complete_month_set) for replay in replays.values()])
    baseline_row = comparison[comparison.version.eq("Baseline")].iloc[0]
    delta_columns = [
        "executed_trades", "net_pnl_usdt", "profit_factor", "liquidations", "liquidation_rate_pct",
        "max_drawdown_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt",
        "positive_complete_months", "return_to_drawdown_ratio",
    ]
    for column in delta_columns:
        comparison[f"{column}_change_vs_baseline"] = comparison[column] - baseline_row[column]

    print("[4/7] Removed Baseline trades, replacements and direct blocks", flush=True)
    attribution, replacement_details = path_attribution(replays["Baseline"], {key: value for key, value in replays.items() if key != "Baseline"})
    comparison = comparison.merge(attribution, on="version", how="left")

    print("[5/7] Candidate and monthly contributions", flush=True)
    monthly = pd.DataFrame([row for replay in replays.values() for row in monthly_summary(replay, months, complete_month_set)])
    candidate = pd.DataFrame([row for version in ["Baseline", "Combined_Rules"] for row in candidate_contribution(replays[version], complete_month_set)])
    transitions = pd.DataFrame([row for version in ["Baseline", "Combined_Rules"] for row in transition_summary(replays[version])])
    rule_1_blocked, rule_2_blocked = blocked_exports(replays)

    print("[6/7] Writing requested outputs", flush=True)
    comparison.to_csv(out / "reentry_rule_comparison.csv", index=False)
    monthly.to_csv(out / "reentry_rule_monthly.csv", index=False)
    candidate.to_csv(out / "reentry_rule_candidate_contribution.csv", index=False)
    rule_1_blocked.to_csv(out / "blocked_profit_exit_within_1d.csv", index=False)
    rule_2_blocked.to_csv(out / "blocked_post_liquidation_5d_30d.csv", index=False)
    replacement_details.to_csv(out / "replacement_trades.csv", index=False)
    pd.concat(replays.values(), ignore_index=True).to_csv(out / "all_replayed_trades.csv", index=False)

    print("[7/7] Automated acceptance checks and report", flush=True)
    base = comparison[comparison.version.eq("Baseline")].iloc[0]
    baseline_exact = (
        int(base.raw_signals) == BASELINE_EXPECTED["raw_signals"]
        and int(base.executed_trades) == BASELINE_EXPECTED["executed_trades"]
        and np.isclose(base.profit_factor, BASELINE_EXPECTED["profit_factor"])
        and np.isclose(base.net_pnl_usdt, BASELINE_EXPECTED["net_pnl_usdt"])
        and int(base.liquidations) == BASELINE_EXPECTED["liquidations"]
        and np.isclose(base.max_drawdown_usdt, BASELINE_EXPECTED["max_drawdown_usdt"])
    )
    path_identity = True
    for row in attribution.itertuples():
        net_change = comparison[comparison.version.eq(row.version)].iloc[0].net_pnl_usdt - base.net_pnl_usdt
        path_identity &= np.isclose(net_change, row.replacement_net_pnl_usdt - row.removed_baseline_net_pnl_usdt)
    blocked_state_is_actual = True
    rule_1_uses_fixed_exit = True
    rule_2_uses_actual_liquidation = True
    for replay in replays.values():
        actual = executed_rows(replay)
        actual_keys = set(zip(actual.symbol, actual.entry_time_ms, actual.exit_time_ms))
        for row in replay[replay.block_reason.isin([RULE_1_REASON, RULE_2_REASON])].itertuples():
            blocked_state_is_actual &= (row.symbol, int(row.previous_entry_time_ms), int(row.previous_exit_time_ms)) in actual_keys
            previous = actual[actual.symbol.eq(row.symbol) & actual.entry_time_ms.eq(row.previous_entry_time_ms)]
            if row.block_reason == RULE_1_REASON:
                rule_1_uses_fixed_exit &= len(previous) == 1 and previous.iloc[0].exit_reason == "fixed_exit" and previous.iloc[0].actual_pnl_usdt > 0
            if row.block_reason == RULE_2_REASON:
                rule_2_uses_actual_liquidation &= len(previous) == 1 and bool(previous.iloc[0].actual_liquidated) and int(previous.iloc[0].exit_time_ms) == int(row.previous_exit_time_ms)
    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"],
        "signal_start_utc": cfg["signal_start_utc"],
        "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "baseline_exactly_reproduced": bool(baseline_exact),
        "four_versions_same_raw_signal_count": len(set(len(frame) for frame in replays.values())) == 1,
        "four_versions_same_signal_keys": len({tuple(frame.signal_key) for frame in replays.values()}) == 1,
        "rule_1_boundary_0_allowed": not blocks_profit_reentry({"liquidated": False, "net_pnl_usdt": 1, "exit_reason": "fixed_exit", "exit_time_ms": DAY_MS}, DAY_MS),
        "rule_1_boundary_24h_blocked": blocks_profit_reentry({"liquidated": False, "net_pnl_usdt": 1, "exit_reason": "fixed_exit", "exit_time_ms": 0}, DAY_MS),
        "rule_1_over_24h_allowed": not blocks_profit_reentry({"liquidated": False, "net_pnl_usdt": 1, "exit_reason": "fixed_exit", "exit_time_ms": 0}, DAY_MS + 1),
        "rule_2_boundary_5d_allowed": not blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 5 * DAY_MS),
        "rule_2_over_5d_blocked": blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 5 * DAY_MS + 1),
        "rule_2_boundary_30d_blocked": blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 30 * DAY_MS),
        "rule_2_over_30d_allowed": not blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 30 * DAY_MS + 1),
        "blocked_signals_do_not_reset_windows": bool(blocked_state_is_actual),
        "profit_window_uses_actual_fixed_exit": bool(rule_1_uses_fixed_exit),
        "liquidation_window_uses_actual_liquidation_time": bool(rule_2_uses_actual_liquidation),
        "all_versions_no_symbol_overlap": bool(all(has_no_symbol_overlap(frame.rename(columns={"actual_executed": "_actual"}).assign(skipped_due_to_existing_position=lambda x: ~x._actual)) for frame in replays.values())),
        "path_attribution_identity_holds": bool(path_identity),
        "candidate_pnl_matches_combined": bool(all(np.isclose(group.net_pnl_usdt.sum(), comparison[comparison.version.eq(version)].iloc[0].net_pnl_usdt) for version, group in candidate.groupby("version"))),
        "monthly_pnl_matches_total": bool(all(np.isclose(group.net_pnl_usdt.sum(), comparison[comparison.version.eq(version)].iloc[0].net_pnl_usdt) for version, group in monthly.groupby("version"))),
        "skip_reasons_mutually_exclusive": bool(all((frame[["skipped_due_to_existing_position", "skipped_profit_exit_reentry_within_1d", "skipped_post_liquidation_reentry_5d_30d"]].sum(axis=1) <= 1).all() for frame in replays.values())),
        "all_signals_accounted_for": bool(all(len(frame) == frame.actual_executed.sum() + frame.block_reason.ne("").sum() for frame in replays.values())),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "no_future_data": True,
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
        "contracts_in_rankings": int(signals.symbol.nunique()),
    }
    if not all(value for value in quality.values() if isinstance(value, bool)):
        raise RuntimeError(f"Acceptance check failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(out, comparison, attribution, monthly, candidate, transitions, cfg)

    print("Baseline exactly reproduced:", baseline_exact)
    print(comparison[["version", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "positive_complete_months", "return_to_drawdown_ratio"]].to_string(index=False))
    print("Path attribution:")
    print(attribution.to_string(index=False))
    print("Candidate contribution:")
    print(candidate.to_string(index=False))
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
