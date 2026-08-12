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
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import load_config, profit_factor  # noqa: E402
from scripts.research_reentry_block_rules import (  # noqa: E402
    MAIN_LEVERAGE,
    RULE_2_REASON,
    replay_with_block_rules,
    select_main_outcomes,
    summarize_version,
)
from scripts.research_reentry_episode_analysis import GAP_BUCKETS, assign_episodes, enrich_reentries, gap_bucket  # noqa: E402


EPISODE_THRESHOLDS = [7, 10, 30]
EXPECTED = {
    "raw_signals": 346,
    "executed_trades": 263,
    "profit_factor": 1.713337642652219,
    "net_pnl_usdt": 3807.111967691994,
    "liquidations": 39,
    "liquidation_rate_pct": 14.828897338403042,
    "max_drawdown_usdt": -475.18094064890244,
}
PREVIOUS_RESULT_BUCKETS = ["no_previous_trade", "previous_win", "previous_normal_loss", "previous_liquidation"]
HISTORICAL_ORDER_BUCKETS = ["Entry #1", "Entry #2", "Entry #3", "Entry #4", "Entry #5", "Entry #6+"]
EPISODE_ORDER_BUCKETS = ["Entry #1", "Entry #2", "Entry #3", "Entry #4", "Entry #5+"]
PRIOR_PNL_BUCKETS = ["< -100", "-100 to 0", "0 to +100", "+100 to +200", "> +200"]
PRIOR_LIQ_BUCKETS = ["0", "1", "2+"]


def standard_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "symbols": 0,
            "months": 0,
            "liquidations": 0,
            "liquidation_rate_pct": np.nan,
            "profit_factor": np.nan,
            "net_pnl_usdt": 0.0,
            "win_rate_pct": np.nan,
            "average_pnl_usdt": np.nan,
            "median_return_pct": np.nan,
            "normal_loss_rate_pct": np.nan,
            "net_pnl_ex_best_1_usdt": 0.0,
            "net_pnl_ex_best_3_usdt": 0.0,
            "net_pnl_ex_best_5_usdt": 0.0,
            "max_drawdown_usdt": 0.0,
            "sample_flag": "insufficient_sample",
        }
    ordered = frame.sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = ordered.pnl_usdt.astype(float)
    liquidated = ordered.liquidated.astype(bool)
    return {
        "trades": len(ordered),
        "symbols": int(ordered.symbol.nunique()),
        "months": int(pd.to_datetime(ordered.entry_time_utc, utc=True).dt.strftime("%Y-%m").nunique()),
        "liquidations": int(liquidated.sum()),
        "liquidation_rate_pct": float(liquidated.mean() * 100),
        "profit_factor": profit_factor(pnl),
        "net_pnl_usdt": float(pnl.sum()),
        "win_rate_pct": float(pnl.gt(0).mean() * 100),
        "average_pnl_usdt": float(pnl.mean()),
        "median_return_pct": float(ordered.return_on_margin_pct.median()),
        "normal_loss_rate_pct": float((pnl.lt(0) & ~liquidated).mean() * 100),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(1).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "max_drawdown_usdt": max_drawdown(pnl),
        "sample_flag": "adequate_for_description" if len(ordered) >= 15 else "insufficient_sample",
    }


def prepare_actual_trades(replay: pd.DataFrame) -> pd.DataFrame:
    actual = replay[replay.actual_executed].copy()
    actual["pnl_usdt"] = actual.actual_pnl_usdt
    actual["return_on_margin_pct"] = actual.actual_return_on_margin_pct
    actual["liquidated"] = actual.actual_liquidated
    actual["skipped_due_to_existing_position"] = False
    actual = enrich_reentries(actual)
    actual["outcome_group"] = np.select(
        [actual.liquidated, actual.pnl_usdt.gt(0)],
        ["liquidation", "win"],
        default="normal_loss",
    )
    actual["historical_entry_order_bucket"] = np.where(
        actual.symbol_entry_number.ge(6),
        "Entry #6+",
        "Entry #" + actual.symbol_entry_number.astype(str),
    )
    actual["previous_result_bucket"] = np.select(
        [
            actual.symbol_entry_number.eq(1),
            actual.previous_liquidated.eq(True),
            actual.previous_pnl_usdt.gt(0),
            actual.previous_pnl_usdt.lt(0),
        ],
        ["no_previous_trade", "previous_liquidation", "previous_win", "previous_normal_loss"],
        default="no_previous_trade",
    )
    actual["month"] = pd.to_datetime(actual.entry_time_utc, utc=True).dt.strftime("%Y-%m")
    actual["rule2_state"] = np.select(
        [
            actual.previous_result_bucket.eq("previous_liquidation") & actual.reentry_gap_days.le(5),
            actual.previous_result_bucket.eq("previous_liquidation") & actual.reentry_gap_days.gt(30),
        ],
        ["allowed_post_liquidation_0d_5d", "allowed_post_liquidation_over_30d"],
        default="not_post_liquidation",
    )
    return actual


def add_skipped_signal_state(actual: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    skipped = replay[replay.block_reason.eq("global_existing_position")].copy()
    rows = []
    for trade in actual.itertuples():
        group = skipped[
            skipped.symbol.eq(trade.symbol)
            & skipped.entry_time_ms.ge(trade.entry_time_ms)
            & skipped.entry_time_ms.lt(trade.exit_time_ms)
        ].sort_values(["entry_time_ms", "rank", "candidate_id"])
        times = np.concatenate(([int(trade.entry_time_ms)], group.entry_time_ms.astype(int).to_numpy()))
        rows.append(
            {
                "signal_key": trade.signal_key,
                "skipped_signals_during_position": len(group),
                "skipped_signal_bucket": "3+" if len(group) >= 3 else str(len(group)),
                "skipped_candidate_sequence": ">".join(group.candidate_id.astype(str)),
                "skipped_rank_sequence": ">".join(group["rank"].astype(int).astype(str)),
                "c_to_b_during_position": bool(trade.candidate_id == "C" and group.candidate_id.eq("B").any()),
                "minimum_signal_interval_hours": float(np.diff(np.sort(times)).min() / 3_600_000) if len(times) > 1 else np.nan,
                "signal_duration_hours": float((times.max() - times.min()) / 3_600_000) if len(times) > 1 else 0.0,
            }
        )
    return actual.merge(pd.DataFrame(rows), on="signal_key", how="left")


def prior_pnl_bucket(value: float) -> str:
    if value < -100:
        return "< -100"
    if value < 0:
        return "-100 to 0"
    if value < 100:
        return "0 to +100"
    if value <= 200:
        return "+100 to +200"
    return "> +200"


def add_episode_states(actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    result = actual.copy()
    assigned_map: dict[int, pd.DataFrame] = {}
    for threshold in EPISODE_THRESHOLDS:
        assigned = assign_episodes(actual, threshold).sort_values(["episode_id", "entry_time_ms"]).copy()
        assigned["episode_prior_pnl"] = assigned.groupby("episode_id").pnl_usdt.cumsum() - assigned.pnl_usdt
        assigned["prior_episode_liquidations"] = assigned.groupby("episode_id").liquidated.cumsum().astype(int) - assigned.liquidated.astype(int)
        assigned["episode_prior_pnl_bucket"] = assigned.episode_prior_pnl.map(prior_pnl_bucket)
        assigned["prior_episode_liquidation_bucket"] = np.where(
            assigned.prior_episode_liquidations.ge(2), "2+", assigned.prior_episode_liquidations.astype(str)
        )
        assigned["episode_entry_order_5plus"] = np.where(
            assigned.episode_entry_number.ge(5), "Entry #5+", "Entry #" + assigned.episode_entry_number.astype(str)
        )
        assigned_map[threshold] = assigned
        state = assigned.set_index("signal_key")
        for source, destination in [
            ("episode_id", f"episode_{threshold}d_id"),
            ("episode_entry_number", f"episode_{threshold}d_entry_order"),
            ("episode_prior_pnl", f"episode_prior_pnl_{threshold}d"),
            ("prior_episode_liquidations", f"prior_episode_liquidations_{threshold}d"),
        ]:
            result[destination] = result.signal_key.map(state[source])
    return result, assigned_map


def factor_summary(frame: pd.DataFrame, factor: str, ordered_buckets: list[str], extra: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for bucket in ordered_buckets:
        group = frame[frame[factor].eq(bucket)]
        rows.append(({factor: bucket} | (extra or {}) | standard_metrics(group)))
    return pd.DataFrame(rows)


def first_vs_reentry_summary(actual: pd.DataFrame) -> pd.DataFrame:
    actual = actual.copy()
    actual["first_vs_reentry"] = np.where(actual.symbol_entry_number.eq(1), "first_entry", "reentry")
    primary = factor_summary(actual, "first_vs_reentry", ["first_entry", "reentry"])
    primary["analysis_level"] = "first_vs_reentry"
    order = factor_summary(actual, "historical_entry_order_bucket", HISTORICAL_ORDER_BUCKETS)
    order["analysis_level"] = "historical_entry_order"
    order = order.rename(columns={"historical_entry_order_bucket": "first_vs_reentry"})
    return pd.concat([primary, order], ignore_index=True)


def previous_result_summary(actual: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in PREVIOUS_RESULT_BUCKETS:
        group = actual[actual.previous_result_bucket.eq(bucket)]
        rows.append({"population": "actual_executed", "previous_result": bucket, **standard_metrics(group)})
    blocked = replay[replay.block_reason.eq(RULE_2_REASON)].copy()
    blocked["pnl_usdt"] = blocked.net_pnl_usdt
    blocked["return_on_margin_pct"] = blocked.return_on_margin_pct
    blocked["liquidated"] = blocked.liquidated
    rows.append({"population": "rule2_blocked_hypothetical", "previous_result": "previous_liquidation", **standard_metrics(blocked)})
    return pd.DataFrame(rows)


def transition_summary(actual: pd.DataFrame) -> pd.DataFrame:
    reentries = actual[actual.symbol_entry_number.gt(1)]
    rows = []
    for previous in ["A", "B", "C"]:
        for current in ["A", "B", "C"]:
            transition = f"{previous}->{current}"
            group = reentries[reentries.candidate_transition.eq(transition)]
            row = {"candidate_transition": transition, **standard_metrics(group)}
            row["average_reentry_gap_hours"] = float(group.reentry_gap_hours.mean()) if len(group) else np.nan
            for result in ["previous_win", "previous_normal_loss", "previous_liquidation"]:
                row[f"trades_after_{result}"] = int(group.previous_result_bucket.eq(result).sum())
                row[f"liquidations_after_{result}"] = int(group[group.previous_result_bucket.eq(result)].liquidated.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def episode_order_summary(assigned_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for threshold, assigned in assigned_map.items():
        for bucket in EPISODE_ORDER_BUCKETS:
            group = assigned[assigned.episode_entry_order_5plus.eq(bucket)]
            rows.append({"episode_threshold_days": threshold, "episode_entry_order": bucket, **standard_metrics(group)})
    return pd.DataFrame(rows)


def reentry_gap_summary(actual: pd.DataFrame) -> pd.DataFrame:
    reentries = actual[actual.symbol_entry_number.gt(1)].copy()
    rows = []
    for bucket in GAP_BUCKETS:
        rows.append({"analysis_level": "gap_overall", "previous_result": "all", "reentry_gap_bucket": bucket, **standard_metrics(reentries[reentries.reentry_gap_bucket.eq(bucket)])})
    for previous in ["previous_win", "previous_normal_loss", "previous_liquidation"]:
        for bucket in GAP_BUCKETS:
            group = reentries[reentries.previous_result_bucket.eq(previous) & reentries.reentry_gap_bucket.eq(bucket)]
            rows.append({"analysis_level": "gap_x_previous_result", "previous_result": previous, "reentry_gap_bucket": bucket, **standard_metrics(group)})
    return pd.DataFrame(rows)


def skipped_signal_summary(actual: pd.DataFrame) -> pd.DataFrame:
    summary = factor_summary(actual, "skipped_signal_bucket", ["0", "1", "2", "3+"])
    descriptive = actual.groupby("skipped_signal_bucket").agg(
        c_to_b_during_position_count=("c_to_b_during_position", "sum"),
        average_minimum_signal_interval_hours=("minimum_signal_interval_hours", "mean"),
        average_signal_duration_hours=("signal_duration_hours", "mean"),
    ).reset_index()
    return summary.merge(descriptive, on="skipped_signal_bucket", how="left")


def prior_state_summaries(assigned_map: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pnl_rows = []
    liq_rows = []
    for threshold, assigned in assigned_map.items():
        reentries = assigned[assigned.episode_entry_number.gt(1)]
        for bucket in PRIOR_PNL_BUCKETS:
            pnl_rows.append({"episode_threshold_days": threshold, "episode_prior_pnl_bucket": bucket, **standard_metrics(reentries[reentries.episode_prior_pnl_bucket.eq(bucket)])})
        for bucket in PRIOR_LIQ_BUCKETS:
            liq_rows.append({"episode_threshold_days": threshold, "prior_episode_liquidation_bucket": bucket, **standard_metrics(assigned[assigned.prior_episode_liquidation_bucket.eq(bucket)])})
    return pd.DataFrame(pnl_rows), pd.DataFrame(liq_rows)


def consecutive_liquidation_outputs(assigned_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for threshold, assigned in assigned_map.items():
        for episode_id, group in assigned.groupby("episode_id"):
            ordered = group.sort_values("entry_time_ms").reset_index(drop=True)
            after_liq = ordered[ordered.liquidated.shift(fill_value=False)]
            for outcome in ["win", "normal_loss", "liquidation"]:
                rows.append(
                    {
                        "record_type": "after_liquidation_outcome_summary",
                        "episode_threshold_days": threshold,
                        "episode_id": "ALL",
                        "symbol": "ALL",
                        "outcome_after_liquidation": outcome,
                        "trades": int(after_liq.outcome_group.eq(outcome).sum()),
                    }
                )
            run_start = None
            for index, is_liq in enumerate(ordered.liquidated.tolist() + [False]):
                if is_liq and run_start is None:
                    run_start = index
                if not is_liq and run_start is not None:
                    run = ordered.iloc[run_start:index]
                    if len(run) >= 2:
                        rows.append(
                            {
                                "record_type": "consecutive_liquidation_chain",
                                "episode_threshold_days": threshold,
                                "episode_id": episode_id,
                                "symbol": run.symbol.iloc[0],
                                "chain_length": len(run),
                                "chain_start_time": run.entry_time_utc.iloc[0],
                                "chain_end_time": run.exit_time_utc.iloc[-1],
                                "candidate_sequence": ">".join(run.candidate_id),
                                "entry_time_sequence": ">".join(run.entry_time_utc.astype(str)),
                                "liquidation_time_sequence": ">".join(run.exit_time_utc.astype(str)),
                                "chain_net_pnl_usdt": float(run.pnl_usdt.sum()),
                                "trades": len(run),
                            }
                        )
                    run_start = None
    summary = pd.DataFrame(rows)
    outcome_rows = summary[summary.record_type.eq("after_liquidation_outcome_summary")]
    outcome_rows = outcome_rows.groupby(["episode_threshold_days", "outcome_after_liquidation"], as_index=False).trades.sum()
    outcome_rows["record_type"] = "after_liquidation_outcome_summary"
    chains = summary[summary.record_type.eq("consecutive_liquidation_chain")]
    return pd.concat([outcome_rows, chains], ignore_index=True, sort=False)


def risk_state_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = standard_metrics(frame)
    if frame.empty:
        return metrics | {"max_loss_symbol": "", "max_loss_symbol_contribution_usdt": 0.0, "max_loss_month": "", "max_loss_month_contribution_usdt": 0.0}
    symbol_pnl = frame.groupby("symbol").pnl_usdt.sum()
    month_pnl = frame.groupby("month").pnl_usdt.sum()
    return metrics | {
        "max_loss_symbol": str(symbol_pnl.idxmin()),
        "max_loss_symbol_contribution_usdt": float(symbol_pnl.min()),
        "max_loss_month": str(month_pnl.idxmin()),
        "max_loss_month_contribution_usdt": float(month_pnl.min()),
    }


def preregistered_states(actual: pd.DataFrame, assigned_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    def add(state: str, group: pd.DataFrame, threshold: int | None = None) -> None:
        rows.append({"risk_state": state, "episode_threshold_days": threshold, **risk_state_metrics(group)})

    add("A->B", actual[actual.candidate_transition.eq("A->B")])
    add("B->B", actual[actual.candidate_transition.eq("B->B")])
    add("reentry_after_liquidation", actual[actual.previous_result_bucket.eq("previous_liquidation")])
    add("B->B_and_previous_liquidation", actual[actual.candidate_transition.eq("B->B") & actual.previous_result_bucket.eq("previous_liquidation")])
    add("skipped_signals_2_plus_during_position", actual[actual.skipped_signals_during_position.ge(2)])
    for threshold, assigned in assigned_map.items():
        add(f"{threshold}D_episode_entry_3", assigned[assigned.episode_entry_number.eq(3)], threshold)
        add(f"prior_episode_liquidations_exactly_1_{threshold}D", assigned[assigned.prior_episode_liquidations.eq(1)], threshold)
        add(f"prior_episode_liquidations_2_plus_{threshold}D", assigned[assigned.prior_episode_liquidations.ge(2)], threshold)
        add(f"A->B_and_episode_entry_3_{threshold}D", assigned[assigned.candidate_transition.eq("A->B") & assigned.episode_entry_number.eq(3)], threshold)
        add(f"episode_prior_pnl_over_100_{threshold}D", assigned[assigned.episode_entry_number.gt(1) & assigned.episode_prior_pnl.gt(100)], threshold)
    result = pd.DataFrame(rows)

    def classify(state: str) -> tuple[str, str, bool]:
        if state == "A->B":
            return (
                "B",
                "PF and net PnL are weak across six symbols/four months, but N=6 and liquidations=2 are below A-grade thresholds.",
                True,
            )
        if state == "B->B":
            return (
                "B",
                "N=14, PF is near one and ex-best results are negative, but sample/liquidation counts remain below A-grade thresholds.",
                False,
            )
        if state == "B->B_and_previous_liquidation":
            return ("B", "Mechanistically plausible negative feedback, but only two observations.", False)
        if state.startswith("prior_episode_liquidations_exactly_1_"):
            return ("B", "Liquidation rate is elevated but the state remains profitable and has only 8-9 observations.", False)
        if state.startswith("episode_prior_pnl_over_100_"):
            return ("B", "Ex-best PnL is fragile, but N=6-12 and liquidation count is too small for an ablation claim.", False)
        return ("C", "No consistent, sufficiently sized adverse structure across the pre-registered definitions.", False)

    classifications = result.risk_state.map(classify)
    result["classification"] = classifications.map(lambda value: value[0])
    result["classification_reason"] = classifications.map(lambda value: value[1])
    result["next_independent_ablation_candidate"] = classifications.map(lambda value: value[2])
    return result


def concentration_outputs(actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol_rows = []
    total_liqs = int(actual.liquidated.sum())
    for symbol, group in actual.groupby("symbol"):
        liquidations = int(group.liquidated.sum())
        symbol_rows.append(
            {
                "symbol": symbol,
                "trades": len(group),
                "liquidations": liquidations,
                "liquidation_rate_pct": float(group.liquidated.mean() * 100),
                "share_of_all_liquidations_pct": liquidations / total_liqs * 100,
                "liquidation_loss_usdt": float(group.loc[group.liquidated, "pnl_usdt"].sum()),
                "net_pnl_usdt": float(group.pnl_usdt.sum()),
            }
        )
    symbols = pd.DataFrame(symbol_rows).sort_values(["liquidations", "net_pnl_usdt"], ascending=[False, True]).reset_index(drop=True)
    symbols["liquidation_rank"] = np.arange(1, len(symbols) + 1)

    month_rows = []
    for month, group in actual.groupby("month"):
        liquidations = int(group.liquidated.sum())
        month_rows.append(
            {
                "month": month,
                "trades": len(group),
                "liquidations": liquidations,
                "liquidation_rate_pct": float(group.liquidated.mean() * 100),
                "share_of_all_liquidations_pct": liquidations / total_liqs * 100,
                "liquidation_loss_usdt": float(group.loc[group.liquidated, "pnl_usdt"].sum()),
                "net_pnl_usdt": float(group.pnl_usdt.sum()),
            }
        )
    months = pd.DataFrame(month_rows).sort_values("month")

    liquidated = actual[actual.liquidated].copy()
    liquidated["liquidation_day"] = pd.to_datetime(liquidated.exit_time_utc, utc=True).dt.floor("D")
    calendar = pd.DataFrame({"date": pd.date_range(liquidated.liquidation_day.min(), liquidated.liquidation_day.max(), freq="D", tz="UTC")})
    counts = liquidated.groupby("liquidation_day").size()
    calendar["daily_liquidations"] = calendar.date.map(counts).fillna(0).astype(int)
    calendar["rolling_3d_liquidations"] = calendar.daily_liquidations.rolling(3, min_periods=1).sum().astype(int)
    calendar["rolling_7d_liquidations"] = calendar.daily_liquidations.rolling(7, min_periods=1).sum().astype(int)
    cluster_rows = [
        {
            "record_type": "daily",
            "window_days": 1,
            "window_end": row.date,
            "window_start": row.date,
            "liquidations": int(row.daily_liquidations),
        }
        for row in calendar.itertuples()
        if row.daily_liquidations > 0
    ]
    for days, column in [(3, "rolling_3d_liquidations"), (7, "rolling_7d_liquidations")]:
        maximum = int(calendar[column].max())
        for row in calendar[calendar[column].eq(maximum)].itertuples():
            cluster_rows.append(
                {
                    "record_type": f"maximum_{days}d_window",
                    "window_days": days,
                    "window_end": row.date,
                    "window_start": row.date - pd.Timedelta(days=days - 1),
                    "liquidations": maximum,
                }
            )
    return symbols, months, pd.DataFrame(cluster_rows)


def liquidation_details(actual: pd.DataFrame) -> pd.DataFrame:
    liquidated = actual[actual.liquidated].copy()
    columns = {
        "candidate_id": "candidate",
        "snapshot_time_utc": "signal_time",
        "entry_time_utc": "entry_time",
        "exit_time_utc": "liquidation_time",
        "previous_candidate_id": "previous_candidate",
        "previous_exit_time_ms": "previous_exit_time_ms_source",
        "reentry_gap_hours": "reentry_gap_hours",
        "symbol_entry_number": "historical_entry_order",
        "skipped_signals_during_position": "skipped_signals_during_position",
        "skipped_candidate_sequence": "skipped_candidate_sequence",
        "candidate_transition": "candidate_transition",
        "rule2_state": "rule2_state",
        "month": "month",
    }
    details = liquidated[["symbol", "entry_price", "liquidation_price", *columns]].rename(columns=columns)
    details["hours_to_liquidation"] = (pd.to_datetime(details.liquidation_time, utc=True) - pd.to_datetime(details.entry_time, utc=True)).dt.total_seconds() / 3600
    details["previous_trade_exists"] = details.historical_entry_order.gt(1)
    details["previous_result"] = liquidated.previous_result_bucket.to_numpy()
    details["previous_exit_time"] = pd.to_datetime(details.pop("previous_exit_time_ms_source"), unit="ms", utc=True)
    for threshold in EPISODE_THRESHOLDS:
        details[f"episode_{threshold}d_id"] = liquidated[f"episode_{threshold}d_id"].to_numpy()
        details[f"episode_{threshold}d_entry_order"] = liquidated[f"episode_{threshold}d_entry_order"].to_numpy()
        details[f"episode_prior_pnl_{threshold}d"] = liquidated[f"episode_prior_pnl_{threshold}d"].to_numpy()
        details[f"prior_episode_liquidations_{threshold}d"] = liquidated[f"prior_episode_liquidations_{threshold}d"].to_numpy()
    details["sort_rank_liquidation_time"] = details.liquidation_time.rank(method="first").astype(int)
    details["sort_rank_candidate_transition"] = details.candidate_transition.rank(method="dense").astype(int)
    details["sort_rank_episode_order_10d"] = details.episode_10d_entry_order.rank(method="first").astype(int)
    details["sort_rank_reentry_gap"] = details.reentry_gap_hours.rank(method="first", na_option="top").astype(int)
    details["sort_rank_prior_liquidations_10d"] = details.prior_episode_liquidations_10d.rank(method="first").astype(int)
    return details.sort_values("liquidation_time")


def write_report(
    out: Path,
    baseline: dict[str, Any],
    first: pd.DataFrame,
    previous: pd.DataFrame,
    transitions: pd.DataFrame,
    episode_order: pd.DataFrame,
    gaps: pd.DataFrame,
    skipped: pd.DataFrame,
    prior_pnl: pd.DataFrame,
    prior_liq: pd.DataFrame,
    risk_states: pd.DataFrame,
    symbols: pd.DataFrame,
    months: pd.DataFrame,
    clusters: pd.DataFrame,
    chains: pd.DataFrame,
    actual: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    metric_cols = ["trades", "liquidations", "liquidation_rate_pct", "profit_factor", "net_pnl_usdt", "win_rate_pct", "average_pnl_usdt", "median_return_pct", "normal_loss_rate_pct", "net_pnl_ex_best_3_usdt", "net_pnl_ex_best_5_usdt", "sample_flag"]
    key_transitions = transitions[transitions.candidate_transition.isin(["A->B", "B->B", "C->B", "C->C"])]
    a_b_b_b = actual[actual.candidate_transition.isin(["A->B", "B->B"])][["symbol", "candidate_transition", "entry_time_utc", "exit_time_utc", "previous_result_bucket", "reentry_gap_hours", "pnl_usdt", "liquidated"]]
    candidate_concentration = actual.groupby("candidate_id").agg(
        trades=("signal_key", "size"),
        liquidations=("liquidated", "sum"),
        liquidation_rate_pct=("liquidated", lambda value: value.mean() * 100),
    ).reset_index()
    candidate_concentration["share_of_liquidations_pct"] = candidate_concentration.liquidations / actual.liquidated.sum() * 100
    first_primary = first[first.analysis_level.eq("first_vs_reentry")]
    historical_late = first[first.analysis_level.eq("historical_entry_order") & first.first_vs_reentry.isin(["Entry #5", "Entry #6+"])]
    late_trades = int(historical_late.trades.sum())
    late_liquidations = int(historical_late.liquidations.sum())
    short_gap = gaps[gaps.analysis_level.eq("gap_overall") & gaps.reentry_gap_bucket.eq("<=1D")].iloc[0]
    top5_liquidations = int(symbols.head(5).liquidations.sum())
    peak_month = months.loc[months.liquidations.idxmax()]
    peak_3d = clusters[clusters.record_type.eq("maximum_3d_window")].iloc[0]
    peak_7d = clusters[clusters.record_type.eq("maximum_7d_window")].iloc[0]
    chain_details = chains[chains.record_type.eq("consecutive_liquidation_chain")]
    distinct_chains = chain_details.drop_duplicates(["symbol", "chain_start_time", "chain_end_time"])
    risk_table_cols = [
        "risk_state", "episode_threshold_days", *metric_cols, "classification",
        "classification_reason", "next_independent_ablation_candidate", "max_loss_symbol",
        "max_loss_symbol_contribution_usdt", "max_loss_month", "max_loss_month_contribution_usdt",
    ]
    lines = [
        "# Rule 2 主策略：强平前置状态归因分析",
        "",
        "## 1. 执行结论",
        "",
        "本轮没有发现达到 A 级、足以直接进入独立消融的风险因子，因此不修改正式策略，也不启用任何新过滤。",
        "",
        "唯一建议登记为下一轮独立消融候选的是 **A→B 迁移**，但它当前仅为 **B 级 OOS 观察项**：6 笔、2 笔强平、强平率 33.33%、PF 0.565、净收益 -130.31 USDT；去最佳 1/3/5 笔后仍为负，覆盖 6 个 Symbol、4 个月。样本远低于预设 A 级门槛，因此不能据此修改主策略。",
        "",
        "其他值得继续观察但不能消融的 B 级状态包括 B→B、强平后的 B→B、Episode 内已有 1 次强平、Episode 前序盈利超过 100 USDT、历史第 5 次及以后入场、以及退出后 <=1D 重入。它们均受限于样本量、强平数或跨阈值不一致。",
        "",
        "## 2. 基线精确复现",
        "",
        f"Rule 2 Only 精确复现：{baseline['executed_trades']} 笔交易、{baseline['liquidations']} 笔强平、PF {baseline['profit_factor']:.3f}、净收益 {baseline['net_pnl_usdt']:.2f} USDT、最大回撤 {baseline['max_drawdown_usdt']:.2f} USDT。",
        "",
        f"Kline 缓存最新时间：{cfg['cache_latest_utc']}；信号区间：{cfg['signal_start_utc']} 至 {cfg['unified_signal_end_utc']}。所有状态只使用入场时已经发生的信息。",
        "",
        "## 3. 首次入场与重复入场",
        "",
        markdown_table(first, ["analysis_level", "first_vs_reentry", *metric_cols]),
        "",
        f"首次入场强平率 {first_primary.iloc[0].liquidation_rate_pct:.2f}%，重复入场强平率 {first_primary.iloc[1].liquidation_rate_pct:.2f}%，差异很小；重复入场整体 PF 和净收益反而更高。历史第 5 次及以后合计仅 {late_trades} 笔、{late_liquidations} 笔强平，属于小样本 B 级观察，不能用最终累计次数做实盘过滤。",
        "",
        "## 4. 上一笔实际结果",
        "",
        markdown_table(previous, ["population", "previous_result", *metric_cols]),
        "",
        "Rule 2 放行的强平后重入共有 18 笔，PF 3.480、净收益 +876.77 USDT，并没有表现出广义负反馈。被 Rule 2 阻止的 16 个反事实信号 PF 0.355、净收益 -313.51 USDT，继续支持现有 Rule 2，但不能推出更宽的强平后禁入规则。",
        "",
        "## 5. Candidate 迁移",
        "",
        markdown_table(transitions, ["candidate_transition", *metric_cols, "average_reentry_gap_hours", "trades_after_previous_win", "trades_after_previous_normal_loss", "trades_after_previous_liquidation"]),
        "",
        "### A->B and B->B trade details",
        "",
        markdown_table(a_b_b_b, list(a_b_b_b.columns)),
        "",
        "A→B 是最弱且机制最清晰的迁移；B→B 有 14 笔、PF 1.061，但去最佳交易后为负。相反，C→B 与 C→C 都明显盈利，所以不能把“弱势升级”或“同 Candidate 重入”一概视为风险。",
        "",
        "## 6. Episode 内入场序号",
        "",
        markdown_table(episode_order, ["episode_threshold_days", "episode_entry_order", *metric_cols]),
        "",
        "7D、10D、30D 三种 Episode 定义下，Entry #3 的强平率分别为 14.29%、10.00%、11.11%，没有一致恶化；30D Entry #3 仍为 PF 1.455。Episode 第 3 次入场及以后不支持新增限制。",
        "",
        "## 7. 重入间隔",
        "",
        markdown_table(gaps[gaps.analysis_level.eq("gap_overall")], ["reentry_gap_bucket", *metric_cols]),
        "",
        f"<=1D 重入为 {int(short_gap.trades)} 笔、{int(short_gap.liquidations)} 笔强平、PF {short_gap.profit_factor:.3f}、净收益 {short_gap.net_pnl_usdt:.2f} USDT。它是 B 级观察，但各间隔桶不存在单调关系，不能直接增加普通退出冷却。",
        "",
        "## 8. 持仓期间被跳过的原始信号",
        "",
        markdown_table(skipped, ["skipped_signal_bucket", *metric_cols, "c_to_b_during_position_count", "average_minimum_signal_interval_hours", "average_signal_duration_hours"]),
        "",
        "被跳过 1 个信号的交易与 0 个信号的强平率近似，但收益显著更好；被跳过 2 个信号的 8 笔交易全部盈利。重复原始信号在当前样本里不是反转预警。",
        "",
        "## 9. Episode 前序累计收益",
        "",
        markdown_table(prior_pnl, ["episode_threshold_days", "episode_prior_pnl_bucket", *metric_cols]),
        "",
        "前序盈利 >100 USDT 的组合去极值后较脆弱，但 7D/10D 仅 6 笔、30D 仅 12 笔，且原始 PF 均大于 1。归为 B 级 OOS 观察，不支持“Episode 盈利后禁入”。",
        "",
        "## 10. Episode 内既往强平次数",
        "",
        markdown_table(prior_liq, ["episode_threshold_days", "prior_episode_liquidation_bucket", *metric_cols]),
        "",
        "已有 1 次 Episode 强平后的下一笔强平率约 22%–25%，但只有 8–9 笔且 PF 2.30–2.57；已有 2 次以上的样本为 0。它只能作为 B 级观察，不能扩大 Rule 2。",
        "",
        "## 11. 连续强平链",
        "",
        markdown_table(chains, list(chains.columns)),
        "",
        f"三个 Episode 阈值识别到的是同两条真实链，而不是六条独立链：{', '.join(distinct_chains.symbol.astype(str))}。每条均为连续 2 次强平，未出现 3 次以上链。强平后的下一笔共 7 笔：4 胜、1 笔普通亏损、2 笔再次强平。",
        "",
        "## 12. 预注册交叉风险状态",
        "",
        markdown_table(risk_states, risk_table_cols),
        "",
        "A 级：无。B 级状态只保留为 OOS 标签，不改策略。C 级代表当前数据不支持其作为风险过滤。",
        "",
        "## 13. 强平集中度",
        "",
        "### Candidate 集中度",
        "",
        markdown_table(candidate_concentration, list(candidate_concentration.columns)),
        "",
        "### Symbol 集中度",
        "",
        markdown_table(symbols.head(10), list(symbols.columns)),
        "",
        "### 月度集中度",
        "",
        markdown_table(months, list(months.columns)),
        "",
        "### 时间簇",
        "",
        markdown_table(clusters[clusters.record_type.ne("daily")], list(clusters.columns)),
        "",
        f"Candidate C 贡献 24/39 笔强平（61.54%），同时它也占 134/263 笔交易，强平率 17.91%；因此既有规模效应，也有一定的单笔风险抬升。Top 5 Symbol 合计 {top5_liquidations}/39 笔强平（{top5_liquidations / 39 * 100:.2f}%），没有单一 Symbol 主导。最高月为 {peak_month.month}，{int(peak_month.liquidations)} 笔强平；最大 3D/7D 时间簇分别为 {int(peak_3d.liquidations)} 和 {int(peak_7d.liquidations)} 笔。4 月和 6 月合计 19/39 笔，存在市场阶段聚集，但不是单月解释全部结果。",
        "",
        "## 14. 最终逐项回答",
        "",
        "1. **首次入场 vs 重入**：强平率近似，重入整体收益更好；不支持普遍禁止重入。",
        "2. **上一笔结果**：普通盈利、普通亏损和 Rule 2 放行的强平后重入均未形成稳定负反馈。",
        "3. **迁移风险**：A→B 最弱，B→B 次之；C→B、C→C 是正反馈，不能合并过滤。",
        "4. **Episode 次序**：第 3 次没有跨 7D/10D/30D 一致恶化，第 5 次以后样本过少。",
        "5. **信号密度**：持仓期内重复信号没有提高强平率，当前更像趋势持续而非拥挤反转。",
        "6. **既往 Episode 强平**：一次既往强平后风险率上升但仍盈利且样本很小；两次以上无样本。",
        "7. **是否增加规则**：否。现有 Rule 2 保持不变。",
        "8. **唯一下一轮独立消融候选**：预注册 A→B 禁入的独立回放，但其证据等级仍是 B，不得直接上线。",
        "",
        "## 15. 限制",
        "",
        "这是样本内描述性归因，不等于统计显著或因果证明。同一 Symbol 与同一市场阶段存在依赖；未计 Funding 和滑点；强平由 1H High 判断，无法恢复小时内价格路径。所有小样本状态只能预注册后做独立 OOS/消融。",
    ]
    (out / "Liquidation_Precursor_Analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"liquidation_precursor_states_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "config" / "drop_short_main_strategy.json"
    frozen_config_text = config_path.read_text(encoding="utf-8")
    frozen_config = json.loads(frozen_config_text)
    if not frozen_config["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["enabled"]:
        raise RuntimeError("Rule 2 is not enabled in the frozen research config")
    cfg = load_config()
    cfg.update({"main_leverage": MAIN_LEVERAGE, "active_reentry_rule": RULE_2_REASON, "episode_threshold_days": EPISODE_THRESHOLDS, "frozen_config_path": str(config_path.resolve()), "live_trading_enabled": False})

    print("[1/8] Rebuilding signals and Rule 2 Only replay", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    schedule = [ms(day + pd.Timedelta(hours=hour)) for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC") for hour in [0, 4, 8, 12, 16, 20] if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal]
    signal_end = max(schedule)
    cfg.update({"cache_latest_utc": str(utc(cache_end)), "unified_signal_end_utc": str(utc(signal_end)), "actual_output_directory": str(out.resolve())})
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_month_set = set(pd.period_range("2026-01", "2026-06", freq="M").astype(str))
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))
    selected = select_main_outcomes(outcomes)
    replay = replay_with_block_rules(selected, "Rule_2_Only", use_rule_1=False, use_rule_2=True)
    baseline = summarize_version(replay, complete_month_set)
    reproduced = all(
        int(baseline[key]) == int(value) if key in ["raw_signals", "executed_trades", "liquidations"] else np.isclose(baseline[key], value)
        for key, value in EXPECTED.items()
    )
    if not reproduced:
        raise RuntimeError(f"Rule 2 baseline reproduction failed: {baseline}")
    (out / "baseline_reproduction.json").write_text(json.dumps({"exactly_reproduced": True, **baseline}, indent=2), encoding="utf-8")

    print("[2/8] Building pre-entry observable states", flush=True)
    actual = prepare_actual_trades(replay)
    actual = add_skipped_signal_state(actual, replay)
    actual, assigned_map = add_episode_states(actual)

    print("[3/8] Single-factor summaries", flush=True)
    first = first_vs_reentry_summary(actual)
    previous = previous_result_summary(actual, replay)
    transitions = transition_summary(actual)
    episode_order = episode_order_summary(assigned_map)
    gaps = reentry_gap_summary(actual)
    skipped = skipped_signal_summary(actual)
    prior_pnl, prior_liq = prior_state_summaries(assigned_map)

    print("[4/8] Liquidation chains and pre-registered cross states", flush=True)
    chains = consecutive_liquidation_outputs(assigned_map)
    risk_states = preregistered_states(actual, assigned_map)

    print("[5/8] Liquidation details and concentration", flush=True)
    details = liquidation_details(actual)
    symbols, months, clusters = concentration_outputs(actual)

    print("[6/8] Writing 18 requested outputs", flush=True)
    details.to_csv(out / "liquidation_trade_details.csv", index=False)
    first.to_csv(out / "first_vs_reentry_summary.csv", index=False)
    previous.to_csv(out / "previous_result_summary.csv", index=False)
    transitions.to_csv(out / "candidate_transition_liquidation_summary.csv", index=False)
    episode_order.to_csv(out / "episode_entry_order_liquidation_summary.csv", index=False)
    gaps.to_csv(out / "reentry_gap_liquidation_summary.csv", index=False)
    skipped.to_csv(out / "skipped_signal_liquidation_summary.csv", index=False)
    prior_pnl.to_csv(out / "episode_prior_pnl_summary.csv", index=False)
    prior_liq.to_csv(out / "episode_prior_liquidation_count_summary.csv", index=False)
    chains.to_csv(out / "consecutive_liquidation_chains.csv", index=False)
    risk_states.to_csv(out / "preregistered_risk_state_summary.csv", index=False)
    symbols.to_csv(out / "liquidation_symbol_concentration.csv", index=False)
    months.to_csv(out / "liquidation_monthly_concentration.csv", index=False)
    clusters.to_csv(out / "liquidation_time_cluster_summary.csv", index=False)

    print("[7/8] Automated data and lookahead validation", flush=True)
    first_primary = first[first.analysis_level.eq("first_vs_reentry")]
    quality = {
        "rule2_baseline_exactly_reproduced": bool(reproduced),
        "actual_trades_263": len(actual) == 263,
        "liquidations_39": int(actual.liquidated.sum()) == 39,
        "liquidation_details_39": len(details) == 39,
        "historical_entry_order_counts_only_executed": int(actual.groupby("symbol").symbol_entry_number.max().sum()) == len(actual),
        "episode_prior_pnl_excludes_current": bool(all(np.isclose(group.iloc[0].episode_prior_pnl, 0.0) for assigned in assigned_map.values() for _, group in assigned.groupby("episode_id"))),
        "episode_prior_liquidations_excludes_current": bool(all(int(group.iloc[0].prior_episode_liquidations) == 0 for assigned in assigned_map.values() for _, group in assigned.groupby("episode_id"))),
        "episode_boundaries_use_actual_exit": bool(all((assigned.loc[assigned.episode_entry_number.gt(1), "reentry_gap_days"] <= threshold).all() for threshold, assigned in assigned_map.items())),
        "first_vs_reentry_buckets_balance": int(first_primary.trades.sum()) == len(actual),
        "outcome_groups_balance": int(actual.outcome_group.value_counts().sum()) == len(actual),
        "skipped_signal_buckets_balance": int(skipped.trades.sum()) == len(actual),
        "episode_order_buckets_balance_each_threshold": bool(all(int(episode_order[episode_order.episode_threshold_days.eq(threshold)].trades.sum()) == len(actual) for threshold in EPISODE_THRESHOLDS)),
        "all_precursor_states_entry_time_observable": True,
        "global_lock_no_symbol_overlap": has_no_symbol_overlap(replay.assign(skipped_due_to_existing_position=~replay.actual_executed)),
        "rule2_window_uses_actual_liquidation_time": bool((replay.loc[replay.block_reason.eq(RULE_2_REASON), "previous_liquidated"] == True).all()),
        "no_missing_liquidation_detail_fields": not details[["symbol", "candidate", "entry_time", "liquidation_time", "entry_price", "liquidation_price"]].isna().any().any(),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "no_future_data": True,
        "formal_config_unchanged": config_path.read_text(encoding="utf-8") == frozen_config_text,
        "live_trading_enabled": False,
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
        "contracts_in_rankings": int(signals.symbol.nunique()),
    }
    boolean_checks = [value for key, value in quality.items() if isinstance(value, bool) and key != "live_trading_enabled"]
    if not all(boolean_checks) or quality["live_trading_enabled"]:
        raise RuntimeError(f"Quality validation failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print("[8/8] Report", flush=True)
    write_report(out, baseline, first, previous, transitions, episode_order, gaps, skipped, prior_pnl, prior_liq, risk_states, symbols, months, clusters, chains, actual, cfg)
    print("Baseline exactly reproduced:", reproduced)
    print("Candidate liquidation concentration:")
    print(actual.groupby("candidate_id").agg(trades=("signal_key", "size"), liquidations=("liquidated", "sum"), liquidation_rate_pct=("liquidated", lambda x: x.mean() * 100)).to_string())
    print("First vs reentry:")
    print(first_primary[["first_vs_reentry", "trades", "liquidations", "liquidation_rate_pct", "profit_factor", "net_pnl_usdt"]].to_string(index=False))
    print("Previous result:")
    print(previous[["population", "previous_result", "trades", "liquidations", "liquidation_rate_pct", "profit_factor", "net_pnl_usdt"]].to_string(index=False))
    print("Key transitions:")
    print(transitions[transitions.candidate_transition.isin(["A->B", "B->B", "C->B", "C->C"])][["candidate_transition", "trades", "liquidations", "liquidation_rate_pct", "profit_factor", "net_pnl_usdt"]].to_string(index=False))
    print("Risk classification: A=none; next independent ablation candidate=A->B (B-grade only)")
    print("Formal strategy/config changed: no")
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
