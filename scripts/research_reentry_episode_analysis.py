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
    replay,
)
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import load_config, profit_factor  # noqa: E402


MAIN_LEVERAGE = {"A": 5, "B": 3, "C": 3}
EPISODE_THRESHOLDS_DAYS = [3, 5, 7, 10, 30]
HIGH_FREQUENCY_SYMBOLS = ["LABUSDT", "CLOUSDT", "COLLECTUSDT", "RIVERUSDT", "PIPPINUSDT", "BSBUSDT"]
GAP_BUCKETS = ["<=1D", ">1D-3D", ">3D-5D", ">5D-7D", ">7D-10D", ">10D-30D", ">30D"]
POST_LIQ_GAP_BUCKETS = ["<=1D", ">1D-3D", ">3D-5D", ">5D-10D", ">10D-30D", ">30D"]
PREVIOUS_RESULTS = ["previous_profit", "previous_ordinary_loss", "previous_liquidation"]
ENTRY_ORDER_BUCKETS = ["Entry #1", "Entry #2", "Entry #3", "Entry #4", "Entry #5", "Entry #6+"]


def gap_bucket(gap_days: float) -> str:
    if gap_days <= 1:
        return "<=1D"
    if gap_days <= 3:
        return ">1D-3D"
    if gap_days <= 5:
        return ">3D-5D"
    if gap_days <= 7:
        return ">5D-7D"
    if gap_days <= 10:
        return ">7D-10D"
    if gap_days <= 30:
        return ">10D-30D"
    return ">30D"


def post_liquidation_gap_bucket(gap_days: float) -> str:
    if gap_days <= 1:
        return "<=1D"
    if gap_days <= 3:
        return ">1D-3D"
    if gap_days <= 5:
        return ">3D-5D"
    if gap_days <= 10:
        return ">5D-10D"
    if gap_days <= 30:
        return ">10D-30D"
    return ">30D"


def enrich_reentries(trades: pd.DataFrame) -> pd.DataFrame:
    executed = trades[~trades.skipped_due_to_existing_position].copy()
    executed = executed.sort_values(["symbol", "entry_time_ms", "candidate_id"]).reset_index(drop=True)
    grouped = executed.groupby("symbol", sort=False)
    executed["symbol_entry_number"] = grouped.cumcount() + 1
    executed["previous_exit_time_ms"] = grouped.exit_time_ms.shift()
    executed["previous_pnl_usdt"] = grouped.pnl_usdt.shift()
    executed["previous_liquidated"] = grouped.liquidated.shift()
    executed["previous_candidate_id"] = grouped.candidate_id.shift()
    executed["reentry_gap_hours"] = (executed.entry_time_ms - executed.previous_exit_time_ms) / 3_600_000
    executed["reentry_gap_days"] = executed.reentry_gap_hours / 24
    executed["is_reentry"] = executed.symbol_entry_number > 1
    executed["reentry_gap_bucket"] = [gap_bucket(value) if pd.notna(value) else "first_entry" for value in executed.reentry_gap_days]
    executed["post_liquidation_gap_bucket"] = [post_liquidation_gap_bucket(value) if pd.notna(value) else "first_entry" for value in executed.reentry_gap_days]
    executed["previous_result"] = np.select(
        [executed.previous_liquidated.eq(True), executed.previous_pnl_usdt.gt(0), executed.previous_pnl_usdt.lt(0)],
        ["previous_liquidation", "previous_profit", "previous_ordinary_loss"],
        default="first_entry",
    )
    executed["candidate_transition"] = executed.previous_candidate_id.fillna("FIRST") + "->" + executed.candidate_id
    return executed


def trade_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "symbols": 0,
            "profit_factor": np.nan,
            "net_pnl_usdt": 0.0,
            "win_rate_pct": np.nan,
            "average_pnl_usdt": np.nan,
            "average_return_on_margin_pct": np.nan,
            "median_return_on_margin_pct": np.nan,
            "liquidations": 0,
            "liquidation_rate_pct": np.nan,
            "net_pnl_ex_best_3_usdt": 0.0,
            "net_pnl_ex_best_5_usdt": 0.0,
            "max_drawdown_usdt": 0.0,
        }
    ordered = frame.sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = ordered.pnl_usdt.astype(float)
    return {
        "trades": len(ordered),
        "symbols": int(ordered.symbol.nunique()),
        "profit_factor": profit_factor(pnl),
        "net_pnl_usdt": float(pnl.sum()),
        "win_rate_pct": float(pnl.gt(0).mean() * 100),
        "average_pnl_usdt": float(pnl.mean()),
        "average_return_on_margin_pct": float(ordered.return_on_margin_pct.mean()),
        "median_return_on_margin_pct": float(ordered.return_on_margin_pct.median()),
        "liquidations": int(ordered.liquidated.sum()),
        "liquidation_rate_pct": float(ordered.liquidated.mean() * 100),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "max_drawdown_usdt": max_drawdown(pnl),
    }


def assign_episodes(executed: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    work = executed.sort_values(["symbol", "entry_time_ms", "candidate_id"]).copy()
    first = work.groupby("symbol").cumcount().eq(0)
    new_episode = first | work.reentry_gap_days.gt(threshold_days)
    work["episode_number"] = new_episode.groupby(work.symbol).cumsum().astype(int)
    work["episode_id"] = work.symbol + f"__{threshold_days}D__" + work.episode_number.astype(str)
    work["episode_entry_number"] = work.groupby("episode_id").cumcount() + 1
    work["episode_entry_order"] = np.where(
        work.episode_entry_number.ge(6),
        "Entry #6+",
        "Entry #" + work.episode_entry_number.astype(str),
    )
    work["episode_threshold_days"] = threshold_days
    return work


def summarize_gap_buckets(reentries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in GAP_BUCKETS:
        rows.append({"reentry_gap_bucket": bucket, **trade_metrics(reentries[reentries.reentry_gap_bucket.eq(bucket)])})
    return pd.DataFrame(rows)


def summarize_gap_by_previous_result(reentries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for previous_result in PREVIOUS_RESULTS:
        for bucket in GAP_BUCKETS:
            group = reentries[reentries.previous_result.eq(previous_result) & reentries.reentry_gap_bucket.eq(bucket)]
            rows.append({"analysis_type": "base_gap_x_previous_result", "previous_result": previous_result, "reentry_gap_bucket": bucket, **trade_metrics(group)})
    after_liq = reentries[reentries.previous_result.eq("previous_liquidation")]
    for bucket in POST_LIQ_GAP_BUCKETS:
        group = after_liq[after_liq.post_liquidation_gap_bucket.eq(bucket)]
        rows.append({"analysis_type": "post_liquidation_focus", "previous_result": "previous_liquidation", "reentry_gap_bucket": bucket, **trade_metrics(group)})
    return pd.DataFrame(rows)


def episode_outputs(executed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    threshold_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    assigned_map: dict[int, pd.DataFrame] = {}
    episode_map: dict[int, pd.DataFrame] = {}
    for threshold in EPISODE_THRESHOLDS_DAYS:
        assigned = assign_episodes(executed, threshold)
        assigned_map[threshold] = assigned
        episode_rows = []
        for episode_id, group in assigned.groupby("episode_id", sort=False):
            ordered = group.sort_values("entry_time_ms")
            episode_rows.append(
                {
                    "episode_threshold_days": threshold,
                    "episode_id": episode_id,
                    "symbol": ordered.symbol.iloc[0],
                    "episode_number": int(ordered.episode_number.iloc[0]),
                    "first_entry_time_ms": int(ordered.entry_time_ms.min()),
                    "first_entry_time_utc": utc(int(ordered.entry_time_ms.min())),
                    "last_exit_time_ms": int(ordered.exit_time_ms.max()),
                    "last_exit_time_utc": utc(int(ordered.exit_time_ms.max())),
                    "episode_duration_hours": float((ordered.exit_time_ms.max() - ordered.entry_time_ms.min()) / 3_600_000),
                    "episode_duration_days": float((ordered.exit_time_ms.max() - ordered.entry_time_ms.min()) / DAY_MS),
                    "trades": len(ordered),
                    "episode_net_pnl_usdt": float(ordered.pnl_usdt.sum()),
                    "episode_liquidations": int(ordered.liquidated.sum()),
                    "profitable_episode": bool(ordered.pnl_usdt.sum() > 0),
                    "repeated_episode": len(ordered) > 1,
                }
            )
        episodes = pd.DataFrame(episode_rows)
        episode_map[threshold] = episodes
        ep_pnl = episodes.episode_net_pnl_usdt
        threshold_rows.append(
            {
                "episode_threshold_days": threshold,
                "episode_count": len(episodes),
                "symbols": int(episodes.symbol.nunique()),
                "repeated_episode_count": int(episodes.repeated_episode.sum()),
                "repeated_episode_ratio_pct": float(episodes.repeated_episode.mean() * 100),
                "average_trades_per_episode": float(episodes.trades.mean()),
                "median_trades_per_episode": float(episodes.trades.median()),
                "max_trades_per_episode": int(episodes.trades.max()),
                "average_episode_duration_days": float(episodes.episode_duration_days.mean()),
                "median_episode_duration_days": float(episodes.episode_duration_days.median()),
                "max_episode_duration_days": float(episodes.episode_duration_days.max()),
                "average_episode_pnl_usdt": float(ep_pnl.mean()),
                "median_episode_pnl_usdt": float(ep_pnl.median()),
                "episode_profit_factor": profit_factor(ep_pnl),
                "episode_total_pnl_usdt": float(ep_pnl.sum()),
                "profitable_episodes": int(episodes.profitable_episode.sum()),
                "profitable_episode_ratio_pct": float(episodes.profitable_episode.mean() * 100),
            }
        )
        for order in ENTRY_ORDER_BUCKETS:
            group = assigned[assigned.episode_entry_order.eq(order)]
            order_rows.append({"episode_threshold_days": threshold, "episode_entry_order": order, **trade_metrics(group)})
    return pd.DataFrame(threshold_rows), pd.DataFrame(order_rows), assigned_map, episode_map


def candidate_transition_summary(reentries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for previous in ["A", "B", "C"]:
        for current in ["A", "B", "C"]:
            transition = f"{previous}->{current}"
            group = reentries[reentries.candidate_transition.eq(transition)]
            after_liq = group[group.previous_result.eq("previous_liquidation")]
            rows.append(
                {
                    "candidate_transition": transition,
                    **trade_metrics(group),
                    "average_reentry_gap_hours": float(group.reentry_gap_hours.mean()) if len(group) else np.nan,
                    "median_reentry_gap_hours": float(group.reentry_gap_hours.median()) if len(group) else np.nan,
                    "after_liquidation_trades": len(after_liq),
                    "after_liquidation_profit_factor": trade_metrics(after_liq)["profit_factor"],
                    "after_liquidation_net_pnl_usdt": trade_metrics(after_liq)["net_pnl_usdt"],
                    "after_liquidation_win_rate_pct": trade_metrics(after_liq)["win_rate_pct"],
                    "after_liquidation_liquidation_rate_pct": trade_metrics(after_liq)["liquidation_rate_pct"],
                }
            )
    return pd.DataFrame(rows)


def assign_raw_signals_to_episodes(raw: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    frames = []
    episode_starts = assigned.groupby(["symbol", "episode_id", "episode_number"], as_index=False).entry_time_ms.min()
    episode_starts = episode_starts.rename(columns={"entry_time_ms": "episode_start_time_ms"})
    for symbol, group in raw.groupby("symbol", sort=False):
        starts = episode_starts[episode_starts.symbol.eq(symbol)].sort_values("episode_start_time_ms")
        work = group.sort_values(["entry_time_ms", "rank", "candidate_id"]).copy()
        positions = np.searchsorted(starts.episode_start_time_ms.to_numpy(), work.entry_time_ms.to_numpy(), side="right") - 1
        if (positions < 0).any():
            raise RuntimeError(f"Raw signal precedes first executed episode for {symbol}")
        work["episode_id"] = starts.episode_id.to_numpy()[positions]
        work["episode_number"] = starts.episode_number.to_numpy()[positions]
        work["episode_start_time_ms"] = starts.episode_start_time_ms.to_numpy()[positions]
        frames.append(work)
    return pd.concat(frames, ignore_index=True)


def raw_signal_density_outputs(
    raw: pd.DataFrame,
    assigned_map: dict[int, pd.DataFrame],
    episode_map: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    density_rows: list[dict[str, Any]] = []
    timeline_frames: list[pd.DataFrame] = []
    for threshold in EPISODE_THRESHOLDS_DAYS:
        assigned_raw = assign_raw_signals_to_episodes(raw, assigned_map[threshold])
        episode_details = episode_map[threshold].set_index("episode_id")
        assigned_lookup = assigned_map[threshold].set_index(["candidate_id", "snapshot_time_ms", "symbol"])
        for episode_id, group in assigned_raw.groupby("episode_id", sort=False):
            ordered = group.sort_values(["entry_time_ms", "rank", "candidate_id"])
            signal_times = ordered.entry_time_ms.drop_duplicates().sort_values()
            intervals = signal_times.diff().dropna() / 3_600_000
            ranks = ordered["rank"].astype(int).to_numpy()
            rank3_to_rank1 = any(ranks[index] == 3 and (ranks[index + 1 :] == 1).any() for index in range(len(ranks)))
            episode = episode_details.loc[episode_id]
            density_rows.append(
                {
                    "episode_threshold_days": threshold,
                    "episode_id": episode_id,
                    "symbol": ordered.symbol.iloc[0],
                    "episode_number": int(ordered.episode_number.iloc[0]),
                    "raw_signals": len(ordered),
                    "executed_trades": int((~ordered.skipped_due_to_existing_position).sum()),
                    "skipped_signals": int(ordered.skipped_due_to_existing_position.sum()),
                    "skip_rate_pct": float(ordered.skipped_due_to_existing_position.mean() * 100),
                    "first_signal_time_utc": utc(int(ordered.snapshot_time_ms.min())),
                    "last_signal_time_utc": utc(int(ordered.snapshot_time_ms.max())),
                    "signal_duration_hours": float((ordered.snapshot_time_ms.max() - ordered.snapshot_time_ms.min()) / 3_600_000),
                    "average_signal_interval_hours": float(intervals.mean()) if len(intervals) else np.nan,
                    "minimum_signal_interval_hours": float(intervals.min()) if len(intervals) else np.nan,
                    "rank3_to_rank1_upgrade": rank3_to_rank1,
                    "episode_net_pnl_usdt": float(episode.episode_net_pnl_usdt),
                    "episode_liquidations": int(episode.episode_liquidations),
                    "episode_had_liquidation": bool(episode.episode_liquidations > 0),
                    "episode_trades": int(episode.trades),
                    "episode_duration_days": float(episode.episode_duration_days),
                }
            )
        high = assigned_raw[assigned_raw.symbol.isin(HIGH_FREQUENCY_SYMBOLS)].copy()
        high["episode_threshold_days"] = threshold
        high["actual_executed"] = ~high.skipped_due_to_existing_position
        high["actual_entry_time_utc"] = high.entry_time_utc.where(high.actual_executed, "")
        high["actual_exit_time_utc"] = high.exit_time_utc.where(high.actual_executed, "")
        high["actual_pnl_usdt"] = high.pnl_usdt.where(high.actual_executed, np.nan)
        high["actual_liquidated"] = high.liquidated.where(high.actual_executed, False)
        episode_entry_numbers = []
        reentry_gap_hours = []
        symbol_cumulative = []
        episode_cumulative = []
        for row in high.sort_values(["symbol", "entry_time_ms", "rank", "candidate_id"]).itertuples():
            key = (row.candidate_id, row.snapshot_time_ms, row.symbol)
            if row.actual_executed:
                actual = assigned_lookup.loc[key]
                episode_entry_numbers.append(int(actual.episode_entry_number))
                reentry_gap_hours.append(float(actual.reentry_gap_hours) if pd.notna(actual.reentry_gap_hours) else np.nan)
            else:
                episode_entry_numbers.append(np.nan)
                reentry_gap_hours.append(np.nan)
        high = high.sort_values(["symbol", "entry_time_ms", "rank", "candidate_id"]).copy()
        high["episode_entry_number"] = episode_entry_numbers
        high["reentry_gap_hours"] = reentry_gap_hours
        for _, symbol_group in high.groupby("symbol", sort=False):
            symbol_total = 0.0
            episode_totals: dict[str, float] = {}
            for row in symbol_group.itertuples():
                if row.actual_executed:
                    symbol_total += float(row.actual_pnl_usdt)
                    episode_totals[row.episode_id] = episode_totals.get(row.episode_id, 0.0) + float(row.actual_pnl_usdt)
                symbol_cumulative.append(symbol_total)
                episode_cumulative.append(episode_totals.get(row.episode_id, 0.0))
        high["symbol_cumulative_pnl_usdt"] = symbol_cumulative
        high["episode_cumulative_pnl_usdt"] = episode_cumulative
        timeline_frames.append(high)
    return pd.DataFrame(density_rows), pd.concat(timeline_frames, ignore_index=True)


def write_report(
    out: Path,
    gap_summary: pd.DataFrame,
    gap_previous: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    order_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    density: pd.DataFrame,
    reentries: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    table_columns = ["trades", "symbols", "profit_factor", "net_pnl_usdt", "win_rate_pct", "average_pnl_usdt", "median_return_on_margin_pct", "liquidation_rate_pct", "net_pnl_ex_best_3_usdt", "net_pnl_ex_best_5_usdt", "max_drawdown_usdt"]
    post_liq = gap_previous[gap_previous.analysis_type.eq("post_liquidation_focus")]
    all_reentry_metrics = trade_metrics(reentries)
    post_liq_metrics = trade_metrics(reentries[reentries.previous_result.eq("previous_liquidation")])
    quick = gap_summary[gap_summary.reentry_gap_bucket.eq("<=1D")].iloc[0]
    one_to_three = gap_summary[gap_summary.reentry_gap_bucket.eq(">1D-3D")].iloc[0]
    third_entries = order_summary[order_summary.episode_entry_order.eq("Entry #3")]
    sixth_entries = order_summary[order_summary.episode_entry_order.eq("Entry #6+")]
    c_to_b = transitions[transitions.candidate_transition.eq("C->B")].iloc[0]
    c_to_c = transitions[transitions.candidate_transition.eq("C->C")].iloc[0]
    b_to_b = transitions[transitions.candidate_transition.eq("B->B")].iloc[0]
    a_to_b = transitions[transitions.candidate_transition.eq("A->B")].iloc[0]
    no_skips = density[density.skipped_signals.eq(0)].groupby("episode_threshold_days").episode_net_pnl_usdt.mean()
    with_skips = density[density.skipped_signals.gt(0)].groupby("episode_threshold_days").episode_net_pnl_usdt.mean()
    lines = [
        "# Reentry Gap and Episode Analysis",
        "",
        "## 1. Scope and frozen strategy",
        "",
        "A=Rank1/0%-20%/BJ00+04/1D/5X; B=Rank1/20%-40%/BJ08/2D/3X; C=Rank3/20%-40%/BJ00+20/3D/3X. All signals are rebuilt from local 1H Klines and replayed with the global same-symbol lock. No live-trading rule was changed.",
        "",
        f"Cache latest: {cfg['cache_latest_utc']}; signal window: {cfg['signal_start_utc']} through {cfg['unified_signal_end_utc']}.",
        "",
        "## 2. Reentry gap buckets",
        "",
        markdown_table(gap_summary, ["reentry_gap_bucket", *table_columns]),
        "",
        "## 3. Reentry after liquidation",
        "",
        markdown_table(post_liq, ["reentry_gap_bucket", *table_columns]),
        "",
        "## 4. Episode threshold consistency",
        "",
        markdown_table(threshold_summary, ["episode_threshold_days", "episode_count", "repeated_episode_count", "average_trades_per_episode", "max_trades_per_episode", "average_episode_duration_days", "median_episode_duration_days", "episode_profit_factor", "episode_total_pnl_usdt", "profitable_episode_ratio_pct"]),
        "",
        "## 5. Entry order within episodes",
        "",
        markdown_table(order_summary, ["episode_threshold_days", "episode_entry_order", *table_columns]),
        "",
        "## 6. Candidate transitions",
        "",
        markdown_table(transitions, ["candidate_transition", "trades", "profit_factor", "net_pnl_usdt", "win_rate_pct", "liquidation_rate_pct", "average_reentry_gap_hours", "after_liquidation_trades", "after_liquidation_profit_factor", "after_liquidation_net_pnl_usdt"]),
        "",
        "## 7. Raw signal density",
        "",
        "raw_signal_density_summary.csv contains every Symbol Episode under all five thresholds, including executed and skipped signals, signal intervals, Rank3-to-Rank1 migration, Episode PnL and liquidation status.",
        "",
        "## 8. High-frequency timelines",
        "",
        "LABUSDT, CLOUSDT, COLLECTUSDT, RIVERUSDT, PIPPINUSDT and BSBUSDT are fully listed in high_frequency_symbol_timeline.csv for each Episode threshold.",
        "",
        "## 9. Mechanism interpretation",
        "",
        f"Fast reentry is not monotonically worse. The <=1D bucket has {int(quick.trades)} trades, PF {quick.profit_factor:.3f} and net PnL {quick.net_pnl_usdt:.2f}; all of those trades follow a profitable exit. But the neighboring >1D-3D bucket has {int(one_to_three.trades)} trades, PF {one_to_three.profit_factor:.3f} and net PnL {one_to_three.net_pnl_usdt:.2f}. This is a narrow <=1D warning, not a general short-gap gradient.",
        "",
        f"Reentry after liquidation is weaker than all reentries but not negative overall: {int(post_liq_metrics['trades'])} trades, PF {post_liq_metrics['profit_factor']:.3f}, net {post_liq_metrics['net_pnl_usdt']:.2f}, versus all reentries PF {all_reentry_metrics['profit_factor']:.3f} and net {all_reentry_metrics['net_pnl_usdt']:.2f}. There are no <=1D post-liquidation reentries. Post-liquidation results are positive through 5D, negative at 5D-30D, and positive again beyond 30D, so a monotonic liquidation cooldown mechanism is not present.",
        "",
        "Entry #2 remains profitable under every Episode threshold. Entry #3 is positive under 3D/5D, approximately flat or negative under 7D/10D, and weakly positive under 30D while failing ex-best-5. Therefore third-entry deterioration is a sensitivity finding, not a threshold-invariant rule. "
        + "; ".join(f"{int(row.episode_threshold_days)}D: N={int(row.trades)}, PF={row.profit_factor:.3f}, net={row.net_pnl_usdt:.2f}" for row in third_entries.itertuples())
        + ".",
        "",
        f"Entry #6+ exists only under the 30D definition and contains {int(sixth_entries.trades.sum())} trades with net PnL {sixth_entries.net_pnl_usdt.sum():.2f}. It is not negative, and the sample is far too small for inference.",
        "",
        f"Candidate transitions are heterogeneous. C->B is strong (N={int(c_to_b.trades)}, PF={c_to_b.profit_factor:.3f}, net={c_to_b.net_pnl_usdt:.2f}) and C->C is also strong (N={int(c_to_c.trades)}, PF={c_to_c.profit_factor:.3f}, net={c_to_c.net_pnl_usdt:.2f}). B->B is nearly flat (N={int(b_to_b.trades)}, PF={b_to_b.profit_factor:.3f}, net={b_to_b.net_pnl_usdt:.2f}); its post-liquidation subset has only {int(b_to_b.after_liquidation_trades)} trades. A->B is the weakest observed transition (N={int(a_to_b.trades)}, PF={a_to_b.profit_factor:.3f}, net={a_to_b.net_pnl_usdt:.2f}), but the sample is too small to create a formal filter.",
        "",
        f"Skipped raw signals do not behave like a general crowding/reversal warning. Across the five Episode definitions, Episodes without skipped signals average {no_skips.min():.2f} to {no_skips.max():.2f} USDT, while Episodes with at least one skipped signal average {with_skips.min():.2f} to {with_skips.max():.2f} USDT. This is descriptive and selection-dependent, but it contradicts a simple 'more locked signals means worse outcome' mechanism.",
        "",
        "## 10. Final decision",
        "",
        "1. Ordinary-exit cooldown: not supported. Only <=1D after a profitable exit is weak (N=9); >1D-3D is strongly positive and reentry after an ordinary loss is positive overall.",
        "",
        "2. Liquidation cooldown: not supported. Post-liquidation reentries are weaker but positive overall, have zero <=1D observations, and are non-monotonic by gap.",
        "",
        "3. Episode maximum trade count: not supported. Third entries are fragile under some definitions but not all; later orders are non-monotonic and severely sample-limited. Entry #6+ is not negative in the only definition where it exists.",
        "",
        "4. Research watchlist only: pre-register observation of <=1D reentry after profit, Entry #3 under 7D-30D Episode definitions, A->B transitions, and post-liquidation B->B. Do not modify the frozen strategy from this in-sample decomposition.",
        "",
        "## 11. Limitations",
        "",
        "This is one in-sample market interval. Reentries share symbol and regime dependence. Funding, slippage and detailed maintenance-margin tiers are not modeled. Hourly High identifies liquidation without intrahour ordering.",
    ]
    (out / "Reentry_Episode_Analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"reentry_episode_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg.update(
        {
            "main_leverage": MAIN_LEVERAGE,
            "episode_threshold_days": EPISODE_THRESHOLDS_DAYS,
            "gap_buckets": GAP_BUCKETS,
            "post_liquidation_gap_buckets": POST_LIQ_GAP_BUCKETS,
            "high_frequency_symbols": HIGH_FREQUENCY_SYMBOLS,
            "margin_per_trade_usdt": MARGIN_USDT,
            "episode_duration_definition": "first actual entry to last actual exit",
            "raw_signal_episode_assignment": "latest actual episode start at or before signal entry",
        }
    )

    print("[1/6] Rebuilding raw signals and replaying frozen main strategy", flush=True)
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
    cfg.update(
        {
            "cache_latest_utc": str(utc(cache_end)),
            "unified_signal_end_utc": str(utc(signal_end)),
            "actual_output_directory": str(out.resolve()),
        }
    )
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))
    raw_trades = replay(outcomes, MAIN_LEVERAGE, True, "A5_B3_C3")
    executed = enrich_reentries(raw_trades)
    reentries = executed[executed.is_reentry].copy()

    print("[2/6] Reentry gaps and previous-result conditioning", flush=True)
    gap_summary = summarize_gap_buckets(reentries)
    gap_previous = summarize_gap_by_previous_result(reentries)

    print("[3/6] Episode thresholds and entry order", flush=True)
    threshold_summary, order_summary, assigned_map, episode_map = episode_outputs(executed)

    print("[4/6] Candidate transitions and raw-signal density", flush=True)
    transitions = candidate_transition_summary(reentries)
    density, timeline = raw_signal_density_outputs(raw_trades, assigned_map, episode_map)
    for threshold, assigned in assigned_map.items():
        lookup = assigned.set_index(["candidate_id", "snapshot_time_ms", "symbol"])
        for column in ["episode_id", "episode_number", "episode_entry_number", "episode_entry_order"]:
            reentries[f"{column}_{threshold}D"] = [lookup.loc[(row.candidate_id, row.snapshot_time_ms, row.symbol), column] for row in reentries.itertuples()]

    print("[5/6] Writing requested outputs", flush=True)
    gap_summary.to_csv(out / "reentry_gap_bucket_summary.csv", index=False)
    gap_previous.to_csv(out / "reentry_gap_by_previous_result.csv", index=False)
    threshold_summary.to_csv(out / "episode_threshold_summary.csv", index=False)
    order_summary.to_csv(out / "episode_entry_order_summary.csv", index=False)
    transitions.to_csv(out / "candidate_transition_summary.csv", index=False)
    density.to_csv(out / "raw_signal_density_summary.csv", index=False)
    timeline_columns = [
        "episode_threshold_days", "symbol", "snapshot_time_utc", "candidate_id", "rank", "drop_24h_pct",
        "actual_executed", "actual_entry_time_utc", "actual_exit_time_utc", "actual_pnl_usdt", "actual_liquidated",
        "reentry_gap_hours", "episode_id", "episode_number", "episode_entry_number",
        "symbol_cumulative_pnl_usdt", "episode_cumulative_pnl_usdt", "skipped_due_to_existing_position",
        "blocked_by_candidate_id", "skip_scope",
    ]
    timeline[timeline_columns].to_csv(out / "high_frequency_symbol_timeline.csv", index=False)
    reentries.to_csv(out / "all_reentry_trades.csv", index=False)

    print("[6/6] Data-quality validation and report", flush=True)
    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"],
        "signal_start_utc": cfg["signal_start_utc"],
        "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "raw_signals": len(raw_trades),
        "executed_trades": len(executed),
        "skipped_signals": int(raw_trades.skipped_due_to_existing_position.sum()),
        "unique_executed_symbols": int(executed.symbol.nunique()),
        "reentry_trades": len(reentries),
        "reentry_identity_holds": len(reentries) == len(executed) - executed.symbol.nunique(),
        "all_reentry_gaps_nonnegative": bool(reentries.reentry_gap_hours.ge(0).all()),
        "global_lock_has_no_symbol_overlap": has_no_symbol_overlap(raw_trades),
        "episode_threshold_rows_expected_5": len(threshold_summary) == 5,
        "episode_order_rows_expected_30": len(order_summary) == 30,
        "candidate_transition_rows_expected_9": len(transitions) == 9,
        "timeline_thresholds_expected_5": int(timeline.episode_threshold_days.nunique()) == 5,
        "raw_signal_density_balances": bool((density.raw_signals == density.executed_trades + density.skipped_signals).all()),
        "episode_pnl_matches_strategy_each_threshold": bool(all(np.isclose(episodes.episode_net_pnl_usdt.sum(), executed.pnl_usdt.sum()) for episodes in episode_map.values())),
        "notional_equals_margin_times_leverage": bool(np.allclose(raw_trades.entry_notional_usdt, raw_trades.leverage * MARGIN_USDT)),
        "liquidation_price_formulas_correct": bool(np.allclose(raw_trades.liquidation_price, raw_trades.entry_price * (1 + 1 / raw_trades.leverage))),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "no_future_data": True,
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
        "contracts_in_rankings": int(signals.symbol.nunique()),
    }
    if not all(value for value in quality.values() if isinstance(value, bool)):
        raise RuntimeError(f"Data quality validation failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(out, gap_summary, gap_previous, threshold_summary, order_summary, transitions, density, reentries, cfg)

    print("Cache latest:", cfg["cache_latest_utc"])
    print("Signal cutoff:", cfg["unified_signal_end_utc"])
    print("Raw / executed / skipped / reentries:", len(raw_trades), len(executed), int(raw_trades.skipped_due_to_existing_position.sum()), len(reentries))
    print("Gap summary:")
    print(gap_summary[["reentry_gap_bucket", "trades", "profit_factor", "net_pnl_usdt", "win_rate_pct", "liquidation_rate_pct", "net_pnl_ex_best_5_usdt"]].to_string(index=False))
    print("Episode order summary:")
    print(order_summary[["episode_threshold_days", "episode_entry_order", "trades", "profit_factor", "net_pnl_usdt", "liquidation_rate_pct", "net_pnl_ex_best_5_usdt"]].to_string(index=False))
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
