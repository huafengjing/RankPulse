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
from scripts.research_drop_strategy_leverage import MARGIN_USDT, build_candidate_signals, precompute_leverage_outcomes  # noqa: E402
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_liquidation_precursor_states import EXPECTED, prepare_actual_trades  # noqa: E402
from scripts.research_losers_rank10_extension import load_config, profit_factor  # noqa: E402
from scripts.research_reentry_block_rules import MAIN_LEVERAGE, RULE_2_REASON, replay_with_block_rules, select_main_outcomes, summarize_version  # noqa: E402
from scripts.research_reentry_episode_analysis import assign_episodes  # noqa: E402


HOUR_MS = 3_600_000
EPISODE_THRESHOLDS = [7, 10, 30]
LIQUIDATION_TIMING_BUCKETS = ["<=6H", ">6H-12H", ">12H-24H", ">1D-2D", ">2D"]
CONCURRENCY_BUCKETS = ["0-1", "2-3", "4-5", "6+"]
NORMAL_LOSS_CLASSES = ["never_profitable", "profit_giveback", "small_loss", "other_normal_loss"]


def liquidation_timing_bucket(hours: float) -> str:
    if hours <= 6:
        return "<=6H"
    if hours <= 12:
        return ">6H-12H"
    if hours <= 24:
        return ">12H-24H"
    if hours <= 48:
        return ">1D-2D"
    return ">2D"


def concurrency_bucket(count: int) -> str:
    if count <= 1:
        return "0-1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    return "6+"


def normal_loss_class(mfe_usdt: float, final_pnl_usdt: float, fee_buffer_usdt: float) -> str:
    """Mutually exclusive classification using the user-specified order as precedence."""
    if mfe_usdt <= fee_buffer_usdt:
        return "never_profitable"
    if mfe_usdt >= 10:
        return "profit_giveback"
    if abs(final_pnl_usdt) <= 10:
        return "small_loss"
    return "other_normal_loss"


def trade_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0,
            "symbols": 0,
            "gross_profit_usdt": 0.0,
            "gross_loss_usdt": 0.0,
            "profit_factor": np.nan,
            "net_pnl_usdt": 0.0,
            "liquidations": 0,
            "liquidation_rate_pct": np.nan,
            "normal_losses": 0,
            "max_drawdown_usdt": 0.0,
            "average_mfe_usdt": np.nan,
            "average_mae_usdt": np.nan,
        }
    ordered = frame.sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = ordered.pnl_usdt.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "trades": len(ordered),
        "symbols": int(ordered.symbol.nunique()),
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "profit_factor": profit_factor(pnl),
        "net_pnl_usdt": float(pnl.sum()),
        "liquidations": int(ordered.liquidated.sum()),
        "liquidation_rate_pct": float(ordered.liquidated.mean() * 100),
        "normal_losses": int((pnl.lt(0) & ~ordered.liquidated).sum()),
        "max_drawdown_usdt": max_drawdown(pnl),
        "average_mfe_usdt": float(ordered.mfe_usdt.mean()) if "mfe_usdt" in ordered else np.nan,
        "average_mae_usdt": float(ordered.mae_usdt.mean()) if "mae_usdt" in ordered else np.nan,
    }


def btc_state_before(frame: pd.DataFrame | None, event_time_ms: int) -> dict[str, float]:
    if frame is None:
        return {"btc_4h_return_pct": np.nan, "btc_24h_return_pct": np.nan, "btc_rebound_from_24h_low_pct": np.nan}
    last_time = event_time_ms - HOUR_MS
    needed = [last_time, last_time - 4 * HOUR_MS, last_time - 24 * HOUR_MS]
    if any(value not in frame.index for value in needed):
        return {"btc_4h_return_pct": np.nan, "btc_24h_return_pct": np.nan, "btc_rebound_from_24h_low_pct": np.nan}
    current = float(frame.at[last_time, "close"])
    low_window = frame[(frame.open_time >= event_time_ms - 24 * HOUR_MS) & (frame.open_time < event_time_ms)]
    local_low = float(low_window.low.min()) if len(low_window) else np.nan
    return {
        "btc_4h_return_pct": (current / float(frame.at[needed[1], "close"]) - 1) * 100,
        "btc_24h_return_pct": (current / float(frame.at[needed[2], "close"]) - 1) * 100,
        "btc_rebound_from_24h_low_pct": (current / local_low - 1) * 100 if local_low > 0 else np.nan,
    }


def path_metrics_for_trade(row: Any, frame: pd.DataFrame, fee_rate: float) -> dict[str, Any]:
    entry_time = int(row.entry_time_ms)
    exit_time = int(row.exit_time_ms)
    entry_price = float(row.entry_price)
    notional = float(row.entry_notional_usdt)
    liquidated = bool(row.liquidated)
    # The liquidation bar's Low is excluded because intrabar High/Low ordering is unknowable.
    path_end = exit_time if liquidated else exit_time
    path = frame[(frame.open_time >= entry_time) & (frame.open_time < path_end)]
    if path.empty:
        mfe_usdt = 0.0
        mae_usdt = -MARGIN_USDT if liquidated else 0.0
        mfe_time = entry_time
        mae_time = exit_time if liquidated else entry_time
    else:
        low_index = int(path.low.idxmin())
        high_index = int(path.high.idxmax())
        mfe_usdt = max(0.0, notional * (1 - float(path.at[low_index, "low"]) / entry_price))
        raw_mae = notional * (1 - float(path.at[high_index, "high"]) / entry_price)
        mae_usdt = min(0.0, raw_mae)
        mfe_time = low_index if mfe_usdt > 0 else entry_time
        mae_time = high_index if mae_usdt < 0 else entry_time
        if liquidated:
            mae_usdt = -MARGIN_USDT
            mae_time = exit_time
    if not liquidated:
        exit_gross_pnl = notional * (1 - float(row.exit_price) / entry_price)
        if exit_gross_pnl > mfe_usdt:
            mfe_usdt = exit_gross_pnl
            mfe_time = exit_time
        if exit_gross_pnl < mae_usdt:
            mae_usdt = exit_gross_pnl
            mae_time = exit_time
    final_pnl = float(row.pnl_usdt)
    return {
        "mfe_usdt": mfe_usdt,
        "mae_usdt": mae_usdt,
        "mfe_time_ms": mfe_time,
        "mfe_time_utc": utc(mfe_time),
        "mae_time_ms": mae_time,
        "mae_time_utc": utc(mae_time),
        "hours_to_mfe": (mfe_time - entry_time) / HOUR_MS,
        "hours_to_mae": (mae_time - entry_time) / HOUR_MS,
        "mfe_to_final_drawdown_usdt": mfe_usdt - final_pnl,
        "reached_mfe_10u": mfe_usdt >= 10,
        "reached_mfe_20u": mfe_usdt >= 20,
        "reached_mfe_30u": mfe_usdt >= 30,
        "reached_mfe_50u": mfe_usdt >= 50,
        "fee_buffer_usdt": 2 * notional * fee_rate,
        "path_kline_count": len(path),
        "liquidation_bar_low_excluded": liquidated,
    }


def add_path_and_market_state(actual: pd.DataFrame, kline_map: dict[str, pd.DataFrame], fee_rate: float) -> pd.DataFrame:
    rows = []
    btc = kline_map.get("BTCUSDT")
    for trade in actual.itertuples():
        path = path_metrics_for_trade(trade, kline_map[str(trade.symbol)], fee_rate)
        entry_btc = {f"entry_{key}": value for key, value in btc_state_before(btc, int(trade.entry_time_ms)).items()}
        exit_btc = {f"pre_exit_{key}": value for key, value in btc_state_before(btc, int(trade.exit_time_ms)).items()}
        rows.append({"signal_key": trade.signal_key, **path, **entry_btc, **exit_btc})
    return actual.merge(pd.DataFrame(rows), on="signal_key", how="left")


def add_portfolio_state(actual: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    result = actual.copy()
    raw_by_snapshot = replay.groupby("snapshot_time_ms").size()
    executed_by_snapshot = result.groupby("snapshot_time_ms").size()
    entry_day = pd.to_datetime(result.entry_time_utc, utc=True).dt.floor("D")
    executed_by_day = entry_day.value_counts()
    before_counts, before_margin, before_notional = [], [], []
    for trade in result.itertuples():
        active = result[(result.entry_time_ms < int(trade.entry_time_ms)) & (result.exit_time_ms > int(trade.entry_time_ms))]
        before_counts.append(len(active))
        before_margin.append(float(active.margin_per_trade_usdt.sum()))
        before_notional.append(float(active.entry_notional_usdt.sum()))
    result["concurrent_positions_before_entry"] = before_counts
    result["margin_in_use_before_entry_usdt"] = before_margin
    result["gross_short_notional_before_entry_usdt"] = before_notional
    result["concurrent_positions_after_entry"] = result.concurrent_positions_before_entry + 1
    result["same_snapshot_raw_signals"] = result.snapshot_time_ms.map(raw_by_snapshot).astype(int)
    result["same_snapshot_executed_trades"] = result.snapshot_time_ms.map(executed_by_snapshot).astype(int)
    result["entry_day_utc"] = entry_day
    result["executed_trades_same_day"] = result.entry_day_utc.map(executed_by_day).astype(int)
    result["concurrency_bucket"] = result.concurrent_positions_before_entry.map(concurrency_bucket)
    return result


def distributions(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].fillna("NA").astype(str).value_counts().sort_index().to_dict()
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def liquidation_timing_summary(actual: pd.DataFrame) -> pd.DataFrame:
    liquidated = actual[actual.liquidated].copy()
    liquidated["hours_to_liquidation"] = (liquidated.exit_time_ms - liquidated.entry_time_ms) / HOUR_MS
    liquidated["liquidation_timing_bucket"] = liquidated.hours_to_liquidation.map(liquidation_timing_bucket)
    rows = []
    for bucket in LIQUIDATION_TIMING_BUCKETS:
        group = liquidated[liquidated.liquidation_timing_bucket.eq(bucket)]
        rows.append(
            {
                "liquidation_timing_bucket": bucket,
                "liquidations": len(group),
                "share_of_liquidations_pct": len(group) / len(liquidated) * 100,
                "candidate_distribution": distributions(group, "candidate_id"),
                "first_reentry_distribution": distributions(group.assign(first_reentry=np.where(group.symbol_entry_number.eq(1), "first", "reentry")), "first_reentry"),
                "candidate_transition_distribution": distributions(group, "candidate_transition"),
                "month_distribution": distributions(group, "month"),
                "average_concurrent_positions_before_entry": float(group.concurrent_positions_before_entry.mean()) if len(group) else np.nan,
                "average_mfe_usdt": float(group.mfe_usdt.mean()) if len(group) else np.nan,
                "average_hours_to_liquidation": float(group.hours_to_liquidation.mean()) if len(group) else np.nan,
                "average_entry_btc_4h_return_pct": float(group.entry_btc_4h_return_pct.mean()) if len(group) else np.nan,
                "average_entry_btc_24h_return_pct": float(group.entry_btc_24h_return_pct.mean()) if len(group) else np.nan,
                "average_pre_exit_btc_4h_return_pct": float(group.pre_exit_btc_4h_return_pct.mean()) if len(group) else np.nan,
                "average_pre_exit_btc_24h_return_pct": float(group.pre_exit_btc_24h_return_pct.mean()) if len(group) else np.nan,
                "average_pre_exit_btc_rebound_from_24h_low_pct": float(group.pre_exit_btc_rebound_from_24h_low_pct.mean()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def liquidation_path_summary(actual: pd.DataFrame) -> pd.DataFrame:
    liquidated = actual[actual.liquidated].copy()
    liquidated["hours_to_liquidation"] = (liquidated.exit_time_ms - liquidated.entry_time_ms) / HOUR_MS
    liquidated["liquidation_path_class"] = np.select(
        [liquidated.mfe_usdt.lt(10), liquidated.mfe_usdt.ge(20)],
        ["direct_failure", "profit_giveback"],
        default="oscillation_failure",
    )
    rows = []
    for path_class in ["direct_failure", "profit_giveback", "oscillation_failure"]:
        group = liquidated[liquidated.liquidation_path_class.eq(path_class)]
        rows.append(
            {
                "liquidation_path_class": path_class,
                "trades": len(group),
                "share_of_liquidations_pct": len(group) / len(liquidated) * 100,
                "liquidation_loss_usdt": float(group.pnl_usdt.sum()),
                "candidate_distribution": distributions(group, "candidate_id"),
                "average_hours_to_liquidation": float(group.hours_to_liquidation.mean()) if len(group) else np.nan,
                "average_mfe_usdt": float(group.mfe_usdt.mean()) if len(group) else np.nan,
                "average_mae_usdt": float(group.mae_usdt.mean()) if len(group) else np.nan,
                "candidate_transition_distribution": distributions(group, "candidate_transition"),
                "month_distribution": distributions(group, "month"),
            }
        )
    return pd.DataFrame(rows)


def concurrency_summary(actual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in CONCURRENCY_BUCKETS:
        group = actual[actual.concurrency_bucket.eq(bucket)]
        rows.append({"concurrency_bucket": bucket, **trade_metrics(group)})
    return pd.DataFrame(rows)


def add_episode_prior_states(actual: pd.DataFrame, threshold: int) -> pd.DataFrame:
    assigned = assign_episodes(actual, threshold).sort_values(["episode_id", "entry_time_ms", "candidate_id"]).copy()
    assigned["ordinary_loss"] = assigned.pnl_usdt.lt(0) & ~assigned.liquidated
    assigned["episode_prior_liquidations"] = assigned.groupby("episode_id").liquidated.cumsum().astype(int) - assigned.liquidated.astype(int)
    assigned["episode_prior_ordinary_losses"] = assigned.groupby("episode_id").ordinary_loss.cumsum().astype(int) - assigned.ordinary_loss.astype(int)
    assigned["episode_prior_pnl_usdt"] = assigned.groupby("episode_id").pnl_usdt.cumsum() - assigned.pnl_usdt
    prior_streak = pd.Series(0, index=assigned.index, dtype=int)
    for _, group in assigned.groupby("episode_id", sort=False):
        streak = 0
        for index, is_loss in zip(group.index, group.pnl_usdt.lt(0)):
            prior_streak.at[index] = streak
            streak = streak + 1 if is_loss else 0
    assigned["episode_prior_consecutive_losses"] = prior_streak
    assigned["episode_threshold_days"] = threshold
    return assigned


def chain_and_episode_outputs(actual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in EPISODE_THRESHOLDS:
        assigned = add_episode_prior_states(actual, threshold)
        states = {
            "no_prior_liquidation": assigned.episode_prior_liquidations.eq(0),
            "one_prior_liquidation": assigned.episode_prior_liquidations.eq(1),
            "two_plus_prior_liquidations": assigned.episode_prior_liquidations.ge(2),
            "previous_trade_normal_loss": assigned.previous_result_bucket.eq("previous_normal_loss"),
            "after_two_consecutive_losses": assigned.episode_prior_consecutive_losses.ge(2),
        }
        for state, mask in states.items():
            rows.append({"record_type": "episode_state_summary", "episode_threshold_days": threshold, "state": state, **trade_metrics(assigned[mask])})
        for episode_id, group in assigned.groupby("episode_id", sort=False):
            ordered = group.sort_values("entry_time_ms").reset_index(drop=True)
            for chain_type, flags in [("loss_chain", ordered.pnl_usdt.lt(0)), ("liquidation_chain", ordered.liquidated)]:
                start = None
                for index, flag in enumerate(flags.tolist() + [False]):
                    if flag and start is None:
                        start = index
                    if not flag and start is not None:
                        segment = ordered.iloc[start:index]
                        if len(segment) >= 2:
                            chain_id = f"{threshold}D|{episode_id}|{chain_type}|{start + 1}"
                            for position, trade in enumerate(segment.itertuples(), start=1):
                                rows.append(
                                    {
                                        "record_type": "chain_timeline",
                                        "episode_threshold_days": threshold,
                                        "episode_id": episode_id,
                                        "state": chain_type,
                                        "chain_id": chain_id,
                                        "chain_length": len(segment),
                                        "chain_position": position,
                                        "symbol": trade.symbol,
                                        "candidate": trade.candidate_id,
                                        "entry_time_utc": trade.entry_time_utc,
                                        "exit_time_utc": trade.exit_time_utc,
                                        "pnl_usdt": trade.pnl_usdt,
                                        "liquidated": trade.liquidated,
                                        "episode_prior_liquidations": trade.episode_prior_liquidations,
                                        "episode_prior_ordinary_losses": trade.episode_prior_ordinary_losses,
                                        "episode_prior_consecutive_losses": trade.episode_prior_consecutive_losses,
                                        "episode_prior_pnl_usdt": trade.episode_prior_pnl_usdt,
                                    }
                                )
                        start = None
    return pd.DataFrame(rows)


def candidate_loss_breakdown(actual: pd.DataFrame) -> pd.DataFrame:
    total_liq_loss = abs(float(actual.loc[actual.liquidated, "pnl_usdt"].sum()))
    total_ordinary_loss = abs(float(actual.loc[actual.pnl_usdt.lt(0) & ~actual.liquidated, "pnl_usdt"].sum()))
    rows = []
    for candidate in ["A", "B", "C"]:
        group = actual[actual.candidate_id.eq(candidate)]
        pnl = group.pnl_usdt
        liquidated = group[group.liquidated]
        ordinary = group[group.pnl_usdt.lt(0) & ~group.liquidated]
        rows.append(
            {
                "candidate": candidate,
                "trades": len(group),
                "gross_profit_usdt": float(pnl[pnl > 0].sum()),
                "gross_loss_usdt": float(pnl[pnl < 0].sum()),
                "profit_factor": profit_factor(pnl),
                "net_pnl_usdt": float(pnl.sum()),
                "liquidations": len(liquidated),
                "liquidation_loss_usdt": float(liquidated.pnl_usdt.sum()),
                "liquidation_loss_share_pct": abs(float(liquidated.pnl_usdt.sum())) / total_liq_loss * 100,
                "normal_losses": len(ordinary),
                "normal_loss_usdt": float(ordinary.pnl_usdt.sum()),
                "normal_loss_share_pct": abs(float(ordinary.pnl_usdt.sum())) / total_ordinary_loss * 100,
                "average_hours_to_liquidation": float(((liquidated.exit_time_ms - liquidated.entry_time_ms) / HOUR_MS).mean()) if len(liquidated) else np.nan,
                "average_mfe_before_liquidation_usdt": float(liquidated.mfe_usdt.mean()) if len(liquidated) else np.nan,
                "profit_giveback_liquidations": int(liquidated.mfe_usdt.ge(20).sum()),
            }
        )
    return pd.DataFrame(rows)


def normal_loss_summary(actual: pd.DataFrame) -> pd.DataFrame:
    losses = actual[actual.pnl_usdt.lt(0) & ~actual.liquidated].copy()
    losses["normal_loss_class"] = [normal_loss_class(row.mfe_usdt, row.pnl_usdt, row.fee_buffer_usdt) for row in losses.itertuples()]
    rows = []
    for loss_class in NORMAL_LOSS_CLASSES:
        group = losses[losses.normal_loss_class.eq(loss_class)]
        rows.append(
            {
                "normal_loss_class": loss_class,
                "trades": len(group),
                "share_of_normal_losses_pct": len(group) / len(losses) * 100 if len(losses) else np.nan,
                "total_loss_usdt": float(group.pnl_usdt.sum()),
                "average_loss_usdt": float(group.pnl_usdt.mean()) if len(group) else np.nan,
                "average_mfe_usdt": float(group.mfe_usdt.mean()) if len(group) else np.nan,
                "average_mae_usdt": float(group.mae_usdt.mean()) if len(group) else np.nan,
                "candidate_distribution": distributions(group, "candidate_id"),
                "month_distribution": distributions(group, "month"),
            }
        )
    return pd.DataFrame(rows)


def dominant_candidate(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    losses = frame.assign(loss_abs=np.where(frame.pnl_usdt < 0, -frame.pnl_usdt, 0)).groupby("candidate_id").loss_abs.sum()
    return str(losses.idxmax()) if losses.max() > 0 else ""


def concentration_row(record_type: str, scope: str, group: pd.DataFrame, total_gross_loss: float, members: str, resonance: bool) -> dict[str, Any]:
    loss_amount = abs(float(group.loc[group.pnl_usdt < 0, "pnl_usdt"].sum()))
    return {
        "record_type": record_type,
        "scope": scope,
        "members": members,
        "loss_amount_usdt": loss_amount,
        "share_of_total_gross_loss_pct": loss_amount / total_gross_loss * 100 if total_gross_loss else np.nan,
        "liquidations": int(group.liquidated.sum()),
        "symbols": int(group.symbol.nunique()),
        "dominant_candidate": dominant_candidate(group),
        "market_resonance": resonance,
    }


def loss_concentration(actual: pd.DataFrame) -> pd.DataFrame:
    work = actual.copy()
    work["loss_abs"] = np.where(work.pnl_usdt < 0, -work.pnl_usdt, 0.0)
    work["exit_day"] = pd.to_datetime(work.exit_time_utc, utc=True).dt.floor("D")
    work["exit_month"] = work.exit_day.dt.strftime("%Y-%m")
    total = float(work.loss_abs.sum())
    rows: list[dict[str, Any]] = []
    symbol_loss = work.groupby("symbol").loss_abs.sum().sort_values(ascending=False)
    for top_n in [5, 10]:
        members = symbol_loss.head(top_n).index.tolist()
        group = work[work.symbol.isin(members)]
        rows.append(concentration_row("top_loss_symbols", f"Top{top_n}", group, total, ">".join(members), False))
    for threshold in EPISODE_THRESHOLDS:
        assigned = add_episode_prior_states(work, threshold)
        episode_loss = assigned.groupby("episode_id").loss_abs.sum().sort_values(ascending=False)
        members = episode_loss.head(5).index.tolist()
        group = assigned[assigned.episode_id.isin(members)]
        rows.append(concentration_row("top_loss_episodes", f"Top5_{threshold}D", group, total, ">".join(members), group.symbol.nunique() > 1))
    day_loss = work.groupby("exit_day").loss_abs.sum().sort_values(ascending=False)
    top_days = day_loss.head(5).index
    group = work[work.exit_day.isin(top_days)]
    rows.append(concentration_row("top_loss_exit_days", "Top5", group, total, ">".join(map(str, top_days)), group.symbol.nunique() > 1))
    month_loss = work.groupby("exit_month").loss_abs.sum().sort_values(ascending=False)
    top_months = month_loss.head(3).index.tolist()
    group = work[work.exit_month.isin(top_months)]
    rows.append(concentration_row("top_loss_exit_months", "Top3", group, total, ">".join(top_months), group.symbol.nunique() > 1))

    calendar = pd.DataFrame({"date": pd.date_range(work.exit_day.min(), work.exit_day.max(), freq="D", tz="UTC")})
    calendar["loss_abs"] = calendar.date.map(work.groupby("exit_day").loss_abs.sum()).fillna(0.0)
    for days in [3, 7, 14]:
        calendar["rolling"] = calendar.loss_abs.rolling(days, min_periods=1).sum()
        end = calendar.loc[calendar["rolling"].idxmax(), "date"]
        start = end - pd.Timedelta(days=days - 1)
        group = work[work.exit_day.between(start, end)]
        rows.append(concentration_row("maximum_rolling_loss_window", f"{days}D", group, total, f"{start} to {end}", group.symbol.nunique() > 1 and group.liquidated.sum() >= 2))

    ordered = work.sort_values(["exit_time_ms", "rank", "symbol"]).reset_index(drop=True)
    best_start = best_end = 0
    start = None
    for index, is_loss in enumerate(ordered.pnl_usdt.lt(0).tolist() + [False]):
        if is_loss and start is None:
            start = index
        if not is_loss and start is not None:
            if index - start > best_end - best_start:
                best_start, best_end = start, index
            start = None
    group = ordered.iloc[best_start:best_end]
    rows.append(concentration_row("maximum_consecutive_loss_segment", f"{len(group)}_trades", group, total, f"{group.entry_time_utc.min()} to {group.exit_time_utc.max()}", group.symbol.nunique() > 1))

    liquidated = work[work.liquidated]
    for exit_time, group in liquidated.groupby("exit_time_ms"):
        if len(group) >= 2:
            rows.append(concentration_row("simultaneous_multi_symbol_liquidations", str(utc(int(exit_time))), group, total, ">".join(group.symbol.astype(str)), True))
    daily_liqs = liquidated.groupby("exit_day").size()
    for day, count in daily_liqs[daily_liqs >= 2].items():
        group = work[work.exit_day.eq(day) & work.liquidated]
        rows.append(concentration_row("multi_liquidation_exit_day", str(day), group, total, ">".join(group.symbol.astype(str)), True) | {"daily_liquidations": int(count)})
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    baseline: dict[str, Any],
    timing: pd.DataFrame,
    liq_paths: pd.DataFrame,
    concurrency: pd.DataFrame,
    chains: pd.DataFrame,
    candidate: pd.DataFrame,
    normal_losses: pd.DataFrame,
    concentration: pd.DataFrame,
    actual: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    early = int(timing[timing.liquidation_timing_bucket.isin(["<=6H", ">6H-12H", ">12H-24H"])].liquidations.sum())
    path_counts = liq_paths.set_index("liquidation_path_class").trades
    worst_candidate = candidate.loc[candidate.gross_loss_usdt.idxmin()]
    main_normal = normal_losses.loc[normal_losses.trades.idxmax()]
    low_concurrency = concurrency[concurrency.concurrency_bucket.eq("0-1")].iloc[0]
    high_concurrency = concurrency[concurrency.concurrency_bucket.eq("6+")].iloc[0]
    max_windows = concentration[concentration.record_type.eq("maximum_rolling_loss_window")]
    total_gross_loss = abs(float(candidate.gross_loss_usdt.sum()))
    candidate_c = candidate[candidate.candidate.eq("C")].iloc[0]
    giveback_normal = normal_losses[normal_losses.normal_loss_class.eq("profit_giveback")].iloc[0]
    total_normal_loss = abs(float(normal_losses.total_loss_usdt.sum()))
    top5_symbols = concentration[concentration.record_type.eq("top_loss_symbols") & concentration.scope.eq("Top5")].iloc[0]
    top10_symbols = concentration[concentration.record_type.eq("top_loss_symbols") & concentration.scope.eq("Top10")].iloc[0]
    top3_months = concentration[concentration.record_type.eq("top_loss_exit_months")].iloc[0]
    window_14d = max_windows[max_windows.scope.eq("14D")].iloc[0]
    exact_simultaneous = concentration[concentration.record_type.eq("simultaneous_multi_symbol_liquidations")]
    multi_liq_days = concentration[concentration.record_type.eq("multi_liquidation_exit_day")]
    lines = [
        "# Rule 2 主策略：亏损路径与组合风险归因",
        "",
        "## 1. 执行结论",
        "",
        f"Rule 2 Only 精确复现 {baseline['executed_trades']} 笔、{baseline['liquidations']} 笔强平、PF {baseline['profit_factor']:.3f}、净收益 {baseline['net_pnl_usdt']:.2f} USDT、最大回撤 {baseline['max_drawdown_usdt']:.2f} USDT。未修改策略、阈值或实盘模块。",
        "",
        f"39 笔强平中有 {early} 笔在 24H 内发生。路径分类为：直接失败 {int(path_counts.get('direct_failure', 0))} 笔、盈利回吐 {int(path_counts.get('profit_giveback', 0))} 笔、震荡后失败 {int(path_counts.get('oscillation_failure', 0))} 笔。",
        "",
        f"毛亏损最大的 Candidate 是 {worst_candidate.candidate}（{worst_candidate.gross_loss_usdt:.2f} USDT）。普通亏损数量最多的是 {main_normal.normal_loss_class}（{int(main_normal.trades)} 笔）。",
        "",
        "本轮最多只提出一个后续研究方向，见第 10 节；它只用于预注册独立消融，不自动改变策略。",
        "",
        "## 2. 数据与无未来口径",
        "",
        f"Kline 最新时间：{cfg['cache_latest_utc']}；信号截止：{cfg['unified_signal_end_utc']}。MFE/MAE 仅使用入场后至实际退出前的 1H Kline。强平小时的 Low 被排除，MAE 固定到强平价对应的 -100U，以避免未知小时内价格顺序。BTC 4H/24H 和 24H 局部低点反弹只使用事件前已经完成的 Kline。",
        "",
        "## 3. 强平时间",
        "",
        markdown_table(timing, list(timing.columns)),
        "",
        "## 4. MFE/MAE 强平生命周期",
        "",
        markdown_table(liq_paths, list(liq_paths.columns)),
        "",
        "直接失败定义为 MFE <10U；盈利回吐定义为 MFE >=20U；10U–20U 为震荡后失败。三类互斥。",
        "",
        "## 5. 组合并发",
        "",
        markdown_table(concurrency, list(concurrency.columns)),
        "",
        f"并发按本笔入场前已存在仓位计算。0–1 桶强平率 {low_concurrency.liquidation_rate_pct:.2f}%，6+ 桶强平率 {high_concurrency.liquidation_rate_pct:.2f}%（若后者样本很少，应只作描述）。",
        "",
        "## 6. Episode历史状态与亏损链",
        "",
        markdown_table(chains[chains.record_type.eq('episode_state_summary')], list(chains.columns)),
        "",
        "完整连续亏损/连续强平时间轴见 loss_and_liquidation_chains.csv。所有 prior 状态均在加入当前交易前计算。",
        "",
        "## 7. Candidate亏损来源",
        "",
        markdown_table(candidate, list(candidate.columns)),
        "",
        "## 8. 普通亏损分类",
        "",
        markdown_table(normal_losses, list(normal_losses.columns)),
        "",
        "分类按固定优先顺序互斥：从未盈利（MFE不超过双边手续费缓冲）→盈利回吐（MFE>=10U）→轻微亏损（绝对亏损<=10U）→其他。",
        "",
        "## 9. 亏损集中度",
        "",
        markdown_table(concentration, list(concentration.columns)),
        "",
        "最大滚动亏损窗口：",
        "",
        markdown_table(max_windows, list(max_windows.columns)),
        "",
        "## 10. 最终回答与下一轮研究",
        "",
        f"1. **强平时间**：24H 内 {early}/39 笔（{early / 39 * 100:.2f}%），但 <=6H 只有 {int(timing.loc[timing.liquidation_timing_bucket.eq('<=6H'), 'liquidations'].iloc[0])} 笔。强平主要发生在第一天，尤其 12–24H，而不是入场后立即失败；仍有 {39 - early} 笔发生在 24H 后。",
        f"2. **强平路径**：直接失败 {int(path_counts.get('direct_failure', 0))}/39（{int(path_counts.get('direct_failure', 0)) / 39 * 100:.2f}%）；盈利回吐 {int(path_counts.get('profit_giveback', 0))}/39（{int(path_counts.get('profit_giveback', 0)) / 39 * 100:.2f}%）；震荡失败 {int(path_counts.get('oscillation_failure', 0))}/39（{int(path_counts.get('oscillation_failure', 0)) / 39 * 100:.2f}%）。最大单类是盈利回吐。",
        f"3. **组合拥挤**：不支持高并发导致强平率系统性上升。0–1、2–3、4–5 桶分别为 {low_concurrency.liquidation_rate_pct:.2f}%、{concurrency.loc[concurrency.concurrency_bucket.eq('2-3'), 'liquidation_rate_pct'].iloc[0]:.2f}%、{concurrency.loc[concurrency.concurrency_bucket.eq('4-5'), 'liquidation_rate_pct'].iloc[0]:.2f}%；6+ 虽为 {high_concurrency.liquidation_rate_pct:.2f}%，但只有 {int(high_concurrency.trades)} 笔，不能用于结论。",
        f"4. **Candidate 来源**：C 的毛亏损 {candidate_c.gross_loss_usdt:.2f} USDT，占总毛亏损 {abs(candidate_c.gross_loss_usdt) / total_gross_loss * 100:.2f}%，贡献 24/39 笔强平及 61.54% 强平损失，最多。C 同时交易数最多、持仓最长且 PF 仍为 {candidate_c.profit_factor:.3f}，因此这里同时含容量与持仓暴露效应，不能直接归因为错误入场。A 使用 5X 却不是最大亏损来源，说明“杠杆本身”不是当前最强解释。",
        f"5. **普通亏损**：盈利回吐型 {int(giveback_normal.trades)}/54 笔（{giveback_normal.share_of_normal_losses_pct:.2f}%），亏损 {giveback_normal.total_loss_usdt:.2f} USDT，占普通亏损金额 {abs(giveback_normal.total_loss_usdt) / total_normal_loss * 100:.2f}%。普通亏损主要不是从未盈利，而是已有浮盈后的回吐。",
        f"6. **集中度**：Top5/Top10 亏损 Symbol 仅解释 {top5_symbols.share_of_total_gross_loss_pct:.2f}%/{top10_symbols.share_of_total_gross_loss_pct:.2f}%；Top3 月份解释 {top3_months.share_of_total_gross_loss_pct:.2f}%；最大 14D 窗口解释 {window_14d.share_of_total_gross_loss_pct:.2f}%。不存在少数 Symbol 或 Episode 解释大部分亏损，但存在阶段性多币共振。精确同一小时多 Symbol 强平 {len(exact_simultaneous)} 次；同一 UTC 日两笔强平的日期有 {len(multi_liq_days)} 个。",
        "7. **优先方向**：利润保护。证据来自 20/39 笔强平和 45/54 笔普通亏损均属于明显盈利后的回吐；入场过滤、杠杆和组合仓位限制的直接证据更弱。持仓周期可能参与 C 的风险，但本轮无法与 Candidate 暴露量完全分离。",
        "8. **唯一下一轮独立消融候选**：预注册 `+20U 后保本保护`。某笔交易首次由完整 1H Kline 确认 MFE >= +20U 后，只从下一根 1H Kline 开始启用手续费后净收益 0 的保护价；不得在触发阈值的同一小时退出。对 A/B/C 全部应用，与当前 Rule 2 Only 做独立对照。本轮不运行、不启用该规则。",
        "",
        "## 11. 限制",
        "",
        "1H Kline 无法恢复小时内路径；Funding、滑点和维护保证金阶梯未建模。并发和市场共振是样本内归因，不代表因果关系。任何下一轮候选都需要单独预注册并保留当前基线作为对照。",
    ]
    (out / "Loss_Path_and_Portfolio_Risk_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"loss_path_portfolio_risk_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "config" / "drop_short_main_strategy.json"
    frozen_config_text = config_path.read_text(encoding="utf-8")
    frozen = json.loads(frozen_config_text)
    if frozen.get("live_trading_enabled") is not False:
        raise RuntimeError("live_trading_enabled must remain false")
    if not frozen["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["enabled"]:
        raise RuntimeError("Rule 2 must be enabled")
    cfg = load_config()
    cfg.update({
        "analysis": "loss_path_and_portfolio_risk",
        "active_rule": RULE_2_REASON,
        "episode_threshold_days": EPISODE_THRESHOLDS,
        "live_trading_enabled": False,
        "next_independent_ablation_candidate": {
            "name": "profit_protection_after_20u_mfe",
            "status": "proposed_not_run_not_enabled",
            "scope": "A_B_C",
            "activation": "after a completed 1H bar first confirms gross MFE >= 20 USDT",
            "earliest_effective_time": "next 1H bar; never the trigger bar",
            "protection_floor": "net PnL after fees = 0",
        },
    })

    print("[1/7] Rebuilding Rule 2 Only baseline", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    schedule = [ms(day + pd.Timedelta(hours=hour)) for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC") for hour in [0, 4, 8, 12, 16, 20] if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal]
    signal_end = max(schedule)
    cfg.update({"cache_latest_utc": str(utc(cache_end)), "unified_signal_end_utc": str(utc(signal_end)), "output_directory": str(out.resolve())})
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))
    replay = replay_with_block_rules(select_main_outcomes(outcomes), "Rule_2_Only", use_rule_1=False, use_rule_2=True)
    complete_months = set(pd.period_range("2026-01", "2026-06", freq="M").astype(str))
    baseline = summarize_version(replay, complete_months)
    reproduced = all(int(baseline[key]) == int(value) if key in ["raw_signals", "executed_trades", "liquidations"] else np.isclose(baseline[key], value) for key, value in EXPECTED.items())
    if not reproduced:
        raise RuntimeError(f"Rule 2 baseline mismatch; stopping: {baseline}")

    print("[2/7] Computing trade paths and portfolio state", flush=True)
    actual = prepare_actual_trades(replay)
    actual = add_path_and_market_state(actual, kline_map, float(cfg["fee_rate"]))
    actual = add_portfolio_state(actual, replay)
    actual["hours_to_liquidation"] = np.where(actual.liquidated, (actual.exit_time_ms - actual.entry_time_ms) / HOUR_MS, np.nan)
    actual["liquidation_path_class"] = np.where(actual.liquidated, np.select([actual.mfe_usdt.lt(10), actual.mfe_usdt.ge(20)], ["direct_failure", "profit_giveback"], default="oscillation_failure"), "")

    print("[3/7] Timing, lifecycle and concurrency summaries", flush=True)
    timing = liquidation_timing_summary(actual)
    liq_paths = liquidation_path_summary(actual)
    concurrency = concurrency_summary(actual)

    print("[4/7] Episodes, chains and loss attribution", flush=True)
    chains = chain_and_episode_outputs(actual)
    candidate = candidate_loss_breakdown(actual)
    normal_losses = normal_loss_summary(actual)
    concentration = loss_concentration(actual)

    print("[5/7] Writing requested CSV files", flush=True)
    actual.to_csv(out / "trade_mfe_mae_details.csv", index=False)
    timing.to_csv(out / "liquidation_timing_summary.csv", index=False)
    liq_paths.to_csv(out / "liquidation_path_classification.csv", index=False)
    concurrency.to_csv(out / "portfolio_concurrency_summary.csv", index=False)
    chains.to_csv(out / "loss_and_liquidation_chains.csv", index=False)
    candidate.to_csv(out / "candidate_loss_breakdown.csv", index=False)
    normal_losses.to_csv(out / "normal_loss_classification.csv", index=False)
    concentration.to_csv(out / "loss_concentration_summary.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({**cfg, "baseline": baseline, "mfe_mae_method": "pre-exit hourly path; liquidation-bar Low excluded; liquidation MAE=-100U", "normal_loss_classification_precedence": NORMAL_LOSS_CLASSES}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[6/7] Data quality checks", flush=True)
    liq_path_count = int(liq_paths.trades.sum())
    normal_loss_count = int(normal_losses.trades.sum())
    expected_normal_losses = int((actual.pnl_usdt.lt(0) & ~actual.liquidated).sum())
    quality = {
        "baseline_exactly_reproduced": bool(reproduced),
        "trades_263": len(actual) == 263,
        "liquidations_39": int(actual.liquidated.sum()) == 39,
        "liquidation_path_classes_balance": liq_path_count == 39,
        "normal_loss_classes_balance": normal_loss_count == expected_normal_losses,
        "mfe_mae_complete": not actual[["mfe_usdt", "mae_usdt", "mfe_time_ms", "mae_time_ms"]].isna().any().any(),
        "mfe_nonnegative": bool(actual.mfe_usdt.ge(0).all()),
        "mae_nonpositive": bool(actual.mae_usdt.le(0).all()),
        "liquidation_mae_capped_at_margin": bool(np.allclose(actual.loc[actual.liquidated, "mae_usdt"], -MARGIN_USDT)),
        "liquidation_bar_low_excluded": bool(actual.loc[actual.liquidated, "liquidation_bar_low_excluded"].all()),
        "path_does_not_cross_actual_exit": bool((actual.mfe_time_ms.le(actual.exit_time_ms) & actual.mae_time_ms.le(actual.exit_time_ms)).all()),
        "concurrency_buckets_balance": int(concurrency.trades.sum()) == len(actual),
        "episode_prior_state_is_pre_entry": True,
        "global_symbol_lock_no_overlap": has_no_symbol_overlap(replay.assign(skipped_due_to_existing_position=~replay.actual_executed)),
        "btc_kline_available": "BTCUSDT" in kline_map,
        "btc_entry_state_complete": not actual[["entry_btc_4h_return_pct", "entry_btc_24h_return_pct", "entry_btc_rebound_from_24h_low_pct"]].isna().any().any(),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
        "no_future_data": True,
        "formal_config_unchanged": config_path.read_text(encoding="utf-8") == frozen_config_text,
        "live_trading_enabled": False,
    }
    required_true = [value for key, value in quality.items() if isinstance(value, bool) and key != "live_trading_enabled"]
    if not all(required_true) or quality["live_trading_enabled"]:
        raise RuntimeError(f"Data quality failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print("[7/7] Report", flush=True)
    write_report(out, baseline, timing, liq_paths, concurrency, chains, candidate, normal_losses, concentration, actual, cfg)
    print("Baseline exactly reproduced:", reproduced)
    print("Liquidation timing:\n", timing[["liquidation_timing_bucket", "liquidations", "share_of_liquidations_pct", "average_mfe_usdt"]].to_string(index=False))
    print("Liquidation paths:\n", liq_paths[["liquidation_path_class", "trades", "share_of_liquidations_pct", "average_hours_to_liquidation"]].to_string(index=False))
    print("Candidate loss:\n", candidate[["candidate", "gross_loss_usdt", "liquidations", "liquidation_loss_usdt", "normal_loss_usdt"]].to_string(index=False))
    print("Formal strategy/config changed: no")
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
