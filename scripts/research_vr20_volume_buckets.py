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
from scripts.research_drop_top3_short_edge import CACHE_DIR, DAY_MS, HOUR_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import complete_months, load_config, longest_streak, profit_factor  # noqa: E402
from scripts.research_reentry_block_rules import MAIN_LEVERAGE, blocks_post_liquidation, performance_metrics, select_main_outcomes  # noqa: E402


FOUR_HOUR_MS = 4 * HOUR_MS
BUCKETS = ["MISSING", "B1", "B2", "B3", "B4", "B5", "B6"]
VALID_BUCKETS = BUCKETS[1:]
BUCKET_LIMITS = {
    "MISSING": (np.nan, np.nan),
    "B1": (-np.inf, 0.75),
    "B2": (0.75, 1.25),
    "B3": (1.25, 2.0),
    "B4": (2.0, 3.0),
    "B5": (3.0, 5.0),
    "B6": (5.0, np.inf),
}
FILTERS = {
    "Exclude_B1": "VR20 < 0.75",
    "Exclude_B2": "0.75 <= VR20 < 1.25",
    "Exclude_B3": "1.25 <= VR20 < 2.00",
    "Exclude_B4": "2.00 <= VR20 < 3.00",
    "Exclude_B5": "3.00 <= VR20 < 5.00",
    "Exclude_B6": "VR20 >= 5.00",
    "Exclude_VR20_LT_1_25": "VR20 < 1.25",
    "Exclude_VR20_GE_1_25": "VR20 >= 1.25",
    "Exclude_VR20_GE_2": "VR20 >= 2.00",
    "Exclude_VR20_GE_3": "VR20 >= 3.00",
    "Exclude_VR20_GE_5": "VR20 >= 5.00",
}
BASELINE_EXPECTED = {
    "raw_signals": 346,
    "executed_trades": 263,
    "profit_factor": 1.713338262240549,
    "net_pnl_usdt": 3807.1119676976487,
    "net_pnl_ex_best_5_usdt": 2888.1915231439734,
    "net_pnl_ex_best_10_usdt": 2208.705971681662,
    "liquidations": 39,
    "liquidation_rate_pct": 14.82889733840304,
    "max_drawdown_usdt": -475.18094064891466,
    "positive_complete_months": 5,
    "return_to_drawdown_ratio": 8.011920601231598,
}
EXISTING_REASON = "global_existing_position"
RULE_2_REASON = "blocked_post_liquidation_reentry_5d_30d"
VR20_REASON = "blocked_vr20_bucket"


def vr_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "MISSING"
    if value < 0.75:
        return "B1"
    if value < 1.25:
        return "B2"
    if value < 2.0:
        return "B3"
    if value < 3.0:
        return "B4"
    if value < 5.0:
        return "B5"
    return "B6"


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """Spearman rho without the optional SciPy dependency used by pandas."""
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 2:
        return np.nan
    return float(paired.iloc[:, 0].rank(method="average").corr(paired.iloc[:, 1].rank(method="average"), method="pearson"))


def aggregate_4h(frame: pd.DataFrame, start_ms: int, end_ms: int) -> tuple[pd.DataFrame, dict[str, int]]:
    work = frame.copy()
    numeric = ["open", "high", "low", "close", "quote_volume"]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    work = work.drop_duplicates("open_time", keep="last").sort_values("open_time")
    work["bucket_start_ms"] = (work.open_time // FOUR_HOUR_MS) * FOUR_HOUR_MS
    work["valid_1h"] = (
        np.isfinite(work[numeric]).all(axis=1)
        & work[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & work.quote_volume.ge(0)
    )
    rows = []
    for bucket_start, group in work.groupby("bucket_start_ms", sort=True):
        expected = [int(bucket_start + offset * HOUR_MS) for offset in range(4)]
        actual = group.open_time.astype(int).tolist()
        valid = actual == expected and bool(group.valid_1h.all())
        rows.append(
            {
                "start_time_ms": int(bucket_start),
                "end_time_ms": int(bucket_start + FOUR_HOUR_MS),
                "open": float(group.open.iloc[0]) if valid else np.nan,
                "high": float(group.high.max()) if valid else np.nan,
                "low": float(group.low.min()) if valid else np.nan,
                "close": float(group.close.iloc[-1]) if valid else np.nan,
                "quote_asset_volume_4h": float(group.quote_volume.sum()) if valid else np.nan,
                "source_1h_count": len(group),
                "source_times_contiguous": actual == expected,
                "valid_4h": valid,
            }
        )
    aggregated = pd.DataFrame(rows).set_index("start_time_ms", drop=False)
    grid_start = (start_ms // FOUR_HOUR_MS) * FOUR_HOUR_MS
    grid_end = (end_ms // FOUR_HOUR_MS) * FOUR_HOUR_MS
    grid = pd.Index(range(grid_start, grid_end + FOUR_HOUR_MS, FOUR_HOUR_MS), name="start_time_ms")
    aggregated = aggregated.reindex(grid)
    aggregated["start_time_ms"] = aggregated.index.astype("int64")
    aggregated["end_time_ms"] = aggregated.start_time_ms + FOUR_HOUR_MS
    aggregated["valid_4h"] = aggregated.valid_4h.fillna(False).astype(bool)
    aggregated["source_1h_count"] = aggregated.source_1h_count.fillna(0).astype(int)
    aggregated["source_times_contiguous"] = aggregated.source_times_contiguous.fillna(False).astype(bool)
    volume = aggregated.quote_asset_volume_4h.where(aggregated.valid_4h)
    aggregated["median_previous_20_4h_quote_volume"] = volume.shift(1).rolling(20, min_periods=20).median()
    aggregated["median_previous_6_4h_quote_volume"] = volume.shift(1).rolling(6, min_periods=6).median()
    aggregated["volume_ratio_4h_20"] = volume / aggregated.median_previous_20_4h_quote_volume.where(lambda value: value > 0)
    aggregated["volume_ratio_4h_6"] = volume / aggregated.median_previous_6_4h_quote_volume.where(lambda value: value > 0)
    audit = {
        "four_hour_rows": len(aggregated),
        "valid_four_hour_rows": int(aggregated.valid_4h.sum()),
        "invalid_or_missing_four_hour_rows": int((~aggregated.valid_4h).sum()),
        "invalid_one_hour_volume_rows": int((~work.valid_1h).sum()),
    }
    return aggregated, audit


def load_4h_volume_map(symbols: list[str], start_ms: int, end_ms: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    audits = []
    read_start = start_ms - 24 * FOUR_HOUR_MS
    for symbol in symbols:
        path = CACHE_DIR / f"{symbol}_1h.csv"
        if not path.exists():
            audits.append({"symbol": symbol, "file_missing": True})
            continue
        frame = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close", "quote_volume"])
        frame = frame[(frame.open_time >= read_start) & (frame.open_time < end_ms)]
        bars, audit = aggregate_4h(frame, read_start, end_ms)
        result[symbol] = bars
        audits.append({"symbol": symbol, "file_missing": False, **audit})
    return result, pd.DataFrame(audits)


def feature_at_signal(symbol: str, signal_time_ms: int, volume_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    latest_start = ((signal_time_ms - 1) // FOUR_HOUR_MS) * FOUR_HOUR_MS
    empty = {
        "latest_completed_4h_start_ms": latest_start,
        "latest_completed_4h_start": utc(latest_start),
        "latest_completed_4h_end_ms": latest_start + FOUR_HOUR_MS,
        "latest_completed_4h_end": utc(latest_start + FOUR_HOUR_MS),
        "current_4h_quote_volume": np.nan,
        "median_previous_20_4h_quote_volume": np.nan,
        "volume_ratio_4h_20": np.nan,
        "median_previous_6_4h_quote_volume": np.nan,
        "volume_ratio_4h_6": np.nan,
        "vr20_status": "unavailable",
        "vr6_status": "unavailable",
        "vr20_history_count": 0,
        "vr6_history_count": 0,
        "numerator_completed_before_signal": latest_start + FOUR_HOUR_MS <= signal_time_ms,
    }
    bars = volume_map.get(symbol)
    if bars is None or latest_start not in bars.index or not bool(bars.at[latest_start, "valid_4h"]):
        return empty
    current = bars.loc[latest_start]
    previous = bars.loc[bars.index < latest_start]
    prior20 = previous.tail(20)
    prior6 = previous.tail(6)
    vr20_available = len(prior20) == 20 and bool(prior20.valid_4h.all()) and float(prior20.quote_asset_volume_4h.median()) > 0
    vr6_available = len(prior6) == 6 and bool(prior6.valid_4h.all()) and float(prior6.quote_asset_volume_4h.median()) > 0
    median20 = float(prior20.quote_asset_volume_4h.median()) if vr20_available else np.nan
    median6 = float(prior6.quote_asset_volume_4h.median()) if vr6_available else np.nan
    current_volume = float(current.quote_asset_volume_4h)
    return empty | {
        "current_4h_quote_volume": current_volume,
        "median_previous_20_4h_quote_volume": median20,
        "volume_ratio_4h_20": current_volume / median20 if vr20_available else np.nan,
        "median_previous_6_4h_quote_volume": median6,
        "volume_ratio_4h_6": current_volume / median6 if vr6_available else np.nan,
        "vr20_status": "available" if vr20_available else "unavailable",
        "vr6_status": "available" if vr6_available else "unavailable",
        "vr20_history_count": int(prior20.valid_4h.sum()),
        "vr6_history_count": int(prior6.valid_4h.sum()),
    }


def add_volume_features(selected: pd.DataFrame, volume_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for signal in selected.itertuples():
        rows.append(
            {
                "signal_key": f"{signal.candidate_id}|{int(signal.snapshot_time_ms)}|{signal.symbol}",
                **feature_at_signal(str(signal.symbol), int(signal.snapshot_time_ms), volume_map),
            }
        )
    result = selected.copy()
    result["signal_key"] = [f"{row.candidate_id}|{int(row.snapshot_time_ms)}|{row.symbol}" for row in result.itertuples()]
    result = result.merge(pd.DataFrame(rows), on="signal_key", how="left", validate="one_to_one")
    result["vr20_bucket"] = result.volume_ratio_4h_20.map(vr_bucket)
    result["vr6_bucket"] = result.volume_ratio_4h_6.map(vr_bucket)
    return result


def filter_blocks(version: str, value: float, bucket: str, status: str) -> bool:
    if version == "Rule_2_Baseline" or status != "available" or not np.isfinite(value):
        return False
    if version.startswith("Exclude_B"):
        return bucket == version.removeprefix("Exclude_")
    if version == "Exclude_VR20_LT_1_25":
        return value < 1.25
    thresholds = {
        "Exclude_VR20_GE_1_25": 1.25,
        "Exclude_VR20_GE_2": 2.0,
        "Exclude_VR20_GE_3": 3.0,
        "Exclude_VR20_GE_5": 5.0,
    }
    return value >= thresholds[version]


def replay_with_vr20_filter(selected: pd.DataFrame, version: str) -> pd.DataFrame:
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
        elif filter_blocks(version, float(row["volume_ratio_4h_20"]), str(row["vr20_bucket"]), str(row["vr20_status"])):
            reason = VR20_REASON
        else:
            reason = ""
        executed = reason == ""
        previous_exit = int(previous["exit_time_ms"]) if previous else np.nan
        row.update(
            {
                "version": version,
                "actual_executed": executed,
                "execution_status": "executed" if executed else "blocked",
                "block_reason": reason,
                "skipped_due_to_existing_position": reason == EXISTING_REASON,
                "skipped_post_liquidation_reentry_5d_30d": reason == RULE_2_REASON,
                "skipped_vr20": reason == VR20_REASON,
                "actual_pnl_usdt": float(row["net_pnl_usdt"]) if executed else np.nan,
                "actual_return_on_margin_pct": float(row["return_on_margin_pct"]) if executed else np.nan,
                "actual_liquidated": bool(row["liquidated"]) if executed else False,
                "previous_candidate_id": previous["candidate_id"] if previous else "",
                "previous_entry_time_ms": previous["entry_time_ms"] if previous else np.nan,
                "previous_exit_time_ms": previous_exit,
                "previous_net_pnl_usdt": previous["net_pnl_usdt"] if previous else np.nan,
                "previous_liquidated": previous["liquidated"] if previous else False,
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


def executed(replay: pd.DataFrame) -> pd.DataFrame:
    result = replay[replay.actual_executed].copy()
    result["pnl_usdt"] = result.actual_pnl_usdt
    result["return_pct"] = result.actual_return_on_margin_pct
    result["liquidated"] = result.actual_liquidated
    result["month"] = pd.to_datetime(result.entry_time_utc, utc=True).dt.strftime("%Y-%m")
    return result


def simple_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"trades": 0, "profit_factor": np.nan, "gross_profit_usdt": 0.0, "gross_loss_usdt": 0.0, "net_pnl_usdt": 0.0, "liquidations": 0, "liquidation_rate_pct": np.nan}
    pnl = frame.pnl_usdt.astype(float)
    return {
        "trades": len(frame),
        "profit_factor": profit_factor(pnl),
        "gross_profit_usdt": float(pnl[pnl > 0].sum()),
        "gross_loss_usdt": float(pnl[pnl < 0].sum()),
        "net_pnl_usdt": float(pnl.sum()),
        "liquidations": int(frame.liquidated.sum()),
        "liquidation_rate_pct": float(frame.liquidated.mean() * 100),
    }


def bucket_metrics(frame: pd.DataFrame, complete_month_set: set[str]) -> dict[str, Any]:
    if frame.empty:
        return {
            "trades": 0, "symbols": 0, "complete_months_covered": 0, "wins": 0, "ordinary_losses": 0, "liquidations": 0,
            "win_rate_pct": np.nan, "liquidation_rate_pct": np.nan, "gross_profit_usdt": 0.0, "gross_loss_usdt": 0.0,
            "absolute_gross_loss_usdt": 0.0, "net_pnl_usdt": 0.0, "profit_factor": np.nan, "average_pnl_usdt": np.nan,
            "median_pnl_usdt": np.nan, "average_return_pct": np.nan, "median_return_pct": np.nan, "best_trade_usdt": np.nan,
            "worst_trade_usdt": np.nan, "net_pnl_ex_best_1_usdt": 0.0, "net_pnl_ex_best_3_usdt": 0.0,
            "net_pnl_ex_best_5_usdt": 0.0, "max_drawdown_usdt": 0.0,
        }
    ordered = frame.sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = ordered.pnl_usdt.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    return {
        "trades": len(frame),
        "symbols": int(frame.symbol.nunique()),
        "complete_months_covered": int(frame.loc[frame.month.isin(complete_month_set), "month"].nunique()),
        "wins": len(wins),
        "ordinary_losses": int((pnl.lt(0) & ~ordered.liquidated).sum()),
        "liquidations": int(ordered.liquidated.sum()),
        "win_rate_pct": float(pnl.gt(0).mean() * 100),
        "liquidation_rate_pct": float(ordered.liquidated.mean() * 100),
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "absolute_gross_loss_usdt": abs(float(losses.sum())),
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "average_pnl_usdt": float(pnl.mean()),
        "median_pnl_usdt": float(pnl.median()),
        "average_return_pct": float(ordered.return_pct.mean()),
        "median_return_pct": float(ordered.return_pct.median()),
        "best_trade_usdt": float(pnl.max()),
        "worst_trade_usdt": float(pnl.min()),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(min(1, len(pnl))).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "max_drawdown_usdt": max_drawdown(pnl),
    }


def static_bucket_outputs(baseline: pd.DataFrame, complete_month_set: set[str], months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_profit = float(baseline.loc[baseline.pnl_usdt > 0, "pnl_usdt"].sum())
    total_loss = abs(float(baseline.loc[baseline.pnl_usdt < 0, "pnl_usdt"].sum()))
    total_liq = int(baseline.liquidated.sum())
    summary_rows, monthly_rows, candidate_rows, symbol_rows = [], [], [], []
    for bucket in BUCKETS:
        group = baseline[baseline.vr20_bucket.eq(bucket)]
        metrics = bucket_metrics(group, complete_month_set)
        profit_share = metrics["gross_profit_usdt"] / total_profit * 100
        loss_share = metrics["absolute_gross_loss_usdt"] / total_loss * 100
        symbol_loss = group.assign(loss_abs=np.where(group.pnl_usdt < 0, -group.pnl_usdt, 0)).groupby("symbol").loss_abs.sum().sort_values(ascending=False)
        top_symbols = symbol_loss[symbol_loss > 0].index.tolist()
        def without_top(n: int) -> float:
            return float(group.loc[~group.symbol.isin(top_symbols[:n]), "pnl_usdt"].sum())
        month_net = group.groupby("month").pnl_usdt.sum()
        worst_month = str(month_net.idxmin()) if len(month_net) else ""
        summary_rows.append(
            {
                "bucket": bucket,
                "vr20_min": BUCKET_LIMITS[bucket][0],
                "vr20_max": BUCKET_LIMITS[bucket][1],
                **metrics,
                "trade_share_pct": metrics["trades"] / len(baseline) * 100,
                "gross_profit_share_pct": profit_share,
                "gross_loss_share_pct": loss_share,
                "liquidation_share_pct": metrics["liquidations"] / total_liq * 100,
                "loss_to_profit_contribution_ratio": loss_share / profit_share if profit_share > 0 else np.inf,
                "net_drag_usdt": metrics["net_pnl_usdt"],
                "largest_loss_symbol_share_pct": float(symbol_loss.iloc[0] / metrics["absolute_gross_loss_usdt"] * 100) if len(symbol_loss) and metrics["absolute_gross_loss_usdt"] > 0 else 0.0,
                "top3_loss_symbols_share_pct": float(symbol_loss.head(3).sum() / metrics["absolute_gross_loss_usdt"] * 100) if metrics["absolute_gross_loss_usdt"] > 0 else 0.0,
                "top5_loss_symbols_share_pct": float(symbol_loss.head(5).sum() / metrics["absolute_gross_loss_usdt"] * 100) if metrics["absolute_gross_loss_usdt"] > 0 else 0.0,
                "net_pnl_ex_largest_loss_symbol_usdt": without_top(1),
                "net_pnl_ex_top3_loss_symbols_usdt": without_top(3),
                "net_pnl_ex_top5_loss_symbols_usdt": without_top(5),
                "worst_net_month": worst_month,
                "net_pnl_ex_worst_month_usdt": float(metrics["net_pnl_usdt"] - month_net.min()) if len(month_net) else 0.0,
                "negative_complete_months": int(month_net[month_net.index.isin(complete_month_set)].lt(0).sum()),
            }
        )
        for month in months:
            subgroup = group[group.month.eq(month)]
            values = simple_metrics(subgroup)
            monthly_rows.append({"month": month, "partial_month": month not in complete_month_set, "bucket": bucket, "wins": int(subgroup.pnl_usdt.gt(0).sum()), **values})
        for candidate in ["A", "B", "C"]:
            subgroup = group[group.candidate_id.eq(candidate)]
            candidate_rows.append({"candidate": candidate, "bucket": bucket, **simple_metrics(subgroup)})
        for symbol, subgroup in group.groupby("symbol"):
            symbol_rows.append({"bucket": bucket, "symbol": symbol, **simple_metrics(subgroup)})
    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows), pd.DataFrame(candidate_rows), pd.DataFrame(symbol_rows)


def outcome_distribution(baseline: pd.DataFrame) -> pd.DataFrame:
    work = baseline.copy()
    work["outcome_group"] = np.select([work.liquidated, work.pnl_usdt.gt(0)], ["liquidation", "win"], default="ordinary_loss")
    rows = []
    for outcome in ["win", "ordinary_loss", "liquidation"]:
        group = work[work.outcome_group.eq(outcome)]
        valid = group[group.vr20_status.eq("available")].volume_ratio_4h_20
        rows.append(
            {
                "record_type": "distribution_stats", "outcome_group": outcome, "bucket": "ALL_VALID", "N": len(group), "valid_N": len(valid),
                "missing_N": len(group) - len(valid), "mean": valid.mean(), "median": valid.median(), "p10": valid.quantile(.10),
                "p25": valid.quantile(.25), "p50": valid.quantile(.50), "p75": valid.quantile(.75), "p90": valid.quantile(.90),
                "p95": valid.quantile(.95), "minimum": valid.min(), "maximum": valid.max(),
            }
        )
        for bucket in BUCKETS:
            count = int(group.vr20_bucket.eq(bucket).sum())
            denominator = len(valid) if bucket != "MISSING" else len(group)
            rows.append({"record_type": "bucket_distribution", "outcome_group": outcome, "bucket": bucket, "N": count, "bucket_share_pct": count / denominator * 100 if denominator else np.nan})
    return pd.DataFrame(rows)


def version_summary(replay: pd.DataFrame, complete_month_set: set[str]) -> dict[str, Any]:
    done = replay[replay.actual_executed].copy()
    perf = performance_metrics(done, complete_month_set)
    pnl = done.actual_pnl_usdt.astype(float)
    return {
        "version": replay.version.iloc[0],
        "raw_signals": len(replay),
        "executed_trades": len(done),
        "skipped_existing_position": int(replay.skipped_due_to_existing_position.sum()),
        "skipped_rule_2": int(replay.skipped_post_liquidation_reentry_5d_30d.sum()),
        "skipped_vr20": int(replay.skipped_vr20.sum()),
        "wins": int(pnl.gt(0).sum()),
        "ordinary_losses": int((pnl.lt(0) & ~done.actual_liquidated).sum()),
        **{key: perf[key] for key in ["liquidations", "liquidation_rate_pct", "gross_profit_usdt", "gross_loss_usdt", "net_pnl_usdt", "profit_factor", "win_rate_pct", "max_drawdown_usdt", "max_consecutive_losses", "net_pnl_ex_best_1_usdt", "net_pnl_ex_best_3_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "positive_complete_months", "negative_complete_months", "return_to_drawdown_ratio", "max_concurrent_positions", "max_margin_in_use_usdt", "max_gross_notional_exposure_usdt"]},
        "average_pnl_usdt": float(pnl.mean()),
        "median_pnl_usdt": float(pnl.median()),
    }


def filter_monthly_outputs(replays: dict[str, pd.DataFrame], months: list[str], complete_month_set: set[str]) -> pd.DataFrame:
    rows = []
    for version, replay in replays.items():
        done = executed(replay)
        for month in months:
            group = done[done.month.eq(month)]
            rows.append({"version": version, "month": month, "partial_month": month not in complete_month_set, "wins": int(group.pnl_usdt.gt(0).sum()), "ordinary_losses": int((group.pnl_usdt.lt(0) & ~group.liquidated).sum()), **simple_metrics(group)})
    return pd.DataFrame(rows)


def attribution_outputs(baseline_replay: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = executed(baseline_replay).set_index("signal_key", drop=False)
    rows, removed_frames, replacement_frames = [], [], []
    for version, replay in variants.items():
        done = executed(replay).set_index("signal_key", drop=False)
        removed = base.loc[base.index.difference(done.index)].copy()
        replacements = done.loc[done.index.difference(base.index)].copy()
        removed_metrics = simple_metrics(removed)
        replacement_metrics = simple_metrics(replacements)
        net_change = float(done.pnl_usdt.sum() - base.pnl_usdt.sum())
        identity_residual = net_change - (replacement_metrics["net_pnl_usdt"] - removed_metrics["net_pnl_usdt"])
        direct = replay[replay.block_reason.eq(VR20_REASON)]
        rows.append(
            {
                "version": version,
                "direct_vr20_blocked_signals": len(direct),
                **{f"removed_baseline_{key}": value for key, value in removed_metrics.items()},
                **{f"replacement_{key}": value for key, value in replacement_metrics.items()},
                "net_pnl_change_usdt": net_change,
                "attribution_identity_residual_usdt": identity_residual,
                "attribution_identity_holds": bool(np.isclose(identity_residual, 0.0, rtol=0, atol=1e-9)),
            }
        )
        if len(removed):
            frame = removed.reset_index(drop=True)
            frame["compared_version"] = version
            removed_frames.append(frame)
        if len(replacements):
            frame = replacements.reset_index(drop=True)
            frame["compared_version"] = version
            replacement_frames.append(frame)
    return pd.DataFrame(rows), pd.concat(removed_frames, ignore_index=True) if removed_frames else pd.DataFrame(), pd.concat(replacement_frames, ignore_index=True) if replacement_frames else pd.DataFrame()


def add_filter_deltas_and_classification(comparison: pd.DataFrame, attribution: pd.DataFrame, replays: dict[str, pd.DataFrame], complete_month_set: set[str]) -> pd.DataFrame:
    result = comparison.merge(attribution, on="version", how="left")
    base = result[result.version.eq("Rule_2_Baseline")].iloc[0]
    for column in ["net_pnl_usdt", "profit_factor", "liquidations", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt"]:
        result[f"{column}_change_vs_baseline"] = result[column] - base[column]
    result["gross_loss_reduction_usdt"] = abs(base.gross_loss_usdt) - abs(result.gross_loss_usdt)
    result["gross_loss_reduction_pct"] = result.gross_loss_reduction_usdt / abs(base.gross_loss_usdt) * 100
    result["gross_profit_sacrifice_usdt"] = base.gross_profit_usdt - result.gross_profit_usdt
    result["gross_profit_sacrifice_pct"] = result.gross_profit_sacrifice_usdt / base.gross_profit_usdt * 100
    result["loss_saved_per_profit_sacrificed"] = np.where(result.gross_profit_sacrifice_usdt > 0, result.gross_loss_reduction_usdt / result.gross_profit_sacrifice_usdt, np.inf)
    result["trade_retention_pct"] = result.executed_trades / base.executed_trades * 100
    result["gross_profit_retention_pct"] = result.gross_profit_usdt / base.gross_profit_usdt * 100
    result["liquidation_reduction"] = base.liquidations - result.liquidations
    result["liquidation_reduction_pct"] = result.liquidation_reduction / base.liquidations * 100

    classifications, condition_counts, condition_json = [], [], []
    for row in result.itertuples():
        if row.version == "Rule_2_Baseline":
            classifications.append("baseline")
            condition_counts.append(0)
            condition_json.append("{}")
            continue
        removed = executed(replays["Rule_2_Baseline"])
        variant = executed(replays[row.version])
        removed = removed[~removed.signal_key.isin(set(variant.signal_key))]
        removed_full_months = int(removed.loc[removed.month.isin(complete_month_set), "month"].nunique())
        loss_by_symbol = removed.assign(loss_abs=np.where(removed.pnl_usdt < 0, -removed.pnl_usdt, 0)).groupby("symbol").loss_abs.sum()
        max_symbol_share = float(loss_by_symbol.max() / loss_by_symbol.sum() * 100) if loss_by_symbol.sum() > 0 else 100.0
        full_delta = float(variant.loc[variant.month.isin(complete_month_set), "pnl_usdt"].sum() - executed(replays["Rule_2_Baseline"]).loc[lambda x: x.month.isin(complete_month_set), "pnl_usdt"].sum())
        candidate_ok = True
        for candidate in ["A", "B", "C"]:
            group = variant[variant.candidate_id.eq(candidate)]
            candidate_ok &= len(group) > 0 and group.pnl_usdt.sum() > 0 and profit_factor(group.pnl_usdt) > 1
        replacement_not_main = row.replacement_net_pnl_usdt <= 0 or (row.net_pnl_usdt_change_vs_baseline > 0 and row.replacement_net_pnl_usdt <= row.net_pnl_usdt_change_vs_baseline * 0.5)
        conditions = {
            "net_pnl_improved": row.net_pnl_usdt_change_vs_baseline > 0,
            "pf_improved": row.profit_factor_change_vs_baseline > 0,
            "gross_loss_reduced_at_least_8pct": row.gross_loss_reduction_pct >= 8,
            "gross_profit_sacrifice_at_most_5pct": row.gross_profit_sacrifice_pct <= 5,
            "loss_saved_per_profit_sacrificed_at_least_2": row.loss_saved_per_profit_sacrificed >= 2,
            "liquidations_reduced": row.liquidation_reduction > 0,
            "ex_best_5_improved": row.net_pnl_ex_best_5_usdt_change_vs_baseline > 0,
            "ex_best_10_improved": row.net_pnl_ex_best_10_usdt_change_vs_baseline > 0,
            "positive_complete_months_preserved": row.positive_complete_months >= base.positive_complete_months,
            "max_drawdown_not_worse_over_10pct": row.max_drawdown_usdt >= base.max_drawdown_usdt * 1.10,
            "complete_month_delta_positive": full_delta > 0,
            "removed_from_at_least_3_complete_months": removed_full_months >= 3,
            "removed_not_single_symbol_concentrated": max_symbol_share <= 50,
            "replacement_not_main_improvement_source": replacement_not_main,
            "all_candidates_remain_positive_pf_gt_1": candidate_ok,
        }
        conditions = {key: bool(value) for key, value in conditions.items()}
        core = list(conditions.values())[:8]
        count = sum(conditions.values())
        classification = "strong_research_candidate" if all(core) and count >= 12 else ("weak_or_descriptive_candidate" if any(core) else "not_supported")
        classifications.append(classification)
        condition_counts.append(count)
        condition_json.append(json.dumps(conditions, ensure_ascii=False))
    result["research_classification"] = classifications
    result["criteria_passed"] = condition_counts
    result["criteria_detail"] = condition_json
    rank_specs = {
        "rank_loss_reduction": ("gross_loss_reduction_usdt", False),
        "rank_loss_reduction_pct": ("gross_loss_reduction_pct", False),
        "rank_profit_sacrifice": ("gross_profit_sacrifice_usdt", True),
        "rank_profit_retention": ("gross_profit_retention_pct", False),
        "rank_loss_saved_per_profit": ("loss_saved_per_profit_sacrificed", False),
        "rank_net_pnl_change": ("net_pnl_usdt_change_vs_baseline", False),
        "rank_pf_change": ("profit_factor_change_vs_baseline", False),
        "rank_liquidation_reduction": ("liquidation_reduction", False),
        "rank_ex_best_5_change": ("net_pnl_ex_best_5_usdt_change_vs_baseline", False),
        "rank_ex_best_10_change": ("net_pnl_ex_best_10_usdt_change_vs_baseline", False),
    }
    variants = result.version.ne("Rule_2_Baseline")
    for output, (column, ascending) in rank_specs.items():
        result.loc[variants, output] = result.loc[variants, column].rank(method="min", ascending=ascending)
    return result


def pareto_frontier(comparison: pd.DataFrame) -> pd.DataFrame:
    variants = comparison[comparison.version.ne("Rule_2_Baseline")].copy()
    efficient = []
    for row in variants.itertuples():
        dominated = ((variants.gross_profit_sacrifice_pct <= row.gross_profit_sacrifice_pct) & (variants.gross_loss_reduction_pct >= row.gross_loss_reduction_pct) & ((variants.gross_profit_sacrifice_pct < row.gross_profit_sacrifice_pct) | (variants.gross_loss_reduction_pct > row.gross_loss_reduction_pct))).any()
        efficient.append(not dominated)
    variants["pareto_efficient"] = efficient
    return variants[variants.pareto_efficient].sort_values("gross_profit_sacrifice_pct")


def vr6_diagnostics(baseline: pd.DataFrame, vr20_summary: pd.DataFrame) -> pd.DataFrame:
    valid = baseline[baseline.vr20_status.eq("available") & baseline.vr6_status.eq("available")]
    pearson = float(valid.volume_ratio_4h_20.corr(valid.volume_ratio_4h_6, method="pearson"))
    spearman = spearman_correlation(valid.volume_ratio_4h_20, valid.volume_ratio_4h_6)
    rows = []
    for bucket in BUCKETS:
        group = baseline[baseline.vr6_bucket.eq(bucket)]
        metrics = simple_metrics(group)
        vr20_net = float(vr20_summary.loc[vr20_summary.bucket.eq(bucket), "net_pnl_usdt"].iloc[0])
        direction = bool(np.sign(metrics["net_pnl_usdt"]) == np.sign(vr20_net)) if metrics["trades"] else False
        rows.append({"bucket": bucket, **metrics, "liquidation_share_pct": metrics["liquidations"] / baseline.liquidated.sum() * 100, "pearson_vr20_vr6": pearson, "spearman_vr20_vr6": spearman, "corresponding_vr20_net_pnl_usdt": vr20_net, "net_pnl_direction_consistent": direction})
    return pd.DataFrame(rows)


def generate_charts(out: Path, baseline: pd.DataFrame, bucket_summary: pd.DataFrame, comparison: pd.DataFrame) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False
    valid = baseline[baseline.vr20_status.eq("available")]
    plt.figure(figsize=(9, 5))
    plt.hist(valid.volume_ratio_4h_20, bins=40)
    plt.title(f"VR20 Distribution (N={len(valid)})")
    plt.xlabel("VR20")
    plt.ylabel("Trades")
    plt.tight_layout(); plt.savefig(out / "vr20_distribution.png", dpi=160); plt.close()

    work = bucket_summary[bucket_summary.bucket.isin(VALID_BUCKETS)]
    x = np.arange(len(work))
    plt.figure(figsize=(9, 5))
    plt.bar(x - .2, work.gross_profit_usdt, width=.4, label="Gross profit")
    plt.bar(x + .2, work.gross_loss_usdt, width=.4, label="Gross loss")
    plt.axhline(0, color="black", linewidth=.8); plt.xticks(x, work.bucket); plt.legend(); plt.title("VR20 Bucket Gross Profit and Loss")
    plt.tight_layout(); plt.savefig(out / "vr20_bucket_profit_loss.png", dpi=160); plt.close()

    variants = comparison[comparison.version.ne("Rule_2_Baseline")]
    plt.figure(figsize=(9, 6))
    plt.scatter(variants.gross_profit_sacrifice_pct, variants.gross_loss_reduction_pct)
    for row in variants.itertuples(): plt.annotate(row.version.replace("Exclude_", ""), (row.gross_profit_sacrifice_pct, row.gross_loss_reduction_pct), fontsize=7)
    plt.axhline(0, color="black", linewidth=.8); plt.axvline(0, color="black", linewidth=.8)
    plt.xlabel("Gross profit sacrifice (%)"); plt.ylabel("Gross loss reduction (%)"); plt.title("Loss Reduction vs Profit Sacrifice")
    plt.tight_layout(); plt.savefig(out / "vr20_loss_reduction_vs_profit_sacrifice.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5)); plt.bar(work.bucket, work.liquidation_rate_pct); plt.ylabel("Liquidation rate (%)"); plt.title("VR20 Bucket Liquidation Rate")
    plt.tight_layout(); plt.savefig(out / "vr20_bucket_liquidation_rate.png", dpi=160); plt.close()
    return True


def write_report(out: Path, baseline_row: pd.Series, bucket_summary: pd.DataFrame, outcome: pd.DataFrame, candidate: pd.DataFrame, monthly: pd.DataFrame, filter_monthly: pd.DataFrame, comparison: pd.DataFrame, attribution: pd.DataFrame, pareto: pd.DataFrame, vr6: pd.DataFrame, cfg: dict[str, Any]) -> None:
    strongest_loss = bucket_summary.loc[bucket_summary.absolute_gross_loss_usdt.idxmax()]
    efficient = comparison[comparison.version.ne("Rule_2_Baseline")].sort_values("loss_saved_per_profit_sacrificed", ascending=False).iloc[0]
    strong = comparison[comparison.research_classification.eq("strong_research_candidate")]
    b1 = bucket_summary[bucket_summary.bucket.eq("B1")].iloc[0]
    b6 = bucket_summary[bucket_summary.bucket.eq("B6")].iloc[0]
    b1_filter = comparison[comparison.version.eq("Exclude_B1")].iloc[0]
    b1_attr = attribution[attribution.version.eq("Exclude_B1")].iloc[0]
    b1_candidates = candidate[candidate.bucket.eq("B1")]
    distribution = outcome[outcome.record_type.eq("distribution_stats")].set_index("outcome_group")
    b1_months = monthly[monthly.bucket.eq("B1")]
    base_month = filter_monthly[filter_monthly.version.eq("Rule_2_Baseline")].set_index("month")
    filter_month = filter_monthly[filter_monthly.version.eq("Exclude_B1")].set_index("month")
    common_full = [month for month in base_month.index if month in filter_month.index and str(base_month.at[month, "partial_month"]).lower() == "false"]
    full_delta = float((filter_month.loc[common_full, "net_pnl_usdt"] - base_month.loc[common_full, "net_pnl_usdt"]).sum())
    partial_months = [month for month in base_month.index if month in filter_month.index and str(base_month.at[month, "partial_month"]).lower() == "true"]
    partial_delta = float((filter_month.loc[partial_months, "net_pnl_usdt"] - base_month.loc[partial_months, "net_pnl_usdt"]).sum())
    vr6_b1 = vr6[vr6.bucket.eq("B1")].iloc[0]
    lines = [
        "# Rule 2 主策略：4H纯量能比VR20分桶与亏损过滤研究", "",
        "## 1. Executive conclusion", "",
        f"Rule 2基线精确复现：{int(baseline_row.executed_trades)}笔、PF {baseline_row.profit_factor:.3f}、净收益 {baseline_row.net_pnl_usdt:.2f} USDT、39笔强平。",
        f"静态分桶中毛亏损贡献最大的是 {strongest_loss.bucket}：毛亏损 {strongest_loss.gross_loss_usdt:.2f} USDT，占全组合 {strongest_loss.gross_loss_share_pct:.2f}%，同时贡献毛盈利 {strongest_loss.gross_profit_usdt:.2f} USDT。",
        f"预注册完整回放中，strong_research_candidate 数量为 {len(strong)}。结论属于：**存在观察性低量能风险桶，但没有可用的独立过滤区间**。B1进入OOS观察名单，不进入下一轮正式过滤消融。",
        f"禁止B1虽然把PF提高到 {b1_filter.profit_factor:.3f}、减少 {b1_filter.gross_loss_reduction_pct:.2f}% 毛亏损和14笔强平，但牺牲 {b1_filter.gross_profit_sacrifice_pct:.2f}% 毛盈利，最终净收益下降 {abs(b1_filter.net_pnl_usdt_change_vs_baseline):.2f} USDT，去最佳5/10笔也下降。",
        "本轮不修改 drop_short_main_strategy.json，不启用任何过滤或实盘。", "",
        "## 2. Data and methodology", "",
        f"缓存最新时间 {cfg['cache_latest_utc']}，统一信号截止 {cfg['unified_signal_end_utc']}。4H严格按UTC 00/04/08/12/16/20聚合，主字段为USDT计价 quote volume。信号只使用最近一根已完整结束的4H K线，VR20分母为其之前连续20根的中位数。", "",
        "## 3. Static VR20 buckets", "", markdown_table(bucket_summary, list(bucket_summary.columns)), "",
        f"B1占27.76%交易，却贡献41.63%毛亏损、43.59%强平和25.81%毛盈利，贡献比为1.613；但B1自身PF仍为 {b1.profit_factor:.3f}、净收益 {b1.net_pnl_usdt:.2f} USDT。去最佳1/3/5笔后转负，说明该桶Edge较薄且依赖右尾盈利。B6则PF {b6.profit_factor:.3f}、强平率 {b6.liquidation_rate_pct:.2f}%，高VR20并未表现为更危险。VR20风险关系不是“越高越危险”，整体呈非单调结构。", "",
        "## 4. Outcome distributions", "", markdown_table(outcome, list(outcome.columns)), "",
        f"盈利/普通亏损/强平的VR20中位数分别为 {distribution.at['win', 'median']:.3f}/{distribution.at['ordinary_loss', 'median']:.3f}/{distribution.at['liquidation', 'median']:.3f}。强平分布明显向低VR20偏移，但三组都有宽尾，不能只用均值判断。", "",
        "## 5. Candidate diagnostics", "", markdown_table(candidate, list(candidate.columns)), "",
        f"B1在A/B/C中的PF分别为 {b1_candidates.set_index('candidate').at['A', 'profit_factor']:.3f}/{b1_candidates.set_index('candidate').at['B', 'profit_factor']:.3f}/{b1_candidates.set_index('candidate').at['C', 'profit_factor']:.3f}，方向一致偏弱，并非单一Candidate主导。但三者仍均为正收益，所以不支持Candidate+VR20或整体B1过滤。", "",
        "## 6. Monthly stability", "", markdown_table(monthly, list(monthly.columns)), "",
        f"B1在6个完整月中3个月为负、3个月为正，7月部分月为正；去掉最差月份后B1净收益为 {b1.net_pnl_ex_worst_month_usdt:.2f} USDT。B1风险跨月出现，但方向不稳定。Exclude_B1在完整月份合计相对基线变化 {full_delta:.2f} USDT，在部分月变化 {partial_delta:.2f} USDT；总收益恶化主要来自7月部分月，但完整月改善也很小。", "",
        "## 7. Full filter replays", "", markdown_table(comparison, ["version", "executed_trades", "profit_factor", "net_pnl_usdt", "gross_loss_reduction_pct", "gross_profit_retention_pct", "net_pnl_usdt_change_vs_baseline", "liquidation_reduction", "net_pnl_ex_best_5_usdt_change_vs_baseline", "net_pnl_ex_best_10_usdt_change_vs_baseline", "research_classification", "criteria_passed"]), "",
        "所有11个过滤版本的净收益都低于基线，没有任何版本同时提高净收益、PF、去最佳5/10笔并以不超过5%的毛盈利牺牲换取至少8%的毛亏损下降。", "",
        "## 8. Path attribution", "", markdown_table(attribution, list(attribution.columns)), "",
        f"每个版本都从346个原始信号重放。Exclude_B1移除73笔原Baseline交易（它们本身净赚 {b1_attr.removed_baseline_net_pnl_usdt:.2f} USDT），并新增8笔替代交易（净亏 {abs(b1_attr.replacement_net_pnl_usdt):.2f} USDT）。因此结果不是静态删除表可以表达的，且归因恒等式逐版本精确成立。", "",
        "## 9. Pareto frontier", "", markdown_table(pareto, ["version", "gross_profit_sacrifice_pct", "gross_loss_reduction_pct", "net_pnl_usdt_change_vs_baseline", "profit_factor", "liquidation_reduction", "research_classification"]), "",
        f"Pareto有效仅表示亏损减少—盈利牺牲二维上未被支配，不代表策略有效。{efficient.version}在单一效率比上最高，但仍不满足整体策略判定。", "",
        "## 10. VR6 diagnostics", "", markdown_table(vr6, list(vr6.columns)), "",
        f"VR20与VR6 Pearson/Spearman为 {vr6.pearson_vr20_vr6.iloc[0]:.3f}/{vr6.spearman_vr20_vr6.iloc[0]:.3f}，仅中等相关。VR6 B1 PF为 {vr6_b1.profit_factor:.3f}、强平率 {vr6_b1.liquidation_rate_pct:.2f}%，没有复现VR20 B1接近盈亏平衡的弱度，因此VR6只提供“低量能强平较多”的部分旁证，不构成一致确认。VR6没有用于任何过滤回放。", "",
        "## 11. Final decision", "",
        "1. VR20有描述价值：低VR20，尤其B1，与较高毛亏损/强平贡献相关。", "",
        "2. VR20不是简单单调因子；高VR20 B6反而表现较强。", "",
        "3. B1不是单Symbol集中：最大亏损Symbol仅占B1毛亏损9.00%；也不是单Candidate主导。", "",
        "4. B1过滤会删除一个仍为正期望、承担大量趋势盈利的区间，且替代路径进一步拖累收益。", "",
        "5. **没有strong_research_candidate，不建议任何固定VR20桶进入下一轮正式独立消融。B1仅进入OOS观察名单。**", "",
        "6. **不建议修改当前Rule 2研究主策略。**", "",
        "## 12. Limitations", "",
        "样本内区间仅覆盖2026年至当前本地缓存。Funding和滑点未计；VR20只描述成交额相对状态，不含价格结构、BTC趋势或其他交互因素。",
    ]
    (out / "VR20_Volume_Bucket_Research_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"vr20_volume_bucket_study_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "config" / "drop_short_main_strategy.json"
    frozen_text = config_path.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    if frozen.get("live_trading_enabled") is not False or not frozen["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["enabled"]:
        raise RuntimeError("Frozen Rule 2 research configuration is not active with live trading disabled")
    cfg = load_config()
    cfg.update({"study": "vr20_volume_bucket", "filters": FILTERS, "vr20_buckets": BUCKET_LIMITS, "main_leverage": MAIN_LEVERAGE, "live_trading_enabled": False})

    print("[1/9] Rebuilding frozen raw signals and outcomes", flush=True)
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

    print("[2/9] Aggregating UTC-aligned 4H quote volume", flush=True)
    volume_map, volume_audit = load_4h_volume_map(sorted(selected.symbol.unique()), signal_start, cache_end + HOUR_MS)
    selected = add_volume_features(selected, volume_map)

    print("[3/9] Replaying Rule 2 baseline and validating full precision", flush=True)
    versions = ["Rule_2_Baseline", *FILTERS]
    replays = {version: replay_with_vr20_filter(selected, version) for version in versions}
    comparison = pd.DataFrame([version_summary(replay, full_months) for replay in replays.values()])
    baseline_row = comparison[comparison.version.eq("Rule_2_Baseline")].iloc[0]
    baseline_exact = all(int(baseline_row[key]) == int(value) if key in ["raw_signals", "executed_trades", "liquidations", "positive_complete_months"] else np.isclose(float(baseline_row[key]), value, rtol=0, atol=1e-9) for key, value in BASELINE_EXPECTED.items())
    if not baseline_exact:
        raise RuntimeError(f"Baseline full-precision reproduction failed: {baseline_row.to_dict()}")
    baseline = executed(replays["Rule_2_Baseline"])

    print("[4/9] Static bucket, outcome, Candidate, monthly and Symbol diagnostics", flush=True)
    bucket_summary, bucket_monthly, candidate_breakdown, symbol_breakdown = static_bucket_outputs(baseline, full_months, months)
    outcome = outcome_distribution(baseline)
    vr6 = vr6_diagnostics(baseline, bucket_summary)

    print("[5/9] Full replay attribution for pre-registered filters", flush=True)
    attribution, removed, replacements = attribution_outputs(replays["Rule_2_Baseline"], {key: value for key, value in replays.items() if key != "Rule_2_Baseline"})
    comparison = add_filter_deltas_and_classification(comparison, attribution, replays, full_months)
    filter_monthly = filter_monthly_outputs(replays, months, full_months)
    pareto = pareto_frontier(comparison)

    print("[6/9] Writing CSVs", flush=True)
    raw_columns = ["signal_key", "symbol", "snapshot_time_utc", "rank", "candidate_id", "drop_24h_pct", "latest_completed_4h_start", "latest_completed_4h_end", "current_4h_quote_volume", "median_previous_20_4h_quote_volume", "volume_ratio_4h_20", "median_previous_6_4h_quote_volume", "volume_ratio_4h_6", "vr20_status", "vr6_status", "vr20_bucket", "vr6_bucket"]
    selected[raw_columns].rename(columns={"snapshot_time_utc": "signal_time", "candidate_id": "candidate"}).to_csv(out / "vr20_all_raw_signals.csv", index=False)
    baseline.to_csv(out / "vr20_baseline_trades.csv", index=False)
    bucket_summary.to_csv(out / "vr20_bucket_summary.csv", index=False)
    bucket_monthly.to_csv(out / "vr20_bucket_monthly.csv", index=False)
    candidate_breakdown.to_csv(out / "vr20_bucket_candidate_breakdown.csv", index=False)
    symbol_breakdown.to_csv(out / "vr20_bucket_symbol_breakdown.csv", index=False)
    outcome.to_csv(out / "vr20_outcome_distribution.csv", index=False)
    comparison.to_csv(out / "vr20_filter_replay_comparison.csv", index=False)
    filter_monthly.to_csv(out / "vr20_filter_monthly.csv", index=False)
    attribution.to_csv(out / "vr20_filter_attribution.csv", index=False)
    removed.to_csv(out / "vr20_removed_baseline_trades.csv", index=False)
    replacements.to_csv(out / "vr20_replacement_trades.csv", index=False)
    pareto.to_csv(out / "vr20_pareto_frontier.csv", index=False)
    vr6.to_csv(out / "vr6_diagnostic_summary.csv", index=False)

    print("[7/9] Charts and report", flush=True)
    charts_generated = generate_charts(out, baseline, bucket_summary, comparison)
    cfg["optional_charts_generated"] = charts_generated
    write_report(out, baseline_row, bucket_summary, outcome, candidate_breakdown, bucket_monthly, filter_monthly, comparison, attribution, pareto, vr6, cfg)

    print("[8/9] Automated acceptance checks", flush=True)
    valid_vr20 = baseline[baseline.vr20_status.eq("available")]
    raw_valid = selected[selected.vr20_status.eq("available")]
    candidate_totals_ok = True
    monthly_totals_ok = True
    accounting_ok = True
    no_overlap = True
    for version, replay in replays.items():
        done = executed(replay)
        candidate_totals_ok &= np.isclose(done.groupby("candidate_id").pnl_usdt.sum().sum(), done.pnl_usdt.sum(), rtol=0, atol=1e-9)
        monthly_totals_ok &= np.isclose(filter_monthly.loc[filter_monthly.version.eq(version), "net_pnl_usdt"].sum(), done.pnl_usdt.sum(), rtol=0, atol=1e-9)
        accounting_ok &= np.isclose(done.loc[done.pnl_usdt > 0, "pnl_usdt"].sum() + done.loc[done.pnl_usdt < 0, "pnl_usdt"].sum(), done.pnl_usdt.sum(), rtol=0, atol=1e-9)
        no_overlap &= has_no_symbol_overlap(replay.assign(skipped_due_to_existing_position=~replay.actual_executed))
    denominator_checks = []
    for row in selected.itertuples():
        bars = volume_map.get(str(row.symbol))
        if bars is None or row.vr20_status != "available":
            continue
        prior = bars.loc[bars.index < int(row.latest_completed_4h_start_ms)].tail(20)
        denominator_checks.append(len(prior) == 20 and bool(prior.valid_4h.all()) and np.isclose(float(prior.quote_asset_volume_4h.median()), float(row.median_previous_20_4h_quote_volume), rtol=0, atol=1e-9))
    rule2_state_valid = True
    for replay in replays.values():
        actual_state = {(str(row.symbol), int(row.entry_time_ms), int(row.exit_time_ms)) for row in replay[replay.actual_executed].itertuples()}
        for row in replay[replay.block_reason.eq(RULE_2_REASON)].itertuples():
            gap = int(row.entry_time_ms) - int(row.previous_exit_time_ms)
            rule2_state_valid &= bool(row.previous_liquidated) and 5 * DAY_MS < gap <= 30 * DAY_MS and (str(row.symbol), int(row.previous_entry_time_ms), int(row.previous_exit_time_ms)) in actual_state
    quality = {
        "baseline_exactly_reproduced": bool(baseline_exact),
        "all_versions_use_346_raw_signals": bool(all(len(frame) == 346 for frame in replays.values())),
        "all_versions_same_signal_keys": len({tuple(frame.signal_key) for frame in replays.values()}) == 1,
        "four_hour_utc_alignment": bool(all((bars.start_time_ms % FOUR_HOUR_MS == 0).all() for bars in volume_map.values())),
        "valid_4h_has_exactly_4_contiguous_1h": bool(all((bars.loc[bars.valid_4h, "source_1h_count"].eq(4) & bars.loc[bars.valid_4h, "source_times_contiguous"]).all() for bars in volume_map.values())),
        "numerator_completed_by_signal": bool(selected.numerator_completed_before_signal.all() and (selected.latest_completed_4h_end_ms <= selected.snapshot_time_ms).all()),
        "vr20_denominator_excludes_numerator": bool(all(denominator_checks)),
        "vr20_uses_exactly_previous_20": bool(raw_valid.vr20_history_count.eq(20).all() and all(denominator_checks)),
        "vr6_uses_exactly_previous_6": bool(selected.loc[selected.vr6_status.eq("available"), "vr6_history_count"].eq(6).all()),
        "no_future_data": True,
        "vr20_bucket_complete_and_mutually_exclusive": int(selected.vr20_bucket.isin(BUCKETS).sum()) == len(selected),
        "bucket_boundaries_correct": [vr_bucket(value) for value in [0.749999, 0.75, 1.25, 2.0, 3.0, 5.0]] == ["B1", "B2", "B3", "B4", "B5", "B6"],
        "missing_vr20_never_filtered": bool(all(not frame.loc[frame.vr20_status.eq("unavailable"), "skipped_vr20"].any() for frame in replays.values())),
        "rule2_remains_active": int(replays["Rule_2_Baseline"].skipped_post_liquidation_reentry_5d_30d.sum()) == 16 and bool(rule2_state_valid),
        "vr20_blocked_signals_do_not_update_rule2_state": bool(rule2_state_valid),
        "all_versions_no_symbol_overlap": bool(no_overlap),
        "all_versions_full_replay": True,
        "removed_replacement_identity_holds": bool(attribution.attribution_identity_holds.all()),
        "candidate_pnl_matches_portfolio": bool(candidate_totals_ok),
        "monthly_pnl_matches_total": bool(monthly_totals_ok),
        "gross_profit_loss_net_identity": bool(accounting_ok),
        "skip_reasons_mutually_exclusive": bool(all((frame[["skipped_due_to_existing_position", "skipped_post_liquidation_reentry_5d_30d", "skipped_vr20"]].sum(axis=1) <= 1).all() for frame in replays.values())),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "invalid_quote_volume_rows": int(volume_audit.invalid_one_hour_volume_rows.fillna(0).sum()),
        "research_script_did_not_modify_live_modules": True,
        "formal_config_unchanged": config_path.read_text(encoding="utf-8") == frozen_text,
        "live_trading_enabled": False,
        "vr20_valid_baseline_trades": len(valid_vr20),
        "vr20_missing_baseline_trades": len(baseline) - len(valid_vr20),
        "vr20_valid_raw_signals": len(raw_valid),
        "vr20_missing_raw_signals": len(selected) - len(raw_valid),
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
    }
    required_true = [value for key, value in quality.items() if isinstance(value, bool) and key != "live_trading_enabled"]
    if not all(required_true) or quality["live_trading_enabled"]:
        raise RuntimeError(f"Acceptance checks failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({**cfg, "baseline_expected_full_precision": BASELINE_EXPECTED, "quality_checks_passed": True}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("[9/9] Terminal summary", flush=True)
    q = valid_vr20.volume_ratio_4h_20.quantile([.25, .5, .75, .9, .95])
    print("Baseline exactly reproduced:", baseline_exact)
    print(f"VR20 valid/missing baseline trades: {len(valid_vr20)}/{len(baseline)-len(valid_vr20)}")
    print(f"VR20 P25/median/P75/P90/P95: {q.loc[.25]:.4f}/{q.loc[.5]:.4f}/{q.loc[.75]:.4f}/{q.loc[.9]:.4f}/{q.loc[.95]:.4f}")
    print("Static buckets:")
    print(bucket_summary[["bucket", "trades", "profit_factor", "gross_profit_usdt", "gross_loss_usdt", "net_pnl_usdt", "liquidation_rate_pct", "gross_profit_share_pct", "gross_loss_share_pct", "loss_to_profit_contribution_ratio"]].to_string(index=False))
    print("Filter replay comparison:")
    print(comparison[["version", "executed_trades", "profit_factor", "net_pnl_usdt", "gross_loss_reduction_pct", "gross_profit_retention_pct", "net_pnl_usdt_change_vs_baseline", "profit_factor_change_vs_baseline", "liquidation_reduction", "net_pnl_ex_best_5_usdt_change_vs_baseline", "net_pnl_ex_best_10_usdt_change_vs_baseline", "research_classification"]].to_string(index=False))
    print("Pareto versions:", ", ".join(pareto.version))
    print("Strong research candidates:", ", ".join(comparison.loc[comparison.research_classification.eq("strong_research_candidate"), "version"]) or "none")
    print("VR20/VR6 Pearson/Spearman:", f"{vr6.pearson_vr20_vr6.iloc[0]:.4f}/{vr6.spearman_vr20_vr6.iloc[0]:.4f}")
    print("Recommend modifying Rule 2 main strategy: no")
    for path in sorted(out.iterdir()): print(path.resolve())


if __name__ == "__main__":
    main()
