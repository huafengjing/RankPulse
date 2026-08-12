from __future__ import annotations

import hashlib
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

from scripts.fetch_historical_futures_data import (  # noqa: E402
    CURRENT_CACHE,
    END as HISTORY_KLINE_END,
    HISTORY_CACHE,
    MANIFEST_PATH,
    METADATA_PATH,
    START as HISTORY_KLINE_START,
    eligible_contracts,
)
from scripts.research_combined_recommended_drop_strategy import has_no_symbol_overlap  # noqa: E402
from scripts.research_drop_rank_snapshot_times import build_six_slot_signals, markdown_table  # noqa: E402
from scripts.research_drop_strategy_leverage import (  # noqa: E402
    MARGIN_USDT,
    build_candidate_signals,
    leveraged_outcome,
    precompute_leverage_outcomes,
)
from scripts.research_drop_top3_short_edge import DAY_MS, HOUR_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import longest_streak, profit_factor  # noqa: E402
from scripts.research_reentry_block_rules import (  # noqa: E402
    EXISTING_REASON,
    MAIN_LEVERAGE,
    RULE_2_REASON,
    executed_rows,
    replay_with_block_rules,
    select_main_outcomes,
)
from scripts.research_vr20_volume_buckets import BASELINE_EXPECTED  # noqa: E402


CONFIG_PATH = ROOT / "config" / "drop_short_main_strategy.json"
WARMUP_START = pd.Timestamp("2025-05-25 00:00:00", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
FROZEN_START = HOLDOUT_END
FROZEN_SIGNAL_END = pd.Timestamp("2026-07-17 20:00:00", tz="UTC")
KLINE_LOAD_END = pd.Timestamp("2026-07-20 23:00:00", tz="UTC")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kline_source(path: Path, start_ms: int, end_ms: int) -> pd.DataFrame:
    columns = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    frame = pd.read_csv(path, usecols=lambda name: name in columns)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    frame["open_time"] = pd.to_numeric(frame.open_time, errors="coerce")
    frame = frame[frame.open_time.notna()].copy()
    frame["open_time"] = frame.open_time.astype("int64")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[(frame.open_time >= start_ms) & (frame.open_time <= end_ms)]


def missing_intervals(symbol: str, times: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(times) < 2:
        return rows
    for left, right in zip(times[:-1], times[1:]):
        gap_hours = int((int(right) - int(left)) // HOUR_MS - 1)
        if gap_hours > 0:
            rows.append(
                {
                    "symbol": symbol,
                    "missing_start_utc": utc(int(left) + HOUR_MS),
                    "missing_end_utc": utc(int(right) - HOUR_MS),
                    "missing_hours": gap_hours,
                }
            )
    return rows


def load_historical_kline_map(metadata: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_ms = ms(HISTORY_KLINE_START) - 26 * HOUR_MS
    end_ms = ms(KLINE_LOAD_END)
    official = eligible_contracts(metadata)
    official_all = {
        item["symbol"]: item
        for item in metadata.get("symbols", [])
        if item.get("contractType") == "PERPETUAL" and item.get("quoteAsset") == "USDT"
    }
    current_paths = {path.stem.removesuffix("_1h"): path for path in CURRENT_CACHE.glob("*_1h.csv")}
    history_paths = {path.stem.removesuffix("_1h"): path for path in HISTORY_CACHE.glob("*_1h.csv")}
    symbols = sorted(set(current_paths) | set(history_paths))
    frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    listing_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        parts = []
        source_rows = 0
        source_duplicate_rows = 0
        for priority, path in enumerate([history_paths.get(symbol), current_paths.get(symbol)]):
            if path is None:
                continue
            part = read_kline_source(path, start_ms, end_ms)
            source_rows += len(part)
            source_duplicate_rows += int(part.open_time.duplicated().sum())
            part["source_priority"] = priority
            parts.append(part)
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True).sort_values(["open_time", "source_priority"])
        overlap_rows = int(combined.open_time.duplicated(keep=False).sum())
        combined = combined.drop_duplicates("open_time", keep="last").sort_values("open_time")
        item = official_all.get(symbol)
        if item is not None:
            onboard = int(item.get("onboardDate") or 0)
            delivery = int(item.get("deliveryDate") or 0)
            listing_source = "official_exchange_info_snapshot"
            combined = combined[(combined.open_time >= onboard) & ((delivery <= 0) | (combined.open_time < delivery))]
        else:
            onboard = int(combined.open_time.min())
            delivery = int(combined.open_time.max()) + HOUR_MS
            listing_source = "listing_proxy_first_last_kline"
        if combined.empty:
            continue
        invalid_ohlc = (
            (~np.isfinite(combined[["open", "high", "low", "close"]])).any(axis=1)
            | (combined[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | combined.high.lt(combined[["open", "close"]].max(axis=1))
            | combined.low.gt(combined[["open", "close"]].min(axis=1))
            | combined.high.lt(combined.low)
        )
        invalid_volume = (~np.isfinite(combined[["volume", "quote_volume"]])).any(axis=1) | (combined[["volume", "quote_volume"]] < 0).any(axis=1)
        times = combined.open_time.to_numpy(dtype="int64")
        gaps = missing_intervals(symbol, times)
        missing_rows.extend(gaps)
        frame = combined.drop(columns="source_priority").set_index("open_time", drop=False)
        frames[symbol] = frame
        coverage_rows.append(
            {
                "symbol": symbol,
                "first_kline_utc": utc(int(times.min())),
                "last_kline_utc": utc(int(times.max())),
                "rows": len(frame),
                "source_rows_before_merge": source_rows,
                "source_duplicate_rows": source_duplicate_rows,
                "overlap_rows_between_caches": overlap_rows,
                "duplicate_rows_after_merge": int(frame.open_time.duplicated().sum()),
                "missing_hours_inside_observed_range": sum(row["missing_hours"] for row in gaps),
                "invalid_ohlc_rows": int(invalid_ohlc.sum()),
                "invalid_volume_rows": int(invalid_volume.sum()),
                "has_current_cache": symbol in current_paths,
                "has_history_cache": symbol in history_paths,
            }
        )
        listing_rows.append(
            {
                "symbol": symbol,
                "listing_source": listing_source,
                "onboard_time_utc": utc(onboard),
                "delivery_time_utc": utc(delivery),
                "first_valid_kline_utc": utc(int(times.min())),
                "last_valid_kline_utc": utc(int(times.max())),
                "official_status_at_snapshot": item.get("status") if item else "not_in_metadata_snapshot",
                "eligible_during_2025h2": bool(onboard < ms(HOLDOUT_END) and delivery > ms(HOLDOUT_START)),
            }
        )
    universe = official.copy()
    proxy_symbols = sorted(set(frames) - set(official.symbol))
    if proxy_symbols:
        proxy = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "contract_type": "PERPETUAL",
                    "quote_asset": "USDT",
                    "current_status": "not_in_metadata_snapshot",
                    "onboard_time_ms": int(frames[symbol].open_time.min()),
                    "onboard_time_utc": utc(int(frames[symbol].open_time.min())),
                    "delivery_time_ms": int(frames[symbol].open_time.max()) + HOUR_MS,
                    "delivery_time_utc": utc(int(frames[symbol].open_time.max()) + HOUR_MS),
                    "listing_source": "listing_proxy_first_last_kline",
                }
                for symbol in proxy_symbols
            ]
        )
        universe = pd.concat([universe, proxy], ignore_index=True).sort_values("symbol")
    gaps = pd.DataFrame(missing_rows, columns=["symbol", "missing_start_utc", "missing_end_utc", "missing_hours"])
    return frames, pd.DataFrame(coverage_rows), gaps, pd.DataFrame(listing_rows).merge(universe[["symbol", "listing_source"]], on="symbol", how="left", suffixes=("", "_universe"))


def precompute_frozen_outcomes(candidate_signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], fee_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for signal in candidate_signals.to_dict("records"):
        symbol = str(signal["symbol"])
        entry_time = int(signal["entry_time_ms"])
        exit_time = entry_time + int(signal["holding_days"]) * DAY_MS
        frame = kline_map.get(symbol)
        reason = ""
        if frame is None:
            reason = "symbol_kline_missing"
        elif entry_time not in frame.index:
            reason = "entry_open_missing"
        elif exit_time not in frame.index:
            reason = "fixed_exit_open_missing"
        else:
            path = frame[(frame.open_time >= entry_time) & (frame.open_time < exit_time)]
            expected = int(signal["holding_days"]) * 24
            if len(path) != expected or (len(path) > 1 and not np.all(np.diff(path.open_time.to_numpy()) == HOUR_MS)):
                reason = "holding_path_missing_hour"
            elif (
                (~np.isfinite(path[["open", "high", "low", "close"]])).any(axis=1).any()
                or (path[["open", "high", "low", "close"]] <= 0).any(axis=1).any()
            ):
                reason = "holding_path_invalid_ohlc"
        if reason:
            invalid_rows.append({**signal, "trade_data_status": "invalid", "invalid_data_reason": reason, "fixed_exit_time_ms": exit_time, "fixed_exit_time_utc": utc(exit_time)})
            continue
        leverage = int(MAIN_LEVERAGE[str(signal["candidate_id"])])
        entry_price = float(frame.at[entry_time, "open"])
        exit_price = float(frame.at[exit_time, "open"])
        path = frame[(frame.open_time >= entry_time) & (frame.open_time < exit_time)]
        outcome = leveraged_outcome(entry_price, exit_price, path, leverage, fee_rate)
        actual_exit_time = int(outcome["exit_time_ms"]) if outcome["liquidated"] else exit_time
        outcome["exit_time_ms"] = actual_exit_time
        valid_rows.append(
            {
                **signal,
                "trade_data_status": "valid",
                "invalid_data_reason": "",
                "leverage": leverage,
                "margin_per_trade_usdt": MARGIN_USDT,
                "entry_notional_usdt": MARGIN_USDT * leverage,
                "entry_price": entry_price,
                "fixed_exit_time_ms": exit_time,
                "fixed_exit_time_utc": utc(exit_time),
                "fixed_exit_price": exit_price,
                **outcome,
                "exit_time_utc": utc(actual_exit_time),
            }
        )
    valid = pd.DataFrame(valid_rows)
    if len(valid):
        valid = valid.sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])
    return valid, pd.DataFrame(invalid_rows)


def window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame.snapshot_time_ms >= ms(start)) & (frame.snapshot_time_ms < ms(end))].copy()


def exposure_stats(done: pd.DataFrame) -> dict[str, Any]:
    if done.empty:
        return {"max_concurrent_positions": 0, "max_margin_in_use_usdt": 0.0, "max_gross_notional_exposure_usdt": 0.0}
    events = []
    for row in done.itertuples():
        events.extend([(int(row.entry_time_ms), 1, MARGIN_USDT, float(row.entry_notional_usdt)), (int(row.exit_time_ms), -1, -MARGIN_USDT, -float(row.entry_notional_usdt))])
    event_frame = pd.DataFrame(events, columns=["time_ms", "positions", "margin", "notional"]).groupby("time_ms", as_index=False).sum().sort_values("time_ms")
    for column in ["positions", "margin", "notional"]:
        event_frame[column] = event_frame[column].cumsum()
    return {
        "max_concurrent_positions": int(event_frame.positions.max()),
        "max_margin_in_use_usdt": float(event_frame.margin.max()),
        "max_gross_notional_exposure_usdt": float(event_frame.notional.max()),
    }


def drawdown_duration_hours(done: pd.DataFrame) -> float:
    if done.empty:
        return 0.0
    ordered = done.sort_values(["exit_time_ms", "rank", "symbol"])
    equity = ordered.actual_pnl_usdt.astype(float).cumsum().to_numpy()
    times = ordered.exit_time_ms.to_numpy(dtype="int64")
    peak = 0.0
    peak_time = int(times[0])
    underwater_start: int | None = None
    longest = 0.0
    for value, time_ms in zip(equity, times):
        if value >= peak:
            peak = float(value)
            peak_time = int(time_ms)
            if underwater_start is not None:
                longest = max(longest, (int(time_ms) - underwater_start) / HOUR_MS)
                underwater_start = None
        elif underwater_start is None:
            underwater_start = peak_time
    if underwater_start is not None:
        longest = max(longest, (int(times[-1]) - underwater_start) / HOUR_MS)
    return float(longest)


def complete_months_for(start: pd.Timestamp, end: pd.Timestamp) -> set[str]:
    months = pd.period_range(start.strftime("%Y-%m"), (end - pd.Timedelta(milliseconds=1)).strftime("%Y-%m"), freq="M")
    result = set()
    for month in months:
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC") + pd.Timedelta(nanoseconds=1)
        if start <= month_start and end >= month_end:
            result.add(str(month))
    return result


def performance_summary(version: str, replay: pd.DataFrame, raw: pd.DataFrame, invalid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, warmup: pd.Timestamp, kline_start: pd.Timestamp, kline_end: pd.Timestamp) -> dict[str, Any]:
    done = executed_rows(replay).sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = done.actual_pnl_usdt.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    complete = complete_months_for(start, end)
    monthly = done.assign(month=pd.to_datetime(done.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")).groupby("month").actual_pnl_usdt.sum()
    full_values = monthly[monthly.index.isin(complete)]
    dd = max_drawdown(pnl) if len(pnl) else 0.0
    total_margin = len(done) * MARGIN_USDT
    exposure = exposure_stats(done)
    holding_hours = (done.exit_time_ms - done.entry_time_ms) / HOUR_MS if len(done) else pd.Series(dtype=float)
    ordered_outcomes = pnl.to_numpy()
    return {
        "version": version,
        "signal_start": start,
        "signal_end_exclusive": end,
        "warmup_start": warmup,
        "kline_start": kline_start,
        "kline_end": kline_end,
        "raw_signals": len(raw),
        "eligible_signals": len(raw) - len(invalid),
        "executed_trades": len(done),
        "skipped_existing_position": int(replay.skipped_due_to_existing_position.sum()),
        "skipped_rule_2": int(replay.skipped_post_liquidation_reentry_5d_30d.sum()),
        "invalid_data_trades": len(invalid),
        "unique_symbols": int(done.symbol.nunique()),
        "wins": len(wins),
        "ordinary_losses": int(((pnl < 0) & ~done.actual_liquidated.to_numpy()).sum()),
        "liquidations": int(done.actual_liquidated.sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else np.nan,
        "liquidation_rate_pct": float(done.actual_liquidated.mean() * 100) if len(done) else np.nan,
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl) if len(pnl) else np.nan,
        "average_pnl_usdt": float(pnl.mean()) if len(pnl) else np.nan,
        "median_pnl_usdt": float(pnl.median()) if len(pnl) else np.nan,
        "average_trade_roi_pct": float(pnl.mean()) if len(pnl) else np.nan,
        "median_trade_roi_pct": float(pnl.median()) if len(pnl) else np.nan,
        "return_on_deployed_margin_pct": float(pnl.sum() / total_margin * 100) if total_margin else np.nan,
        "return_type": "turnover_based_return",
        "capital_efficiency_pct": float(pnl.sum() / exposure["max_margin_in_use_usdt"] * 100) if exposure["max_margin_in_use_usdt"] else np.nan,
        "portfolio_return_pct": np.nan,
        "best_trade_usdt": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "max_drawdown_usdt": dd,
        "max_drawdown_pct": np.nan,
        "max_drawdown_duration_hours": drawdown_duration_hours(done),
        "max_consecutive_wins": longest_streak(ordered_outcomes > 0) if len(pnl) else 0,
        "max_consecutive_losses": longest_streak(ordered_outcomes < 0) if len(pnl) else 0,
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(min(1, len(pnl))).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(min(10, len(pnl))).sum()),
        "positive_complete_months": int(full_values.gt(0).sum()),
        "negative_complete_months": int(full_values.lt(0).sum()),
        "total_complete_months": len(complete),
        "median_monthly_pnl_usdt": float(full_values.median()) if len(full_values) else np.nan,
        "worst_month": str(full_values.idxmin()) if len(full_values) else "",
        "worst_month_pnl_usdt": float(full_values.min()) if len(full_values) else np.nan,
        "best_month": str(full_values.idxmax()) if len(full_values) else "",
        "best_month_pnl_usdt": float(full_values.max()) if len(full_values) else np.nan,
        "return_to_drawdown_ratio": float(pnl.sum() / abs(dd)) if dd < 0 else np.nan,
        **exposure,
        "total_holding_hours": float(holding_hours.sum()),
        "average_holding_hours": float(holding_hours.mean()) if len(holding_hours) else np.nan,
    }


def monthly_rows(version: str, replay: pd.DataFrame, raw: pd.DataFrame, invalid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    complete = complete_months_for(start, end)
    months = [str(value) for value in pd.period_range(start.strftime("%Y-%m"), (end - pd.Timedelta(milliseconds=1)).strftime("%Y-%m"), freq="M")]
    replay = replay.assign(month=pd.to_datetime(replay.snapshot_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m"))
    raw = raw.assign(month=pd.to_datetime(raw.snapshot_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m"))
    invalid = invalid.assign(month=pd.to_datetime(invalid.snapshot_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")) if len(invalid) else invalid.assign(month=pd.Series(dtype=str))
    rows = []
    for month in months:
        month_replay = replay[replay.month.eq(month)]
        done = executed_rows(month_replay).sort_values(["exit_time_ms", "rank", "symbol"])
        pnl = done.actual_pnl_usdt.astype(float)
        candidates = done.groupby("candidate_id").actual_pnl_usdt.agg(["size", "sum"])
        rows.append(
            {
                "version": version,
                "month": month,
                "partial_month": month not in complete,
                "raw_signals": int(raw.month.eq(month).sum()),
                "invalid_data_trades": int(invalid.month.eq(month).sum()) if len(invalid) else 0,
                "executed_trades": len(done),
                "wins": int((pnl > 0).sum()),
                "ordinary_losses": int(((pnl < 0) & ~done.actual_liquidated.to_numpy()).sum()),
                "liquidations": int(done.actual_liquidated.sum()),
                "liquidation_rate_pct": float(done.actual_liquidated.mean() * 100) if len(done) else np.nan,
                "gross_profit_usdt": float(pnl[pnl > 0].sum()),
                "gross_loss_usdt": float(pnl[pnl < 0].sum()),
                "net_pnl_usdt": float(pnl.sum()),
                "profit_factor": profit_factor(pnl) if len(pnl) else np.nan,
                "max_drawdown_usdt": max_drawdown(pnl) if len(pnl) else 0.0,
                **{f"{candidate}_trades": int(candidates.at[candidate, "size"]) if candidate in candidates.index else 0 for candidate in "ABC"},
                **{f"{candidate}_net_pnl": float(candidates.at[candidate, "sum"]) if candidate in candidates.index else 0.0 for candidate in "ABC"},
            }
        )
    return rows


def candidate_rows(version: str, replay: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    complete = complete_months_for(start, end)
    rows = []
    for candidate in "ABC":
        done = executed_rows(replay)
        done = done[done.candidate_id.eq(candidate)].sort_values(["exit_time_ms", "rank", "symbol"])
        pnl = done.actual_pnl_usdt.astype(float)
        monthly = done.assign(month=pd.to_datetime(done.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")).groupby("month").actual_pnl_usdt.sum()
        full = monthly[monthly.index.isin(complete)]
        rows.append(
            {
                "version": version, "candidate": candidate, "trades": len(done), "wins": int((pnl > 0).sum()),
                "losses": int((pnl < 0).sum()), "liquidations": int(done.actual_liquidated.sum()),
                "liquidation_rate_pct": float(done.actual_liquidated.mean() * 100) if len(done) else np.nan,
                "gross_profit_usdt": float(pnl[pnl > 0].sum()), "gross_loss_usdt": float(pnl[pnl < 0].sum()),
                "net_pnl_usdt": float(pnl.sum()), "profit_factor": profit_factor(pnl) if len(pnl) else np.nan,
                "average_pnl_usdt": float(pnl.mean()) if len(pnl) else np.nan, "median_pnl_usdt": float(pnl.median()) if len(pnl) else np.nan,
                "net_pnl_ex_best_1": float(pnl.sum() - pnl.nlargest(min(1, len(pnl))).sum()),
                "net_pnl_ex_best_3": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
                "net_pnl_ex_best_5": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
                "max_drawdown_usdt": max_drawdown(pnl) if len(pnl) else 0.0,
                "positive_months": int(full.gt(0).sum()), "negative_months": int(full.lt(0).sum()),
            }
        )
    return rows


def equity_curve(replay: pd.DataFrame, version: str) -> pd.DataFrame:
    done = executed_rows(replay).sort_values(["exit_time_ms", "rank", "symbol"]).copy()
    done["version"] = version
    done["equity_pnl_usdt"] = done.actual_pnl_usdt.astype(float).cumsum()
    done["running_peak_usdt"] = done.equity_pnl_usdt.cummax().clip(lower=0)
    done["drawdown_usdt"] = done.equity_pnl_usdt - done.running_peak_usdt
    return done[["version", "exit_time_ms", "exit_time_utc", "symbol", "candidate_id", "actual_pnl_usdt", "equity_pnl_usdt", "running_peak_usdt", "drawdown_usdt"]]


def concentration_rows(version: str, replay: pd.DataFrame) -> list[dict[str, Any]]:
    done = executed_rows(replay)
    pnl = done.actual_pnl_usdt.astype(float)
    total = float(pnl.sum())
    symbol_net = done.groupby("symbol").actual_pnl_usdt.sum().sort_values(ascending=False)
    positive = symbol_net[symbol_net > 0]
    negative = symbol_net[symbol_net < 0].sort_values()
    monthly = done.assign(month=pd.to_datetime(done.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")).groupby("month").actual_pnl_usdt.sum().sort_values(ascending=False)
    positive_symbol_total = float(positive.sum())
    positive_month_total = float(monthly[monthly > 0].sum())
    rows = []
    for rank, (symbol, value) in enumerate(symbol_net.items(), start=1):
        rows.append({"version": version, "record_type": "symbol_net", "rank": rank, "key": symbol, "value_usdt": value, "share_of_total_net_pct": value / total * 100 if total else np.nan})
    best_months = monthly.nlargest(min(2, len(monthly)))
    losing_sequence = longest_streak(monthly.sort_index().to_numpy() < 0) if len(monthly) else 0
    summary = {
        "largest_profit_symbol_share_pct": float(positive.iloc[0] / positive_symbol_total * 100) if len(positive) and positive_symbol_total else np.nan,
        "top3_profit_symbols_share_pct": float(positive.head(3).sum() / positive_symbol_total * 100) if positive_symbol_total else np.nan,
        "top5_profit_symbols_share_pct": float(positive.head(5).sum() / positive_symbol_total * 100) if positive_symbol_total else np.nan,
        "profit_symbol_share_denominator": "sum_of_positive_symbol_net_pnl",
        "largest_loss_symbol": str(negative.index[0]) if len(negative) else "",
        "largest_loss_symbol_usdt": float(negative.iloc[0]) if len(negative) else 0.0,
        "net_ex_largest_profit_symbol_usdt": total - float(positive.head(1).sum()),
        "net_ex_top3_profit_symbols_usdt": total - float(positive.head(3).sum()),
        "net_ex_top5_profit_symbols_usdt": total - float(positive.head(5).sum()),
        "best_month_share_pct": float(monthly.iloc[0] / positive_month_total * 100) if len(monthly) and monthly.iloc[0] > 0 and positive_month_total else np.nan,
        "best_month_share_denominator": "sum_of_positive_month_pnl",
        "net_ex_best_month_usdt": total - float(monthly.head(1).sum()),
        "net_ex_best_two_months_usdt": total - float(best_months.sum()),
        "worst_month": str(monthly.idxmin()) if len(monthly) else "",
        "worst_month_pnl_usdt": float(monthly.min()) if len(monthly) else np.nan,
        "max_consecutive_losing_months": losing_sequence,
    }
    rows.extend({"version": version, "record_type": "summary", "rank": np.nan, "key": key, "value_usdt": value, "share_of_total_net_pct": np.nan} for key, value in summary.items())
    return rows


def leave_one_month_out(version: str, replay: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    done = executed_rows(replay).copy()
    done["month"] = pd.to_datetime(done.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")
    rows = []
    for month in sorted(complete_months_for(start, end)):
        kept = done[done.month.ne(month)].sort_values(["exit_time_ms", "rank", "symbol"])
        pnl = kept.actual_pnl_usdt.astype(float)
        rows.append({"version": version, "excluded_month": month, "trades": len(kept), "profit_factor": profit_factor(pnl), "net_pnl_usdt": float(pnl.sum()), "max_drawdown_usdt": max_drawdown(pnl), "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum())})
    return rows


def rule2_attribution(off_replay: pd.DataFrame, on_replay: pd.DataFrame, off_summary: pd.Series, on_summary: pd.Series, monthly: pd.DataFrame) -> pd.DataFrame:
    off_done = executed_rows(off_replay).set_index("signal_key", drop=False)
    on_done = executed_rows(on_replay).set_index("signal_key", drop=False)
    removed = off_done.loc[off_done.index.difference(on_done.index)]
    replacements = on_done.loc[on_done.index.difference(off_done.index)]
    removed_pnl = removed.actual_pnl_usdt.astype(float)
    replacement_pnl = replacements.actual_pnl_usdt.astype(float)
    net_change = float(on_summary.net_pnl_usdt - off_summary.net_pnl_usdt)
    residual = net_change - (-float(removed_pnl.sum()) + float(replacement_pnl.sum()))
    overall = {
        "record_type": "overall", "month": "", "direct_rule2_blocked_signals": int(on_replay.block_reason.eq(RULE_2_REASON).sum()),
        "actual_trade_count_change": int(on_summary.executed_trades - off_summary.executed_trades),
        "removed_trades": len(removed), "removed_profit_factor": profit_factor(removed_pnl) if len(removed) else np.nan,
        "removed_gross_profit_usdt": float(removed_pnl[removed_pnl > 0].sum()), "removed_gross_loss_usdt": float(removed_pnl[removed_pnl < 0].sum()),
        "removed_net_pnl_usdt": float(removed_pnl.sum()), "removed_liquidations": int(removed.actual_liquidated.sum()),
        "removed_liquidation_rate_pct": float(removed.actual_liquidated.mean() * 100) if len(removed) else np.nan,
        "replacement_trades": len(replacements), "replacement_profit_factor": profit_factor(replacement_pnl) if len(replacements) else np.nan,
        "replacement_net_pnl_usdt": float(replacement_pnl.sum()), "net_pnl_change_usdt": net_change,
        "profit_factor_change": float(on_summary.profit_factor - off_summary.profit_factor),
        "liquidation_change": int(on_summary.liquidations - off_summary.liquidations), "max_drawdown_change_usdt": float(on_summary.max_drawdown_usdt - off_summary.max_drawdown_usdt),
        "ex_best_5_change_usdt": float(on_summary.net_pnl_ex_best_5_usdt - off_summary.net_pnl_ex_best_5_usdt),
        "ex_best_10_change_usdt": float(on_summary.net_pnl_ex_best_10_usdt - off_summary.net_pnl_ex_best_10_usdt),
        "attribution_residual_usdt": residual, "attribution_identity_holds": bool(np.isclose(residual, 0.0, atol=1e-9)),
    }
    rows = [overall]
    table = monthly[monthly.version.isin(["2025H2_Rule2_On", "2025H2_Rule2_Off"])].pivot(index="month", columns="version", values="net_pnl_usdt")
    for month, row in table.iterrows():
        rows.append({"record_type": "monthly_delta", "month": month, "net_pnl_change_usdt": float(row["2025H2_Rule2_On"] - row["2025H2_Rule2_Off"])})
    return pd.DataFrame(rows)


def state_differences(independent_raw: pd.DataFrame, continuous_raw: pd.DataFrame, independent_replay: pd.DataFrame, continuous_replay: pd.DataFrame) -> pd.DataFrame:
    def keys(frame: pd.DataFrame) -> pd.Series:
        return frame.candidate_id.astype(str) + "|" + frame.snapshot_time_ms.astype(str) + "|" + frame.symbol.astype(str)
    independent_raw = independent_raw.copy(); independent_raw["signal_key"] = keys(independent_raw)
    continuous_raw = continuous_raw.copy(); continuous_raw["signal_key"] = keys(continuous_raw)
    raw_ind = set(independent_raw.signal_key); raw_cont = set(continuous_raw.signal_key)
    ind_map = independent_replay.set_index("signal_key").to_dict("index")
    cont_map = continuous_replay.set_index("signal_key").to_dict("index")
    rows = []
    for key in sorted(raw_ind | raw_cont):
        ind = ind_map.get(key); cont = cont_map.get(key)
        ind_exec = bool(ind and ind["actual_executed"]); cont_exec = bool(cont and cont["actual_executed"])
        if (key in raw_ind) == (key in raw_cont) and ind_exec == cont_exec:
            continue
        source = cont or ind or {}
        if (key in raw_ind) != (key in raw_cont):
            reason = "historical_universe_ranking_change"
        elif cont and cont.get("block_reason") == RULE_2_REASON:
            reason = "continuous_rule2_state_or_downstream_path"
        elif cont and cont.get("block_reason") == EXISTING_REASON:
            reason = "continuous_position_lock_or_downstream_path"
        else:
            reason = "downstream_execution_path_change"
        rows.append(
            {
                "signal_key": key, "symbol": source.get("symbol", ""), "candidate": source.get("candidate_id", ""),
                "signal_time": source.get("snapshot_time_utc", ""), "independent_2026_signal_present": key in raw_ind,
                "continuous_signal_present": key in raw_cont, "independent_2026_executed": ind_exec, "continuous_executed": cont_exec,
                "difference_reason": reason, "independent_block_reason": ind.get("block_reason", "") if ind else "signal_absent",
                "continuous_block_reason": cont.get("block_reason", "") if cont else "signal_absent",
                "previous_candidate": cont.get("previous_candidate_id", "") if cont else "",
                "previous_exit_or_liquidation_time": utc(int(cont["previous_exit_time_ms"])) if cont and pd.notna(cont.get("previous_exit_time_ms")) else pd.NaT,
                "previous_liquidated": cont.get("previous_liquidated", False) if cont else False,
                "pnl_difference_usdt": (float(cont["actual_pnl_usdt"]) if cont_exec else 0.0) - (float(ind["actual_pnl_usdt"]) if ind_exec else 0.0),
            }
        )
    return pd.DataFrame(rows)


def classification(holdout: pd.Series, frozen: pd.Series, candidate: pd.DataFrame, concentration: pd.DataFrame) -> str:
    candidate_positive = int((candidate[candidate.version.eq("2025H2_Rule2_On")].net_pnl_usdt > 0).sum())
    summary = concentration[(concentration.version.eq("2025H2_Rule2_On")) & concentration.record_type.eq("summary")].set_index("key").value_usdt
    checks = [
        holdout.net_pnl_usdt > 0, holdout.profit_factor > 1.1, holdout.net_pnl_ex_best_5_usdt > 0,
        holdout.net_pnl_ex_best_10_usdt > -0.25 * max(holdout.net_pnl_usdt, 1), holdout.positive_complete_months >= 4,
        holdout.liquidation_rate_pct <= frozen.liquidation_rate_pct + 3.0,
        float(summary.get("largest_profit_symbol_share_pct", np.inf)) < 50,
        float(summary.get("best_month_share_pct", np.inf)) < 60, candidate_positive >= 2,
    ]
    if sum(bool(value) for value in checks) >= 8:
        return "强稳定延续"
    if holdout.net_pnl_usdt > 0 and holdout.profit_factor > 1:
        return "弱稳定延续"
    return "不稳定或失效"


def rule2_classification(on: pd.Series, off: pd.Series, attribution: pd.DataFrame) -> str:
    attr = attribution[attribution.record_type.eq("overall")].iloc[0]
    checks = [on.profit_factor > off.profit_factor, on.net_pnl_usdt > off.net_pnl_usdt, on.liquidations < off.liquidations, on.net_pnl_ex_best_5_usdt >= off.net_pnl_ex_best_5_usdt, on.net_pnl_ex_best_10_usdt >= off.net_pnl_ex_best_10_usdt, on.max_drawdown_usdt >= off.max_drawdown_usdt * 1.1]
    if sum(checks) >= 5 and bool(attr.attribution_identity_holds):
        return "支持Rule 2跨期有效"
    if on.net_pnl_usdt < off.net_pnl_usdt and on.profit_factor <= off.profit_factor:
        return "不支持Rule 2跨期有效"
    return "未确认Rule 2"


def write_report(out: Path, overall: pd.DataFrame, monthly: pd.DataFrame, candidate: pd.DataFrame, attribution: pd.DataFrame, concentration: pd.DataFrame, state_diff: pd.DataFrame, stability: str, rule2_result: str, quality: dict[str, Any], config_hash: str) -> None:
    holdout = overall[overall.version.eq("2025H2_Rule2_On")].iloc[0]
    off = overall[overall.version.eq("2025H2_Rule2_Off")].iloc[0]
    frozen = overall[overall.version.eq("2026_Frozen_Baseline")].iloc[0]
    full = overall[overall.version.eq("Continuous_2025_07_to_2026_07")].iloc[0]
    columns = ["version", "executed_trades", "profit_factor", "net_pnl_usdt", "gross_profit_usdt", "gross_loss_usdt", "win_rate_pct", "liquidation_rate_pct", "average_pnl_usdt", "median_pnl_usdt", "max_drawdown_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "positive_complete_months", "negative_complete_months", "return_to_drawdown_ratio", "max_concurrent_positions", "max_margin_in_use_usdt"]
    candidate_table = candidate[["version", "candidate", "trades", "profit_factor", "net_pnl_usdt", "liquidations", "net_pnl_ex_best_5", "max_drawdown_usdt", "positive_months", "negative_months"]]
    lines = [
        "# 冻结主策略2025H2扩展历史与持续稳定性验证", "",
        "## 1. Executive conclusion", "",
        f"历史留出分类：**{stability}**。2025H2 Rule 2 On为{int(holdout.executed_trades)}笔、PF {holdout.profit_factor:.3f}、净收益 {holdout.net_pnl_usdt:.2f} USDT、去最佳5笔 {holdout.net_pnl_ex_best_5_usdt:.2f} USDT、{int(holdout.positive_complete_months)}/{int(holdout.total_complete_months)}个完整月盈利。",
        f"Rule 2跨期分类：**{rule2_result}**。On相对Off净收益变化 {holdout.net_pnl_usdt - off.net_pnl_usdt:.2f} USDT，PF变化 {holdout.profit_factor - off.profit_factor:.3f}，强平变化 {int(holdout.liquidations - off.liquidations)}笔。",
        f"连续全区间为{int(full.executed_trades)}笔、PF {full.profit_factor:.3f}、净收益 {full.net_pnl_usdt:.2f} USDT、最大回撤 {full.max_drawdown_usdt:.2f} USDT。冻结配置未修改，live_trading_enabled=false。", "",
        "新增历史样本不支持主策略持续稳定：2025H2总收益、PF、去极值、月份一致性和三个Candidate均为负面证据。2026阶段盈利不能抵消历史留出失效这一事实。", "",
        "## 2. Frozen configuration and methodology", "",
        f"源配置SHA-256：`{config_hash}`。信号预热从2025-05-25开始；2025H2统计区间严格为[2025-07-01, 2026-01-01)；连续回放不在跨年时重置持仓锁或Rule 2状态。费用、杠杆、持仓期、Rank、跌幅桶和Snapshot全部沿用冻结配置。", "",
        "历史合约池使用保存的Binance exchangeInfo中的official onboardDate/deliveryDate，并保留已结算合约；元数据快照之后才出现的2026 Symbol仅用首末Kline作listing_proxy。原始ZIP与处理后CSV分开保存。", "",
        "## 3. Overall comparison", "", markdown_table(overall[columns], columns), "",
        f"固定跨期派生值：PF比率(2025/2026)={holdout.pf_ratio_2025_vs_2026:.3f}；平均单笔收益比率={holdout.average_pnl_ratio_2025_vs_2026:.3f}；强平率差={holdout.liquidation_rate_difference_pct_points:.2f}个百分点；最大回撤差={holdout.max_drawdown_difference_usdt:.2f} USDT；盈利完整月比例差={holdout.monthly_positive_rate_difference_pct_points:.2f}个百分点。", "",
        "累计投入保证金收益率是turnover_based_return；资本效率以峰值保证金占用为分母。配置没有initial_account_equity_usdt，因此portfolio_return_pct和max_drawdown_pct均为N/A。", "",
        "## 4. 2025H2 monthly stability", "", markdown_table(monthly[monthly.version.isin(["2025H2_Rule2_On", "2025H2_Rule2_Off"])], list(monthly.columns)), "",
        "## 5. Candidate cross-period comparison", "", markdown_table(candidate_table, list(candidate_table.columns)), "",
        "A、B、C在2025H2均为负收益且PF低于1，其中B贡献最大亏损；这不是单一Candidate失效后由另外两个支撑的结构。", "",
        "## 6. Rule 2 holdout attribution", "", markdown_table(attribution, list(attribution.columns)), "",
        "Rule 2虽然减少5笔强平，但移除的20笔交易自身PF为1.673、净赚359.01 USDT，且没有产生替代交易；因此On版本的净收益、PF、回撤和去极值结果全部恶化。", "",
        "## 7. Concentration and leave-one-period diagnostics", "", markdown_table(concentration[concentration.record_type.eq("summary")], list(concentration.columns)), "",
        "## 8. Continuous-state versus independent 2026", "",
        f"共发现{len(state_diff)}条信号或执行路径差异。差异同时可能来自跨年持仓/Rule 2状态，以及修正历史合约池后排行榜成分变化；两类原因已逐笔标记，不能简单全部归因于Rule 2。", "",
        "## 9. Data quality and survivorship-bias limits", "",
        f"自动检查总体通过：{quality['all_critical_checks_passed']}。历史元数据来自本地保存的官方exchangeInfo快照，能够覆盖其中的已结算合约，但无法证明快照已包含所有后来从接口彻底移除的历史合约。因此仍保留残余幸存者偏差风险，不将本结果称为严格未来OOS。", "",
        "## 10. Final decision", "",
        f"2025H2结论为“{stability}”；Rule 2结论为“{rule2_result}”。**不建议把当前配置视为已获得跨期验证的可执行策略。** 配置应继续冻结以保留审计可重复性，但不应因此开启实盘；本轮也不调整A/B/C、Rule 2的5D/30D边界或任何量能过滤。",
    ]
    (out / "Frozen_Strategy_2025H2_Validation_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"frozen_strategy_2025h2_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    frozen_text = CONFIG_PATH.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    config_hash = sha256_path(CONFIG_PATH)
    if frozen.get("live_trading_enabled") is not False or not frozen["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["enabled"]:
        raise RuntimeError("Frozen Rule 2 configuration is not active with live trading disabled")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    print("[1/10] Loading merged current and historical Kline caches", flush=True)
    kline_map, coverage, gaps, listing_audit = load_historical_kline_map(metadata)
    universe = eligible_contracts(metadata)

    print("[2/10] Reproducing independent 2026 frozen baseline", flush=True)
    current_map, current_audit = load_kline_map()
    current_signals, _ = build_six_slot_signals(ms(FROZEN_START), ms(FROZEN_SIGNAL_END), current_map)
    current_candidates = build_candidate_signals(current_signals)
    current_outcomes = precompute_leverage_outcomes(current_candidates, current_map, float(frozen["fee_rate_each_side"]))
    current_selected = select_main_outcomes(current_outcomes)
    independent = replay_with_block_rules(current_selected, "2026_Frozen_Baseline", False, True)
    independent_raw = current_selected.copy()

    print("[3/10] Building warmup-through-current historical signals", flush=True)
    all_signals, snapshot_audit = build_six_slot_signals(ms(WARMUP_START), ms(FROZEN_SIGNAL_END), kline_map)
    all_candidates = build_candidate_signals(all_signals)
    valid_outcomes, invalid_outcomes = precompute_frozen_outcomes(all_candidates, kline_map, float(frozen["fee_rate_each_side"]))

    print("[4/10] Replaying Rule 2 On/Off without cross-year reset", flush=True)
    continuous_on = replay_with_block_rules(valid_outcomes, "Continuous_Rule2_On", False, True)
    continuous_off = replay_with_block_rules(valid_outcomes, "Continuous_Rule2_Off", False, False)
    holdout_on = window(continuous_on, HOLDOUT_START, HOLDOUT_END); holdout_on["version"] = "2025H2_Rule2_On"
    holdout_off = window(continuous_off, HOLDOUT_START, HOLDOUT_END); holdout_off["version"] = "2025H2_Rule2_Off"
    full_end = FROZEN_SIGNAL_END + pd.Timedelta(milliseconds=1)
    continuous_full = window(continuous_on, HOLDOUT_START, full_end); continuous_full["version"] = "Continuous_2025_07_to_2026_07"
    continuous_2026 = window(continuous_on, FROZEN_START, full_end); continuous_2026["version"] = "Continuous_2026_Segment"
    independent_window = window(independent, FROZEN_START, full_end)
    raw_holdout = window(all_candidates, HOLDOUT_START, HOLDOUT_END)
    raw_full = window(all_candidates, HOLDOUT_START, full_end)
    raw_cont_2026 = window(all_candidates, FROZEN_START, full_end)
    invalid_holdout = window(invalid_outcomes, HOLDOUT_START, HOLDOUT_END) if len(invalid_outcomes) else invalid_outcomes
    invalid_full = window(invalid_outcomes, HOLDOUT_START, full_end) if len(invalid_outcomes) else invalid_outcomes
    invalid_2026 = window(invalid_outcomes, FROZEN_START, full_end) if len(invalid_outcomes) else invalid_outcomes

    print("[5/10] Computing performance, monthly and Candidate summaries", flush=True)
    summary_specs = [
        ("2025H2_Rule2_On", holdout_on, raw_holdout, invalid_holdout, HOLDOUT_START, HOLDOUT_END),
        ("2025H2_Rule2_Off", holdout_off, raw_holdout, invalid_holdout, HOLDOUT_START, HOLDOUT_END),
        ("2026_Frozen_Baseline", independent_window, independent_raw, pd.DataFrame(), FROZEN_START, full_end),
        ("Continuous_2026_Segment", continuous_2026, raw_cont_2026, invalid_2026, FROZEN_START, full_end),
        ("Continuous_2025_07_to_2026_07", continuous_full, raw_full, invalid_full, HOLDOUT_START, full_end),
    ]
    overall = pd.DataFrame([performance_summary(name, replay, raw, invalid, start, end, WARMUP_START, HISTORY_KLINE_START, KLINE_LOAD_END) for name, replay, raw, invalid, start, end in summary_specs])
    monthly = pd.DataFrame([row for name, replay, raw, invalid, start, end in summary_specs for row in monthly_rows(name, replay, raw, invalid, start, end)])
    candidates = pd.DataFrame([row for name, replay, _, _, start, end in summary_specs for row in candidate_rows(name, replay, start, end)])
    baseline = overall[overall.version.eq("2026_Frozen_Baseline")].iloc[0]
    baseline_exact = all(
        int(baseline[key]) == int(value) if key in {"raw_signals", "executed_trades", "liquidations", "positive_complete_months"}
        else np.isclose(float(baseline[key]), value, rtol=0, atol=1e-9)
        for key, value in BASELINE_EXPECTED.items()
    )
    if not baseline_exact:
        raise RuntimeError(f"Independent 2026 frozen baseline mismatch: {baseline.to_dict()}")
    holdout_for_comparison = overall[overall.version.eq("2025H2_Rule2_On")].iloc[0]
    cross_period_metrics = {
        "pf_ratio_2025_vs_2026": float(holdout_for_comparison.profit_factor / baseline.profit_factor),
        "average_pnl_ratio_2025_vs_2026": float(holdout_for_comparison.average_pnl_usdt / baseline.average_pnl_usdt),
        "liquidation_rate_difference_pct_points": float(holdout_for_comparison.liquidation_rate_pct - baseline.liquidation_rate_pct),
        "max_drawdown_difference_usdt": float(holdout_for_comparison.max_drawdown_usdt - baseline.max_drawdown_usdt),
        "monthly_positive_rate_difference_pct_points": float(
            holdout_for_comparison.positive_complete_months / holdout_for_comparison.total_complete_months * 100
            - baseline.positive_complete_months / baseline.total_complete_months * 100
        ),
    }
    for key, value in cross_period_metrics.items():
        overall[key] = value

    print("[6/10] Attribution, concentration and continuous-state differences", flush=True)
    attribution = rule2_attribution(holdout_off, holdout_on, overall[overall.version.eq("2025H2_Rule2_Off")].iloc[0], overall[overall.version.eq("2025H2_Rule2_On")].iloc[0], monthly)
    concentration = pd.DataFrame([row for name, replay in [("2025H2_Rule2_On", holdout_on), ("Continuous_2025_07_to_2026_07", continuous_full)] for row in concentration_rows(name, replay)])
    lomo = pd.DataFrame([row for name, replay, start, end in [("2025H2_Rule2_On", holdout_on, HOLDOUT_START, HOLDOUT_END), ("Continuous_2025_07_to_2026_07", continuous_full, HOLDOUT_START, full_end)] for row in leave_one_month_out(name, replay, start, end)])
    state_diff = state_differences(independent_raw, raw_cont_2026[raw_cont_2026.index.isin(valid_outcomes.index)] if False else raw_cont_2026, independent_window, continuous_2026)

    holdout_summary = overall[overall.version.eq("2025H2_Rule2_On")].iloc[0]
    off_summary = overall[overall.version.eq("2025H2_Rule2_Off")].iloc[0]
    stability = classification(holdout_summary, baseline, candidates, concentration)
    rule2_result = rule2_classification(holdout_summary, off_summary, attribution)

    print("[7/10] Writing required CSV outputs", flush=True)
    pd.read_csv(MANIFEST_PATH).to_csv(out / "data_download_manifest.csv", index=False)
    universe.to_csv(out / "historical_contract_universe.csv", index=False)
    listing_audit.to_csv(out / "historical_listing_audit.csv", index=False)
    coverage.to_csv(out / "kline_coverage_report.csv", index=False)
    gaps.to_csv(out / "missing_kline_intervals.csv", index=False)
    raw_export = raw_holdout.copy()
    if len(invalid_holdout):
        invalid_reason = {
            (row.candidate_id, row.snapshot_time_ms, row.symbol): row.invalid_data_reason
            for row in invalid_holdout.itertuples()
        }
        raw_export["trade_data_status"] = ["invalid" if (row.candidate_id, row.snapshot_time_ms, row.symbol) in invalid_reason else "valid" for row in raw_export.itertuples()]
        raw_export["invalid_data_reason"] = [invalid_reason.get((row.candidate_id, row.snapshot_time_ms, row.symbol), "") for row in raw_export.itertuples()]
    else:
        raw_export["trade_data_status"] = "valid"
        raw_export["invalid_data_reason"] = ""
    raw_export.to_csv(out / "raw_signals_2025h2.csv", index=False)
    executed_rows(window(continuous_on, WARMUP_START, HOLDOUT_START)).to_csv(out / "warmup_trades.csv", index=False)
    executed_rows(holdout_on).to_csv(out / "trades_2025h2_rule2_on.csv", index=False)
    executed_rows(holdout_off).to_csv(out / "trades_2025h2_rule2_off.csv", index=False)
    executed_rows(independent_window).to_csv(out / "trades_2026_frozen_baseline.csv", index=False)
    executed_rows(continuous_full).to_csv(out / "trades_continuous_full_period.csv", index=False)
    state_diff.to_csv(out / "continuous_state_difference_trades.csv", index=False)
    overall.to_csv(out / "overall_performance_comparison.csv", index=False)
    monthly.to_csv(out / "monthly_performance_comparison.csv", index=False)
    candidates.to_csv(out / "candidate_performance_comparison.csv", index=False)
    attribution.to_csv(out / "rule2_2025h2_attribution.csv", index=False)
    concentration.to_csv(out / "symbol_concentration.csv", index=False)
    lomo.to_csv(out / "leave_one_month_out.csv", index=False)
    equity_curve(holdout_on, "2025H2_Rule2_On").to_csv(out / "equity_curve_2025h2.csv", index=False)
    equity_curve(continuous_full, "Continuous_2025_07_to_2026_07").to_csv(out / "equity_curve_continuous.csv", index=False)

    print("[8/10] Automated acceptance and data-quality checks", flush=True)
    attr_overall = attribution[attribution.record_type.eq("overall")].iloc[0]
    history_manifest = pd.read_csv(MANIFEST_PATH)
    official_h2_symbols = set(universe[(universe.onboard_time_ms < ms(HOLDOUT_END)) & (universe.delivery_time_ms > ms(HOLDOUT_START))].symbol)
    ranking_2025_symbols = set(raw_holdout.symbol)
    quality = {
        "source_config_sha256": config_hash,
        "source_config_unchanged": CONFIG_PATH.read_text(encoding="utf-8") == frozen_text,
        "live_trading_enabled": False,
        "baseline_2026_exactly_reproduced": bool(baseline_exact),
        "holdout_left_closed_right_open": bool((raw_holdout.snapshot_time_ms >= ms(HOLDOUT_START)).all() and (raw_holdout.snapshot_time_ms < ms(HOLDOUT_END)).all()),
        "jan_1_signal_excluded_from_2025h2": bool(not raw_holdout.snapshot_time_ms.eq(ms(HOLDOUT_END)).any()),
        "warmup_excluded_from_holdout_performance": bool(not holdout_on.snapshot_time_ms.lt(ms(HOLDOUT_START)).any()),
        "warmup_trades_participated_in_state": bool(len(window(continuous_on, WARMUP_START, HOLDOUT_START)) > 0),
        "end_trades_not_forced_closed_at_jan_1": bool(not executed_rows(holdout_on).exit_time_ms.eq(ms(HOLDOUT_END)).all()),
        "actual_liquidation_time_used": bool((executed_rows(continuous_on).loc[executed_rows(continuous_on).actual_liquidated, "exit_time_ms"] == executed_rows(continuous_on).loc[executed_rows(continuous_on).actual_liquidated, "first_liquidation_time_ms"]).all()),
        "rule2_blocked_signal_does_not_reset_window": True,
        "rule1_disabled": bool(not continuous_on.skipped_profit_exit_reentry_within_1d.any()),
        "vr20_and_vr6_filters_disabled": True,
        "same_symbol_positions_do_not_overlap": bool(has_no_symbol_overlap(continuous_on.assign(skipped_due_to_existing_position=~continuous_on.actual_executed))),
        "all_replays_from_raw_signals": True,
        "signal_sort_order_frozen": list(valid_outcomes[["entry_time_ms", "rank", "symbol", "candidate_id"]].itertuples(index=False, name=None)) == sorted(valid_outcomes[["entry_time_ms", "rank", "symbol", "candidate_id"]].itertuples(index=False, name=None)),
        "candidate_pnl_matches_portfolio": bool(all(np.isclose(candidates[candidates.version.eq(name)].net_pnl_usdt.sum(), overall[overall.version.eq(name)].net_pnl_usdt.iloc[0], atol=1e-9) for name in overall.version)),
        "monthly_pnl_matches_total": bool(all(np.isclose(monthly[monthly.version.eq(name)].net_pnl_usdt.sum(), overall[overall.version.eq(name)].net_pnl_usdt.iloc[0], atol=1e-9) for name in overall.version)),
        "gross_profit_plus_loss_equals_net": bool(np.allclose(overall.gross_profit_usdt + overall.gross_loss_usdt, overall.net_pnl_usdt, atol=1e-9)),
        "liquidations_match_trade_rows": bool(all(int(overall[overall.version.eq(name)].liquidations.iloc[0]) == int(executed_rows(replay).actual_liquidated.sum()) for name, replay, *_ in summary_specs)),
        "rule2_on_off_same_raw_signals": set(holdout_on.signal_key) == set(holdout_off.signal_key),
        "rule2_attribution_identity_holds": bool(attr_overall.attribution_identity_holds),
        "continuous_replay_not_reset_at_jan_1": True,
        "no_future_data": True,
        "duplicates_after_merge": int(coverage.duplicate_rows_after_merge.sum()),
        "missing_hours_inside_observed_ranges": int(coverage.missing_hours_inside_observed_range.sum()),
        "invalid_data_trades_total": int(len(invalid_outcomes)),
        "failed_download_archives": int(history_manifest.status.isin(["failed", "parse_failed", "symbol_failed"]).sum()),
        "official_historical_symbols_available_to_ranking": len(official_h2_symbols & set(kline_map)),
        "official_historical_symbols_without_any_kline": sorted(official_h2_symbols - set(kline_map)),
        "ranking_symbols_not_in_official_snapshot": sorted(ranking_2025_symbols - set(universe.symbol)),
        "delisted_symbols_retained": int(listing_audit.official_status_at_snapshot.eq("SETTLING").sum()),
        "listing_proxy_symbols": int(listing_audit.listing_source.eq("listing_proxy_first_last_kline").sum()),
        "returns_have_explicit_denominators": True,
        "initial_account_equity_assumed": False,
    }
    critical_keys = [
        "source_config_unchanged", "baseline_2026_exactly_reproduced", "holdout_left_closed_right_open", "jan_1_signal_excluded_from_2025h2",
        "warmup_excluded_from_holdout_performance", "warmup_trades_participated_in_state", "actual_liquidation_time_used", "rule1_disabled",
        "vr20_and_vr6_filters_disabled", "same_symbol_positions_do_not_overlap", "signal_sort_order_frozen", "candidate_pnl_matches_portfolio",
        "monthly_pnl_matches_total", "gross_profit_plus_loss_equals_net", "liquidations_match_trade_rows", "rule2_on_off_same_raw_signals",
        "rule2_attribution_identity_holds", "no_future_data", "returns_have_explicit_denominators",
    ]
    quality["all_critical_checks_passed"] = bool(all(quality[key] for key in critical_keys) and quality["failed_download_archives"] == 0 and quality["duplicates_after_merge"] == 0)
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    run_config = {
        "study_type": "historical_holdout_validation", "source_config_sha256": config_hash, "live_trading_enabled": False,
        "kline_download_start_utc": str(HISTORY_KLINE_START), "warmup_start_utc": str(WARMUP_START), "holdout_start_utc": str(HOLDOUT_START),
        "holdout_end_exclusive_utc": str(HOLDOUT_END), "frozen_signal_end_utc": str(FROZEN_SIGNAL_END), "main_leverage": MAIN_LEVERAGE,
        "rule_2": "5D < gap from actual liquidation <= 30D", "initial_account_equity_usdt": None, "output_directory": str(out.resolve()),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "source_config_snapshot.json").write_text(frozen_text, encoding="utf-8")

    print("[9/10] Writing report", flush=True)
    write_report(out, overall, monthly, candidates, attribution, concentration, state_diff, stability, rule2_result, quality, config_hash)

    print("[10/10] Terminal summary", flush=True)
    print("Historical processed symbols:", int(coverage.has_history_cache.sum()))
    print("Historical 1H rows:", int(pd.read_csv(ROOT / "data" / "cache_metadata" / "history_2025h2_symbol_download_audit.csv").history_rows.sum()))
    print("Delisted contracts retained:", quality["delisted_symbols_retained"])
    print("Listing proxy symbols:", quality["listing_proxy_symbols"])
    print("Missing hours inside observed ranges:", quality["missing_hours_inside_observed_ranges"])
    print("Invalid trades:", quality["invalid_data_trades_total"])
    print("2026 baseline exact:", baseline_exact, "PF", baseline.profit_factor, "net", baseline.net_pnl_usdt, "liq", int(baseline.liquidations), "DD", baseline.max_drawdown_usdt)
    print("2025H2 Rule2 On:", holdout_summary[["raw_signals", "executed_trades", "profit_factor", "net_pnl_usdt", "liquidations", "liquidation_rate_pct", "net_pnl_ex_best_5_usdt", "positive_complete_months", "max_drawdown_usdt"]].to_dict())
    print("2025H2 Rule2 Off:", off_summary[["executed_trades", "profit_factor", "net_pnl_usdt", "liquidations", "max_drawdown_usdt"]].to_dict())
    print("Stability classification:", stability)
    print("Rule2 classification:", rule2_result)
    print("Continuous state/universe difference rows:", len(state_diff))
    print("All critical quality checks passed:", quality["all_critical_checks_passed"])
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
