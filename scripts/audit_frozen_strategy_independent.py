from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
UTC = "UTC"
BEIJING = "Asia/Shanghai"


def to_ms(value: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def as_utc(value: int) -> pd.Timestamp:
    return pd.Timestamp(value, unit="ms", tz=UTC)


def beijing_slot_to_utc(slot: str) -> tuple[int, int]:
    hour = int(slot[:2])
    utc_hour = (hour - 8) % 24
    day_offset = -1 if hour < 8 else 0
    return utc_hour, day_offset


def load_metadata(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Merge historical snapshots without deleting symbols absent from a newer snapshot."""
    observations: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("symbols", []):
            observations.setdefault(str(item["symbol"]), []).append(item)

    merged: dict[str, dict[str, Any]] = {}
    distant_future = 4_000_000_000_000
    for symbol, items in observations.items():
        perpetual_usdt = [
            item
            for item in items
            if item.get("contractType") == "PERPETUAL" and item.get("quoteAsset") == "USDT"
        ]
        if not perpetual_usdt:
            continue
        finite_deliveries = [
            int(item.get("deliveryDate", distant_future))
            for item in perpetual_usdt
            if int(item.get("deliveryDate", distant_future)) < distant_future
        ]
        merged[symbol] = {
            "symbol": symbol,
            "contractType": "PERPETUAL",
            "quoteAsset": "USDT",
            "onboardDate": min(int(item["onboardDate"]) for item in perpetual_usdt),
            "deliveryDate": min(finite_deliveries) if finite_deliveries else distant_future,
            "observations": len(perpetual_usdt),
            "observed_statuses": sorted({str(item.get("status", "")) for item in perpetual_usdt}),
        }
    return merged


def load_klines(
    directory: Path,
    earliest_ms: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    columns = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "close_time"]
    for path in sorted(directory.glob("*_1h.csv")):
        symbol = path.stem.removesuffix("_1h")
        source = pd.read_csv(path, usecols=columns)
        source_rows = len(source)
        source = source[source.open_time >= earliest_ms].copy()
        duplicates = int(source.open_time.duplicated(keep="last").sum())
        source = source.drop_duplicates("open_time", keep="last").sort_values("open_time")
        invalid = (
            ~np.isfinite(source[["open", "high", "low", "close", "volume", "quote_volume"]]).all(axis=1)
            | (source[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (source.high < source[["open", "close"]].max(axis=1))
            | (source.low > source[["open", "close"]].min(axis=1))
            | (source.high < source.low)
        )
        invalid_rows = int(invalid.sum())
        source = source.loc[~invalid].set_index("open_time", drop=False)
        if source.empty:
            continue
        frames[symbol] = source
        expected = (int(source.open_time.max()) - int(source.open_time.min())) // HOUR_MS + 1
        audit_rows.append(
            {
                "symbol": symbol,
                "absolute_path": str(path.resolve()),
                "source_rows": source_rows,
                "loaded_rows": len(source),
                "duplicate_rows_removed": duplicates,
                "invalid_rows_removed": invalid_rows,
                "missing_hour_count": max(0, expected - len(source)),
                "first_open_time": as_utc(int(source.open_time.min())),
                "last_open_time": as_utc(int(source.open_time.max())),
            }
        )
    if not frames:
        raise RuntimeError(f"No raw 1H CSV files found in {directory.resolve()}")
    return frames, pd.DataFrame(audit_rows)


def schedule_times(start_ms: int, end_ms: int) -> list[int]:
    result: list[int] = []
    for day in pd.date_range(as_utc(start_ms).floor("D"), as_utc(end_ms).floor("D"), freq="D", tz=UTC):
        for hour in (0, 4, 8, 12, 16, 20):
            value = to_ms(day + pd.Timedelta(hours=hour))
            if start_ms <= value <= end_ms:
                result.append(value)
    return result


def _metadata_reason(item: dict[str, Any] | None, snapshot_ms: int) -> str | None:
    if item is None:
        return "missing_historical_metadata"
    if item["contractType"] != "PERPETUAL":
        return "not_perpetual"
    if item["quoteAsset"] != "USDT":
        return "not_usdt_m"
    if int(item["onboardDate"]) > snapshot_ms:
        return "not_yet_onboarded"
    if int(item["deliveryDate"]) <= snapshot_ms:
        return "already_delisted"
    return None


def build_universe_and_rankings(
    snapshots: Iterable[int],
    klines: dict[str, pd.DataFrame],
    metadata: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    all_symbols = sorted(set(metadata) | set(klines))
    for snapshot_ms in snapshots:
        current_open = snapshot_ms - HOUR_MS
        prior_open = current_open - 24 * HOUR_MS
        eligible: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        for symbol in all_symbols:
            reason = _metadata_reason(metadata.get(symbol), snapshot_ms)
            frame = klines.get(symbol)
            if reason is None and frame is None:
                reason = "missing_raw_kline_file"
            if reason is None and current_open not in frame.index:
                reason = "missing_completed_snapshot_kline"
            if reason is None and prior_open not in frame.index:
                reason = "missing_strict_24h_kline"
            if reason is not None:
                reasons[reason] += 1
                continue
            close_now = float(frame.at[current_open, "close"])
            close_24h_ago = float(frame.at[prior_open, "close"])
            drop_pct = (close_24h_ago - close_now) / close_24h_ago * 100.0
            eligible.append(
                {
                    "snapshot_time_ms": snapshot_ms,
                    "snapshot_time": as_utc(snapshot_ms),
                    "symbol": symbol,
                    "close_now": close_now,
                    "close_24h_ago": close_24h_ago,
                    "drop_24h_pct": drop_pct,
                }
            )
        eligible.sort(key=lambda item: (-item["drop_24h_pct"], item["symbol"]))
        size = len(eligible)
        for rank, item in enumerate(eligible, 1):
            ranking_rows.append({**item, "rank": rank, "eligible_universe_size": size})
        universe_rows.append(
            {
                "snapshot_time_ms": snapshot_ms,
                "snapshot_time": as_utc(snapshot_ms),
                "eligible_symbol_count": size,
                "excluded_symbol_count": len(all_symbols) - size,
                "eligible_symbols_json": json.dumps([item["symbol"] for item in eligible], ensure_ascii=False),
                "exclusion_reason": json.dumps(dict(sorted(reasons.items())), ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(universe_rows), pd.DataFrame(ranking_rows)


def build_raw_signals(rankings: pd.DataFrame, frozen: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_candidate = frozen["main_candidates"]
    relevant = rankings[rankings["rank"].isin({int(spec["rank"]) for spec in by_candidate.values()})]
    for item in relevant.to_dict("records"):
        beijing_time = as_utc(int(item["snapshot_time_ms"])).tz_convert(BEIJING)
        slot = beijing_time.strftime("%H:%M")
        for candidate, spec in by_candidate.items():
            low, high = map(float, spec["drop_bucket_pct"])
            if (
                int(item["rank"]) == int(spec["rank"])
                and low <= float(item["drop_24h_pct"]) < high
                and slot in spec["snapshot_times_beijing"]
            ):
                rows.append(
                    {
                        **item,
                        "signal_time_ms": int(item["snapshot_time_ms"]),
                        "signal_time": item["snapshot_time"],
                        "snapshot_time_beijing": beijing_time,
                        "snapshot_hour_beijing": slot,
                        "candidate": candidate,
                        "direction": frozen["direction"],
                        "holding_days": int(spec["holding_days"]),
                        "leverage": int(spec["leverage"]),
                    }
                )
    columns = [
        "signal_time_ms", "signal_time", "snapshot_time_beijing", "snapshot_hour_beijing",
        "candidate", "symbol", "rank", "close_now", "close_24h_ago", "drop_24h_pct",
        "eligible_universe_size", "direction", "holding_days", "leverage",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["signal_time_ms", "rank", "symbol", "candidate"]
    ).reset_index(drop=True)


def calculate_outcomes(
    signals: pd.DataFrame,
    klines: dict[str, pd.DataFrame],
    frozen: dict[str, Any],
) -> pd.DataFrame:
    margin = float(frozen["margin_per_trade_usdt"])
    fee_rate = float(frozen["fee_rate_each_side"])
    rows: list[dict[str, Any]] = []
    for signal in signals.to_dict("records"):
        frame = klines[str(signal["symbol"])]
        entry_time = int(signal["signal_time_ms"])
        planned_exit = entry_time + int(signal["holding_days"]) * DAY_MS
        outcome = dict(signal)
        outcome.update(
            {
                "entry_time_ms": entry_time,
                "entry_time": as_utc(entry_time),
                "planned_exit_time_ms": planned_exit,
                "planned_exit_time": as_utc(planned_exit),
                "margin_usdt": margin,
                "notional_usdt": margin * int(signal["leverage"]),
            }
        )
        if entry_time not in frame.index or planned_exit not in frame.index:
            outcome.update({"outcome_available": False, "incomplete_reason": "missing_entry_or_full_fixed_exit_kline"})
            rows.append(outcome)
            continue
        entry_price = float(frame.at[entry_time, "open"])
        fixed_exit_price = float(frame.at[planned_exit, "open"])
        leverage = int(signal["leverage"])
        liquidation_price = entry_price * (1.0 + 1.0 / leverage)
        path = frame[(frame.open_time >= entry_time) & (frame.open_time < planned_exit)]
        hits = path[path.high >= liquidation_price]
        liquidated = not hits.empty
        if liquidated:
            actual_exit = int(hits.iloc[0].open_time)
            exit_price = liquidation_price
            gross_pnl = -margin
            entry_fee = 0.0
            exit_fee = 0.0
            net_pnl = -margin
            exit_reason = f"liquidation_{leverage}x_short"
        else:
            actual_exit = planned_exit
            exit_price = fixed_exit_price
            notional = margin * leverage
            ratio = exit_price / entry_price
            gross_pnl = notional * (1.0 - ratio)
            entry_fee = notional * fee_rate
            exit_fee = notional * ratio * fee_rate
            net_pnl = gross_pnl - entry_fee - exit_fee
            exit_reason = "fixed_exit"
        outcome.update(
            {
                "outcome_available": True,
                "incomplete_reason": "",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "liquidation_price": liquidation_price,
                "actual_exit_time_ms": actual_exit,
                "actual_exit_time": as_utc(actual_exit),
                "gross_pnl_usdt": gross_pnl,
                "entry_fee_usdt": entry_fee,
                "exit_fee_usdt": exit_fee,
                "fees_usdt": entry_fee + exit_fee,
                "net_pnl_usdt": net_pnl,
                "return_on_margin_pct": net_pnl / margin * 100.0,
                "liquidated": liquidated,
                "exit_reason": exit_reason,
            }
        )
        rows.append(outcome)
    return pd.DataFrame(rows)


def replay_rule_2(outcomes: pd.DataFrame) -> pd.DataFrame:
    open_positions: dict[str, dict[str, Any]] = {}
    last_completed: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    ordered = outcomes.sort_values(["entry_time_ms", "rank", "symbol", "candidate"])
    for source in ordered.to_dict("records"):
        row = dict(source)
        symbol = str(row["symbol"])
        entry_time = int(row["entry_time_ms"])
        blocker = open_positions.get(symbol)
        if blocker is not None and entry_time >= int(blocker["actual_exit_time_ms"]):
            last_completed[symbol] = blocker
            del open_positions[symbol]
            blocker = None
        previous = last_completed.get(symbol)
        gap_ms = entry_time - int(previous["actual_exit_time_ms"]) if previous else None
        rule_2 = bool(previous and previous["liquidated"] and 5 * DAY_MS < gap_ms <= 30 * DAY_MS)
        if not bool(row.get("outcome_available", False)):
            reason = "incomplete_holding_data"
        elif blocker is not None:
            reason = "global_existing_position"
        elif rule_2:
            reason = "blocked_post_liquidation_reentry_5d_30d"
        else:
            reason = ""
        executed = reason == ""
        row.update(
            {
                "executed": executed,
                "skip_reason": reason,
                "existing_position": blocker is not None,
                "position_release_time_ms": int(blocker["actual_exit_time_ms"]) if blocker else np.nan,
                "previous_actual_liquidation_time_ms": (
                    int(previous["actual_exit_time_ms"]) if previous and previous["liquidated"] else np.nan
                ),
                "previous_trade_liquidated": bool(previous and previous["liquidated"]),
                "rule_2_triggered": rule_2 and blocker is None and bool(row.get("outcome_available", False)),
                "rule_2_gap_days": gap_ms / DAY_MS if gap_ms is not None else np.nan,
            }
        )
        if executed:
            open_positions[symbol] = {
                "candidate": row["candidate"],
                "entry_time_ms": entry_time,
                "actual_exit_time_ms": int(row["actual_exit_time_ms"]),
                "net_pnl_usdt": float(row["net_pnl_usdt"]),
                "liquidated": bool(row["liquidated"]),
                "exit_reason": row["exit_reason"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def longest_streak(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def complete_months(start_ms: int, end_ms: int) -> set[str]:
    result: set[str] = set()
    start = as_utc(start_ms)
    end = as_utc(end_ms)
    for month_start in pd.date_range(start.floor("D").replace(day=1), end.floor("D"), freq="MS", tz=UTC):
        if start <= month_start and end >= month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1):
            result.add(month_start.strftime("%Y-%m"))
    return result


def drawdown_stats(executed: pd.DataFrame) -> tuple[float, float]:
    if executed.empty:
        return 0.0, 0.0
    ordered = executed.sort_values(["actual_exit_time_ms", "rank", "symbol"])
    equity = ordered.net_pnl_usdt.astype(float).cumsum().to_numpy()
    times = ordered.actual_exit_time_ms.astype("int64").to_numpy()
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity - running_peak
    max_dd = float(drawdowns.min())
    longest = 0.0
    peak_time = int(times[0])
    for time_ms, value, peak in zip(times, equity, running_peak):
        if value >= peak:
            peak_time = int(time_ms)
        else:
            longest = max(longest, (int(time_ms) - peak_time) / HOUR_MS)
    return max_dd, longest


def exposure_stats(executed: pd.DataFrame) -> dict[str, Any]:
    if executed.empty:
        return {
            "max_concurrent_positions": 0,
            "max_margin_in_use_usdt": 0.0,
            "max_gross_notional_exposure_usdt": 0.0,
        }
    events: list[tuple[int, int, float, float]] = []
    for row in executed.itertuples(index=False):
        events.append((int(row.entry_time_ms), 1, float(row.margin_usdt), float(row.notional_usdt)))
        events.append((int(row.actual_exit_time_ms), -1, -float(row.margin_usdt), -float(row.notional_usdt)))
    frame = pd.DataFrame(events, columns=["time", "positions", "margin", "notional"])
    frame = frame.groupby("time", as_index=False).sum().sort_values("time")
    for column in ("positions", "margin", "notional"):
        frame[column] = frame[column].cumsum()
    return {
        "max_concurrent_positions": int(frame.positions.max()),
        "max_margin_in_use_usdt": float(frame.margin.max()),
        "max_gross_notional_exposure_usdt": float(frame.notional.max()),
    }


def performance_metrics(replay: pd.DataFrame, start_ms: int, end_ms: int) -> dict[str, Any]:
    executed = replay[replay.executed].sort_values(["actual_exit_time_ms", "rank", "symbol"]).copy()
    pnl = executed.net_pnl_usdt.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    max_dd, dd_hours = drawdown_stats(executed)
    complete = complete_months(start_ms, end_ms)
    monthly = (
        executed.assign(month=pd.to_datetime(executed.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m"))
        .groupby("month").net_pnl_usdt.sum()
    )
    complete_values = monthly[monthly.index.isin(complete)]
    result = {
        "raw_signals": len(replay),
        "eligible_signals": int(replay.outcome_available.sum()),
        "executed_trades": len(executed),
        "skipped_existing_position": int(replay.skip_reason.eq("global_existing_position").sum()),
        "skipped_rule_2": int(replay.skip_reason.eq("blocked_post_liquidation_reentry_5d_30d").sum()),
        "unique_symbols": int(executed.symbol.nunique()),
        "wins": len(wins),
        "ordinary_losses": int(((pnl < 0) & ~executed.liquidated.astype(bool)).sum()),
        "liquidations": int(executed.liquidated.sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else np.nan,
        "liquidation_rate_pct": float(executed.liquidated.mean() * 100) if len(pnl) else np.nan,
        "gross_profit_usdt": gross_profit,
        "gross_loss_usdt": gross_loss,
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss else (math.inf if gross_profit else np.nan),
        "average_pnl_usdt": float(pnl.mean()) if len(pnl) else np.nan,
        "median_pnl_usdt": float(pnl.median()) if len(pnl) else np.nan,
        "best_trade_usdt": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "max_drawdown_usdt": max_dd,
        "max_drawdown_duration_hours": dd_hours,
        "max_consecutive_wins": longest_streak(pnl > 0),
        "max_consecutive_losses": longest_streak(pnl < 0),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(min(1, len(pnl))).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(min(10, len(pnl))).sum()),
        "positive_complete_months": int((complete_values > 0).sum()),
        "negative_complete_months": int((complete_values < 0).sum()),
        "total_complete_months": len(complete_values),
        "return_to_drawdown_ratio": float(pnl.sum() / abs(max_dd)) if max_dd < 0 else np.nan,
        **exposure_stats(executed),
    }
    return result


def monthly_summary(replay: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    complete = complete_months(start_ms, end_ms)
    replay = replay.copy()
    replay["month"] = pd.to_datetime(replay.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")
    months = pd.period_range(as_utc(start_ms).strftime("%Y-%m"), as_utc(end_ms).strftime("%Y-%m"), freq="M").astype(str)
    rows = []
    for month in months:
        raw = replay[replay.month.eq(month)]
        done = raw[raw.executed]
        pnl = done.net_pnl_usdt.astype(float)
        profit = float(pnl[pnl > 0].sum())
        loss = float(pnl[pnl < 0].sum())
        rows.append(
            {
                "month": month,
                "complete_month": month in complete,
                "raw_signals": len(raw),
                "executed_trades": len(done),
                "wins": int((pnl > 0).sum()),
                "ordinary_losses": int(((pnl < 0) & ~done.liquidated.astype(bool)).sum()),
                "liquidations": int(done.liquidated.sum()),
                "net_pnl_usdt": float(pnl.sum()),
                "profit_factor": profit / abs(loss) if loss else (math.inf if profit else np.nan),
            }
        )
    return pd.DataFrame(rows)


def candidate_summary(replay: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    for candidate in sorted(replay.candidate.unique()):
        group = replay[replay.candidate.eq(candidate)].copy()
        rows.append({"candidate": candidate, **performance_metrics(group, start_ms, end_ms)})
    return pd.DataFrame(rows)
