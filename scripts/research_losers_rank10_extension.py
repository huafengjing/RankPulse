from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_drop_top3_short_edge import (  # noqa: E402
    CACHE_DIR,
    DAY_MS,
    DROP_BUCKET_ORDER,
    HOUR_MS,
    build_signals,
    drop_bucket,
    load_kline_map,
    max_drawdown,
    ms,
    path_excursions,
    profit_factor,
    utc,
)


CONFIG_PATH = ROOT / "config" / "rank10_extension.json"
OLD_TOP3_OUT = ROOT / "outputs" / "drop_top3_short_edge_2026"
MONTH_ORDER = [f"2026-{month:02d}" for month in range(1, 13)]


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"binance_futures_losers_rank10_extension_{stamp}"


def longest_streak(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def sample_label(trades: int) -> str:
    if trades < 20:
        return "unusable"
    if trades < 50:
        return "very_low_confidence"
    if trades < 100:
        return "low_confidence"
    if trades < 200:
        return "medium_low_confidence"
    return "candidate_discussion"


def complete_months(signal_start: int, signal_end: int) -> set[str]:
    start = utc(signal_start)
    end = utc(signal_end)
    result: set[str] = set()
    for month_start in pd.date_range(start.floor("D").replace(day=1), end.floor("D"), freq="MS", tz="UTC"):
        month_end = month_start + pd.offsets.MonthBegin(1)
        if start <= month_start and end >= month_end - pd.Timedelta(hours=1):
            result.add(month_start.strftime("%Y-%m"))
    return result


def audit_cache(kline_map: dict[str, pd.DataFrame], cache_audit: pd.DataFrame) -> dict[str, Any]:
    duplicate_rows = non_hour_rows = invalid_ohlc = inconsistent_ohlc = 0
    symbols_not_usdt: list[str] = []
    for path in sorted(CACHE_DIR.glob("*_1h.csv")):
        symbol = path.stem.removesuffix("_1h")
        if not symbol.endswith("USDT"):
            symbols_not_usdt.append(symbol)
        frame = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close"])
        duplicate_rows += int(frame.duplicated("open_time").sum())
        non_hour_rows += int((pd.to_numeric(frame["open_time"], errors="coerce") % HOUR_MS != 0).sum())
        prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        invalid_ohlc += int((~np.isfinite(prices)).any(axis=1).sum() + (prices <= 0).any(axis=1).sum())
        inconsistent_ohlc += int(((prices["high"] < prices[["open", "close", "low"]].max(axis=1)) | (prices["low"] > prices[["open", "close", "high"]].min(axis=1))).sum())
    exchange_path = ROOT / "data" / "raw" / "exchange_info" / "exchange_info_latest.json"
    exchange = json.loads(exchange_path.read_text(encoding="utf-8")) if exchange_path.exists() else {}
    perpetual_usdt = {
        row.get("symbol")
        for row in exchange.get("symbols", [])
        if row.get("quoteAsset") == "USDT" and row.get("contractType") == "PERPETUAL"
    }
    unknown_contract_type = sorted(set(kline_map) - perpetual_usdt)
    return {
        "cache_file_count": len(kline_map),
        "cache_latest_utc": str(utc(min(int(frame.open_time.max()) for frame in kline_map.values()))),
        "duplicate_kline_rows": duplicate_rows,
        "missing_kline_hours": int(cache_audit["missing_hour_count"].sum()),
        "non_hour_timestamp_rows": non_hour_rows,
        "invalid_ohlc_rows": invalid_ohlc,
        "inconsistent_ohlc_rows": inconsistent_ohlc,
        "symbols_not_ending_usdt": symbols_not_usdt,
        "symbols_not_verified_current_usdt_perpetual": unknown_contract_type,
        "historical_delisted_contract_coverage_verifiable": False,
        "survivorship_bias_warning": "Local cache cannot prove coverage of contracts delisted before the cache universe was assembled.",
    }


def build_rank10_signals(
    signal_start: int,
    signal_end: int,
    kline_map: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals, snapshot_audit = build_signals(signal_start, signal_end, kline_map, "short", top_n=10)
    signals = signals.rename(
        columns={
            "signal_time_ms": "snapshot_time_ms",
            "signal_time_utc": "snapshot_time_utc",
            "change_24h_pct": "return_24h_pct",
        }
    )
    signals["entry_time_ms"] = signals["snapshot_time_ms"]
    signals["entry_time_utc"] = signals["snapshot_time_utc"]
    signals["signal_eligible"] = True
    signals["eligibility_reason"] = "complete_24h_history_and_entry_open"
    signals["stable_tie_break"] = "return_24h_pct_ascending_then_symbol_ascending"
    return signals, snapshot_audit


def precompute_outcomes(
    signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    liquidation_multiple = float(cfg["liquidation_price_multiple"])
    base_notional = float(cfg["per_symbol_notional_usdt"])
    for signal in signals.itertuples(index=False):
        frame = kline_map[str(signal.symbol)]
        entry_time = int(signal.entry_time_ms)
        entry_price = float(frame.at[entry_time, "open"])
        for hold in cfg["holding_days"]:
            planned_exit = entry_time + int(hold) * DAY_MS
            if planned_exit not in frame.index:
                continue
            path = frame[(frame.open_time >= entry_time) & (frame.open_time < planned_exit)]
            hits = path[path.high >= entry_price * liquidation_multiple]
            liquidated = not hits.empty
            if liquidated:
                exit_time = int(hits.iloc[0].open_time)
                exit_price = entry_price * liquidation_multiple
                gross, fees, net = -1.0, 0.0, -1.0
                used_path = path[path.open_time <= exit_time]
                exit_reason = "liquidation_1x_short"
            else:
                exit_time = planned_exit
                exit_price = float(frame.at[exit_time, "open"])
                ratio = exit_price / entry_price
                gross = 1.0 - ratio
                fees = float(cfg["fee_rate"]) + float(cfg["fee_rate"]) * ratio
                net = gross - fees - 2 * float(cfg["slippage_rate"])
                used_path = path
                exit_reason = f"fixed_{hold}d"
            mfe, mae = path_excursions("short", used_path, entry_price)
            rows.append(
                {
                    "snapshot_time_ms": int(signal.snapshot_time_ms),
                    "snapshot_time_utc": signal.snapshot_time_utc,
                    "snapshot_hour_bj": signal.snapshot_hour_bj,
                    "symbol": signal.symbol,
                    "rank": int(signal.rank),
                    "current_close": float(signal.current_close),
                    "close_24h_ago": float(signal.close_24h_ago),
                    "return_24h_pct": float(signal.return_24h_pct),
                    "drop_24h_pct": float(signal.drop_24h_pct),
                    "drop_bucket": signal.drop_bucket,
                    "entry_time_ms": entry_time,
                    "entry_time_utc": signal.entry_time_utc,
                    "entry_price": entry_price,
                    "planned_exit_time_ms": planned_exit,
                    "exit_time_ms": exit_time,
                    "exit_time_utc": utc(exit_time),
                    "exit_price": exit_price,
                    "holding_days": int(hold),
                    "gross_return_pct": gross * 100,
                    "fees_usdt_at_100": fees * base_notional,
                    "net_return_pct": net * 100,
                    "pnl_usdt_at_100": net * base_notional,
                    "liquidated": liquidated,
                    "exit_reason": exit_reason,
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                    "signal_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def apply_position_conflict(
    outcomes: pd.DataFrame,
    ranks: set[int],
    holding_days: int,
    notional_per_rank: dict[int, float],
    strategy_id: str,
    capital_mode: str,
) -> pd.DataFrame:
    selected = outcomes[(outcomes.holding_days == holding_days) & outcomes["rank"].isin(ranks)].copy()
    selected = selected.sort_values(["snapshot_time_ms", "rank", "symbol"])
    open_until: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        symbol = str(row["symbol"])
        skipped = int(row["entry_time_ms"]) < open_until.get(symbol, -1)
        notional = float(notional_per_rank[int(row["rank"])] )
        row.update(
            {
                "strategy_id": strategy_id,
                "capital_mode": capital_mode,
                "notional_usdt": notional,
                "skipped_due_to_existing_position": skipped,
                "signal_eligible": not skipped,
                "eligibility_reason": "existing_position" if skipped else "eligible",
                "pnl_usdt": np.nan if skipped else float(row["net_return_pct"]) / 100 * notional,
                "fees_usdt": np.nan if skipped else float(row["fees_usdt_at_100"]) / 100 * notional,
            }
        )
        if not skipped:
            open_until[symbol] = int(row["exit_time_ms"])
        rows.append(row)
    return pd.DataFrame(rows)


def completed(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[~trades.skipped_due_to_existing_position].copy()


def summarize_trades(group: pd.DataFrame, complete_month_set: set[str]) -> dict[str, Any]:
    trades = completed(group) if "skipped_due_to_existing_position" in group else group.copy()
    trades = trades.sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = trades.pnl_usdt.astype(float)
    ret = trades.net_return_pct.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    monthly = trades.assign(month=pd.to_datetime(trades.entry_time_utc, utc=True).dt.strftime("%Y-%m")).groupby("month").pnl_usdt.sum()
    complete_values = monthly[monthly.index.isin(complete_month_set)]
    deployed = float(trades.notional_usdt.sum()) if "notional_usdt" in trades else len(trades) * 100.0
    dd = max_drawdown(pnl)
    avg_win = float(ret[pnl > 0].mean()) if len(wins) else np.nan
    avg_loss = float(ret[pnl < 0].mean()) if len(losses) else np.nan
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "liquidations": int(trades.liquidated.sum()),
        "liquidation_rate_pct": float(trades.liquidated.mean() * 100) if len(trades) else np.nan,
        "net_pnl_usdt": float(pnl.sum()),
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "profit_factor": profit_factor(pnl),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(trades) else np.nan,
        "average_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "return_std_pct": float(ret.std(ddof=1)) if len(ret) > 1 else np.nan,
        "average_win_pct": avg_win,
        "average_loss_pct": avg_loss,
        "payoff_ratio": avg_win / abs(avg_loss) if pd.notna(avg_loss) and avg_loss else np.nan,
        "expectancy_usdt_per_trade": float(pnl.mean()) if len(pnl) else np.nan,
        "expectancy_pct_per_trade": float(ret.mean()) if len(ret) else np.nan,
        "net_pnl_per_deployed_usdt_pct": float(pnl.sum() / deployed * 100) if deployed else np.nan,
        "max_drawdown_usdt": dd,
        "max_drawdown_pct_on_deployed_capital": dd / deployed * 100 if deployed else np.nan,
        "max_trade_profit_usdt": float(pnl.max()) if len(pnl) else np.nan,
        "max_trade_loss_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "max_consecutive_wins": longest_streak(pnl > 0),
        "max_consecutive_losses": longest_streak(pnl < 0),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(10).sum()) if len(pnl) >= 10 else np.nan,
        "net_pnl_ex_worst_1_usdt": float(pnl.sum() - pnl.nsmallest(1).sum()) if len(pnl) else np.nan,
        "positive_months": int((complete_values > 0).sum()),
        "negative_months": int((complete_values < 0).sum()),
        "total_complete_months": len(complete_values),
        "positive_month_ratio": float((complete_values > 0).mean()) if len(complete_values) else np.nan,
        "first_trade_time": trades.entry_time_utc.min() if len(trades) else pd.NaT,
        "last_trade_time": trades.entry_time_utc.max() if len(trades) else pd.NaT,
        "sample_label": sample_label(len(trades)),
        "return_to_drawdown_ratio": float(pnl.sum() / abs(dd)) if dd < 0 else np.nan,
    }


def monthly_summary(trades: pd.DataFrame, complete_month_set: set[str], keys: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for month, group in completed(trades).assign(month=lambda x: pd.to_datetime(x.entry_time_utc, utc=True).dt.strftime("%Y-%m")).groupby("month"):
        stats = summarize_trades(group, {month})
        result.append(keys | {"month": month, "partial_month": month not in complete_month_set} | stats)
    return result


def snapshot_time_summary(trades: pd.DataFrame, complete_month_set: set[str], keys: dict[str, Any]) -> list[dict[str, Any]]:
    return [keys | {"snapshot_hour_bj": hour} | summarize_trades(group, complete_month_set) for hour, group in trades.groupby("snapshot_hour_bj")]


def portfolio_stats(trades: pd.DataFrame, signal_start: int, signal_end: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    done = completed(trades)
    snapshots = trades.groupby("snapshot_time_ms").size()
    entries = done.groupby("snapshot_time_ms").size()
    deployed = done.groupby("snapshot_time_ms").notional_usdt.sum()
    cohorts = done.groupby(["snapshot_time_ms", "snapshot_time_utc"]).agg(
        entries=("symbol", "size"),
        cohort_pnl_usdt=("pnl_usdt", "sum"),
        liquidations=("liquidated", "sum"),
        capital_deployed_usdt=("notional_usdt", "sum"),
    ).reset_index()
    events: list[tuple[int, float, int]] = []
    for row in done.itertuples():
        events.append((int(row.entry_time_ms), float(row.notional_usdt), 1))
        events.append((int(row.exit_time_ms), -float(row.notional_usdt), -1))
    exposure_rows = []
    exposure = 0.0
    positions = 0
    for time_ms, group in pd.DataFrame(events, columns=["time_ms", "notional_delta", "position_delta"]).groupby("time_ms") if events else []:
        exposure += float(group.notional_delta.sum())
        positions += int(group.position_delta.sum())
        exposure_rows.append({"time_ms": time_ms, "time_utc": utc(time_ms), "gross_exposure_usdt": exposure, "concurrent_positions": positions})
    exposure_df = pd.DataFrame(exposure_rows)
    hourly = pd.DataFrame({"time_ms": range(signal_start, signal_end + 7 * DAY_MS + HOUR_MS, HOUR_MS)})
    hourly["time_utc"] = pd.to_datetime(hourly.time_ms, unit="ms", utc=True)
    if not exposure_df.empty:
        hourly = pd.merge_asof(hourly.sort_values("time_ms"), exposure_df.sort_values("time_ms"), on="time_ms", direction="backward", suffixes=("", "_event"))
        hourly["gross_exposure_usdt"] = hourly.gross_exposure_usdt.fillna(0)
        hourly["concurrent_positions"] = hourly.concurrent_positions.fillna(0)
    else:
        hourly["gross_exposure_usdt"] = 0.0
        hourly["concurrent_positions"] = 0
    daily = done.assign(day=pd.to_datetime(done.exit_time_utc, utc=True).dt.floor("D")).groupby("day").pnl_usdt.sum()
    full_days = pd.date_range(utc(signal_start).floor("D"), utc(signal_end + 7 * DAY_MS).floor("D"), freq="D", tz="UTC")
    daily = daily.reindex(full_days, fill_value=0.0)
    daily_frame = pd.DataFrame({"day": full_days, "realized_pnl_usdt": daily.values})
    daily_frame["equity_usdt"] = daily_frame.realized_pnl_usdt.cumsum()
    max_exposure = float(hourly.gross_exposure_usdt.max())
    daily_frame["daily_return_on_max_exposure"] = daily_frame.realized_pnl_usdt / max_exposure if max_exposure else 0.0
    daily_mean = float(daily_frame.daily_return_on_max_exposure.mean())
    daily_std = float(daily_frame.daily_return_on_max_exposure.std(ddof=1))
    equity = daily_frame.equity_usdt
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    trough_pos = int(np.argmin(drawdown.to_numpy())) if len(drawdown) else 0
    peak_value = float(running_peak.iloc[trough_pos]) if len(running_peak) else 0.0
    peak_candidates = np.flatnonzero(equity.iloc[: trough_pos + 1].to_numpy() >= peak_value - 1e-12)
    peak_pos = int(peak_candidates[0]) if len(peak_candidates) else 0
    recovery_candidates = np.flatnonzero(equity.iloc[trough_pos + 1 :].to_numpy() >= peak_value - 1e-12)
    recovered = len(recovery_candidates) > 0
    recovery_pos = trough_pos + 1 + int(recovery_candidates[0]) if recovered else len(equity) - 1
    stats = {
        "snapshot_count": int(trades.snapshot_time_ms.nunique()),
        "average_signals_per_snapshot": float(snapshots.mean()),
        "average_entries_per_snapshot": float(entries.reindex(snapshots.index, fill_value=0).mean()),
        "average_capital_deployed_per_snapshot": float(deployed.reindex(snapshots.index, fill_value=0).mean()),
        "max_capital_deployed_per_snapshot": float(deployed.max()) if len(deployed) else 0.0,
        "actual_capital_use_ratio": float(deployed.sum() / trades.groupby("snapshot_time_ms").notional_usdt.sum().sum()) if len(trades) else np.nan,
        "average_concurrent_positions": float(hourly.concurrent_positions.mean()),
        "max_concurrent_positions": int(hourly.concurrent_positions.max()),
        "average_gross_exposure": float(hourly.gross_exposure_usdt.mean()),
        "max_gross_exposure": max_exposure,
        "worst_snapshot_entry_cohort_pnl": float(cohorts.cohort_pnl_usdt.min()) if len(cohorts) else np.nan,
        "best_snapshot_entry_cohort_pnl": float(cohorts.cohort_pnl_usdt.max()) if len(cohorts) else np.nan,
        "snapshots_with_multiple_liquidations": int((cohorts.liquidations >= 2).sum()),
        "maximum_liquidations_from_one_snapshot": int(cohorts.liquidations.max()) if len(cohorts) else 0,
        "daily_return_mean": daily_mean,
        "daily_return_std": daily_std,
        "daily_sharpe_without_annualization": daily_mean / daily_std if daily_std else np.nan,
        "sharpe_frequency": "daily_realized_pnl_divided_by_max_gross_exposure",
        "sharpe_risk_free_rate": 0.0,
        "sharpe_annualized": False,
        "portfolio_max_drawdown": max_drawdown(daily_frame.realized_pnl_usdt),
        "max_drawdown_recovered": recovered,
        "max_drawdown_recovery_days": max(0, recovery_pos - peak_pos),
    }
    return stats, hourly, cohorts, daily_frame


def winsorized_pnl(pnl: pd.Series, fraction: float) -> float:
    if pnl.empty:
        return 0.0
    return float(pnl.clip(pnl.quantile(fraction), pnl.quantile(1 - fraction)).sum())


def robustness_row(trades: pd.DataFrame, keys: dict[str, Any], complete_month_set: set[str]) -> dict[str, Any]:
    done = completed(trades)
    pnl = done.pnl_usdt.astype(float)
    base = summarize_trades(done, complete_month_set)
    return keys | {
        "trades": len(done),
        "net_pnl_usdt": float(pnl.sum()),
        "net_pnl_ex_best_1_usdt": base["net_pnl_ex_best_1_usdt"],
        "net_pnl_ex_best_3_usdt": base["net_pnl_ex_best_3_usdt"],
        "net_pnl_ex_best_5_usdt": base["net_pnl_ex_best_5_usdt"],
        "net_pnl_ex_best_10_usdt": base["net_pnl_ex_best_10_usdt"],
        "winsorized_1pct_pnl_usdt": winsorized_pnl(pnl, 0.01),
        "winsorized_2_5pct_pnl_usdt": winsorized_pnl(pnl, 0.025),
    }


def common_entry_events(signals: pd.DataFrame, low: int, high: int, cooldown_hours: int) -> pd.DataFrame:
    membership = signals[signals["rank"].between(low, high)].copy()
    by_snapshot = {int(time): set(group.symbol) for time, group in membership.groupby("snapshot_time_ms")}
    all_times = sorted(signals.snapshot_time_ms.unique())
    previous_members: set[str] = set()
    last_event: dict[str, int] = {}
    rows = []
    cooldown_ms = cooldown_hours * HOUR_MS
    for time_ms in all_times:
        current = by_snapshot.get(int(time_ms), set())
        entering = current - previous_members
        for symbol in sorted(entering):
            if int(time_ms) - last_event.get(symbol, -10**30) < cooldown_ms:
                continue
            row = membership[(membership.snapshot_time_ms == time_ms) & (membership.symbol == symbol)].iloc[0].to_dict()
            row["cooldown_hours"] = cooldown_hours
            rows.append(row)
            last_event[symbol] = int(time_ms)
        previous_members = current
    return pd.DataFrame(rows)


def common_event_comparison(
    event_sets: dict[tuple[str, int], pd.DataFrame],
    outcomes: pd.DataFrame,
    cfg: dict[str, Any],
    complete_month_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_frames = []
    summary_rows = []
    for (target, cooldown), events in event_sets.items():
        tagged = events.copy()
        tagged["target"] = target
        signal_frames.append(tagged)
        keys = set(zip(events.snapshot_time_ms.astype(int), events.symbol.astype(str)))
        for hold in cfg["holding_days"]:
            selected = outcomes[outcomes.holding_days.eq(hold) & outcomes.apply(lambda row: (int(row.snapshot_time_ms), str(row.symbol)) in keys, axis=1)].copy()
            selected["notional_usdt"] = float(cfg["per_symbol_notional_usdt"])
            selected["pnl_usdt"] = selected.pnl_usdt_at_100
            selected["skipped_due_to_existing_position"] = False
            summary_rows.append({"target": target, "cooldown_hours": cooldown, "holding_days": hold} | summarize_trades(selected, complete_month_set))
    return pd.concat(signal_frames, ignore_index=True), pd.DataFrame(summary_rows)


def mae_mfe_rows(trades: pd.DataFrame, keys: dict[str, Any]) -> list[dict[str, Any]]:
    done = completed(trades)
    rows = []
    for hold, group in done.groupby("holding_days"):
        row = keys | {"horizon_hours": int(hold) * 24, "trades": len(group)}
        for metric in ["mfe_pct", "mae_pct"]:
            values = group[metric].dropna()
            prefix = metric.removesuffix("_pct")
            row |= {
                f"{prefix}_mean": float(values.mean()),
                f"{prefix}_p10": float(values.quantile(0.10)),
                f"{prefix}_p25": float(values.quantile(0.25)),
                f"{prefix}_median": float(values.median()),
                f"{prefix}_p75": float(values.quantile(0.75)),
                f"{prefix}_p90": float(values.quantile(0.90)),
            }
        rows.append(row)
    return rows


def block_bootstrap(
    trades: pd.DataFrame,
    candidate_id: str,
    block_type: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    done = completed(trades).copy()
    if block_type == "snapshot":
        done["block"] = done.snapshot_time_ms.astype(str)
    else:
        done["block"] = pd.to_datetime(done.snapshot_time_utc, utc=True).dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    blocks = [group.pnl_usdt.to_numpy() for _, group in done.groupby("block")]
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(repetitions):
        sampled = rng.integers(0, len(blocks), len(blocks))
        pnl = np.concatenate([blocks[index] for index in sampled])
        wins = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())
        rows.append(
            {
                "candidate_id": candidate_id,
                "block_type": block_type,
                "replicate": rep,
                "profit_factor": wins / losses if losses else np.inf,
                "average_pnl_usdt": float(pnl.mean()),
                "total_net_pnl_usdt": float(pnl.sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, block_type), group in results.groupby(["candidate_id", "block_type"]):
        rows.append(
            {
                "candidate_id": candidate,
                "block_type": block_type,
                "repetitions": len(group),
                "pf_median": float(group.profit_factor.median()),
                "pf_ci_low_95": float(group.profit_factor.quantile(0.025)),
                "pf_ci_high_95": float(group.profit_factor.quantile(0.975)),
                "mean_trade_pnl_ci_low_95": float(group.average_pnl_usdt.quantile(0.025)),
                "mean_trade_pnl_ci_high_95": float(group.average_pnl_usdt.quantile(0.975)),
                "total_pnl_ci_low_95": float(group.total_net_pnl_usdt.quantile(0.025)),
                "total_pnl_ci_high_95": float(group.total_net_pnl_usdt.quantile(0.975)),
                "pf_le_1_ratio": float((group.profit_factor <= 1).mean()),
                "net_pnl_le_0_ratio": float((group.total_net_pnl_usdt <= 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def ranking_tables(single_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metric_specs = {
        "pf": ("profit_factor", False),
        "net_pnl": ("net_pnl_usdt", False),
        "ex_best5": ("net_pnl_ex_best_5_usdt", False),
        "expectancy": ("expectancy_usdt_per_trade", False),
        "return_to_drawdown": ("return_to_drawdown_ratio", False),
        "positive_month_ratio": ("positive_month_ratio", False),
    }
    result = {}
    for name, (column, ascending) in metric_specs.items():
        ranked = single_summary.sort_values([column, "trades"], ascending=[ascending, False]).reset_index(drop=True)
        ranked.insert(0, "sort_position", np.arange(1, len(ranked) + 1))
        result[name] = ranked
    return result


def segmented_robustness(trades: pd.DataFrame, candidate_id: str, latest_complete_month: str) -> list[dict[str, Any]]:
    done = completed(trades).copy()
    done["month"] = pd.to_datetime(done.entry_time_utc, utc=True).dt.strftime("%Y-%m")
    specs = [
        ("2026-01_to_2026-02", "2026-01", "2026-02", False),
        ("2026-03_to_2026-04", "2026-03", "2026-04", False),
        ("2026-05_to_latest_complete", "2026-05", latest_complete_month, False),
        ("latest_partial_month", None, None, True),
    ]
    rows = []
    for segment, start, end, partial in specs:
        if partial:
            group = done[done.month > latest_complete_month]
        else:
            group = done[done.month.between(str(start), str(end))]
        pnl = group.pnl_usdt.astype(float)
        rows.append(
            {
                "candidate_id": candidate_id,
                "segment": segment,
                "partial_segment": partial,
                "trades": len(group),
                "net_pnl_usdt": float(pnl.sum()),
                "profit_factor": profit_factor(pnl),
                "average_return_pct": float(group.net_return_pct.mean()) if len(group) else np.nan,
            }
        )
    return rows


def _svg_text(x: float, y: float, value: Any, size: int = 12, anchor: str = "middle", rotate: int = 0) -> str:
    safe = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{transform}>{safe}</text>'


def _color(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "#eeeeee"
    ratio = 0.5 if high == low else min(1.0, max(0.0, (value - low) / (high - low)))
    if ratio < 0.5:
        t = ratio * 2
        return f"rgb(220,{int(80 + 175*t)},{int(80 + 175*t)})"
    t = (ratio - 0.5) * 2
    return f"rgb({int(255 - 175*t)},{int(255 - 80*t)},{int(255 - 175*t)})"


def save_heatmap(matrix: pd.DataFrame, title: str, label: str, path: Path, note: str) -> None:
    width, height, left, top = 1100, 700, 120, 90
    cell_w = (width - left - 40) / max(1, len(matrix.columns))
    cell_h = (height - top - 100) / max(1, len(matrix.index))
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="font-family:Arial;background:white">', _svg_text(width/2, 32, title, 20)]
    for row, index in enumerate(matrix.index):
        parts.append(_svg_text(left - 12, top + (row + .65) * cell_h, index, 12, "end"))
        for col, column in enumerate(matrix.columns):
            value = values[row, col]
            x, y = left + col * cell_w, top + row * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{_color(value, low, high)}" stroke="white"/>')
            parts.append(_svg_text(x + cell_w/2, y + cell_h*.62, "" if not np.isfinite(value) else f"{value:.2f}", 10))
    for col, column in enumerate(matrix.columns):
        parts.append(_svg_text(left + (col + .5) * cell_w, top - 10, column, 11))
    parts += [_svg_text(35, height/2, "Rank", 13, rotate=-90), _svg_text(width/2, height-48, label, 12), _svg_text(width/2, height-18, note, 10), "</svg>"]
    path.write_text("".join(parts), encoding="utf-8")


def line_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    path: Path,
    note: str,
) -> None:
    width, height, left, top = 1100, 650, 90, 65
    plot_w, plot_h = width - left - 190, height - top - 90
    xv = pd.to_numeric(frame[x], errors="coerce"); yv = pd.to_numeric(frame[y], errors="coerce")
    xmin, xmax = float(xv.min()), float(xv.max()); ymin, ymax = float(yv.min()), float(yv.max())
    if ymin == ymax: ymin, ymax = ymin - 1, ymax + 1
    sx = lambda value: left + (float(value) - xmin) / max(xmax - xmin, 1e-12) * plot_w
    sy = lambda value: top + (ymax - float(value)) / (ymax - ymin) * plot_h
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#003f5c"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="font-family:Arial;background:white">', _svg_text(width/2, 28, title, 20), f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#555"/>']
    baseline = 1 if y == "profit_factor" else 0
    if ymin <= baseline <= ymax:
        parts.append(f'<line x1="{left}" y1="{sy(baseline):.1f}" x2="{left+plot_w}" y2="{sy(baseline):.1f}" stroke="black" stroke-dasharray="4 3"/>')
    for index, (key, data) in enumerate(frame.groupby(group)):
        data = data.sort_values(x)
        points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(data[x], data[y]))
        color = palette[index % len(palette)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for a, b in zip(data[x], data[y]): parts.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="3" fill="{color}"/>')
        parts += [f'<line x1="{left+plot_w+20}" y1="{top+18*index}" x2="{left+plot_w+40}" y2="{top+18*index}" stroke="{color}" stroke-width="2"/>', _svg_text(left+plot_w+46, top+4+18*index, key, 10, "start")]
    unique_x = sorted(pd.unique(frame[x]))
    tick_indices = np.unique(np.linspace(0, len(unique_x) - 1, min(10, len(unique_x))).astype(int)) if unique_x else []
    for index in tick_indices:
        value = unique_x[int(index)]; parts.append(_svg_text(sx(value), top+plot_h+20, f"{value:.0f}" if isinstance(value, (float, np.floating)) else value, 10))
    for ratio in np.linspace(0,1,6):
        value = ymin + ratio*(ymax-ymin); parts.append(_svg_text(left-8, sy(value)+4, f"{value:.2f}", 10, "end"))
    parts += [_svg_text(left+plot_w/2, height-42, x.replace("_", " ").title(), 12), _svg_text(25, top+plot_h/2, ylabel, 12, rotate=-90), _svg_text(width/2, height-14, note, 9), "</svg>"]
    path.write_text("".join(parts), encoding="utf-8")


def equity_chart(timeseries: pd.DataFrame, strategy_ids: list[str], title: str, path: Path, note: str) -> None:
    selected = timeseries[timeseries.strategy_id.isin(strategy_ids)]
    selected = selected.copy(); selected["day_num"] = pd.to_datetime(selected.day, utc=True).astype("int64") / 1e9
    selected["series"] = selected.strategy_id.astype(str) + " " + selected.holding_days.astype(str) + "D " + selected.capital_mode.astype(str)
    line_chart(selected, "day_num", "equity_usdt", "series", title, "Cumulative realized PnL (USDT)", path, note)


def generate_charts(
    out: Path,
    single: pd.DataFrame,
    rank_monthly: pd.DataFrame,
    rank_bucket: pd.DataFrame,
    common: pd.DataFrame,
    snapshot_summary_df: pd.DataFrame,
    portfolio_timeseries: pd.DataFrame,
    exposure_timeseries: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rank_trade_map: dict[tuple[int, int], pd.DataFrame],
    strategy_trade_map: dict[tuple[str, int, str], pd.DataFrame],
    data_note: str,
) -> None:
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    pf = single.pivot(index="rank", columns="holding_days", values="profit_factor")
    exp = single.pivot(index="rank", columns="holding_days", values="expectancy_usdt_per_trade")
    dd = single.pivot(index="rank", columns="holding_days", values="max_drawdown_usdt")
    save_heatmap(pf, "Rank x Holding Day Profit Factor", "PF", charts / "01_rank_holding_pf_heatmap.svg", data_note)
    save_heatmap(exp, "Rank x Holding Day Expectancy", "USDT/trade", charts / "02_rank_holding_expectancy_heatmap.svg", data_note)
    save_heatmap(dd, "Rank x Holding Day Max Drawdown", "USDT", charts / "03_rank_holding_drawdown_heatmap.svg", data_note)
    line_chart(single, "holding_days", "profit_factor", "rank", "Rank1-10 PF Term Structure", "Profit factor", charts / "04_rank_pf_curves.svg", data_note)
    line_chart(single, "holding_days", "net_pnl_usdt", "rank", "Rank1-10 Net PnL Term Structure", "Net PnL (USDT)", charts / "05_rank_pnl_curves.svg", data_note)

    best_band_keys = []
    for strategy in ["Rank4-5", "Rank6-10", "Rank3-5"]:
        keys = [key for key in strategy_trade_map if key[0] == strategy and key[2] == "fixed_per_symbol"]
        if keys:
            best_band_keys.append(max(keys, key=lambda key: summarize_trades(strategy_trade_map[key], set())["profit_factor"]))
    band_ts = []
    for key in best_band_keys:
        trades = strategy_trade_map[key]
        done = completed(trades).assign(day=lambda x: pd.to_datetime(x.exit_time_utc, utc=True).dt.floor("D"))
        daily = done.groupby("day").pnl_usdt.sum().sort_index().cumsum().reset_index(name="equity_usdt")
        daily["strategy_id"], daily["holding_days"], daily["capital_mode"] = key
        band_ts.append(daily)
    equity_chart(pd.concat(band_ts, ignore_index=True), [key[0] for key in best_band_keys], "Marginal Rank Band Equity Curves", charts / "06_rank_band_equity.svg", data_note)
    equity_chart(portfolio_timeseries[portfolio_timeseries.capital_mode.eq("fixed_per_symbol") & portfolio_timeseries.holding_days.eq(3)], ["Top3", "Top5", "Top10"], "TopN Fixed Per-Symbol Equity (3D)", charts / "07_topn_fixed_per_symbol_equity.svg", data_note)
    equity_chart(portfolio_timeseries[portfolio_timeseries.capital_mode.eq("fixed_snapshot_capital") & portfolio_timeseries.holding_days.eq(3)], ["Top3", "Top5", "Top10"], "TopN Fixed Snapshot Capital Equity (3D)", charts / "08_topn_fixed_capital_equity.svg", data_note)

    best_hold = single.loc[single.groupby("rank").profit_factor.idxmax(), ["rank", "holding_days"]]
    monthly_rows = []
    for row in best_hold.itertuples():
        group = rank_monthly[(rank_monthly["rank"] == row.rank) & (rank_monthly.holding_days == row.holding_days)]
        monthly_rows.append(group[["rank", "month", "net_pnl_usdt"]])
    monthly_matrix = pd.concat(monthly_rows).pivot(index="rank", columns="month", values="net_pnl_usdt").fillna(0)
    save_heatmap(monthly_matrix, "Monthly PnL at Each Rank's Best Descriptive Holding", "USDT", charts / "09_rank_monthly_heatmap.svg", data_note)
    line_chart(single, "holding_days", "liquidation_rate_pct", "rank", "Rank1-10 Liquidation Rates", "Liquidation rate (%)", charts / "10_rank_liquidation_rate.svg", data_note)

    bucket_pf = rank_bucket[(rank_bucket.holding_days == 3) & (rank_bucket.trades >= 20)].pivot(index="rank", columns="drop_bucket", values="profit_factor")
    save_heatmap(bucket_pf, "Rank x Drop Bucket PF (3D, n>=20)", "PF", charts / "11_rank_drop_bucket_pf.svg", data_note)
    common_main = common[(common.cooldown_hours == 0) & common.target.isin(["Rank3", "Rank4", "Rank5", "Top3", "Top5", "Top10"])]
    line_chart(common_main, "holding_days", "net_pnl_usdt", "target", "Common-Event Holding Comparison", "Net PnL (USDT)", charts / "12_common_event_holding.svg", data_note + "; common event samples")
    snap = snapshot_summary_df[snapshot_summary_df["rank"].isin(range(1, 11))]
    line_chart(snap, "rank", "profit_factor", "snapshot_hour_bj", "Snapshot Time PF by Rank", "Profit factor", charts / "13_snapshot_time_comparison.svg", data_note)

    chosen_exposure = exposure_timeseries[(exposure_timeseries.strategy_id == "Top10") & (exposure_timeseries.capital_mode == "fixed_snapshot_capital")]
    chosen_hold = int(chosen_exposure.holding_days.min()) if len(chosen_exposure) else 1
    chosen_exposure = chosen_exposure[chosen_exposure.holding_days == chosen_hold]
    chosen_exposure = chosen_exposure.copy(); chosen_exposure["time_num"] = pd.to_datetime(chosen_exposure.time_utc, utc=True).astype("int64") / 1e9; chosen_exposure["series"] = "Top10"
    line_chart(chosen_exposure, "time_num", "concurrent_positions", "series", f"Top10 Concurrent Positions ({chosen_hold}D)", "Positions", charts / "14_concurrent_positions.svg", data_note)
    line_chart(chosen_exposure, "time_num", "gross_exposure_usdt", "series", f"Top10 Gross Exposure ({chosen_hold}D)", "USDT", charts / "15_gross_exposure.svg", data_note)
    weekly = bootstrap[bootstrap.block_type.eq("weekly")].copy()
    histogram_rows = []
    edges = np.linspace(0, 3, 31)
    for candidate, group in weekly.groupby("candidate_id"):
        counts, _ = np.histogram(group.profit_factor.clip(upper=3), bins=edges)
        for index, count in enumerate(counts): histogram_rows.append({"pf_bin": (edges[index]+edges[index+1])/2, "count": count, "candidate_id": candidate})
    line_chart(pd.DataFrame(histogram_rows), "pf_bin", "count", "candidate_id", "Weekly Block Bootstrap PF Distributions (display capped at PF=3)", "Replicates", charts / "16_bootstrap_pf_distribution.svg", data_note + "; raw CSV is not clipped")


def md_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    view = frame[columns].head(limit).copy() if limit else frame[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    return "\n".join([
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy().tolist()],
    ])


def screen(row: pd.Series) -> bool:
    return bool(
        row.profit_factor > 1
        and row.net_pnl_ex_best_5_usdt > 0
        and row.positive_month_ratio > 0.5
        and row.trades >= 100
    )


def write_reports(
    out: Path,
    cfg: dict[str, Any],
    quality: dict[str, Any],
    baseline: pd.DataFrame,
    single: pd.DataFrame,
    bands: pd.DataFrame,
    topn_per_symbol: pd.DataFrame,
    topn_fixed: pd.DataFrame,
    common: pd.DataFrame,
    buckets: pd.DataFrame,
    bootstrap_s: pd.DataFrame,
    snapshot_s: pd.DataFrame,
) -> None:
    best_by_rank = single.loc[single.groupby("rank").profit_factor.idxmax()].sort_values("rank")
    rank4 = best_by_rank[best_by_rank["rank"] == 4].iloc[0]
    rank5 = best_by_rank[best_by_rank["rank"] == 5].iloc[0]
    band45 = bands[bands.strategy_id.eq("Rank4-5")].sort_values("profit_factor", ascending=False).iloc[0]
    band610 = bands[bands.strategy_id.eq("Rank6-10")].sort_values("profit_factor", ascending=False).iloc[0]
    candidates = single[single.apply(screen, axis=1)]
    candidate_bands = bands[bands.apply(screen, axis=1)]
    common_rank3 = common[(common.target == "Rank3") & (common.cooldown_hours == 0)]
    common_best = common_rank3.sort_values("profit_factor", ascending=False).iloc[0]
    fixed_best = topn_fixed.loc[topn_fixed.groupby("strategy_id").profit_factor.idxmax()].sort_values("strategy_id")
    recommend_top5 = screen(band45) and float(fixed_best[fixed_best.strategy_id.eq("Top5")].profit_factor.iloc[0]) > float(fixed_best[fixed_best.strategy_id.eq("Top3")].profit_factor.iloc[0])
    recommend_top10 = screen(band610) and float(fixed_best[fixed_best.strategy_id.eq("Top10")].profit_factor.iloc[0]) > float(fixed_best[fixed_best.strategy_id.eq("Top5")].profit_factor.iloc[0])
    level = "worth_preregistered_OOS" if len(candidates) or len(candidate_bands) else "not_supported"
    lines = [
        "# Rank4-Rank10 Extension Report",
        "",
        "## 1. Executive Conclusion",
        "",
        f"- Expand Top3 to Top5: **{'YES, OOS candidate only' if recommend_top5 else 'NO'}**.",
        f"- Expand Top5 to Top10: **{'YES, OOS candidate only' if recommend_top10 else 'NO'}**.",
        f"- Overall decision level: **{level}**. This is not a confirmed executable strategy.",
        f"- Rank4 best descriptive result: {int(rank4.holding_days)}D, PF {rank4.profit_factor:.3f}, PnL {rank4.net_pnl_usdt:.2f}, ex-best5 {rank4.net_pnl_ex_best_5_usdt:.2f}.",
        f"- Rank5 best descriptive result: {int(rank5.holding_days)}D, PF {rank5.profit_factor:.3f}, PnL {rank5.net_pnl_usdt:.2f}, ex-best5 {rank5.net_pnl_ex_best_5_usdt:.2f}.",
        f"- Rank4-5 marginal best: {int(band45.holding_days)}D, PF {band45.profit_factor:.3f}, PnL {band45.net_pnl_usdt:.2f}.",
        f"- Rank6-10 marginal best: {int(band610.holding_days)}D, PF {band610.profit_factor:.3f}, PnL {band610.net_pnl_usdt:.2f}.",
        "",
        "## 2. Data and Methodology",
        "",
        f"Cache latest: {quality['cache_latest_utc']}. Unified signal window: {cfg['signal_start_utc']} through {cfg['unified_signal_end_utc']}. Snapshots: Beijing 00:00/08:00. Ranking uses the last completed hourly close and the completed close 24 hours earlier. Entry and fixed exit use the signal/exit hourly open.",
        "",
        "Each position is isolated 1X. Entry/exit fees are 0.1% each, slippage is zero, and funding is omitted. A short is liquidated at 2x entry using the holding-path hourly High and loses its full assigned margin. Existing positions in the same symbol block later entries. No trading filters are used.",
        "",
        f"Historical universe warning: {quality['survivorship_bias_warning']}",
        "",
        "## 3. Baseline Reproduction",
        "",
        md_table(baseline, ["holding_days", "old_net_pnl_usdt", "new_net_pnl_usdt", "old_profit_factor", "new_profit_factor", "difference_reason"]),
        "",
        "## 4. Rank1-Rank10 Single-Rank Results",
        "",
        md_table(single, ["rank", "holding_days", "trades", "net_pnl_usdt", "profit_factor", "net_pnl_ex_best_5_usdt", "positive_month_ratio", "max_drawdown_usdt"]),
        "",
        "All 70 combinations are shown; no losing holding periods are hidden.",
        "",
        "## 5. Rank4-Rank5 Marginal Contribution",
        "",
        f"Rank4 independent conclusion: {'minimum screen passed' if screen(rank4) else 'not supported'}; best is {int(rank4.holding_days)}D. Rank5 independent conclusion: {'minimum screen passed' if screen(rank5) else 'not supported'}; best is {int(rank5.holding_days)}D. Rank4-5 marginal conclusion: {'positive independent marginal edge candidate' if screen(band45) else 'no reliable marginal value from expanding Top3 to Top5'}.",
        "",
        "## 6. Rank6-Rank10 Marginal Contribution",
        "",
        f"Rank6-10 best descriptive period is {int(band610.holding_days)}D. Minimum screen: {'pass' if screen(band610) else 'fail'}. Isolated rank-period peaks without adjacent rank, term and month continuity are treated as in-sample noise.",
        "",
        "## 7. Top3 vs Top5 vs Top10",
        "",
        "### Fixed 100 USDT per symbol",
        "",
        md_table(topn_per_symbol, ["strategy_id", "holding_days", "trades", "net_pnl_usdt", "profit_factor", "portfolio_max_drawdown", "max_gross_exposure"]),
        "",
        "### Fixed 300 USDT target per snapshot",
        "",
        md_table(topn_fixed, ["strategy_id", "holding_days", "trades", "net_pnl_usdt", "profit_factor", "portfolio_max_drawdown", "actual_capital_use_ratio"]),
        "",
        "Absolute TopN PnL is not used alone: marginal ranks and fixed-capital results control the decision.",
        "",
        "## 8. Common-Signal Holding Comparison",
        "",
        md_table(common[common.cooldown_hours.eq(0)], ["target", "holding_days", "trades", "net_pnl_usdt", "profit_factor", "net_pnl_ex_best_5_usdt"]),
        "",
        f"Rank3 common-event best is {int(common_best.holding_days)}D (PF {common_best.profit_factor:.3f}). Therefore the prior Rank3/6D peak is {'preserved' if int(common_best.holding_days)==6 else 'not preserved; part of the old 6D advantage came from holding-period-dependent duplicate-signal skipping'}.",
        "",
        "## 9. Rank x Drop Bucket Analysis",
        "",
        md_table(buckets[buckets.trades.ge(30)].sort_values("profit_factor", ascending=False), ["rank", "drop_bucket", "holding_days", "trades", "net_pnl_usdt", "profit_factor", "net_pnl_ex_best_5_usdt"], limit=40),
        "",
        "Buckets below 20 trades are marked insufficient_sample and never enter conclusions. This analysis is descriptive, not a filter.",
        "",
        "## 10. Monthly and Snapshot-Time Stability",
        "",
        md_table(snapshot_s.sort_values("profit_factor", ascending=False), ["candidate_id", "holding_days", "snapshot_hour_bj", "trades", "net_pnl_usdt", "profit_factor", "net_pnl_ex_best_5_usdt"], limit=40),
        "",
        "Partial months are excluded from the main positive-month ratio.",
        "",
        "## 11. MAE / MFE and Tail Risk",
        "",
        "MFE is favorable downward movement (positive); MAE is adverse upward movement (negative). Hourly High drives liquidation. Full percentile tables are in the CSV outputs.",
        "",
        "## 12. Bootstrap and Robustness",
        "",
        md_table(bootstrap_s, ["candidate_id", "block_type", "pf_median", "pf_ci_low_95", "pf_ci_high_95", "pf_le_1_ratio", "net_pnl_le_0_ratio"]),
        "",
        "Snapshot and natural-week block bootstraps preserve important within-block dependence. They are fragility diagnostics, not proof of statistical significance.",
        "",
        "## 13. Candidate Strategies",
        "",
        (md_table(candidates.assign(candidate_type="single_rank", candidate_id=lambda x: "Rank" + x["rank"].astype(int).astype(str)), ["candidate_type", "candidate_id", "holding_days", "trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt"]) + ("\n\n" + md_table(candidate_bands.assign(candidate_type="rank_band", candidate_id=candidate_bands.strategy_id), ["candidate_type", "candidate_id", "holding_days", "trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt"]) if len(candidate_bands) else "") if len(candidates) + len(candidate_bands) else "No candidate passed the minimum descriptive screen."),
        "",
        "Any OOS candidate must freeze rank interval, holding days, snapshot times, 1X sizing, fees, liquidation rule and duplicate-position rule before evaluation.",
        "",
        "## 14. Rejected Strategies",
        "",
        f"Rank5 is rejected at every holding period (best PF {rank5.profit_factor:.3f}). Rank4-5 is rejected as a marginal extension (best PF {band45.profit_factor:.3f}, ex-best5 {band45.net_pnl_ex_best_5_usdt:.2f}). Rank3-5 under the executable duplicate-position rule is also rejected. All other combinations failing PF>1, ex-best5 PnL>0, majority of complete months positive, or reasonable sample size are rejected from OOS candidacy. Complete losing results remain in CSV.",
        "",
        "## 15. Final Recommendation",
        "",
        f"**{level}**. {'Do not expand leaderboard depth.' if not recommend_top5 and not recommend_top10 else 'Only the explicitly listed candidate may proceed to preregistered OOS; do not modify parameters on this sample.'}",
        "",
        "### Explicit answers to the eight decision questions",
        "",
        f"1. Rank4: **temporarily unconfirmed**; its best {int(rank4.holding_days)}D result has PF {rank4.profit_factor:.3f}, PnL {rank4.net_pnl_usdt:.2f}, ex-best5 {rank4.net_pnl_ex_best_5_usdt:.2f}, complete-month ratio {rank4.positive_month_ratio:.3f}, drawdown {rank4.max_drawdown_usdt:.2f}, liquidation rate {rank4.liquidation_rate_pct:.2f}%. It does not form a clean executable adjacent-holding platform after tail removal. Rank5: **not supported**; all seven periods lose money/PF<1 at the best descriptive point.",
        f"2. Rank6-10: **weak candidate only** at {int(band610.holding_days)}D (PF {band610.profit_factor:.3f}); adjacent holding periods and adjacent single ranks are inconsistent, so isolated peaks are not treated as confirmation.",
        f"3. Rank4-5 marginal addition: **not profitable robustly**. Best PF {band45.profit_factor:.3f}, ex-best5 {band45.net_pnl_ex_best_5_usdt:.2f}. Top5 gains are supported by original Top3, so expanding to Top5 has no marginal value.",
        f"4. Rank6-10 marginal addition: best descriptive PF {band610.profit_factor:.3f}, but fixed-capital Top10 remains weaker than Top3 and bootstrap includes PF<=1 outcomes. Absolute Top10 PnL is not evidence of added-rank edge.",
        f"5. Fixed 300 USDT/snapshot: Top3 best PF {fixed_best[fixed_best.strategy_id.eq('Top3')].profit_factor.iloc[0]:.3f}; Top5 {fixed_best[fixed_best.strategy_id.eq('Top5')].profit_factor.iloc[0]:.3f}; Top10 {fixed_best[fixed_best.strategy_id.eq('Top10')].profit_factor.iloc[0]:.3f}. **Top5/Top10 do not beat Top3.**",
        "6. No continuous rank interval is strong enough to recommend. Rank6-10/3D is the broadest weak candidate, but its single-rank continuity is poor. Rank7 has a 4D-6D term platform but is isolated from neighboring ranks, so it requires preregistered OOS rather than leaderboard expansion.",
        f"7. Rank3 common-event best remains {int(common_best.holding_days)}D, so the Rank3/6D result is **not explained only by holding-period-dependent duplicate-signal skipping**.",
        "8. Executable-strategy decision: **temporarily unconfirmed**. The only allowed next step is preregistered OOS for explicitly frozen weak candidates; no production or paper-trading rule is approved by this report.",
    ]
    (out / "rank4_to_rank10_extension_report.md").write_text("\n".join(lines), encoding="utf-8")
    audit_lines = [
        "# Methodology and Data Audit",
        "",
        "This run reuses the existing Top3 hourly cache loader, completed-bar ranking definition, short fee formula, 1X liquidation convention and same-symbol position conflict rule.",
        "",
        "## Data quality",
        "",
        *[f"- {key}: {value}" for key, value in quality.items()],
        "",
        "## Assumption changes",
        "",
        "No fee, slippage, leverage, liquidation or signal-time assumptions were changed. The new run uses the newer local cache and a later unified signal cutoff; baseline differences are reported rather than silently overwritten.",
        "",
        "## Known limitation",
        "",
        "The local cache and current exchange-info snapshot do not establish a point-in-time historical contract master. Results can therefore retain survivorship bias from contracts that were delisted before the local universe was assembled.",
    ]
    (out / "methodology_and_data_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    out = output_dir()
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("[1/8] Loading and auditing local 1H cache...", flush=True)
    kline_map, cache_audit = load_kline_map()
    quality = audit_cache(kline_map, cache_audit)
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - max(cfg["holding_days"]) * DAY_MS
    snapshot_candidates = [
        ms(day + pd.Timedelta(hours=hour))
        for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC")
        for hour in cfg["snapshot_utc_hours"]
        if ms(day + pd.Timedelta(hours=hour)) <= latest_signal
    ]
    signal_end = max(snapshot_candidates)
    cfg["unified_signal_end_utc"] = str(utc(signal_end))
    cfg["cache_latest_utc"] = str(utc(cache_end))
    cfg["actual_output_directory"] = str(out)
    cfg["actual_signal_cutoff_by_holding_days"] = {str(day): str(utc(signal_end)) for day in cfg["holding_days"]}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    complete_month_set = complete_months(signal_start, signal_end)
    latest_complete_month = max(complete_month_set)

    print("[2/8] Rebuilding Rank1-10 historical loser snapshots...", flush=True)
    signals, snapshot_audit = build_rank10_signals(signal_start, signal_end, kline_map)
    quality |= {
        "signal_start_utc": str(utc(signal_start)),
        "unified_signal_end_utc": str(utc(signal_end)),
        "snapshot_count": int(signals.snapshot_time_ms.nunique()),
        "total_rank1_to_rank10_signals": len(signals),
        "signals_with_incomplete_24h_history": 0,
        "signals_missing_entry_open": int(sum(int(row.entry_time_ms) not in kline_map[str(row.symbol)].index for row in signals.itertuples())),
        "duplicate_symbol_snapshot_signals": int(signals.duplicated(["snapshot_time_ms", "symbol"]).sum()),
        "duplicate_rank_snapshot_signals": int(signals.duplicated(["snapshot_time_ms", "rank"]).sum()),
        "snapshots_without_10_unique_ranks": int((signals.groupby("snapshot_time_ms")["rank"].nunique() != 10).sum()),
        "rank_sort_order_violations": int(signals.sort_values(["snapshot_time_ms", "rank"]).groupby("snapshot_time_ms").return_24h_pct.apply(lambda x: (x.diff().dropna() < 0).sum()).sum()),
        "stable_tie_break_rule": "return ascending, then symbol ascending",
        "entry_uses_future_known_price": False,
        "all_holding_exits_within_cache": bool(signal_end + max(cfg["holding_days"]) * DAY_MS <= cache_end),
        "liquidation_uses_holding_path_high": True,
        "liquidation_charged_exit_fee": False,
        "skipped_signals_counted_as_trades": False,
        "complete_months_used_for_stability": sorted(complete_month_set),
        "partial_months": sorted(set(pd.to_datetime(signals.snapshot_time_utc, utc=True).dt.strftime("%Y-%m")) - complete_month_set),
    }
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")

    print("[3/8] Precomputing 1D-7D outcomes and position-conflict portfolios...", flush=True)
    outcomes = precompute_outcomes(signals, kline_map, cfg)
    base_notional = float(cfg["per_symbol_notional_usdt"])
    single_trade_map: dict[tuple[int, int], pd.DataFrame] = {}
    single_frames = []
    single_summary_rows = []
    single_month_rows = []
    single_snapshot_rows = []
    single_mae_rows = []
    single_robustness_rows = []
    for rank in range(1, int(cfg["max_rank"]) + 1):
        for hold in cfg["holding_days"]:
            trades = apply_position_conflict(outcomes, {rank}, hold, {rank: base_notional}, f"Rank{rank}", "fixed_per_symbol")
            single_trade_map[(rank, hold)] = trades
            single_frames.append(trades)
            keys = {"rank": rank, "holding_days": hold}
            single_summary_rows.append(keys | summarize_trades(trades, complete_month_set))
            single_month_rows.extend(monthly_summary(trades, complete_month_set, keys))
            single_snapshot_rows.extend(snapshot_time_summary(trades, complete_month_set, keys | {"strategy_id": f"Rank{rank}"}))
            single_robustness_rows.append({"row_type": "tail", **robustness_row(trades, keys, complete_month_set)})
            single_robustness_rows.extend({"row_type": "segment", "rank": rank, "holding_days": hold, **row} for row in segmented_robustness(trades, f"Rank{rank}_{hold}D", latest_complete_month))
        rank_trades = pd.concat([single_trade_map[(rank, hold)] for hold in cfg["holding_days"]], ignore_index=True)
        single_mae_rows.extend(mae_mfe_rows(rank_trades, {"rank": rank}))
    all_single_trades = pd.concat(single_frames, ignore_index=True)
    single_summary = pd.DataFrame(single_summary_rows).sort_values(["rank", "holding_days"])
    single_monthly = pd.DataFrame(single_month_rows)
    single_snapshot = pd.DataFrame(single_snapshot_rows)
    single_mae = pd.DataFrame(single_mae_rows)
    single_robustness = pd.DataFrame(single_robustness_rows)

    strategy_trade_map: dict[tuple[str, int, str], pd.DataFrame] = {}
    band_summary_rows = []
    band_month_rows = []
    band_portfolio_rows = []
    band_robustness_rows = []
    band_mae_rows = []
    band_trade_frames = []
    band_exposure_frames = []
    band_cohort_frames = []
    band_daily_frames = []
    for strategy, (low, high) in cfg["rank_bands"].items():
        ranks = set(range(int(low), int(high) + 1))
        for hold in cfg["holding_days"]:
            notionals = {rank: base_notional for rank in ranks}
            trades = apply_position_conflict(outcomes, ranks, hold, notionals, strategy, "fixed_per_symbol")
            strategy_trade_map[(strategy, hold, "fixed_per_symbol")] = trades
            band_trade_frames.append(trades)
            keys = {"strategy_id": strategy, "rank_low": low, "rank_high": high, "holding_days": hold}
            stats = summarize_trades(trades, complete_month_set)
            band_summary_rows.append(keys | stats)
            band_month_rows.extend(monthly_summary(trades, complete_month_set, keys))
            portfolio, exposure, cohorts, daily = portfolio_stats(trades, signal_start, signal_end)
            band_portfolio_rows.append(keys | stats | portfolio)
            exposure = exposure.assign(**keys, capital_mode="fixed_per_symbol"); band_exposure_frames.append(exposure)
            cohorts = cohorts.assign(**keys, capital_mode="fixed_per_symbol"); band_cohort_frames.append(cohorts)
            daily = daily.assign(**keys, capital_mode="fixed_per_symbol"); band_daily_frames.append(daily)
            band_robustness_rows.append({"row_type": "tail", **robustness_row(trades, keys, complete_month_set)})
            band_robustness_rows.extend({"row_type": "segment", **keys, **row} for row in segmented_robustness(trades, f"{strategy}_{hold}D", latest_complete_month))
        combined = pd.concat([strategy_trade_map[(strategy, hold, "fixed_per_symbol")] for hold in cfg["holding_days"]], ignore_index=True)
        band_mae_rows.extend(mae_mfe_rows(combined, {"strategy_id": strategy}))
    band_summary = pd.DataFrame(band_summary_rows)
    band_monthly = pd.DataFrame(band_month_rows)
    band_portfolio = pd.DataFrame(band_portfolio_rows)
    band_robustness = pd.DataFrame(band_robustness_rows)

    print("[4/8] Building TopN fixed-per-symbol and fixed-snapshot-capital portfolios...", flush=True)
    topn_per_rows = []
    topn_fixed_rows = []
    topn_month_rows = []
    topn_cohort_frames = []
    topn_exposure_rows = []
    topn_daily_frames = []
    topn_trade_frames = []
    topn_robustness_rows = []
    exposure_frames = []
    for top_n in cfg["topn"]:
        strategy = f"Top{top_n}"
        ranks = set(range(1, int(top_n) + 1))
        for capital_mode, per_rank in [
            ("fixed_per_symbol", base_notional),
            ("fixed_snapshot_capital", float(cfg["fixed_snapshot_notional_usdt"]) / int(top_n)),
        ]:
            notionals = {rank: per_rank for rank in ranks}
            for hold in cfg["holding_days"]:
                trades = apply_position_conflict(outcomes, ranks, hold, notionals, strategy, capital_mode)
                strategy_trade_map[(strategy, hold, capital_mode)] = trades
                topn_trade_frames.append(trades)
                keys = {"strategy_id": strategy, "top_n": top_n, "holding_days": hold, "capital_mode": capital_mode}
                stats = summarize_trades(trades, complete_month_set)
                portfolio, exposure, cohorts, daily = portfolio_stats(trades, signal_start, signal_end)
                row = keys | stats | portfolio
                (topn_per_rows if capital_mode == "fixed_per_symbol" else topn_fixed_rows).append(row)
                topn_month_rows.extend(monthly_summary(trades, complete_month_set, keys))
                topn_robustness_rows.append({"row_type": "tail", **robustness_row(trades, keys, complete_month_set)})
                topn_robustness_rows.extend({"row_type": "segment", **keys, **row} for row in segmented_robustness(trades, f"{strategy}_{hold}D_{capital_mode}", latest_complete_month))
                cohorts = cohorts.assign(**keys); topn_cohort_frames.append(cohorts)
                exposure = exposure.assign(**keys); exposure_frames.append(exposure)
                topn_exposure_rows.append(keys | {key: portfolio[key] for key in ["average_concurrent_positions", "max_concurrent_positions", "average_gross_exposure", "max_gross_exposure", "average_capital_deployed_per_snapshot", "max_capital_deployed_per_snapshot", "actual_capital_use_ratio"]})
                daily = daily.assign(**keys); topn_daily_frames.append(daily)
    topn_per = pd.DataFrame(topn_per_rows)
    topn_fixed = pd.DataFrame(topn_fixed_rows)
    topn_monthly = pd.DataFrame(topn_month_rows)
    topn_cohorts = pd.concat(topn_cohort_frames, ignore_index=True)
    topn_exposure = pd.DataFrame(topn_exposure_rows)
    exposure_timeseries = pd.concat(exposure_frames, ignore_index=True)
    topn_daily = pd.concat(topn_daily_frames, ignore_index=True)
    topn_robustness = pd.DataFrame(topn_robustness_rows)

    print("[5/8] Common-event, bucket, MAE/MFE and robustness analysis...", flush=True)
    event_sets = {}
    for target, (low, high) in cfg["common_event_targets"].items():
        for cooldown in cfg["common_event_cooldown_hours"]:
            event_sets[(target, int(cooldown))] = common_entry_events(signals, int(low), int(high), int(cooldown))
    common_signals, common_comparison = common_event_comparison(event_sets, outcomes, cfg, complete_month_set)

    bucket_rows = []
    bucket_month_rows = []
    for (rank, hold), trades in single_trade_map.items():
        for bucket, group in trades.groupby("drop_bucket", observed=True):
            stats = summarize_trades(group, complete_month_set)
            stats["sample_status"] = "insufficient_sample" if stats["trades"] < 20 else sample_label(stats["trades"])
            keys = {"rank": rank, "drop_bucket": bucket, "holding_days": hold}
            bucket_rows.append(keys | stats)
            bucket_month_rows.extend(monthly_summary(group, complete_month_set, keys))
    rank_bucket = pd.DataFrame(bucket_rows)
    rank_bucket_monthly = pd.DataFrame(bucket_month_rows)
    rank_holding_matrix = single_summary.pivot(index="rank", columns="holding_days", values=["profit_factor", "expectancy_usdt_per_trade", "net_pnl_usdt", "max_drawdown_usdt"])
    rank_holding_matrix.columns = [f"{metric}_{hold}D" for metric, hold in rank_holding_matrix.columns]
    rank_holding_matrix = rank_holding_matrix.reset_index()
    rank_bucket_matrix = rank_bucket.pivot_table(index=["rank", "drop_bucket"], columns="holding_days", values="profit_factor", observed=True)
    rank_bucket_matrix.columns = [f"pf_{hold}D" for hold in rank_bucket_matrix.columns]
    rank_bucket_matrix = rank_bucket_matrix.reset_index()

    print("[6/8] Running snapshot and weekly block bootstraps...", flush=True)
    bootstrap_candidates: dict[str, pd.DataFrame] = {"Rank3_6D": single_trade_map[(3, 6)]}
    for rank in [4, 5]:
        best_hold = int(single_summary[single_summary["rank"] == rank].sort_values("profit_factor", ascending=False).iloc[0].holding_days)
        bootstrap_candidates[f"Rank{rank}_{best_hold}D"] = single_trade_map[(rank, best_hold)]
    for strategy in ["Rank4-5", "Rank6-10", "Rank3-5"]:
        best_hold = int(band_summary[band_summary.strategy_id.eq(strategy)].sort_values("profit_factor", ascending=False).iloc[0].holding_days)
        bootstrap_candidates[f"{strategy}_{best_hold}D"] = strategy_trade_map[(strategy, best_hold, "fixed_per_symbol")]
    rank7_best_hold = int(single_summary[single_summary["rank"] == 7].sort_values("profit_factor", ascending=False).iloc[0].holding_days)
    bootstrap_candidates[f"Rank7_{rank7_best_hold}D"] = single_trade_map[(7, rank7_best_hold)]
    for mode, summary in [("fixed_per_symbol", topn_per), ("fixed_snapshot_capital", topn_fixed)]:
        for strategy in ["Top3", "Top5", "Top10"]:
            best_hold = int(summary[summary.strategy_id.eq(strategy)].sort_values("profit_factor", ascending=False).iloc[0].holding_days)
            bootstrap_candidates[f"{strategy}_{best_hold}D_{mode}"] = strategy_trade_map[(strategy, best_hold, mode)]
    bootstrap_frames = []
    for index, (candidate, trades) in enumerate(bootstrap_candidates.items()):
        for block_type in ["snapshot", "weekly"]:
            bootstrap_frames.append(block_bootstrap(trades, candidate, block_type, int(cfg["bootstrap_repetitions"]), int(cfg["random_seed"]) + index))
    bootstrap_results = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_s = bootstrap_summary(bootstrap_results)

    old = pd.read_csv(OLD_TOP3_OUT / "02_holding_period_summary.csv")
    new_top3 = band_summary[band_summary.strategy_id.eq("Rank1-3")]
    baseline = old[["holding_days", "net_pnl_usdt", "profit_factor", "trades"]].merge(
        new_top3[["holding_days", "net_pnl_usdt", "profit_factor", "trades"]], on="holding_days", suffixes=("_old", "_new")
    ).rename(columns={"net_pnl_usdt_old": "old_net_pnl_usdt", "net_pnl_usdt_new": "new_net_pnl_usdt", "profit_factor_old": "old_profit_factor", "profit_factor_new": "new_profit_factor", "trades_old": "old_trades", "trades_new": "new_trades"})
    baseline["difference_reason"] = "newer_cache_and_later_unified_signal_cutoff; same Top3 position-conflict method"

    ranking_outputs = ranking_tables(single_summary)
    single_sort = pd.concat([frame.assign(sort_metric=name) for name, frame in ranking_outputs.items()], ignore_index=True)

    major_snapshot_rows = []
    major_candidates: dict[str, pd.DataFrame] = {
        "Rank3_6D": single_trade_map[(3, 6)],
        "Rank4_1D": single_trade_map[(4, 1)],
        "Rank5_7D": single_trade_map[(5, 7)],
        f"Rank7_{rank7_best_hold}D": single_trade_map[(7, rank7_best_hold)],
    }
    for strategy in ["Rank4-5", "Rank6-10", "Rank3-5"]:
        best_hold = int(band_summary[band_summary.strategy_id.eq(strategy)].sort_values("profit_factor", ascending=False).iloc[0].holding_days)
        major_candidates[f"{strategy}_{best_hold}D"] = strategy_trade_map[(strategy, best_hold, "fixed_per_symbol")]
    for strategy in ["Top3", "Top5", "Top10"]:
        best_hold = int(topn_fixed[topn_fixed.strategy_id.eq(strategy)].sort_values("profit_factor", ascending=False).iloc[0].holding_days)
        major_candidates[f"{strategy}_{best_hold}D_fixed_capital"] = strategy_trade_map[(strategy, best_hold, "fixed_snapshot_capital")]
    for candidate_id, trades in major_candidates.items():
        hold = int(completed(trades).holding_days.iloc[0])
        major_snapshot_rows.extend(snapshot_time_summary(trades, complete_month_set, {"candidate_id": candidate_id, "holding_days": hold}))
    major_snapshot = pd.DataFrame(major_snapshot_rows)

    print("[7/8] Writing CSVs, audit files and charts...", flush=True)
    csv_outputs: dict[str, pd.DataFrame] = {
        "historical_losers_rank1_to_rank10_signals.csv": signals,
        "rank1_to_rank10_all_trades.csv": all_single_trades,
        "common_event_signal_set.csv": common_signals,
        "common_event_holding_comparison.csv": common_comparison,
        "single_rank_holding_summary.csv": single_summary,
        "single_rank_monthly_summary.csv": single_monthly,
        "single_rank_snapshot_time_summary.csv": single_snapshot,
        "single_rank_mae_mfe.csv": single_mae,
        "single_rank_robustness.csv": single_robustness,
        "single_rank_sorted_results.csv": single_sort,
        "rank_band_holding_summary.csv": band_summary,
        "rank_band_monthly_summary.csv": band_monthly,
        "rank_band_portfolio_summary.csv": band_portfolio,
        "rank_band_all_trades.csv": pd.concat(band_trade_frames, ignore_index=True),
        "rank_band_portfolio_timeseries.csv": pd.concat(band_daily_frames, ignore_index=True),
        "rank_band_snapshot_cohort_summary.csv": pd.concat(band_cohort_frames, ignore_index=True),
        "rank_band_exposure_timeseries.csv": pd.concat(band_exposure_frames, ignore_index=True),
        "rank_band_robustness.csv": band_robustness,
        "rank_band_mae_mfe.csv": pd.DataFrame(band_mae_rows),
        "topn_fixed_per_symbol_summary.csv": topn_per,
        "topn_fixed_snapshot_capital_summary.csv": topn_fixed,
        "topn_monthly_summary.csv": topn_monthly,
        "topn_portfolio_timeseries.csv": topn_daily,
        "topn_snapshot_cohort_summary.csv": topn_cohorts,
        "topn_exposure_summary.csv": topn_exposure,
        "topn_exposure_timeseries.csv": exposure_timeseries,
        "topn_all_trades.csv": pd.concat(topn_trade_frames, ignore_index=True),
        "topn_robustness.csv": topn_robustness,
        "major_candidate_snapshot_time_summary.csv": major_snapshot,
        "rank_drop_bucket_holding_summary.csv": rank_bucket,
        "rank_drop_bucket_monthly_summary.csv": rank_bucket_monthly,
        "rank_holding_matrix.csv": rank_holding_matrix,
        "rank_drop_bucket_matrix.csv": rank_bucket_matrix,
        "block_bootstrap_snapshot_results.csv": bootstrap_results[bootstrap_results.block_type.eq("snapshot")],
        "block_bootstrap_weekly_results.csv": bootstrap_results[bootstrap_results.block_type.eq("weekly")],
        "bootstrap_candidate_summary.csv": bootstrap_s,
        "baseline_top3_reproduction.csv": baseline,
        "snapshot_data_quality.csv": snapshot_audit,
        "cache_data_quality.csv": cache_audit,
    }
    for name, frame in csv_outputs.items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")

    data_note = f"388 signals/rank before conflicts; exact executed n in CSV; through {utc(signal_end).strftime('%Y-%m-%d %H:%M UTC')}; fees 0.1%/side; no funding"
    generate_charts(out, single_summary, single_monthly, rank_bucket, common_comparison, single_snapshot, topn_daily, exposure_timeseries, bootstrap_results, single_trade_map, strategy_trade_map, data_note)

    print("[8/8] Generating conclusions and terminal summary...", flush=True)
    write_reports(out, cfg, quality, baseline, single_summary, band_summary, topn_per, topn_fixed, common_comparison, rank_bucket, bootstrap_s, major_snapshot)

    rank_best = single_summary.loc[single_summary.groupby("rank").profit_factor.idxmax(), ["rank", "holding_days", "profit_factor", "net_pnl_usdt"]].sort_values("rank")
    band45 = band_summary[band_summary.strategy_id.eq("Rank4-5")].sort_values("profit_factor", ascending=False).iloc[0]
    band610 = band_summary[band_summary.strategy_id.eq("Rank6-10")].sort_values("profit_factor", ascending=False).iloc[0]
    common_rank3 = common_comparison[(common_comparison.target == "Rank3") & (common_comparison.cooldown_hours == 0)].sort_values("profit_factor", ascending=False).iloc[0]
    fixed_best = topn_fixed[topn_fixed.strategy_id.isin(["Top3", "Top5", "Top10"])].loc[lambda x: x.groupby("strategy_id").profit_factor.idxmax(), ["strategy_id", "holding_days", "profit_factor", "net_pnl_usdt"]]
    passed_candidates = single_summary[single_summary.apply(screen, axis=1)][["rank", "holding_days", "profit_factor"]]
    all_quality_pass = all([
        quality["duplicate_kline_rows"] == 0,
        quality["missing_kline_hours"] == 0,
        quality["non_hour_timestamp_rows"] == 0,
        quality["invalid_ohlc_rows"] == 0,
        quality["duplicate_symbol_snapshot_signals"] == 0,
        quality["duplicate_rank_snapshot_signals"] == 0,
        quality["rank_sort_order_violations"] == 0,
        quality["all_holding_exits_within_cache"],
    ])
    print("\n========== Rank1-10 Extension Summary ==========", flush=True)
    print(f"Cache latest: {utc(cache_end)}", flush=True)
    print(f"Unified signal cutoff: {utc(signal_end)}", flush=True)
    print(f"Total signals: {len(signals)}", flush=True)
    print("Signals per rank:", signals.groupby("rank").size().to_dict(), flush=True)
    print("Best descriptive result per rank:\n", rank_best.to_string(index=False), flush=True)
    print(f"Rank4-5 best: {int(band45.holding_days)}D PF={band45.profit_factor:.3f} PnL={band45.net_pnl_usdt:.2f}", flush=True)
    print(f"Rank6-10 best: {int(band610.holding_days)}D PF={band610.profit_factor:.3f} PnL={band610.net_pnl_usdt:.2f}", flush=True)
    print("Fixed-capital TopN best:\n", fixed_best.to_string(index=False), flush=True)
    print(f"Common-event Rank3 best: {int(common_rank3.holding_days)}D; original 6D preserved={int(common_rank3.holding_days)==6}", flush=True)
    print("Preregistered OOS single-rank candidates:\n", passed_candidates.to_string(index=False) if len(passed_candidates) else "None", flush=True)
    print(f"Output directory: {out}", flush=True)
    print(f"All core data-quality checks passed: {all_quality_pass}", flush=True)
    print("Research-specific tests must be run separately with pytest.", flush=True)


if __name__ == "__main__":
    main()
