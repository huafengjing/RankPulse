from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, HOUR_MS, OUT, load_kline_map, ms_to_utc, skipped_open_position_trade
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import (
    BResponsePlan,
    EXCLUDE_SYMBOLS,
    IndicatorSpec,
    build_health_timeline,
    opportunity_sets,
    pareto,
    summarize_candidate,
)
from scripts.regime_adaptive_leverage_walkforward import bucket_for_signal, simulate_trade_with_leverage
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, apply_entry_rules, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "rank3_fast_recovery_vs_monthly_reset"
RECOVERY_WINDOWS = [3, 5, 7]
LEVERS = [1, 2, 3, 5]
BASELINE_NAME = "baseline_green__baseline_no_adaptive"
D15_NAME = "D15_Red1x__base"


@dataclass(frozen=True)
class RecoverySpec:
    name: str
    metric: str
    window: int
    threshold: str


def quantile(values: list[float], q: float) -> float:
    good = [x for x in values if np.isfinite(x)]
    return float(np.quantile(good, q)) if good else np.nan


def metric_value(hist: pd.DataFrame, metric: str, window: int) -> float:
    recent = hist.tail(window)
    if recent.empty:
        return np.nan
    if metric == "avg_return24":
        return float(pd.to_numeric(recent["return_24h"], errors="coerce").mean())
    if metric == "positive_rate24":
        return float((pd.to_numeric(recent["return_24h"], errors="coerce") > 0).mean())
    if metric == "new_high_rate24":
        return float(recent["new_high_24h"].astype(bool).mean())
    if metric == "cq24":
        mfe = float(pd.to_numeric(recent["mfe_24h"], errors="coerce").median())
        mae = abs(float(pd.to_numeric(recent["mae_24h"], errors="coerce").median()))
        return mfe / mae if mae > 0 else np.nan
    if metric == "liq_hit24":
        return float(recent["liq_hit_5x_24h"].astype(bool).mean())
    raise ValueError(metric)


def pass_recovery(value: float, threshold: str, seen: list[float]) -> tuple[bool, float]:
    if not np.isfinite(value):
        return False, np.nan
    if threshold == "gt_0":
        return value > 0, 0.0
    if threshold == "gt_1":
        return value > 1, 1.0
    if threshold == "gt_median":
        t = quantile(seen, 0.50)
        return np.isfinite(t) and value > t, t
    if threshold == "gt_q60":
        t = quantile(seen, 0.60)
        return np.isfinite(t) and value > t, t
    if threshold.startswith("gte_"):
        t = float(threshold.split("_", 1)[1])
        return value >= t, t
    if threshold == "lt_median":
        t = quantile(seen, 0.50)
        return np.isfinite(t) and value < t, t
    if threshold == "lt_q60":
        t = quantile(seen, 0.60)
        return np.isfinite(t) and value < t, t
    if threshold == "eq_0":
        return value <= 0, 0.0
    raise ValueError(threshold)


def recovery_specs() -> list[RecoverySpec]:
    specs: list[RecoverySpec] = []
    metric_thresholds = {
        "avg_return24": ["gt_0", "gt_median", "gt_q60"],
        "positive_rate24": ["gte_0.5", "gte_0.6", "gte_0.67"],
        "new_high_rate24": ["gte_0.5", "gte_0.6", "gte_0.67"],
        "cq24": ["gt_1", "gt_median", "gt_q60"],
        "liq_hit24": ["eq_0", "lt_median", "lt_q60"],
    }
    for metric, thresholds in metric_thresholds.items():
        for window in RECOVERY_WINDOWS:
            for threshold in thresholds:
                specs.append(RecoverySpec(f"{metric}_l{window}_{threshold}", metric, window, threshold))
    return specs


def build_recovery_timeline(signal_times: list[int], opportunities: pd.DataFrame, spec: RecoverySpec) -> pd.DataFrame:
    opp = opportunities.copy()
    opp["mature24"] = opp["signal_time"] + 24 * HOUR_MS
    opp = opp.sort_values(["signal_time", "symbol"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    seen: list[float] = []
    for t in sorted(signal_times):
        hist = opp[(opp["mature24"] <= t) & (opp["path_status"].eq("ok"))].copy()
        value = metric_value(hist, spec.metric, spec.window) if len(hist) >= spec.window else np.nan
        ok, threshold_value = pass_recovery(value, spec.threshold, seen)
        rows.append(
            {
                "model": spec.name,
                "signal_time": t,
                "signal_time_utc": ms_to_utc(t).strftime("%Y-%m-%d %H:%M:%S"),
                "month": ms_to_utc(t).strftime("%Y-%m"),
                "metric": spec.metric,
                "window": spec.window,
                "threshold": spec.threshold,
                "value": value,
                "threshold_value": threshold_value,
                "hist_count": len(hist),
                "recovery_signal": bool(ok),
            }
        )
        if np.isfinite(value):
            seen.append(value)
    return pd.DataFrame(rows)


def d15_action(state: str) -> tuple[int, int]:
    if state == "RED":
        return 2, 1
    if state == "YELLOW":
        return 3, 3
    return 3, 5


def make_base_action_timeline(d15: pd.DataFrame, name: str = "D15_Red1x") -> pd.DataFrame:
    rows = []
    for row in d15.sort_values("signal_time").itertuples(index=False):
        r2, r3 = d15_action(row.regime_state)
        rows.append(
            {
                "signal_time": int(row.signal_time),
                "signal_time_utc": row.signal_time_utc,
                "month": row.month,
                "strategy_state": row.regime_state,
                "base_state": row.regime_state,
                "recovery_signal": False,
                "r2_lev": r2,
                "r3_lev": r3,
                "controller": name,
            }
        )
    return pd.DataFrame(rows)


def fast_recovery_action_timeline(d15: pd.DataFrame, recovery: pd.DataFrame, fr_mode: str, yr_mode: str, name: str) -> pd.DataFrame:
    merged = d15[["signal_time", "signal_time_utc", "month", "regime_state"]].merge(
        recovery[["signal_time", "recovery_signal", "value", "threshold_value"]],
        on="signal_time",
        how="left",
    )
    rows = []
    recovery_streak = 0
    prev_month = None
    for row in merged.sort_values("signal_time").itertuples(index=False):
        if row.month != prev_month:
            recovery_streak = 0
            prev_month = row.month
        base_state = row.regime_state
        recovery_signal = bool(row.recovery_signal)
        if base_state == "GREEN":
            recovery_streak = 0
            state = "GREEN"
            r2, r3 = 3, 5
        else:
            recovery_streak = recovery_streak + 1 if recovery_signal else 0
            if base_state == "YELLOW":
                if yr_mode == "yr1" and recovery_signal:
                    state, r2, r3 = "FR_GREEN", 3, 5
                elif yr_mode == "yr2" and recovery_streak >= 2:
                    state, r2, r3 = "FR_GREEN", 3, 5
                else:
                    state, r2, r3 = "YELLOW", 3, 3
            elif fr_mode == "fr1":
                if recovery_signal:
                    state, r2, r3 = "FR_YELLOW", 3, 3
                else:
                    state, r2, r3 = "RED", 2, 1
            elif fr_mode == "fr2":
                if recovery_streak >= 2:
                    state, r2, r3 = "FR_YELLOW", 3, 3
                elif recovery_streak == 1:
                    state, r2, r3 = "FR_STEP2X", 2, 2
                else:
                    state, r2, r3 = "RED", 2, 1
            elif fr_mode == "fr3":
                if recovery_streak >= 2:
                    state, r2, r3 = "FR_GREEN", 3, 5
                elif recovery_streak == 1:
                    state, r2, r3 = "FR_YELLOW", 3, 3
                else:
                    state, r2, r3 = "RED", 2, 1
            else:
                raise ValueError(fr_mode)
        rows.append(
            {
                "signal_time": int(row.signal_time),
                "signal_time_utc": row.signal_time_utc,
                "month": row.month,
                "strategy_state": state,
                "base_state": base_state,
                "recovery_signal": recovery_signal,
                "recovery_value": row.value,
                "recovery_threshold": row.threshold_value,
                "r2_lev": r2,
                "r3_lev": r3,
                "controller": name,
            }
        )
    return pd.DataFrame(rows)


def monthly_reset_timeline(d15: pd.DataFrame, opportunities: pd.DataFrame, mode: str, value: int, name: str, reset_day: int = 1) -> pd.DataFrame:
    opp = opportunities.copy()
    opp["mature48"] = opp["signal_time"] + 48 * HOUR_MS
    rows = []
    current_reset_key = None
    bad_streak = 0
    first_after_reset = False
    for row in d15.sort_values("signal_time").itertuples(index=False):
        ts = pd.Timestamp(row.signal_time_utc, tz="UTC")
        month_start = pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")
        reset_ts = month_start + pd.Timedelta(days=reset_day - 1)
        if ts < reset_ts:
            prev_month = month_start - pd.offsets.MonthBegin(1)
            reset_ts = pd.Timestamp(prev_month).tz_convert("UTC") + pd.Timedelta(days=reset_day - 1)
        reset_key = reset_ts.strftime("%Y-%m-%d")
        if reset_key != current_reset_key:
            current_reset_key = reset_key
            bad_streak = 0
            first_after_reset = True
        base_state = row.regime_state
        allow_bad = True
        if first_after_reset:
            allow_bad = False
            first_after_reset = False
        elif mode == "mr1":
            allow_bad = True
        elif mode == "mr2":
            if base_state in {"YELLOW", "RED"}:
                bad_streak += 1
            else:
                bad_streak = 0
            allow_bad = bad_streak >= value
        elif mode == "mr3":
            matured_this_month = opp[
                (opp["month"].eq(row.month))
                & (opp["mature48"] <= int(row.signal_time))
                & (opp["path_status"].eq("ok"))
            ]
            allow_bad = len(matured_this_month) >= value
        else:
            raise ValueError(mode)
        state = base_state if allow_bad else "GREEN"
        r2, r3 = d15_action(state)
        rows.append(
            {
                "signal_time": int(row.signal_time),
                "signal_time_utc": row.signal_time_utc,
                "month": row.month,
                "strategy_state": state,
                "base_state": base_state,
                "recovery_signal": False,
                "r2_lev": r2,
                "r3_lev": r3,
                "controller": name,
                "reset_day": reset_day,
            }
        )
    return pd.DataFrame(rows)


def hybrid_timeline(d15: pd.DataFrame, recovery: pd.DataFrame, cap_days: int, name: str) -> pd.DataFrame:
    merged = d15[["signal_time", "signal_time_utc", "month", "regime_state"]].merge(
        recovery[["signal_time", "recovery_signal"]],
        on="signal_time",
        how="left",
    )
    rows = []
    risk_start: int | None = None
    for row in merged.sort_values("signal_time").itertuples(index=False):
        base_state = row.regime_state
        if base_state == "GREEN":
            risk_start = None
            state, r2, r3 = "GREEN", 3, 5
        else:
            if risk_start is None:
                risk_start = int(row.signal_time)
            age_days = (int(row.signal_time) - risk_start) / DAY_MS
            if bool(row.recovery_signal) and age_days >= cap_days:
                if base_state == "RED":
                    state, r2, r3 = "HYBRID_UP1", 3, 3
                else:
                    state, r2, r3 = "HYBRID_GREEN", 3, 5
            else:
                state = base_state
                r2, r3 = d15_action(base_state)
        rows.append(
            {
                "signal_time": int(row.signal_time),
                "signal_time_utc": row.signal_time_utc,
                "month": row.month,
                "strategy_state": state,
                "base_state": base_state,
                "recovery_signal": bool(row.recovery_signal),
                "r2_lev": r2,
                "r3_lev": r3,
                "controller": name,
            }
        )
    return pd.DataFrame(rows)


def precompute_outcomes(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], common_end: int) -> dict[tuple[int, int], dict[str, Any]]:
    outcomes: dict[tuple[int, int], dict[str, Any]] = {}
    for row in signals.itertuples(index=False):
        signal = pd.Series(row._asdict())
        sid = int(signal["signal_id"])
        for lev in LEVERS:
            outcomes[(sid, lev)] = simulate_trade_with_leverage(signal, kline_map, common_end, lev)
    return outcomes


def simulate_actions(signals: pd.DataFrame, outcomes: dict[tuple[int, int], dict[str, Any]], action_timeline: pd.DataFrame, candidate: str) -> pd.DataFrame:
    action_by_time = action_timeline.set_index("signal_time").to_dict("index")
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        bucket = bucket_for_signal(signal)
        action = action_by_time.get(signal_time, {})
        if bucket == "B" and int(signal["rank"]) == 2:
            lev = int(action.get("r2_lev", 3))
        elif bucket == "B" and int(signal["rank"]) == 3:
            lev = int(action.get("r3_lev", 5))
        else:
            lev = int(signal["leverage"])
        common = {
            "candidate": candidate,
            "regime_model": action.get("controller", candidate),
            "response_plan": "D15_Red1x_recovery_reset",
            "bucket": bucket,
            "original_leverage": int(signal["leverage"]),
            "adaptive_leverage": lev,
            "regime_state": action.get("strategy_state", "GREEN"),
            "base_state": action.get("base_state", "GREEN"),
            "recovery_signal": action.get("recovery_signal", False),
        }
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["leverage"] = int(signal["leverage"])
            rows.append(row | common | {"status": "skipped", "skip_reason": "symbol_already_open"})
            continue
        trade = outcomes[(int(signal["signal_id"]), lev)].copy()
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            # Mark-to-market rows are still open at the cutoff, so they must
            # also block another signal on the same cutoff timestamp.
            lock_extra_ms = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + lock_extra_ms
    return pd.DataFrame(rows)


def summarize_all(logs: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.concat(logs, ignore_index=True)
    summary, monthly, rank_summary = summarize_candidate(trades, BASELINE_NAME)
    summary = pareto(summary)
    return trades, summary, monthly, rank_summary


def add_compare_fields(summary: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    base_july = float(monthly[(monthly["strategy"].eq(BASELINE_NAME)) & (monthly["month"].eq("2026-07"))]["net_pnl_u"].sum())
    out = summary.copy()
    for strat in out["strategy"]:
        aug = monthly[(monthly["strategy"].eq(strat)) & (monthly["month"].eq("2026-08"))]["net_pnl_u"].sum()
        apr = monthly[(monthly["strategy"].eq(strat)) & (monthly["month"].eq("2026-04"))]["net_pnl_u"].sum()
        out.loc[out["strategy"].eq(strat), "april_pnl_u"] = apr
        out.loc[out["strategy"].eq(strat), "august_pnl_u_incomplete"] = aug
        out.loc[out["strategy"].eq(strat), "july_protection_pct"] = out.loc[out["strategy"].eq(strat), "july_loss_saved_u"] / abs(base_july) * 100
    return out


def monthly_comparison(monthly: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    frames = []
    for strat in strategies:
        frames.append(monthly[monthly["strategy"].eq(strat)][["month", "net_pnl_u"]].rename(columns={"net_pnl_u": strat}))
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="month", how="outer")
    return out.sort_values("month")


def episode_audit(d15: pd.DataFrame, actions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    d = d15.sort_values("signal_time").reset_index(drop=True)
    in_episode = False
    start = None
    start_state = None
    for row in d.itertuples(index=False):
        bad = row.regime_state in {"YELLOW", "RED"}
        if bad and not in_episode:
            in_episode = True
            start = int(row.signal_time)
            start_state = row.regime_state
        if in_episode and not bad:
            end = int(row.signal_time)
            rows.append({"episode_start": start, "episode_end": end, "start_state": start_state})
            in_episode = False
    if in_episode:
        rows.append({"episode_start": start, "episode_end": int(d.iloc[-1]["signal_time"]), "start_state": start_state})
    ep = pd.DataFrame(rows)
    if ep.empty:
        return ep
    ep["episode_start_utc"] = ep["episode_start"].apply(lambda x: ms_to_utc(int(x)).strftime("%Y-%m-%d %H:%M:%S"))
    ep["episode_end_utc"] = ep["episode_end"].apply(lambda x: ms_to_utc(int(x)).strftime("%Y-%m-%d %H:%M:%S"))
    ep["month"] = ep["episode_start"].apply(lambda x: ms_to_utc(int(x)).strftime("%Y-%m"))
    for name, action in actions.items():
        first_recovery = action[
            (action["signal_time"] >= ep["episode_start"].min())
            & action["recovery_signal"].astype(bool)
            & action["base_state"].isin(["YELLOW", "RED"])
        ]
        vals = []
        for erow in ep.itertuples(index=False):
            sub = action[
                (action["signal_time"] >= erow.episode_start)
                & (action["signal_time"] <= erow.episode_end)
                & (action["recovery_signal"].astype(bool))
            ]
            vals.append(sub.iloc[0]["signal_time_utc"] if not sub.empty else "")
        ep[f"{name}_first_recovery_utc"] = vals
    return ep


def timeline_slice(name: str, action: pd.DataFrame, trades: pd.DataFrame, month: str) -> pd.DataFrame:
    checkpoints = {
        "2026-04": ["2026-04-01 00:00:00", "2026-04-04 00:00:00", "2026-04-08 00:00:00", "2026-04-12 00:00:00", "2026-04-16 00:00:00", "2026-04-20 00:00:00"],
        "2026-07": ["2026-07-01 00:00:00", "2026-07-04 00:00:00", "2026-07-06 00:00:00", "2026-07-10 00:00:00", "2026-07-15 00:00:00", "2026-07-20 00:00:00"],
        "2026-08": ["2026-08-01 00:00:00", "2026-08-03 00:00:00", "2026-08-05 00:00:00", "2026-08-08 00:00:00"],
    }[month]
    rows = []
    strat_trades = trades[trades["candidate"].eq(name)]
    for text in checkpoints:
        t = int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)
        a = action[action["signal_time"].eq(t)]
        so_far = strat_trades[
            (strat_trades["month"].eq(month))
            & (strat_trades["entry_time_ms"] <= t)
            & (strat_trades["status"].isin(["completed", "open_mark_to_market"]))
        ]
        rows.append(
            {
                "time_utc": text,
                "strategy_state": a.iloc[0]["strategy_state"] if not a.empty else "",
                "base_state": a.iloc[0]["base_state"] if not a.empty else "",
                "recovery_signal": bool(a.iloc[0]["recovery_signal"]) if not a.empty else False,
                "r2_lev": a.iloc[0]["r2_lev"] if not a.empty else np.nan,
                "r3_lev": a.iloc[0]["r3_lev"] if not a.empty else np.nan,
                "cum_pnl_u": float(pd.to_numeric(so_far["pnl_u"], errors="coerce").sum()),
                "liqs_so_far": int(so_far["liquidated"].sum()) if "liquidated" in so_far else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [s for s in cached_symbols() if s not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)
    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    filtered = apply_entry_rules(raw, kline_map).copy()
    filtered["bucket"] = filtered.apply(bucket_for_signal, axis=1)
    filtered = filtered.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    filtered["signal_id"] = filtered.index.astype(int)
    signal_times = sorted(filtered["signal_time"].astype(int).unique())

    sets = opportunity_sets(raw, kline_map)
    d15_spec = IndicatorSpec("D_b_r3_decay_l15", "B_R3", "mean_decay48", "lower_bad", 15)
    d15 = build_health_timeline(signal_times, sets["B_R3"], d15_spec)
    d15.to_csv(OUT_DIR / "d15_risk_off_timeline.csv", index=False, encoding="utf-8-sig")

    recovery_frames = []
    for spec in recovery_specs():
        recovery_frames.append(build_recovery_timeline(signal_times, sets["B_R3"], spec))
    recovery_all = pd.concat(recovery_frames, ignore_index=True)
    recovery_all.to_csv(OUT_DIR / "fast_recovery_single_sensor.csv", index=False, encoding="utf-8-sig")
    recovery_map = {name: df for name, df in recovery_all.groupby("model")}

    outcomes = precompute_outcomes(filtered, kline_map, common_end)
    baseline_action = pd.DataFrame(
        {
            "signal_time": signal_times,
            "signal_time_utc": [ms_to_utc(t).strftime("%Y-%m-%d %H:%M:%S") for t in signal_times],
            "month": [ms_to_utc(t).strftime("%Y-%m") for t in signal_times],
            "strategy_state": "GREEN",
            "base_state": "GREEN",
            "recovery_signal": False,
            "r2_lev": 3,
            "r3_lev": 5,
            "controller": "baseline_green",
        }
    )
    d15_action_tl = make_base_action_timeline(d15)

    logs = [
        simulate_actions(filtered, outcomes, baseline_action, BASELINE_NAME),
        simulate_actions(filtered, outcomes, d15_action_tl, D15_NAME),
    ]
    action_tables: dict[str, pd.DataFrame] = {BASELINE_NAME: baseline_action, D15_NAME: d15_action_tl}

    fast_rows = []
    for sensor_name, rec in recovery_map.items():
        for fr_mode in ["fr1", "fr2", "fr3"]:
            for yr_mode in ["yr1", "yr2"]:
                name = f"FR_{sensor_name}_{fr_mode}_{yr_mode}"
                action = fast_recovery_action_timeline(d15, rec, fr_mode, yr_mode, name)
                logs.append(simulate_actions(filtered, outcomes, action, name))
                action_tables[name] = action
                fast_rows.append({"strategy": name, "sensor": sensor_name, "fr_mode": fr_mode, "yr_mode": yr_mode})

    monthly_rows = []
    for mode, values in {"mr1": [1], "mr2": [1, 2, 3], "mr3": [3, 5]}.items():
        for value in values:
            name = f"MR_{mode}_{value}_day1"
            action = monthly_reset_timeline(d15, sets["B_R3"], mode, value, name, reset_day=1)
            logs.append(simulate_actions(filtered, outcomes, action, name))
            action_tables[name] = action
            monthly_rows.append({"strategy": name, "mode": mode, "value": value, "reset_day": 1})
    for reset_day in [8, 15, 22]:
        name = f"MR_placebo_mr2_2_day{reset_day}"
        action = monthly_reset_timeline(d15, sets["B_R3"], "mr2", 2, name, reset_day=reset_day)
        logs.append(simulate_actions(filtered, outcomes, action, name))
        action_tables[name] = action
        monthly_rows.append({"strategy": name, "mode": "mr2_placebo", "value": 2, "reset_day": reset_day})

    # Keep hybrid search small and tied to a simple high-signal recovery sensor.
    hybrid_sensor = "avg_return24_l5_gt_0"
    for cap in [7, 10, 14]:
        name = f"HY_{hybrid_sensor}_cap{cap}d"
        action = hybrid_timeline(d15, recovery_map[hybrid_sensor], cap, name)
        logs.append(simulate_actions(filtered, outcomes, action, name))
        action_tables[name] = action

    trades, summary, monthly, rank_summary = summarize_all(logs)
    summary = add_compare_fields(summary, monthly)
    trades.to_csv(OUT_DIR / "strategy_trade_log.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "strategy_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_results.csv", index=False, encoding="utf-8-sig")
    rank_summary.to_csv(OUT_DIR / "bucket_b_rank_results.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(fast_rows).merge(summary, on="strategy", how="left").to_csv(
        OUT_DIR / "fast_recovery_strategies.csv", index=False, encoding="utf-8-sig"
    )
    mr_meta = pd.DataFrame(monthly_rows).merge(summary, on="strategy", how="left")
    mr_meta[~mr_meta["mode"].astype(str).str.contains("placebo")].to_csv(
        OUT_DIR / "monthly_reset_strategies.csv", index=False, encoding="utf-8-sig"
    )
    mr_meta[mr_meta["mode"].astype(str).str.contains("placebo")].to_csv(
        OUT_DIR / "monthly_reset_placebo.csv", index=False, encoding="utf-8-sig"
    )
    summary[summary["strategy"].str.startswith("HY_")].to_csv(OUT_DIR / "hybrid_strategies.csv", index=False, encoding="utf-8-sig")

    baseline_compare_names = [BASELINE_NAME, D15_NAME]
    best_fast = summary[summary["strategy"].str.startswith("FR_")].sort_values(["net_pnl_u", "july_loss_saved_u"], ascending=False).iloc[0]["strategy"]
    best_mr = summary[summary["strategy"].str.startswith("MR_") & ~summary["strategy"].str.contains("placebo")].sort_values(["net_pnl_u", "july_loss_saved_u"], ascending=False).iloc[0]["strategy"]
    best_hy = summary[summary["strategy"].str.startswith("HY_")].sort_values(["net_pnl_u", "july_loss_saved_u"], ascending=False).iloc[0]["strategy"]
    selected = baseline_compare_names + [best_fast, best_mr, best_hy]
    summary[summary["strategy"].isin(selected)].to_csv(OUT_DIR / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    monthly_comparison(monthly, selected).to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    summary.sort_values(["pareto_frontier", "net_pnl_u", "july_loss_saved_u"], ascending=[False, False, False]).head(30).to_csv(
        OUT_DIR / "candidate_summary.csv", index=False, encoding="utf-8-sig"
    )

    # Stability around the simplest recovery family.
    stability = summary[summary["strategy"].str.contains("avg_return24_l(3|5|7)_gt_0", regex=True)].copy()
    stability.to_csv(OUT_DIR / "parameter_stability.csv", index=False, encoding="utf-8-sig")

    ep = episode_audit(d15, {name: action_tables[name] for name in [D15_NAME, best_fast, best_mr, best_hy] if name in action_tables})
    ep.to_csv(OUT_DIR / "recovery_episode_audit.csv", index=False, encoding="utf-8-sig")
    roc = summary[summary["strategy"].isin(selected)][
        [
            "strategy",
            "net_pnl_u",
            "jan_jun_pnl_u",
            "jan_jun_retention",
            "april_pnl_u",
            "july_pnl_u",
            "july_loss_saved_u",
            "august_pnl_u_incomplete",
            "pf",
            "max_drawdown_u",
            "liquidations",
            "drop_top1_u",
            "drop_top5_u",
            "profit_ex_top10_u",
        ]
    ].copy()
    d15_row = roc[roc["strategy"].eq(D15_NAME)].iloc[0]
    for col in ["net_pnl_u", "jan_jun_pnl_u", "april_pnl_u", "july_pnl_u", "august_pnl_u_incomplete"]:
        roc[f"diff_vs_d15_{col}"] = roc[col] - float(d15_row[col])
    roc.to_csv(OUT_DIR / "recovery_opportunity_cost.csv", index=False, encoding="utf-8-sig")

    for month, filename in [("2026-04", "april_timeline.csv"), ("2026-07", "july_timeline.csv"), ("2026-08", "august_shadow_timeline.csv")]:
        timeline_slice(best_fast, action_tables[best_fast], trades, month).to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")

    final = summary[summary["strategy"].isin(selected)].sort_values("net_pnl_u", ascending=False)
    lines = [
        "# Fast Recovery vs Monthly Reset Final Judgment",
        "",
        f"- Cutoff: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "- Scope: original/VELVET-aligned universe, `BTWUSDT` excluded, `VELVETUSDT` retained.",
        "- Frozen main strategy is unchanged. Only Bucket B Rank2/Rank3 leverage is changed in research variants.",
        "- D15_Red1x: GREEN R2 3x/R3 5x; YELLOW R2 3x/R3 3x; RED R2 2x/R3 1x.",
        "- Fees: 0.1% per side; no extra slippage; 6D default exit and early exits unchanged.",
        "",
        "## Selected Comparison",
        "",
        final[
            [
                "strategy",
                "net_pnl_u",
                "pf",
                "jan_jun_retention",
                "april_pnl_u",
                "july_pnl_u",
                "july_loss_saved_u",
                "august_pnl_u_incomplete",
                "max_drawdown_u",
                "liquidations",
            ]
        ].round(3).to_string(index=False),
        "",
        "## Notes",
        "",
        "- August is incomplete and is shadow/OOS only.",
        "- Monthly reset placebo is saved separately; do not select a day-1 reset unless it beats placebo offsets robustly.",
        "- Fast Recovery variants are economically preferable when performance is close, because the trigger is market-path based rather than calendar based.",
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")
    print("output", OUT_DIR)
    print(final[["strategy", "net_pnl_u", "pf", "jan_jun_retention", "april_pnl_u", "july_pnl_u", "july_loss_saved_u", "august_pnl_u_incomplete", "max_drawdown_u", "liquidations"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
