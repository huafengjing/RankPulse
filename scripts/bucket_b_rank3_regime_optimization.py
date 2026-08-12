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

from scripts.analyze_july_regime import load_cache, rebuild_snapshots
from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    HOUR_MS,
    OUT,
    add_entry_factors,
    load_kline_map,
    max_drawdown,
    mfe_mae,
    ms_to_utc,
    profit_factor,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals, latest_signal_end_dt
from scripts.regime_adaptive_leverage_walkforward import (
    RegimeModel,
    apply_hysteresis,
    bucket_for_signal,
    mature_decay_table,
    simple_markdown_table,
    simulate_trade_with_leverage,
    summarize_extended,
)
from scripts.run_current_main_strategy_2026_jan_jun import (
    HOLD_DAYS,
    SIGNAL_START_MS,
    SNAPSHOT_HOURS_BJ,
    apply_entry_rules,
    cache_common_end_ms,
    cached_symbols,
    gain_bucket,
)


OUT_DIR = OUT / "bucket_b_rank3_regime_optimization"
EXCLUDE_SYMBOLS = {"BTWUSDT"}
LIQ_THRESHOLDS = {3: -33.0, 5: -20.0}
WARMUP_DAYS = 30
MIN_PRIOR_VALUES = 10
WINDOWS = [3, 5, 7, 10, 15]


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    source: str
    metric: str
    direction: str
    window: int
    yellow_q: float = 0.10
    red_q: float = 0.05
    hysteresis: int = 0
    red_confirm: str | None = None
    broad_confirm: str | None = None


@dataclass(frozen=True)
class BResponsePlan:
    name: str
    yellow_r2: int | str
    yellow_r3: int | str
    red_r2: int | str
    red_r3: int | str
    description: str


def response_plans() -> list[BResponsePlan]:
    return [
        BResponsePlan("b_only_r3_y4_r3", 3, 4, 3, 3, "Only reduce Rank3 mildly; Rank2 unchanged."),
        BResponsePlan("b_only_r3_y3_r2", 3, 3, 3, 2, "Rank3 5x -> 3x in YELLOW, 2x in RED."),
        BResponsePlan("asym_a_y3_red_r2_roff", 3, 3, 2, "off", "Asymmetric A: Rank3 off only in RED, Rank2 to 2x."),
        BResponsePlan("asym_b_y_r2_2_r3_3_red_1_2", 2, 3, 1, 2, "Asymmetric B: both ranks reduce, no OFF."),
        BResponsePlan("asym_c_y_r2_3_r3_2_red_2_off", 3, 2, 2, "off", "Asymmetric C: strong Rank3 protection, Rank2 moderate in RED."),
        BResponsePlan("red_only_r3_off", 3, 5, 3, "off", "Only RED turns Rank3 off; YELLOW is observe-only."),
        BResponsePlan("red_no_off_2x", 3, 3, 2, 2, "No OFF; RED caps both ranks at 2x."),
    ]


def pct_return(price: float, ref: float) -> float:
    return (price / ref - 1.0) * 100.0 if np.isfinite(price) and ref > 0 else np.nan


def get_open(frame: pd.DataFrame, open_time: int) -> float:
    row = frame[frame["open_time"].eq(open_time)]
    if row.empty:
        return np.nan
    return float(row.iloc[-1]["open"])


def path_metrics_for_signal(signal: pd.Series, kline_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    h1 = kline_map.get(symbol, pd.DataFrame())
    base = {
        "signal_time": entry_time,
        "signal_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
        "month": ms_to_utc(entry_time).strftime("%Y-%m"),
        "symbol": symbol,
        "rank": int(signal["rank"]),
        "gain_24h": float(signal["gain_24h"]),
        "gain_24h_bucket": gain_bucket(float(signal["gain_24h"])),
        "volume_24h_ratio_7d": float(signal.get("volume_24h_ratio_7d", np.nan)),
        "entry_price": np.nan,
    }
    if h1.empty:
        return base | {"path_status": "missing_symbol"}
    entry_price = get_open(h1, entry_time)
    if not np.isfinite(entry_price):
        return base | {"path_status": "missing_entry"}
    out: dict[str, Any] = base | {"entry_price": entry_price, "path_status": "ok"}
    for hours in [4, 8, 12, 24, 48, 72]:
        maturity = entry_time + hours * HOUR_MS
        exit_price = get_open(h1, maturity)
        out[f"mature_time_{hours}h"] = maturity
        out[f"return_{hours}h"] = pct_return(exit_price, entry_price)
        path = h1[(h1["open_time"] >= entry_time) & (h1["open_time"] <= maturity - HOUR_MS)].copy()
        if path.empty:
            out[f"mfe_{hours}h"] = np.nan
            out[f"mae_{hours}h"] = np.nan
            out[f"time_to_mfe_{hours}h"] = np.nan
            out[f"time_to_mae_{hours}h"] = np.nan
            out[f"new_high_{hours}h"] = False
            continue
        mfe, mae, max_price, min_price = mfe_mae(path, entry_price)
        out[f"mfe_{hours}h"] = mfe
        out[f"mae_{hours}h"] = mae
        max_row = path[path["high"].eq(max_price)].iloc[0]
        min_row = path[path["low"].eq(min_price)].iloc[0]
        out[f"time_to_mfe_{hours}h"] = (int(max_row["open_time"]) - entry_time) / HOUR_MS
        out[f"time_to_mae_{hours}h"] = (int(min_row["open_time"]) - entry_time) / HOUR_MS
        out[f"new_high_{hours}h"] = bool(max_price > entry_price)
    out["decay48"] = out.get("return_48h", np.nan) - out.get("return_24h", np.nan)
    out["decay72"] = out.get("return_72h", np.nan) - out.get("return_24h", np.nan)
    out["liq_hit_5x_24h"] = bool(out.get("mae_24h", np.nan) <= LIQ_THRESHOLDS[5])
    out["liq_hit_5x_48h"] = bool(out.get("mae_48h", np.nan) <= LIQ_THRESHOLDS[5])
    out["liq_hit_5x_72h"] = bool(out.get("mae_72h", np.nan) <= LIQ_THRESHOLDS[5])
    out["liq_hit_3x_24h"] = bool(out.get("mae_24h", np.nan) <= LIQ_THRESHOLDS[3])
    out["liq_hit_3x_48h"] = bool(out.get("mae_48h", np.nan) <= LIQ_THRESHOLDS[3])
    out["liq_hit_3x_72h"] = bool(out.get("mae_72h", np.nan) <= LIQ_THRESHOLDS[3])
    return out


def opportunity_sets(raw_signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    base = raw_signals[
        raw_signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & raw_signals["rank"].isin([2, 3])
        & raw_signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    base = add_entry_factors(base, kline_map)
    b = base[
        base["gain_24h"].between(0.20, 0.40, inclusive="left")
        & base["volume_24h_ratio_7d"].between(1.5, 5.0, inclusive="left")
    ].copy()
    r3_all = base[base["rank"].eq(3)].copy()
    r3_gain20_40 = base[base["rank"].eq(3) & base["gain_24h"].between(0.20, 0.40, inclusive="left")].copy()
    sets = {
        "B_R2": b[b["rank"].eq(2)].copy(),
        "B_R3": b[b["rank"].eq(3)].copy(),
        "Rank3_All": r3_all,
        "Rank3_20_40": r3_gain20_40,
        "B_All": b,
    }
    for name, frame in list(sets.items()):
        rows = [path_metrics_for_signal(row, kline_map) for _, row in frame.iterrows()]
        sets[name] = pd.DataFrame(rows)
        sets[name]["source_set"] = name
    return sets


def aggregate_metric(hist: pd.DataFrame, metric: str, window: int) -> float:
    recent = hist.tail(window)
    if recent.empty:
        return np.nan
    if metric == "mean_decay48":
        return float(pd.to_numeric(recent["decay48"], errors="coerce").mean())
    if metric == "median_decay48":
        return float(pd.to_numeric(recent["decay48"], errors="coerce").median())
    if metric == "mean_decay72":
        return float(pd.to_numeric(recent["decay72"], errors="coerce").mean())
    if metric == "liq_rate48_5x":
        return float(recent["liq_hit_5x_48h"].astype(bool).mean())
    if metric == "liq_rate72_5x":
        return float(recent["liq_hit_5x_72h"].astype(bool).mean())
    if metric == "liq_rate48_3x":
        return float(recent["liq_hit_3x_48h"].astype(bool).mean())
    if metric == "positive_rate48":
        return float((pd.to_numeric(recent["return_48h"], errors="coerce") > 0).mean())
    if metric == "new_high_rate48":
        return float(recent["new_high_48h"].astype(bool).mean())
    if metric == "cq48":
        mfe = float(pd.to_numeric(recent["mfe_48h"], errors="coerce").mean())
        mae = abs(float(pd.to_numeric(recent["mae_48h"], errors="coerce").mean()))
        return mfe / mae if mae > 0 else np.nan
    if metric == "avg_return48":
        return float(pd.to_numeric(recent["return_48h"], errors="coerce").mean())
    raise ValueError(metric)


def state_from_value(value: float, q_y: float, q_r: float, direction: str) -> str:
    if not all(np.isfinite(x) for x in [value, q_y, q_r]):
        return "GREEN"
    if direction == "lower_bad":
        if value <= q_r:
            return "RED"
        if value <= q_y:
            return "YELLOW"
    else:
        if value >= q_r:
            return "RED"
        if value >= q_y:
            return "YELLOW"
    return "GREEN"


def build_health_timeline(signal_times: list[int], opportunities: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    values_seen: list[float] = []
    min_time = SIGNAL_START_MS + WARMUP_DAYS * DAY_MS
    opp = opportunities.copy()
    if opp.empty:
        return pd.DataFrame()
    opp["metric_mature_time"] = opp["signal_time"] + (72 if "72" in spec.metric else 48) * HOUR_MS
    opp = opp.sort_values(["signal_time", "symbol"]).reset_index(drop=True)
    for t in sorted(signal_times):
        hist = opp[(opp["metric_mature_time"] <= t) & (opp["path_status"].eq("ok"))].copy()
        value = np.nan
        q_y = q_r = np.nan
        warm = bool(t >= min_time or len(hist) >= spec.window)
        if warm and len(hist) >= spec.window:
            value = aggregate_metric(hist, spec.metric, spec.window)
            if len(values_seen) >= MIN_PRIOR_VALUES:
                q_y = float(np.quantile(values_seen, spec.yellow_q))
                q_r = float(np.quantile(values_seen, spec.red_q))
        state = state_from_value(value, q_y, q_r, spec.direction) if warm else "GREEN"
        rows.append(
            {
                "model": spec.name,
                "source_set": spec.source,
                "metric": spec.metric,
                "window": spec.window,
                "signal_time": t,
                "signal_time_utc": ms_to_utc(t).strftime("%Y-%m-%d %H:%M:%S"),
                "month": ms_to_utc(t).strftime("%Y-%m"),
                "value": value,
                "hist_yellow_threshold": q_y,
                "hist_red_threshold": q_r,
                "hist_count": len(hist),
                "warmup_complete": warm,
                "raw_state": state,
            }
        )
        if warm and np.isfinite(value):
            values_seen.append(value)
    frame = pd.DataFrame(rows)
    frame["regime_state"] = apply_hysteresis(frame["raw_state"].tolist(), spec.hysteresis)
    return frame


def broad_health(signal_times: list[int], snapshot_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    snapshots, _ = rebuild_snapshots(snapshot_map, include_august=True)
    mature = mature_decay_table(snapshots)
    model = RegimeModel("Broad_R4_10_5d_count_h0", "count", count=70, hysteresis=0)
    from scripts.regime_adaptive_leverage_walkforward import build_regime_timeline

    broad = build_regime_timeline(signal_times, mature, model).rename(
        columns={"post24_decay": "value", "hist_q10": "hist_yellow_threshold", "hist_q5": "hist_red_threshold"}
    )
    broad["source_set"] = "Broad_R4_10"
    broad["metric"] = "post24_decay"
    broad["window"] = 70
    return broad


def indicator_specs() -> list[IndicatorSpec]:
    return [
        IndicatorSpec("A_broad_r4_10_decay_5d", "Broad_R4_10", "post24_decay", "lower_bad", 70),
        IndicatorSpec("B_rank3_all_decay_l7", "Rank3_All", "mean_decay48", "lower_bad", 7),
        IndicatorSpec("C_rank3_gain20_40_decay_l7", "Rank3_20_40", "mean_decay48", "lower_bad", 7),
        IndicatorSpec("D_b_r3_decay_l7", "B_R3", "mean_decay48", "lower_bad", 7),
        IndicatorSpec("E_b_r3_liq48_l7", "B_R3", "liq_rate48_5x", "higher_bad", 7, yellow_q=0.90, red_q=0.95),
        IndicatorSpec("F_b_r3_cq48_l7", "B_R3", "cq48", "lower_bad", 7),
        IndicatorSpec("G_b_r3_pos48_l7", "B_R3", "positive_rate48", "lower_bad", 7),
    ]


def window_specs() -> list[IndicatorSpec]:
    return [IndicatorSpec(f"D_b_r3_decay_l{w}", "B_R3", "mean_decay48", "lower_bad", w) for w in WINDOWS]


def threshold_specs() -> list[IndicatorSpec]:
    pairs = [(0.20, 0.10), (0.15, 0.05), (0.10, 0.05), (0.20, 0.05), (0.15, 0.10)]
    return [
        IndicatorSpec(f"D_b_r3_decay_l7_q{int(y*100)}_{int(r*100)}", "B_R3", "mean_decay48", "lower_bad", 7, yellow_q=y, red_q=r)
        for y, r in pairs
    ]


def combine_states(primary: pd.DataFrame, confirm: pd.DataFrame, name: str, mode: str) -> pd.DataFrame:
    left = primary[["signal_time", "signal_time_utc", "month", "value", "hist_yellow_threshold", "hist_red_threshold", "hist_count", "warmup_complete", "raw_state", "regime_state"]].copy()
    right = confirm[["signal_time", "regime_state"]].rename(columns={"regime_state": "confirm_state"})
    out = left.merge(right, on="signal_time", how="left")
    states = []
    for row in out.itertuples(index=False):
        p = row.regime_state
        c = row.confirm_state if isinstance(row.confirm_state, str) else "GREEN"
        if mode == "decay_plus_liq_red_confirm":
            if p == "RED" and c in {"YELLOW", "RED"}:
                states.append("RED")
            elif p in {"YELLOW", "RED"}:
                states.append("YELLOW")
            else:
                states.append("GREEN")
        elif mode == "decay_or_confirm":
            if "RED" in {p, c}:
                states.append("RED")
            elif "YELLOW" in {p, c}:
                states.append("YELLOW")
            else:
                states.append("GREEN")
        else:
            states.append(p)
    out["model"] = name
    out["source_set"] = mode
    out["metric"] = mode
    out["window"] = np.nan
    out["raw_state"] = states
    out["regime_state"] = states
    return out.drop(columns=["confirm_state"])


def with_hysteresis(timeline: pd.DataFrame, name: str, hysteresis: int) -> pd.DataFrame:
    out = timeline.copy()
    out["model"] = name
    out["regime_state"] = apply_hysteresis(out["raw_state"].tolist(), hysteresis)
    out["hysteresis"] = hysteresis
    return out


def b_action(action: int | str, original: int) -> tuple[int | None, str]:
    if action == "base":
        return original, "base"
    if action == "off":
        return None, "off"
    target = int(action)
    return min(original, target), f"cap_{target}x"


def adjusted_b_leverage(plan: BResponsePlan, state: str, rank: int, original: int, bucket: str) -> tuple[int | None, bool, str]:
    if bucket != "B" or state == "GREEN":
        return original, False, "base"
    if rank == 2:
        action = plan.yellow_r2 if state == "YELLOW" else plan.red_r2
    elif rank == 3:
        action = plan.yellow_r3 if state == "YELLOW" else plan.red_r3
    else:
        action = "base"
    lev, label = b_action(action, original)
    return lev, lev is None, label


def simulate_b_controller(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], current_time: int, timeline: pd.DataFrame, plan: BResponsePlan, model_name: str) -> pd.DataFrame:
    state_by_time = timeline.set_index("signal_time").to_dict("index")
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        bucket = bucket_for_signal(signal)
        original = int(signal["leverage"])
        regime = state_by_time.get(signal_time, {})
        state = str(regime.get("regime_state", "GREEN"))
        adaptive, skip_off, action = adjusted_b_leverage(plan, state, int(signal["rank"]), original, bucket)
        common = {
            "candidate": f"{model_name}__{plan.name}",
            "regime_model": model_name,
            "response_plan": plan.name,
            "bucket": bucket,
            "original_leverage": original,
            "adaptive_leverage": adaptive if adaptive is not None else np.nan,
            "regime_state": state,
            "regime_value": regime.get("value", np.nan),
            "hist_yellow_threshold": regime.get("hist_yellow_threshold", np.nan),
            "hist_red_threshold": regime.get("hist_red_threshold", np.nan),
            "hist_count": regime.get("hist_count", np.nan),
            "warmup_complete": regime.get("warmup_complete", False),
            "leverage_action": action,
        }
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["leverage"] = original
            rows.append(row | common | {"status": "skipped", "skip_reason": "symbol_already_open"})
            continue
        if skip_off or adaptive is None:
            row = skipped_open_position_trade(signal, signal_time)
            row["leverage"] = original
            rows.append(row | common | {"status": "skipped", "skip_reason": "regime_off"})
            continue
        trade = simulate_trade_with_leverage(signal, kline_map, current_time, int(adaptive))
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            # Mark-to-market rows are still open at the cutoff, so they must
            # also block another signal on the same cutoff timestamp.
            lock_extra_ms = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + lock_extra_ms
    return pd.DataFrame(rows)


def summarize_candidate(all_trades: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    monthly_rows = []
    rank_rows = []
    for candidate, group in all_trades.groupby("candidate", sort=True):
        summary_rows.append({"strategy": candidate, **summarize_extended(group)})
        for month, mg in group.groupby("month", sort=True):
            monthly_rows.append({"strategy": candidate, "month": month, **summarize_extended(mg)})
        b = group[group.apply(lambda r: bucket_for_signal(r) == "B" if pd.notna(r.get("gain_24h", np.nan)) else False, axis=1)].copy()
        for rank, rg in b.groupby("rank", sort=True):
            rank_rows.append({"strategy": candidate, "rank": int(rank), **summarize_extended(rg)})
    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    rank_summary = pd.DataFrame(rank_rows)
    base = summary[summary["strategy"].eq(baseline)].iloc[0]
    base_jj = monthly[(monthly["strategy"].eq(baseline)) & (monthly["month"].between("2026-01", "2026-06"))]["net_pnl_u"].sum()
    base_july = float(monthly[(monthly["strategy"].eq(baseline)) & (monthly["month"].eq("2026-07"))]["net_pnl_u"].sum())
    for idx, row in summary.iterrows():
        strat = row["strategy"]
        jj = monthly[(monthly["strategy"].eq(strat)) & (monthly["month"].between("2026-01", "2026-06"))]["net_pnl_u"].sum()
        july = monthly[(monthly["strategy"].eq(strat)) & (monthly["month"].eq("2026-07"))]["net_pnl_u"].sum()
        summary.loc[idx, "jan_jun_pnl_u"] = jj
        summary.loc[idx, "jan_jun_retention"] = jj / base_jj if base_jj else np.nan
        summary.loc[idx, "july_pnl_u"] = july
        summary.loc[idx, "july_loss_saved_u"] = july - base_july
        summary.loc[idx, "july_protection_ratio"] = (july - base_july) / abs(base_july) if base_july < 0 else np.nan
        summary.loc[idx, "jan_jun_profit_sacrifice_u"] = base_jj - jj
        summary.loc[idx, "protection_efficiency"] = (july - base_july) / (base_jj - jj) if (base_jj - jj) > 0 else np.nan
        summary.loc[idx, "baseline_total_pnl_u"] = base["net_pnl_u"]
    return summary, monthly, rank_summary


def pareto(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ["net_pnl_u", "july_loss_saved_u", "jan_jun_retention", "pf", "max_drawdown_u"]
    rows = []
    for _, row in summary.iterrows():
        dominated = False
        for _, other in summary.iterrows():
            if row["strategy"] == other["strategy"]:
                continue
            better_or_equal = (
                other["net_pnl_u"] >= row["net_pnl_u"]
                and other["july_loss_saved_u"] >= row["july_loss_saved_u"]
                and other["jan_jun_retention"] >= row["jan_jun_retention"]
                and other["pf"] >= row["pf"]
                and other["liquidations"] <= row["liquidations"]
                and other["max_drawdown_u"] >= row["max_drawdown_u"]
            )
            strictly = (
                other["net_pnl_u"] > row["net_pnl_u"]
                or other["july_loss_saved_u"] > row["july_loss_saved_u"]
                or other["jan_jun_retention"] > row["jan_jun_retention"]
                or other["pf"] > row["pf"]
                or other["liquidations"] < row["liquidations"]
                or other["max_drawdown_u"] > row["max_drawdown_u"]
            )
            if better_or_equal and strictly:
                dominated = True
                break
        rows.append({"strategy": row["strategy"], "pareto_frontier": not dominated})
    return summary.merge(pd.DataFrame(rows), on="strategy", how="left")


def state_counts(timeline: pd.DataFrame, model: str) -> dict[str, Any]:
    g = timeline[timeline["model"].eq(model)]
    return {
        "july_first_yellow_utc": g[(g["month"].eq("2026-07")) & (g["regime_state"].eq("YELLOW"))]["signal_time_utc"].min(),
        "july_first_red_utc": g[(g["month"].eq("2026-07")) & (g["regime_state"].eq("RED"))]["signal_time_utc"].min(),
        "jan_jun_yellow_red_obs": int(g[g["month"].between("2026-01", "2026-06")]["regime_state"].isin(["YELLOW", "RED"]).sum()),
        "july_yellow_red_obs": int(g[g["month"].eq("2026-07")]["regime_state"].isin(["YELLOW", "RED"]).sum()),
    }


def july_timeline_rows(timeline_map: dict[str, pd.DataFrame], baseline_trades: pd.DataFrame, strategy_trades: pd.DataFrame, model_name: str, plan: BResponsePlan) -> pd.DataFrame:
    times = [
        "2026-07-01 00:00:00",
        "2026-07-02 00:00:00",
        "2026-07-03 00:00:00",
        "2026-07-04 00:00:00",
        "2026-07-05 00:00:00",
        "2026-07-07 00:00:00",
        "2026-07-10 00:00:00",
    ]
    r3_l5 = timeline_map.get("D_b_r3_decay_l5", pd.DataFrame())
    r3_l7 = timeline_map.get("D_b_r3_decay_l7", pd.DataFrame())
    liq = timeline_map.get("E_b_r3_liq48_l7", pd.DataFrame())
    r2 = timeline_map.get("R2_decay_l7", pd.DataFrame())
    broad = timeline_map.get("A_broad_r4_10_decay_5d", pd.DataFrame())
    rows = []
    for text_time in times:
        t = int(pd.Timestamp(text_time, tz="UTC").timestamp() * 1000)
        b_so_far = baseline_trades[
            (baseline_trades["month"].eq("2026-07"))
            & (baseline_trades["entry_time_ms"] <= t)
            & (baseline_trades["status"].isin(["completed", "open_mark_to_market"]))
        ]
        state_row = timeline_map[model_name][timeline_map[model_name]["signal_time"].eq(t)]
        state = state_row.iloc[0]["regime_state"] if not state_row.empty else "GREEN"
        r2_lev, _, _ = adjusted_b_leverage(plan, state, 2, 3, "B")
        r3_lev, r3_off, _ = adjusted_b_leverage(plan, state, 3, 5, "B")
        def val(df: pd.DataFrame, col: str = "value") -> float:
            rr = df[df["signal_time"].eq(t)]
            return float(rr.iloc[0][col]) if not rr.empty and pd.notna(rr.iloc[0][col]) else np.nan
        rows.append(
            {
                "time_utc": text_time,
                "r3_last5_decay48": val(r3_l5),
                "r3_last7_decay48": val(r3_l7),
                "liq_hit48_rate": val(liq),
                "r2_health_decay48": val(r2),
                "broad_health": val(broad),
                "actual_liqs_so_far": int(b_so_far["liquidated"].sum()) if "liquidated" in b_so_far else 0,
                "cum_pnl_so_far_u": float(pd.to_numeric(b_so_far["pnl_u"], errors="coerce").sum()),
                "regime": state,
                "r2_lev": r2_lev,
                "r3_lev": "OFF" if r3_off else r3_lev,
            }
        )
    return pd.DataFrame(rows)


def february_vs_july(sets: dict[str, pd.DataFrame], timelines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for source in ["B_R3", "B_R2"]:
        df = sets[source]
        for month in ["2026-02", "2026-07"]:
            g = df[df["month"].eq(month)].copy()
            rows.append(
                {
                    "source": source,
                    "month": month,
                    "opportunities": len(g),
                    "mean_decay48": pd.to_numeric(g["decay48"], errors="coerce").mean(),
                    "mean_decay72": pd.to_numeric(g["decay72"], errors="coerce").mean(),
                    "mean_mfe48": pd.to_numeric(g["mfe_48h"], errors="coerce").mean(),
                    "mean_mae48": pd.to_numeric(g["mae_48h"], errors="coerce").mean(),
                    "liq_rate48_5x": g["liq_hit_5x_48h"].astype(bool).mean() if len(g) else np.nan,
                    "liq_rate72_5x": g["liq_hit_5x_72h"].astype(bool).mean() if len(g) else np.nan,
                    "positive_rate48": (pd.to_numeric(g["return_48h"], errors="coerce") > 0).mean() if len(g) else np.nan,
                    "new_high_rate48": g["new_high_48h"].astype(bool).mean() if len(g) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_frozen_baseline(common_end: int) -> None:
    lines = [
        "# Frozen Baseline",
        "",
        "- Scope: original/VELVET-aligned universe.",
        "- Ranking universe excluded symbols: BTWUSDT.",
        "- VELVETUSDT retained.",
        "- Observation: Beijing 00:00 and 08:00.",
        "- Buckets A/C unchanged; this research only changes Bucket B Rank2/Rank3 leverage or skip.",
        "- Fees: 0.1% per side.",
        f"- Holding period: {HOLD_DAYS} days.",
        "- Early exits: 4H extreme weak and 12H weak, unchanged.",
        "- Liquidation thresholds: 3x MAE <= -33%, 5x MAE <= -20%.",
        f"- Cache common end: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
    ]
    (OUT_DIR / "frozen_baseline.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [s for s in cached_symbols() if s not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    write_frozen_baseline(common_end)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)
    raw_signals = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    filtered = apply_entry_rules(raw_signals, kline_map)
    filtered["bucket"] = filtered.apply(bucket_for_signal, axis=1)
    signal_times = sorted(filtered["signal_time"].astype(int).unique())

    sets = opportunity_sets(raw_signals, kline_map)
    bucket_b = pd.concat([sets["B_R2"], sets["B_R3"]], ignore_index=True)
    bucket_b.to_csv(OUT_DIR / "bucket_b_eligible_opportunities.csv", index=False, encoding="utf-8-sig")
    path_all = pd.concat([sets["B_R2"], sets["B_R3"], sets["Rank3_All"], sets["Rank3_20_40"]], ignore_index=True)
    path_all.to_csv(OUT_DIR / "liquidation_path_metrics.csv", index=False, encoding="utf-8-sig")

    snapshot_map = {s: df for s, df in load_cache().items() if s not in EXCLUDE_SYMBOLS}
    timelines: dict[str, pd.DataFrame] = {}
    timelines["A_broad_r4_10_decay_5d"] = broad_health(signal_times, snapshot_map)
    for spec in indicator_specs()[1:] + window_specs() + threshold_specs():
        if spec.name in timelines:
            continue
        timelines[spec.name] = build_health_timeline(signal_times, sets[spec.source], spec)
    r2_spec = IndicatorSpec("R2_decay_l7", "B_R2", "mean_decay48", "lower_bad", 7)
    timelines["R2_decay_l7"] = build_health_timeline(signal_times, sets["B_R2"], r2_spec)

    # Simple multi-indicator confirmations.
    timelines["M_decay_plus_liq_red_confirm"] = combine_states(
        timelines["D_b_r3_decay_l7"], timelines["E_b_r3_liq48_l7"], "M_decay_plus_liq_red_confirm", "decay_plus_liq_red_confirm"
    )
    timelines["M_decay_or_pos"] = combine_states(
        timelines["D_b_r3_decay_l7"], timelines["G_b_r3_pos48_l7"], "M_decay_or_pos", "decay_or_confirm"
    )
    timelines["M_decay_plus_broad_red_confirm"] = combine_states(
        timelines["D_b_r3_decay_l7"], timelines["A_broad_r4_10_decay_5d"], "M_decay_plus_broad_red_confirm", "decay_plus_liq_red_confirm"
    )
    for base_name in ["D_b_r3_decay_l7", "M_decay_or_pos", "M_decay_plus_broad_red_confirm"]:
        timelines[f"{base_name}_h1"] = with_hysteresis(timelines[base_name], f"{base_name}_h1", 1)
        timelines[f"{base_name}_h2"] = with_hysteresis(timelines[base_name], f"{base_name}_h2", 2)

    all_timeline = pd.concat(timelines.values(), ignore_index=True)
    all_timeline[all_timeline["source_set"].astype(str).str.contains("B_R3|Rank3", na=False)].to_csv(
        OUT_DIR / "rank3_health_walkforward.csv", index=False, encoding="utf-8-sig"
    )
    timelines["R2_decay_l7"].to_csv(OUT_DIR / "rank2_health_walkforward.csv", index=False, encoding="utf-8-sig")
    timelines["A_broad_r4_10_decay_5d"].to_csv(OUT_DIR / "broad_health_walkforward.csv", index=False, encoding="utf-8-sig")

    plans = response_plans()
    pd.DataFrame([plan.__dict__ for plan in plans]).to_csv(OUT_DIR / "leverage_response_matrix.csv", index=False, encoding="utf-8-sig")

    trade_logs = []
    baseline_timeline = pd.DataFrame(
        {
            "signal_time": signal_times,
            "regime_state": "GREEN",
            "value": np.nan,
            "hist_yellow_threshold": np.nan,
            "hist_red_threshold": np.nan,
            "hist_count": 0,
            "warmup_complete": False,
        }
    )
    base_plan = BResponsePlan("baseline_no_adaptive", 3, 5, 3, 5, "No adaptive response.")
    trade_logs.append(simulate_b_controller(filtered, kline_map, common_end, baseline_timeline, base_plan, "baseline_green"))

    default_plan = [p for p in plans if p.name == "asym_a_y3_red_r2_roff"][0]
    test_rows = []
    for spec_name in [s.name for s in indicator_specs()] + [s.name for s in window_specs()] + [s.name for s in threshold_specs()]:
        test_rows.append((spec_name, default_plan, "single_or_window_threshold"))
    for model_name in ["D_b_r3_decay_l7", "M_decay_plus_liq_red_confirm", "M_decay_or_pos", "M_decay_plus_broad_red_confirm"]:
        for plan in plans:
            test_rows.append((model_name, plan, "response_or_multi"))
    for model_name in [
        "D_b_r3_decay_l7_h1",
        "D_b_r3_decay_l7_h2",
        "M_decay_or_pos_h1",
        "M_decay_or_pos_h2",
        "M_decay_plus_broad_red_confirm_h1",
        "M_decay_plus_broad_red_confirm_h2",
    ]:
        test_rows.append((model_name, default_plan, "hysteresis"))
    seen: set[tuple[str, str]] = set()
    run_rows = []
    for model_name, plan, phase in test_rows:
        key = (model_name, plan.name)
        if key in seen:
            continue
        seen.add(key)
        log = simulate_b_controller(filtered, kline_map, common_end, timelines[model_name], plan, model_name)
        log["research_phase"] = phase
        trade_logs.append(log)
        run_rows.append({"model": model_name, "plan": plan.name, "phase": phase})

    all_trades = pd.concat(trade_logs, ignore_index=True)
    all_trades.to_csv(OUT_DIR / "adaptive_trade_log.csv", index=False, encoding="utf-8-sig")
    summary, monthly, rank_summary = summarize_candidate(all_trades, "baseline_green__baseline_no_adaptive")
    summary = pareto(summary)
    summary.to_csv(OUT_DIR / "strategy_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_results.csv", index=False, encoding="utf-8-sig")
    rank_summary.to_csv(OUT_DIR / "bucket_b_rank_results.csv", index=False, encoding="utf-8-sig")

    single = summary[summary["strategy"].str.contains("__asym_a_y3_red_r2_roff")].copy()
    single.to_csv(OUT_DIR / "single_indicator_ablation.csv", index=False, encoding="utf-8-sig")
    summary[summary["strategy"].str.contains("D_b_r3_decay_l(3|5|7|10|15)__", regex=True)].to_csv(
        OUT_DIR / "window_ablation.csv", index=False, encoding="utf-8-sig"
    )
    summary[summary["strategy"].str.contains("q20_|q15_|q10_", regex=True)].to_csv(
        OUT_DIR / "threshold_ablation.csv", index=False, encoding="utf-8-sig"
    )
    summary[summary["strategy"].str.startswith("M_")].to_csv(OUT_DIR / "multi_indicator_confirmation.csv", index=False, encoding="utf-8-sig")
    summary[summary["strategy"].str.contains("_h[12]__", regex=True)].to_csv(
        OUT_DIR / "hysteresis_ablation.csv", index=False, encoding="utf-8-sig"
    )

    pareto(summary).to_csv(OUT_DIR / "pareto_frontier.csv", index=False, encoding="utf-8-sig")
    param = summary[summary["strategy"].str.contains("D_b_r3_decay_l(3|5|7|10|15)__asym_a", regex=True)].copy()
    param.to_csv(OUT_DIR / "parameter_stability.csv", index=False, encoding="utf-8-sig")
    top = summary.sort_values(["pareto_frontier", "net_pnl_u", "july_loss_saved_u"], ascending=[False, False, False]).head(20)
    top.to_csv(OUT_DIR / "candidate_summary.csv", index=False, encoding="utf-8-sig")

    chosen = "D_b_r3_decay_l7__asym_a_y3_red_r2_roff"
    if chosen not in set(all_trades["candidate"]):
        chosen = top.iloc[0]["strategy"]
    july_timeline_rows(timelines, all_trades[all_trades["candidate"].eq("baseline_green__baseline_no_adaptive")], all_trades[all_trades["candidate"].eq(chosen)], chosen.split("__")[0], default_plan).to_csv(
        OUT_DIR / "july_timeline.csv", index=False, encoding="utf-8-sig"
    )
    february_vs_july(sets, timelines).to_csv(OUT_DIR / "february_vs_july.csv", index=False, encoding="utf-8-sig")

    baseline = summary[summary["strategy"].eq("baseline_green__baseline_no_adaptive")].iloc[0]
    best_total = summary[summary["strategy"].ne("baseline_green__baseline_no_adaptive")].sort_values("net_pnl_u", ascending=False).iloc[0]
    best_july = summary[summary["strategy"].ne("baseline_green__baseline_no_adaptive")].sort_values("july_loss_saved_u", ascending=False).iloc[0]
    best_pf = summary[summary["strategy"].ne("baseline_green__baseline_no_adaptive")].sort_values(["pf", "net_pnl_u"], ascending=False).iloc[0]
    report = [
        "# Bucket B / Rank3 Regime Optimization Final Judgment",
        "",
        "## Scope",
        "",
        "Original/VELVET-aligned universe; `BTWUSDT` excluded from the ranking universe; main strategy rules are frozen. Only Bucket B Rank2/Rank3 leverage or skip is changed.",
        "",
        "## Baseline",
        "",
        f"- Net PnL: {baseline['net_pnl_u']:.2f}U",
        f"- PF: {baseline['pf']:.2f}",
        f"- July PnL: {baseline['july_pnl_u']:.2f}U",
        f"- Jan-Jun PnL: {baseline['jan_jun_pnl_u']:.2f}U",
        f"- Max DD: {baseline['max_drawdown_u']:.2f}U",
        f"- Liquidations: {int(baseline['liquidations'])}",
        "",
        "## Leading Candidates",
        "",
        simple_markdown_table(
            pd.DataFrame([best_total, best_pf, best_july])[
                [
                    "strategy",
                    "net_pnl_u",
                    "pf",
                    "july_pnl_u",
                    "july_loss_saved_u",
                    "jan_jun_retention",
                    "max_drawdown_u",
                    "liquidations",
                    "drop_top5_u",
                    "profit_ex_top10_u",
                ]
            ].round(3)
        ),
        "",
        "## Q1-Q12 Short Answers",
        "",
        "Q1: Bucket B Rank3 is closer to the actual 5x risk object than broad Rank4-10; see single indicator ablation.",
        "Q2: Last5/Last7/Last10 must be compared in `window_ablation.csv`; prefer a plateau, not a single spike.",
        "Q3: Post24Decay and LiqHitRate48 are the primary candidates; CQ48/PositiveRate48 are confirmation candidates.",
        "Q4: Multi-indicator confirmation reduces false positives but can be later; compare `multi_indicator_confirmation.csv`.",
        "Q5: Rank2 should not automatically mirror Rank3; asymmetric plans are tested in `leverage_response_matrix.csv`.",
        "Q6: Yellow Rank3 candidates include 4x/3x/2x/OFF via the response matrix.",
        "Q7: Red Rank3 candidates include 3x/2x/1x/OFF via the response matrix.",
        "Q8: See `strategy_results.csv` for >=90% and >=95% Jan-Jun retention filters.",
        "Q9: Highest total profit is listed above and in `candidate_summary.csv`.",
        "Q10: PF/DD/liquidation/exTop5/exTop10 comparison is in `strategy_results.csv`.",
        "Q11: February vs July path differences are in `february_vs_july.csv`.",
        "Q12: Shadow-mode candidate exists only if it stays on Pareto frontier and preserves normal-month retention.",
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(report), encoding="utf-8")
    print("output", OUT_DIR)
    print(top[["strategy", "net_pnl_u", "pf", "july_pnl_u", "july_loss_saved_u", "jan_jun_retention", "max_drawdown_u", "liquidations", "pareto_frontier"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
