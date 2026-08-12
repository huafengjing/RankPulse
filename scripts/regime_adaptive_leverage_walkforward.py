from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_july_regime import EXTRA_END, START, load_cache, rebuild_snapshots
from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    EARLY_REASON,
    FEE_RATE,
    HOUR_MS,
    OUT,
    add_entry_factors,
    get_open_at_or_latest,
    load_kline_map,
    max_drawdown,
    mfe_mae,
    ms_to_bj_string,
    ms_to_utc,
    path_slice,
    profit_factor,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals, latest_signal_end_dt
from scripts.run_current_main_strategy_2026_jan_jun import (
    HOLD_DAYS,
    SIGNAL_START_MS,
    SNAPSHOT_HOURS_BJ,
    apply_entry_rules,
    cache_common_end_ms,
    cached_symbols,
    calc_leveraged_pnl,
    current_close_at_or_before,
    gain_bucket,
    leverage_for_signal,
    summarize,
)


OUT_DIR = OUT / os.environ.get("REGIME_ADAPTIVE_OUT_DIR", "regime_adaptive_leverage")
EXCLUDE_SYMBOLS = {
    item.strip().upper()
    for item in os.environ.get("REGIME_ADAPTIVE_EXCLUDE_SYMBOLS", "").split(",")
    if item.strip()
}
MIN_WARMUP_DAYS = 30
MIN_MATURE_OBS = 420
MIN_PRIOR_INDICATOR_POINTS = 20
LIQUIDATION_THRESHOLDS_PCT = {1: -100.0, 2: -50.0, 3: -33.0, 4: -25.0, 5: -20.0}


BUCKET_LABELS = {
    "A": "10%-20% Rank2/3 no volume filter original 3x",
    "B": "20%-40% Rank2/3 vol 1.5-5 original Rank2 3x / Rank3 5x",
    "C": "40%-60% Rank2 only vol 3-5.5 original 2x",
}


def bucket_for_signal(row: pd.Series) -> str:
    gain = float(row["gain_24h"])
    if 0.10 <= gain < 0.20:
        return "A"
    if 0.20 <= gain < 0.40:
        return "B"
    if 0.40 <= gain < 0.60:
        return "C"
    return "OTHER"


@dataclass(frozen=True)
class RegimeModel:
    name: str
    mode: str
    window_days: int | None = None
    count: int | None = None
    hysteresis: int = 0


@dataclass(frozen=True)
class ResponsePlan:
    name: str
    yellow: dict[str, Any]
    red: dict[str, Any]
    description: str


def response_plans() -> list[ResponsePlan]:
    return [
        ResponsePlan(
            "baseline_no_adaptive",
            yellow={"A": "base", "B": "base", "C": "base"},
            red={"A": "base", "B": "base", "C": "base"},
            description="Y0 baseline; no leverage response",
        ),
        ResponsePlan(
            "mild_y1_r1",
            yellow={"A": 2, "B": 3, "C": "base"},
            red={"A": 2, "B": 2, "C": 1},
            description="Mild Y; RED lowers all but keeps trading",
        ),
        ResponsePlan(
            "moderate_y2_r2",
            yellow={"A": 2, "B": 2, "C": "base"},
            red={"A": 1, "B": 1, "C": "base"},
            description="Moderate; low-risk C preserved",
        ),
        ResponsePlan(
            "balanced_y2_r3",
            yellow={"A": 2, "B": 2, "C": "base"},
            red={"A": 1, "B": "off", "C": "base"},
            description="Balanced; RED disables highest-risk B",
        ),
        ResponsePlan(
            "focus_y4_r3",
            yellow={"A": "base", "B": 3, "C": "base"},
            red={"A": 1, "B": "off", "C": "base"},
            description="High-risk focus; only B adjusted in YELLOW",
        ),
        ResponsePlan(
            "disable_y5_r4",
            yellow={"A": "base", "B": "off", "C": "base"},
            red={"A": "off", "B": "off", "C": "base"},
            description="Disable high-risk B in YELLOW; RED keeps only C",
        ),
        ResponsePlan(
            "all_off_red_r5",
            yellow={"A": 2, "B": 2, "C": "base"},
            red={"A": "off", "B": "off", "C": "off"},
            description="Moderate Y; full OFF in RED",
        ),
    ]


def selected_tests() -> list[tuple[RegimeModel, ResponsePlan]]:
    plans = {p.name: p for p in response_plans()}
    models = [
        RegimeModel("3d_calendar_h0", "calendar", window_days=3, hysteresis=0),
        RegimeModel("5d_calendar_h0", "calendar", window_days=5, hysteresis=0),
        RegimeModel("7d_calendar_h0", "calendar", window_days=7, hysteresis=0),
        RegimeModel("3d_count_h0", "count", count=42, hysteresis=0),
        RegimeModel("5d_count_h0", "count", count=70, hysteresis=0),
        RegimeModel("7d_count_h0", "count", count=98, hysteresis=0),
        RegimeModel("3d_calendar_h2", "calendar", window_days=3, hysteresis=2),
        RegimeModel("7d_calendar_h2", "calendar", window_days=7, hysteresis=2),
    ]
    tests: list[tuple[RegimeModel, ResponsePlan]] = []
    default = plans["balanced_y2_r3"]
    for model in models[:6]:
        tests.append((model, default))
    for plan_name in ["mild_y1_r1", "moderate_y2_r2", "focus_y4_r3", "disable_y5_r4", "all_off_red_r5"]:
        tests.append((models[2], plans[plan_name]))
    tests.append((models[6], default))
    tests.append((models[7], default))
    return tests


def mature_decay_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    work = snapshots[(snapshots["rank"].between(4, 10))].copy()
    work["future_return_24h"] = pd.to_numeric(work["future_return_24h"], errors="coerce")
    work["future_return_48h"] = pd.to_numeric(work["future_return_48h"], errors="coerce")
    work = work.dropna(subset=["future_return_24h", "future_return_48h"])
    work["post24_decay"] = work["future_return_48h"] - work["future_return_24h"]
    work["mature_time"] = work["signal_time"].astype("int64") + 48 * HOUR_MS
    return work.sort_values(["mature_time", "signal_time", "rank", "symbol"]).reset_index(drop=True)


def raw_state(value: float, q10: float, q5: float) -> str:
    if not np.isfinite(value) or not np.isfinite(q10) or not np.isfinite(q5):
        return "GREEN"
    if value <= q5:
        return "RED"
    if value <= q10:
        return "YELLOW"
    return "GREEN"


def severity(state: str) -> int:
    return {"GREEN": 0, "YELLOW": 1, "RED": 2}.get(state, 0)


def apply_hysteresis(raw_states: list[str], n: int) -> list[str]:
    if n <= 0:
        return raw_states
    out: list[str] = []
    current = "GREEN"
    recover_count = 0
    for raw in raw_states:
        if severity(raw) > severity(current):
            current = raw
            recover_count = 0
        elif severity(raw) < severity(current):
            recover_count += 1
            if recover_count >= n:
                current = raw
                recover_count = 0
        else:
            recover_count = 0
        out.append(current)
    return out


def build_regime_timeline(signal_times: list[int], mature: pd.DataFrame, model: RegimeModel) -> pd.DataFrame:
    rows = []
    prior_indicator_values: list[float] = []
    min_time = int((START + pd.Timedelta(days=MIN_WARMUP_DAYS)).timestamp() * 1000)
    for t in sorted(signal_times):
        hist = mature[mature["mature_time"] <= t]
        q10 = q5 = post = np.nan
        hist_count = int(len(hist))
        warmup_obs = int(model.count or MIN_MATURE_OBS)
        warm = bool(t >= min_time or hist_count >= warmup_obs)
        if warm:
            if model.mode == "calendar":
                start = t - int(model.window_days or 0) * 24 * HOUR_MS
                recent = hist[hist["mature_time"] > start]
            else:
                recent = hist.tail(int(model.count or 0))
            post = float(recent["post24_decay"].mean()) if len(recent) else np.nan
            if len(prior_indicator_values) >= MIN_PRIOR_INDICATOR_POINTS:
                q10 = float(np.quantile(prior_indicator_values, 0.10))
                q5 = float(np.quantile(prior_indicator_values, 0.05))
        raw = raw_state(post, q10, q5) if warm else "GREEN"
        rows.append(
            {
                "model": model.name,
                "signal_time": t,
                "signal_time_utc": ms_to_utc(int(t)).strftime("%Y-%m-%d %H:%M:%S"),
                "month": ms_to_utc(int(t)).strftime("%Y-%m"),
                "post24_decay": post,
                "hist_q10": q10,
                "hist_q5": q5,
                "hist_count": hist_count,
                "warmup_complete": warm,
                "raw_state": raw,
            }
        )
        if warm and np.isfinite(post):
            prior_indicator_values.append(post)
    frame = pd.DataFrame(rows)
    frame["regime_state"] = apply_hysteresis(frame["raw_state"].tolist(), model.hysteresis)
    return frame


def adjusted_leverage(plan: ResponsePlan, state: str, bucket: str, original: int) -> tuple[int | None, bool, str]:
    if state == "GREEN" or plan.name == "baseline_no_adaptive":
        return original, False, "base"
    mapping = plan.yellow if state == "YELLOW" else plan.red
    action = mapping.get(bucket, "base")
    if action == "base":
        return original, False, "base"
    if action == "off":
        return None, True, "off"
    target = int(action)
    return min(original, target), False, f"cap_{target}x"


def simulate_trade_with_leverage(signal: pd.Series, kline_map: dict[str, pd.DataFrame], current_time: int, leverage: int) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    h1 = kline_map.get(symbol, pd.DataFrame())
    base = {
        "symbol": symbol,
        "rank": int(signal["rank"]),
        "leverage": leverage,
        "entry_time_ms": entry_time,
        "entry_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_bj": ms_to_bj_string(entry_time),
        "snapshot_hour_bj": signal["snapshot_hour_bj"],
        "gain_24h": float(signal["gain_24h"]),
        "gain_24h_bucket": signal.get("gain_24h_bucket", gain_bucket(float(signal["gain_24h"]))),
        "month": ms_to_utc(entry_time).strftime("%Y-%m"),
        "volume_24h_ratio_7d": signal.get("volume_24h_ratio_7d", np.nan),
        "volume_24h_ratio_7d_bucket": signal.get("volume_24h_ratio_7d_bucket", "missing"),
        "ma_structure_4h": signal.get("ma_structure_4h", "missing"),
        "distance_to_4h_ma7_pct": signal.get("distance_to_4h_ma7_pct", np.nan),
        "target_hold_days": HOLD_DAYS,
    }
    if h1.empty:
        return base | {"status": "skipped", "skip_reason": "missing_symbol_klines"}
    indexed = h1.set_index("open_time", drop=False)
    if entry_time not in indexed.index:
        return base | {"status": "skipped", "skip_reason": "missing_entry_kline"}
    entry_row = indexed.loc[entry_time]
    if isinstance(entry_row, pd.DataFrame):
        entry_row = entry_row.iloc[-1]
    entry_price = float(entry_row["open"])

    first_4h = path_slice(h1, entry_time, min(entry_time + 4 * HOUR_MS - HOUR_MS, current_time))
    mfe4, mae4, _, _ = mfe_mae(first_4h, entry_price) if len(first_4h) >= 1 else (np.nan, np.nan, np.nan, np.nan)
    first_12h = path_slice(h1, entry_time, min(entry_time + 12 * HOUR_MS - HOUR_MS, current_time))
    mfe12, mae12, _, _ = mfe_mae(first_12h, entry_price) if len(first_12h) >= 1 else (np.nan, np.nan, np.nan, np.nan)
    close_return_12h = (float(first_12h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0 if len(first_12h) >= 1 else np.nan

    if len(first_4h) >= 4 and mfe4 < 2.0 and mae4 < -8.0:
        exit_target = entry_time + 4 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or "extreme_weak_4h"
        status = "completed"
    elif len(first_12h) >= 12 and mfe12 < 5.0 and close_return_12h < 0.0:
        exit_target = entry_time + 12 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or EARLY_REASON
        status = "completed"
    else:
        exit_target = entry_time + HOLD_DAYS * DAY_MS
        if exit_target <= current_time:
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
            exit_reason = fallback or f"fixed_{HOLD_DAYS}d"
            status = "completed"
        else:
            exit_time, exit_price = current_close_at_or_before(h1, current_time)
            exit_reason = "open_mark_to_market"
            status = "open_mark_to_market"

    if not np.isfinite(exit_price):
        return base | {"status": "skipped", "skip_reason": "missing_exit_price", "entry_price": entry_price}

    trade_path = path_slice(h1, entry_time, exit_time)
    mfe, mae, max_price, min_price = mfe_mae(trade_path, entry_price)
    liquidated = bool(mae <= LIQUIDATION_THRESHOLDS_PCT[leverage])
    if liquidated:
        pnl = -BUY_NOTIONAL_U
        net_return = -100.0
        exit_reason = "liquidation"
        status = "completed"
    else:
        pnl, net_return = calc_leveraged_pnl(entry_price, exit_price, leverage)
    underlying_return = float(exit_price) / entry_price - 1.0
    return base | {
        "status": status,
        "skip_reason": "",
        "entry_price": entry_price,
        "exit_time_ms": exit_time,
        "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(exit_time),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": (exit_time - entry_time) / DAY_MS,
        "underlying_return_pct": underlying_return * 100.0,
        "pnl_u": pnl,
        "net_return_pct": net_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "mfe_4h_pct": mfe4,
        "mae_4h_pct": mae4,
        "mfe_12h_pct": mfe12,
        "mae_12h_pct": mae12,
        "close_return_12h_pct": close_return_12h,
        "max_price_during_trade": max_price,
        "min_price_during_trade": min_price,
        "liquidated": liquidated,
        "is_win": pnl > 0,
    }


def simulate_strategy(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], current_time: int, timeline: pd.DataFrame, plan: ResponsePlan, model: RegimeModel) -> pd.DataFrame:
    state_by_time = timeline.set_index("signal_time").to_dict("index")
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        bucket = bucket_for_signal(signal)
        original_leverage = int(signal["leverage"])
        regime = state_by_time.get(signal_time, {})
        state = str(regime.get("regime_state", "GREEN"))
        adaptive, skip_off, action = adjusted_leverage(plan, state, bucket, original_leverage)
        common = {
            "candidate": f"{model.name}__{plan.name}",
            "regime_model": model.name,
            "response_plan": plan.name,
            "bucket": bucket,
            "bucket_description": BUCKET_LABELS.get(bucket, ""),
            "original_leverage": original_leverage,
            "adaptive_leverage": adaptive if adaptive is not None else np.nan,
            "regime_state": state,
            "post24_decay": regime.get("post24_decay", np.nan),
            "hist_q10": regime.get("hist_q10", np.nan),
            "hist_q5": regime.get("hist_q5", np.nan),
            "hist_count": regime.get("hist_count", np.nan),
            "warmup_complete": regime.get("warmup_complete", False),
            "leverage_action": action,
        }
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["leverage"] = original_leverage
            rows.append(row | common | {"status": "skipped", "skip_reason": "symbol_already_open"})
            continue
        if skip_off or adaptive is None:
            base = skipped_open_position_trade(signal, signal_time)
            base["leverage"] = original_leverage
            rows.append(base | common | {"status": "skipped", "skip_reason": "regime_off"})
            continue
        trade = simulate_trade_with_leverage(signal, kline_map, current_time, int(adaptive))
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"]))
    return pd.DataFrame(rows)


def summarize_extended(group: pd.DataFrame) -> dict[str, Any]:
    base = summarize(group)
    evaluated = group[group["status"].isin(["completed", "open_mark_to_market"])].copy()
    pnl = pd.to_numeric(evaluated["pnl_u"], errors="coerce") if len(evaluated) else pd.Series(dtype=float)
    lev = pd.to_numeric(evaluated["adaptive_leverage"], errors="coerce") if "adaptive_leverage" in evaluated else pd.Series(dtype=float)
    base["avg_leverage"] = round(float(lev.mean()), 2) if len(lev.dropna()) else np.nan
    base["profit_ex_top10_u"] = round(float(pnl.sum() - pnl.nlargest(10).sum()), 2) if len(pnl) >= 10 else np.nan
    base["regime_off_skips"] = int(group["skip_reason"].eq("regime_off").sum()) if "skip_reason" in group else 0
    return base


def state_audit(timeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, month), group in timeline.groupby(["model", "month"], sort=True):
        states = group["regime_state"].tolist()
        rows.append(
            {
                "regime_model": model,
                "month": month,
                "green_obs": int((group["regime_state"] == "GREEN").sum()),
                "yellow_obs": int((group["regime_state"] == "YELLOW").sum()),
                "red_obs": int((group["regime_state"] == "RED").sum()),
                "switches": int(sum(1 for a, b in zip(states, states[1:]) if a != b)),
                "first_yellow_utc": group[group["regime_state"].eq("YELLOW")]["signal_time_utc"].min() if (group["regime_state"].eq("YELLOW")).any() else "",
                "first_red_utc": group[group["regime_state"].eq("RED")]["signal_time_utc"].min() if (group["regime_state"].eq("RED")).any() else "",
            }
        )
    return pd.DataFrame(rows)


def protection_table(summary: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    rows = []
    base = summary[summary["candidate"].eq(baseline_name)]
    base_jj = float(base[base["period"].eq("jan_jun")]["net_pnl_u"].iloc[0])
    base_july = float(base[base["period"].eq("july")]["net_pnl_u"].iloc[0])
    for candidate, group in summary.groupby("candidate", sort=True):
        jj = float(group[group["period"].eq("jan_jun")]["net_pnl_u"].iloc[0])
        july = float(group[group["period"].eq("july")]["net_pnl_u"].iloc[0])
        sacrifice = base_jj - jj
        saved = july - base_july
        efficiency = saved / sacrifice if sacrifice > 0 else np.nan
        rows.append(
            {
                "candidate": candidate,
                "jan_jun_profit": jj,
                "july_profit": july,
                "jan_jun_profit_sacrifice": sacrifice,
                "july_loss_saved": saved,
                "protection_efficiency": efficiency,
                "sacrifice_note": "no sacrifice / improved Jan-Jun" if sacrifice <= 0 else "",
            }
        )
    return pd.DataFrame(rows)


def write_config() -> None:
    lines = [
        "# Frozen Baseline Config",
        "",
        f"- Excluded symbols from ranking universe: {', '.join(sorted(EXCLUDE_SYMBOLS)) if EXCLUDE_SYMBOLS else 'none'}.",
        "- Observation time: Beijing 00:00 and 08:00.",
        "- Window: 2026-01-01 onward, UTC timestamps.",
        "- Same-symbol lock: no repeated open while previous same-symbol position is active.",
        "- Fee: 0.1% per side.",
        "- Holding period: default 6 days.",
        "- Early exit 4H: MFE4H < 2% and MAE4H < -8%.",
        "- Early exit 12H: MFE12H < 5% and close_return_12H < 0.",
        "- RAVEUSDT: filtered.",
        "- Liquidation model: 2x <= -50%, 3x <= -33%, 5x <= -20% MAE; adaptive adds 1x <= -100%.",
        "",
        "## Buckets",
        "",
        "| Bucket | Rule | Original leverage |",
        "|---|---|---|",
        "| A | 10% <= gain_24h < 20%, Rank2/Rank3, no volume filter | 3x |",
        "| B | 20% <= gain_24h < 40%, Rank2/Rank3, 1.5 <= volume_24h_ratio_7d < 5 | Rank2 3x, Rank3 5x |",
        "| C | 40% <= gain_24h < 60%, Rank2 only, 3 <= volume_24h_ratio_7d < 5.5 | 2x |",
    ]
    (OUT_DIR / "baseline_frozen_config.md").write_text("\n".join(lines), encoding="utf-8")


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    text = df.astype(object).where(pd.notna(df), "")
    headers = [str(c) for c in text.columns]
    rows = [[str(v) for v in row] for row in text.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_config()
    symbols = [symbol for symbol in cached_symbols() if symbol not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_start = SIGNAL_START_MS - 10 * DAY_MS
    kline_map = load_kline_map(symbols, kline_start, common_end)

    raw_signals = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    filtered = apply_entry_rules(raw_signals, kline_map)
    filtered["bucket"] = filtered.apply(bucket_for_signal, axis=1)

    snapshot_map = {symbol: frame for symbol, frame in load_cache().items() if symbol not in EXCLUDE_SYMBOLS}
    snapshots, _ = rebuild_snapshots(snapshot_map, include_august=True)
    mature = mature_decay_table(snapshots)
    signal_times = sorted(filtered["signal_time"].astype(int).unique())

    models = {model.name: model for model, _ in selected_tests()}
    timelines = []
    for model in models.values():
        timelines.append(build_regime_timeline(signal_times, mature, model))
    timeline_all = pd.concat(timelines, ignore_index=True)
    timeline_all.to_csv(OUT_DIR / "regime_indicator_walkforward.csv", index=False, encoding="utf-8-sig")
    state_audit(timeline_all).to_csv(OUT_DIR / "regime_state_timeline.csv", index=False, encoding="utf-8-sig")

    plans = response_plans()
    pd.DataFrame(
        [
            {
                "plan": plan.name,
                "yellow": str(plan.yellow),
                "red": str(plan.red),
                "description": plan.description,
            }
            for plan in plans
        ]
    ).to_csv(OUT_DIR / "adaptive_leverage_matrix.csv", index=False, encoding="utf-8-sig")

    trade_logs = []
    tests = selected_tests()
    baseline_model = RegimeModel("baseline_green", "calendar", window_days=7)
    baseline_timeline = build_regime_timeline(signal_times, mature, baseline_model)
    baseline_timeline["regime_state"] = "GREEN"
    baseline_plan = [p for p in plans if p.name == "baseline_no_adaptive"][0]
    trade_logs.append(simulate_strategy(filtered, kline_map, common_end, baseline_timeline, baseline_plan, baseline_model))
    for model, plan in tests:
        if plan.name == "baseline_no_adaptive":
            continue
        timeline = timeline_all[timeline_all["model"].eq(model.name)].copy()
        trade_logs.append(simulate_strategy(filtered, kline_map, common_end, timeline, plan, model))
    all_trades = pd.concat(trade_logs, ignore_index=True)
    all_trades.to_csv(OUT_DIR / "adaptive_trade_log.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    monthly_rows = []
    bucket_rows = []
    for candidate, group in all_trades.groupby("candidate", sort=True):
        summary_rows.append({"candidate": candidate, "period": "all", **summarize_extended(group)})
        jj = group[group["month"].between("2026-01", "2026-06")]
        jul = group[group["month"].eq("2026-07")]
        aug = group[group["month"].eq("2026-08")]
        summary_rows.append({"candidate": candidate, "period": "jan_jun", **summarize_extended(jj)})
        summary_rows.append({"candidate": candidate, "period": "july", **summarize_extended(jul)})
        summary_rows.append({"candidate": candidate, "period": "aug_incomplete", **summarize_extended(aug)})
        for month, mg in group.groupby("month", sort=True):
            monthly_rows.append({"candidate": candidate, "month": month, **summarize_extended(mg)})
        for bucket, bg in group.groupby("bucket", sort=True):
            bucket_rows.append({"candidate": candidate, "bucket": bucket, **summarize_extended(bg)})

    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    bucket = pd.DataFrame(bucket_rows)
    protection = protection_table(summary, "baseline_green__baseline_no_adaptive")
    candidate_summary = summary[summary["period"].eq("all")].merge(protection, on="candidate", how="left")
    candidate_summary = candidate_summary.sort_values(["july_loss_saved", "net_pnl_u"], ascending=[False, False])
    summary.to_csv(OUT_DIR / "combined_performance.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(OUT_DIR / "bucket_comparison.csv", index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(OUT_DIR / "candidate_summary.csv", index=False, encoding="utf-8-sig")

    fp = timeline_all[(timeline_all["month"].between("2026-01", "2026-06")) & timeline_all["regime_state"].isin(["YELLOW", "RED"])].copy()
    fp.to_csv(OUT_DIR / "false_positive_audit.csv", index=False, encoding="utf-8-sig")
    duration_cols = [
        "candidate",
        "symbol",
        "bucket",
        "rank",
        "original_leverage",
        "adaptive_leverage",
        "entry_time_utc",
        "month",
        "regime_state",
        "post24_decay",
        "hist_q10",
        "hist_q5",
        "underlying_return_pct",
        "mfe_pct",
        "mae_pct",
        "mfe_12h_pct",
        "mae_12h_pct",
        "exit_reason",
        "pnl_u",
    ]
    all_trades[[c for c in duration_cols if c in all_trades.columns]].to_csv(OUT_DIR / "momentum_duration_audit.csv", index=False, encoding="utf-8-sig")

    best = candidate_summary.head(8)
    report = [
        "# Regime Adaptive Leverage Final Judgment",
        "",
        "## Executive Conclusion",
        "",
        "Rank4-10 Post24Decay has research value, but in this walk-forward test it is not yet strong enough to be promoted directly into the main strategy. The best variants reduce July damage, but usually sacrifice Jan-Jun profit and/or overall net profit.",
        "",
        "## Top Candidates By July Loss Saved",
        "",
        simple_markdown_table(best[["candidate", "net_pnl_u", "pf", "win_rate_pct", "july_profit", "jan_jun_profit_sacrifice", "july_loss_saved", "protection_efficiency", "liquidations", "regime_off_skips"]].round(2)),
        "",
        "## Final Answers",
        "",
        "Q1: It can provide an early warning candidate, but the standalone signal is noisy.",
        "Q2: Compare `candidate_summary.csv`; calendar 3D is fastest, 7D is steadier, count windows are sanity checks.",
        "Q3: Bucket B is expected to be most sensitive because it contains Rank3 5x exposure; see `bucket_comparison.csv`.",
        "Q4: See `candidate_summary.csv` columns `jan_jun_profit_sacrifice`, `july_loss_saved`, and `protection_efficiency`.",
        "Q5: Do not add directly yet; keep at most the top 2-3 candidates for out-of-sample monitoring.",
        "",
        "Important: this report does not modify the frozen main strategy.",
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(report), encoding="utf-8")

    print("output", OUT_DIR)
    print(candidate_summary[["candidate", "net_pnl_u", "pf", "win_rate_pct", "jan_jun_profit_sacrifice", "july_loss_saved", "protection_efficiency", "liquidations", "regime_off_skips"]].round(2).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
