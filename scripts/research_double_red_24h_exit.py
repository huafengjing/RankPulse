from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (  # noqa: E402
    DAY_MS,
    EARLY_REASON,
    HOUR_MS,
    OUT,
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
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, FEE_RATE, generate_signals, latest_signal_end_dt  # noqa: E402
from scripts.bucket_b_rank3_regime_optimization import (  # noqa: E402
    EXCLUDE_SYMBOLS,
    IndicatorSpec,
    build_health_timeline,
    opportunity_sets,
)
from scripts.rank3_fast_recovery_vs_monthly_reset import (  # noqa: E402
    RecoverySpec,
    build_recovery_timeline,
    fast_recovery_action_timeline,
    precompute_outcomes,
    simulate_actions,
)
from scripts.regime_adaptive_leverage_walkforward import (  # noqa: E402
    LIQUIDATION_THRESHOLDS_PCT,
    bucket_for_signal,
    simulate_trade_with_leverage,
)
from scripts.run_current_main_strategy_2026_jan_jun import (  # noqa: E402
    HOLD_DAYS,
    SIGNAL_START_MS,
    apply_entry_rules,
    cache_common_end_ms,
    cached_symbols,
    calc_leveraged_pnl,
    current_close_at_or_before,
)


OUT_DIR = OUT / "double_red_24h_exit"
BASELINE = "baseline_fr3_yr1"
DOUBLE_RED = "double_red_v0"
MODEL_NAME = "FR_avg_return24_l3_gt_0_fr3_yr1"


def entry_anchored_24h_bars(frame: pd.DataFrame, entry_time: int, current_time: int) -> list[dict[str, Any]]:
    bars = []
    indexed = frame.set_index("open_time", drop=False)
    for number in range(1, HOLD_DAYS + 1):
        start = entry_time + (number - 1) * DAY_MS
        end = entry_time + number * DAY_MS
        if end > current_time or start not in indexed.index:
            break
        bar_path = path_slice(frame, start, end - HOUR_MS)
        if len(bar_path) < 24:
            break
        start_row = indexed.loc[start]
        if isinstance(start_row, pd.DataFrame):
            start_row = start_row.iloc[-1]
        open_price = float(start_row["open"])
        close_price = float(bar_path.iloc[-1]["close"])
        bars.append(
            {
                "bar_number": number,
                "bar_start_ms": start,
                "bar_end_ms": end,
                "bar_start_utc": ms_to_utc(start).strftime("%Y-%m-%d %H:%M:%S"),
                "bar_end_utc": ms_to_utc(end).strftime("%Y-%m-%d %H:%M:%S"),
                "open": open_price,
                "close": close_price,
                "red": close_price < open_price,
            }
        )
    return bars


def double_red_trigger(frame: pd.DataFrame, entry_time: int, current_time: int) -> dict[str, Any] | None:
    bars = entry_anchored_24h_bars(frame, entry_time, current_time)
    for idx in range(1, len(bars)):
        prev_bar = bars[idx - 1]
        bar = bars[idx]
        if prev_bar["red"] and bar["red"]:
            return {
                "double_red_exit_time_ms": int(bar["bar_end_ms"]),
                "double_red_trigger_bar": int(bar["bar_number"]),
                "double_red_pair": f"Day{prev_bar['bar_number']}+Day{bar['bar_number']}",
                "first_red_open": float(prev_bar["open"]),
                "second_red_close": float(bar["close"]),
                "two_bar_return_pct": (float(bar["close"]) / float(prev_bar["open"]) - 1.0) * 100.0,
            }
    return None


def simulate_trade_double_red(signal: pd.Series, kline_map: dict[str, pd.DataFrame], current_time: int, leverage: int) -> dict[str, Any]:
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
        "gain_24h_bucket": signal.get("gain_24h_bucket", ""),
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

    trigger = None
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
        trigger = double_red_trigger(h1, entry_time, current_time)
        fixed_exit_target = entry_time + HOLD_DAYS * DAY_MS
        if trigger is not None and int(trigger["double_red_exit_time_ms"]) < fixed_exit_target:
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, int(trigger["double_red_exit_time_ms"]), entry_time)
            exit_reason = fallback or "double_red_24h"
            status = "completed"
        elif fixed_exit_target <= current_time:
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, fixed_exit_target, entry_time)
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
        "underlying_return_pct": (float(exit_price) / entry_price - 1.0) * 100.0,
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
        **(trigger or {}),
    }


def precompute_double_red_outcomes(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], common_end: int) -> dict[tuple[int, int], dict[str, Any]]:
    outcomes: dict[tuple[int, int], dict[str, Any]] = {}
    for row in signals.itertuples(index=False):
        signal = pd.Series(row._asdict())
        sid = int(signal["signal_id"])
        for lev in [1, 2, 3, 5]:
            outcomes[(sid, lev)] = simulate_trade_double_red(signal, kline_map, common_end, lev)
    return outcomes


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    evaluated = group[group["status"].isin(["completed", "open_mark_to_market"])].sort_values("entry_time_ms").copy()
    pnl = pd.to_numeric(evaluated.get("pnl_u", pd.Series(dtype=float)), errors="coerce")
    ret = pd.to_numeric(evaluated.get("net_return_pct", pd.Series(dtype=float)), errors="coerce")
    hold = pd.to_numeric(evaluated.get("holding_days", pd.Series(dtype=float)), errors="coerce")
    fees = len(evaluated) * BUY_NOTIONAL_U * 0.0
    if "entry_price" in evaluated and "exit_price" in evaluated and "leverage" in evaluated:
        fees = 0.0
        for row in evaluated.itertuples(index=False):
            if pd.notna(row.entry_price) and pd.notna(row.exit_price):
                nominal = BUY_NOTIONAL_U * int(row.leverage)
                qty = nominal * (1.0 - FEE_RATE) / float(row.entry_price)
                fees += nominal * FEE_RATE + qty * float(row.exit_price) * FEE_RATE
    return {
        "signals": int(len(group)),
        "trade_count": int(len(evaluated)),
        "closed": int(evaluated["status"].eq("completed").sum()) if len(evaluated) else 0,
        "open_mtm": int(evaluated["status"].eq("open_mark_to_market").sum()) if len(evaluated) else 0,
        "skipped": int(group["status"].eq("skipped").sum()) if "status" in group else 0,
        "total_pnl_u": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "pf": round(float(profit_factor(pnl)), 2) if len(pnl) else 0.0,
        "win_rate_pct": round(float((pnl > 0).sum() / len(evaluated) * 100), 2) if len(evaluated) else np.nan,
        "avg_pnl_u": round(float(pnl.mean()), 2) if len(pnl) else np.nan,
        "median_pnl_u": round(float(pnl.median()), 2) if len(pnl) else np.nan,
        "avg_return_pct": round(float(ret.mean()), 2) if len(ret) else np.nan,
        "median_return_pct": round(float(ret.median()), 2) if len(ret) else np.nan,
        "max_dd_u": round(max_drawdown(pnl), 2),
        "liquidations": int(evaluated["liquidated"].fillna(False).sum()) if "liquidated" in evaluated else 0,
        "avg_hold_days": round(float(hold.mean()), 2) if len(hold) else np.nan,
        "median_hold_days": round(float(hold.median()), 2) if len(hold) else np.nan,
        "fees_u": round(float(fees), 2),
        "total_return_pct": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "ex_top1_u": round(float(pnl.sum() - pnl.nlargest(1).sum()), 2) if len(pnl) >= 1 else np.nan,
        "ex_top3_u": round(float(pnl.sum() - pnl.nlargest(3).sum()), 2) if len(pnl) >= 3 else np.nan,
        "ex_top5_u": round(float(pnl.sum() - pnl.nlargest(5).sum()), 2) if len(pnl) >= 5 else np.nan,
        "ex_top10_u": round(float(pnl.sum() - pnl.nlargest(10).sum()), 2) if len(pnl) >= 10 else np.nan,
    }


def trade_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["status"].isin(["completed", "open_mark_to_market"])].copy()
    out["trade_key"] = out["entry_time_ms"].astype("int64").astype(str) + "|" + out["symbol"].astype(str) + "|" + out["rank"].astype("int64").astype(str)
    return out


def attach_signal_id(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    keys = ["entry_time_ms", "symbol", "rank"]
    lookup = signals[["signal_id", "signal_time", "symbol", "rank"]].rename(columns={"signal_time": "entry_time_ms"})
    return trades.merge(lookup, on=keys, how="left")


def comparison_rows(baseline: pd.DataFrame, double_red: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    if group_col is None:
        rows.append({"group": "ALL", "strategy": BASELINE, **summarize(baseline)})
        rows.append({"group": "ALL", "strategy": DOUBLE_RED, **summarize(double_red)})
    else:
        keys = sorted(set(baseline[group_col].dropna().astype(str)) | set(double_red[group_col].dropna().astype(str)))
        for key in keys:
            rows.append({"group": key, "strategy": BASELINE, **summarize(baseline[baseline[group_col].astype(str).eq(key)])})
            rows.append({"group": key, "strategy": DOUBLE_RED, **summarize(double_red[double_red[group_col].astype(str).eq(key)])})
    return pd.DataFrame(rows)


def add_delta_table(comp: pd.DataFrame, key_col: str = "group") -> pd.DataFrame:
    metrics = [c for c in comp.columns if c not in {key_col, "strategy"}]
    rows = []
    for key, g in comp.groupby(key_col, sort=True):
        b = g[g["strategy"].eq(BASELINE)]
        d = g[g["strategy"].eq(DOUBLE_RED)]
        if b.empty or d.empty:
            continue
        row = {key_col: key}
        for m in metrics:
            row[f"{m}_baseline"] = b.iloc[0][m]
            row[f"{m}_double_red"] = d.iloc[0][m]
            if pd.api.types.is_number(row[f"{m}_baseline"]) and pd.api.types.is_number(row[f"{m}_double_red"]):
                row[f"{m}_delta"] = row[f"{m}_double_red"] - row[f"{m}_baseline"]
        rows.append(row)
    return pd.DataFrame(rows)


def post_exit_path(events: pd.DataFrame, kline_map: dict[str, pd.DataFrame], baseline_map: dict[tuple[int, int], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in events.itertuples(index=False):
        h1 = kline_map.get(row.symbol, pd.DataFrame())
        if h1.empty:
            continue
        baseline = baseline_map.get((int(row.signal_id), int(row.adaptive_leverage)), {})
        exit_time = int(row.exit_time_ms)
        entry_price = float(row.entry_price)
        for horizon, hours in [("plus_24h", 24), ("plus_48h", 48)]:
            target = exit_time + hours * HOUR_MS
            _, price, _ = get_open_at_or_latest(h1, target, int(row.entry_time_ms))
            rows.append(
                {
                    "symbol": row.symbol,
                    "entry_time_utc": row.entry_time_utc,
                    "exit_time_utc": row.exit_time_utc,
                    "horizon": horizon,
                    "return_from_entry_pct": (price / entry_price - 1.0) * 100.0 if np.isfinite(price) else np.nan,
                }
            )
        baseline_exit = int(baseline.get("exit_time_ms", row.exit_time_ms))
        post = path_slice(h1, exit_time, baseline_exit)
        mfe, mae, _, _ = mfe_mae(post, entry_price) if len(post) else (np.nan, np.nan, np.nan, np.nan)
        rows.append(
            {
                "symbol": row.symbol,
                "entry_time_utc": row.entry_time_utc,
                "exit_time_utc": row.exit_time_utc,
                "horizon": "to_baseline_exit",
                "return_from_entry_pct": baseline.get("net_return_pct", np.nan),
                "post_exit_mfe_pct": mfe,
                "post_exit_mae_pct": mae,
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
    d15 = build_health_timeline(signal_times, sets["B_R3"], IndicatorSpec("D_b_r3_decay_l15", "B_R3", "mean_decay48", "lower_bad", 15))
    recovery = build_recovery_timeline(signal_times, sets["B_R3"], RecoverySpec("avg_return24_l3_gt_0", "avg_return24", 3, "gt_0"))
    action = fast_recovery_action_timeline(d15, recovery, "fr3", "yr1", MODEL_NAME)

    baseline_outcomes = precompute_outcomes(filtered, kline_map, common_end)
    double_red_outcomes = precompute_double_red_outcomes(filtered, kline_map, common_end)
    baseline = attach_signal_id(simulate_actions(filtered, baseline_outcomes, action, BASELINE), filtered)
    double_red = attach_signal_id(simulate_actions(filtered, double_red_outcomes, action, DOUBLE_RED), filtered)

    baseline_keys = trade_key_frame(baseline)
    double_keys = trade_key_frame(double_red)
    baseline_key_map = {row.trade_key: row for row in baseline_keys.itertuples(index=False)}
    event_rows = []
    for row in double_keys[double_keys["exit_reason"].eq("double_red_24h")].itertuples(index=False):
        cf = baseline_outcomes.get((int(row.signal_id), int(row.adaptive_leverage)), {})
        cf_pnl = float(cf.get("pnl_u", np.nan))
        event_rows.append(
            {
                **row._asdict(),
                "baseline_exit_time_utc": cf.get("exit_time_utc", ""),
                "baseline_exit_reason": cf.get("exit_reason", ""),
                "baseline_pnl_u": cf_pnl,
                "baseline_return_pct": cf.get("net_return_pct", np.nan),
                "exit_value_added_u": float(row.pnl_u) - cf_pnl if np.isfinite(cf_pnl) else np.nan,
                "true_exit": bool(float(row.pnl_u) > cf_pnl) if np.isfinite(cf_pnl) else False,
                "false_exit": bool(float(row.pnl_u) < cf_pnl) if np.isfinite(cf_pnl) else False,
                "prior_mfe_bucket": pd.cut([float(row.mfe_pct)], [-np.inf, 5, 15, 30, 50, np.inf], labels=["<5", "5-15", "15-30", "30-50", ">=50"])[0],
            }
        )
    events = pd.DataFrame(event_rows)

    base_summary = pd.DataFrame(
        [
            {"strategy": BASELINE, "cutoff_utc": ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"), **summarize(baseline)},
            {"strategy": DOUBLE_RED, "cutoff_utc": ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"), **summarize(double_red)},
        ]
    )
    monthly = add_delta_table(comparison_rows(baseline, double_red, "month"), "group").rename(columns={"group": "month"})
    exit_reason = pd.concat(
        [
            comparison_rows(baseline[baseline["status"].isin(["completed", "open_mark_to_market"])], double_red[double_red["status"].isin(["completed", "open_mark_to_market"])], "exit_reason")
        ],
        ignore_index=True,
    )
    bucket_comp = add_delta_table(comparison_rows(baseline, double_red, "bucket"), "group").rename(columns={"group": "bucket"})
    rank_comp = add_delta_table(comparison_rows(baseline, double_red, "rank"), "group").rename(columns={"group": "rank"})
    regime_comp = add_delta_table(comparison_rows(baseline, double_red, "regime_state"), "group").rename(columns={"group": "regime_state"})
    hour_comp = add_delta_table(comparison_rows(baseline, double_red, "snapshot_hour_bj"), "group").rename(columns={"group": "snapshot_hour_bj"})
    trigger_day = events.groupby("double_red_pair").agg(
        count=("symbol", "count"),
        total_value_added_u=("exit_value_added_u", "sum"),
        mean_value_added_u=("exit_value_added_u", "mean"),
        median_value_added_u=("exit_value_added_u", "median"),
        false_exit_rate=("false_exit", "mean"),
    ).reset_index() if len(events) else pd.DataFrame()

    counter = events[
        [
            "symbol",
            "entry_time_utc",
            "rank",
            "bucket",
            "adaptive_leverage",
            "regime_state",
            "exit_time_utc",
            "pnl_u",
            "net_return_pct",
            "baseline_exit_time_utc",
            "baseline_exit_reason",
            "baseline_pnl_u",
            "baseline_return_pct",
            "exit_value_added_u",
            "true_exit",
            "false_exit",
            "mfe_pct",
            "mae_pct",
            "double_red_pair",
            "two_bar_return_pct",
        ]
    ].copy() if len(events) else pd.DataFrame()

    winner_rows = []
    for threshold in [10, 20, 30, 50, 100]:
        b = baseline_keys[pd.to_numeric(baseline_keys["mfe_pct"], errors="coerce").ge(threshold)].copy()
        dr_keys = set(events["trade_key"]) if len(events) else set()
        winner_rows.append(
            {
                "mfe_threshold_pct": threshold,
                "baseline_trade_count": len(b),
                "double_red_triggered_count": int(b["trade_key"].isin(dr_keys).sum()) if len(b) else 0,
                "baseline_final_pnl_u": float(pd.to_numeric(b["pnl_u"], errors="coerce").sum()) if len(b) else 0.0,
            }
        )
    winner_giveback = pd.DataFrame(winner_rows)

    top_base = baseline_keys.sort_values("pnl_u", ascending=False).head(20)
    top_damage = top_base.merge(
        events[["trade_key", "pnl_u", "exit_value_added_u", "exit_time_utc"]].rename(
            columns={"pnl_u": "double_red_pnl_u", "exit_time_utc": "double_red_exit_time_utc"}
        ),
        on="trade_key",
        how="left",
    )
    top_damage["cut_by_double_red"] = top_damage["double_red_pnl_u"].notna()
    top_damage["lost_profit_u"] = -pd.to_numeric(top_damage["exit_value_added_u"], errors="coerce")

    post = post_exit_path(events, kline_map, baseline_outcomes) if len(events) else pd.DataFrame()
    phase2 = pd.DataFrame(
        [
            {"candidate": "Double Red + giveback 10/20/30pct", "allowed_next": bool(len(events) and events["false_exit"].mean() > 0 and events["exit_value_added_u"].sum() > 0)},
            {"candidate": "Only after prior MFE 10/20/30pct", "allowed_next": bool(len(events) and events["false_exit"].mean() > 0)},
            {"candidate": "Regime conditional YELLOW/RED", "allowed_next": bool(len(events) and len(regime_comp))},
            {"candidate": "Bucket conditional", "allowed_next": bool(len(events) and len(bucket_comp))},
        ]
    )

    base_summary.to_csv(OUT_DIR / "baseline_vs_double_red.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "double_red_events.csv", index=False, encoding="utf-8-sig")
    counter.to_csv(OUT_DIR / "double_red_counterfactual.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    bucket_comp.to_csv(OUT_DIR / "bucket_comparison.csv", index=False, encoding="utf-8-sig")
    rank_comp.to_csv(OUT_DIR / "rank_comparison.csv", index=False, encoding="utf-8-sig")
    regime_comp.to_csv(OUT_DIR / "regime_comparison.csv", index=False, encoding="utf-8-sig")
    hour_comp.to_csv(OUT_DIR / "entry_hour_comparison.csv", index=False, encoding="utf-8-sig")
    trigger_day.to_csv(OUT_DIR / "trigger_day_comparison.csv", index=False, encoding="utf-8-sig")
    winner_giveback.to_csv(OUT_DIR / "winner_giveback_analysis.csv", index=False, encoding="utf-8-sig")
    top_damage.to_csv(OUT_DIR / "top_winner_damage.csv", index=False, encoding="utf-8-sig")
    post.to_csv(OUT_DIR / "post_exit_path.csv", index=False, encoding="utf-8-sig")
    phase2.to_csv(OUT_DIR / "phase2_candidates.csv", index=False, encoding="utf-8-sig")

    event_value = float(events["exit_value_added_u"].sum()) if len(events) else 0.0
    false_cost = abs(float(events.loc[events["false_exit"], "exit_value_added_u"].sum())) if len(events) else 0.0
    true_benefit = float(events.loc[events["true_exit"], "exit_value_added_u"].sum()) if len(events) else 0.0
    judgment = "KEEP BASELINE"
    if len(events) and event_value > 0 and float(base_summary.iloc[1]["total_pnl_u"]) >= float(base_summary.iloc[0]["total_pnl_u"]) and false_cost <= true_benefit:
        judgment = "DOUBLE RED v0 SHADOW"
    elif len(events) and event_value > 0:
        judgment = "CONDITIONAL DOUBLE RED SHADOW"

    lines = [
        "# Double Red 24H Exit v0 Judgment",
        "",
        f"- Cutoff: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "- Baseline: current main strategy, FR3/YR1, Rank3 B volume floor 1.2, 4H/12H exits, 6D default.",
        "- Test: add plain entry-anchored consecutive two 24H red bars; exit at the second bar end using the existing open-at-target execution path.",
        "- Fees: 0.1% per side; no extra slippage; 100U margin per trade.",
        "",
        "## Core",
        "",
        base_summary.to_string(index=False),
        "",
        "## Double Red Event Value",
        "",
        f"- Events: {len(events)}",
        f"- True exits: {int(events['true_exit'].sum()) if len(events) else 0}",
        f"- False exits: {int(events['false_exit'].sum()) if len(events) else 0}",
        f"- True exit benefit: {true_benefit:.2f}U",
        f"- False exit opportunity cost: {false_cost:.2f}U",
        f"- Net exit value: {event_value:.2f}U",
        "",
        "## Final Recommendation",
        "",
        judgment,
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")

    print(base_summary.to_string(index=False))
    print()
    print(f"events={len(events)} true_benefit={true_benefit:.2f} false_cost={false_cost:.2f} net_event_value={event_value:.2f}")
    print("judgment", judgment)
    print("output", OUT_DIR)


if __name__ == "__main__":
    main()
