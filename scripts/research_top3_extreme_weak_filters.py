from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (
    CACHE_DIR,
    DAY_MS,
    EARLY_REASON,
    FEE_RATE,
    HOUR_MS,
    OUT,
    calc_pnl,
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
from scripts.test_main_strategy_hold_days_leverage import apply_current_main_filters


PREFIX = "top3_extreme_weak_filter_research"
LIQ_THRESHOLD = {2: -50.0, 3: -33.0, 5: -20.0}
WINDOWS = [4, 8]


@dataclass(frozen=True)
class WeakRule:
    rule_id: str
    rule_desc: str
    window_h: int
    rule_type: str
    fn: Callable[[pd.Series], bool]


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def add_gain_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gain_bucket"] = pd.cut(
        out["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%", ">=80%"],
        right=False,
    )
    return out


def leverage_for_current_main(row: pd.Series) -> int:
    gain = float(row["gain_24h"])
    rank = int(row["rank"])
    if 0.10 <= gain < 0.20:
        return 3
    if 0.20 <= gain < 0.40:
        return 3 if rank == 2 else 5
    if 0.40 <= gain < 0.60:
        return 2
    raise ValueError(f"unexpected traded row: gain={gain}, rank={rank}, symbol={row.get('symbol')}")


def leveraged_pnl(row: pd.Series, leverage: int) -> tuple[float, float, bool]:
    mae = float(row["mae_pct"])
    if leverage > 1 and mae <= LIQ_THRESHOLD[leverage]:
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def prepare_paths(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        entry_time = int(signal["signal_time"])
        h1 = kline_map.get(symbol, pd.DataFrame())
        base = {
            **signal.to_dict(),
            "entry_time_ms": entry_time,
            "entry_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
            "entry_time_bj": ms_to_bj_string(entry_time),
            "month": ms_to_bj_string(entry_time)[:7],
            "path_status": "ok",
            "path_skip_reason": "",
            "entry_price": np.nan,
        }
        if h1.empty:
            rows.append(base | {"path_status": "skipped", "path_skip_reason": "missing_symbol_klines"})
            continue
        indexed = h1.set_index("open_time", drop=False)
        if entry_time not in indexed.index:
            rows.append(base | {"path_status": "skipped", "path_skip_reason": "missing_entry_kline"})
            continue
        entry_row = indexed.loc[entry_time]
        if isinstance(entry_row, pd.DataFrame):
            entry_row = entry_row.iloc[-1]
        entry_price = float(entry_row["open"])
        base["entry_price"] = entry_price

        for window in [4, 8, 12]:
            segment = path_slice(h1, entry_time, entry_time + window * HOUR_MS - HOUR_MS)
            if segment.empty:
                base[f"w{window}_available"] = False
                base[f"w{window}_mfe"] = np.nan
                base[f"w{window}_mae"] = np.nan
                base[f"w{window}_close_return"] = np.nan
                base[f"w{window}_range"] = np.nan
                base[f"w{window}_score"] = np.nan
                base[f"w{window}_exit_time_ms"] = np.nan
                base[f"w{window}_exit_price"] = np.nan
                continue
            mfe, mae, _, _ = mfe_mae(segment, entry_price)
            close_return = (float(segment.iloc[-1]["close"]) / entry_price - 1.0) * 100.0
            exit_time, exit_price, _ = get_open_at_or_latest(h1, entry_time + window * HOUR_MS, entry_time)
            base[f"w{window}_available"] = bool(np.isfinite(exit_price))
            base[f"w{window}_mfe"] = float(mfe)
            base[f"w{window}_mae"] = float(mae)
            base[f"w{window}_close_return"] = float(close_return)
            base[f"w{window}_range"] = float(mfe - mae)
            base[f"w{window}_score"] = float(mfe + close_return)
            base[f"w{window}_exit_time_ms"] = int(exit_time)
            base[f"w{window}_exit_price"] = float(exit_price) if np.isfinite(exit_price) else np.nan

        exit_6d_time, exit_6d_price, fallback = get_open_at_or_latest(h1, entry_time + 6 * DAY_MS, entry_time)
        if not np.isfinite(exit_6d_price):
            rows.append(base | {"path_status": "skipped", "path_skip_reason": "missing_6d_exit_price"})
            continue
        base["exit_6d_time_ms"] = int(exit_6d_time)
        base["exit_6d_price"] = float(exit_6d_price)
        base["exit_6d_reason"] = fallback or "fixed_6d"
        rows.append(base)
    return add_gain_bucket(pd.DataFrame(rows))


def make_rules() -> list[WeakRule]:
    rules: list[WeakRule] = []
    for window in WINDOWS:
        prefix = f"W{window}"

        for threshold in [1.0, 2.0, 3.0]:
            rules.append(
                WeakRule(
                    f"{prefix}_MFE_lt_{threshold:g}",
                    f"{window}H MFE < {threshold:g}%",
                    window,
                    "single_mfe",
                    lambda row, w=window, t=threshold: float(row[f"w{w}_mfe"]) < t,
                )
            )

        for threshold in [0.0, -3.0, -5.0]:
            rules.append(
                WeakRule(
                    f"{prefix}_Close_lt_{threshold:g}",
                    f"{window}H close_return < {threshold:g}%",
                    window,
                    "single_close",
                    lambda row, w=window, t=threshold: float(row[f"w{w}_close_return"]) < t,
                )
            )

        for threshold in [-8.0, -10.0, -15.0]:
            rules.append(
                WeakRule(
                    f"{prefix}_MAE_lt_{threshold:g}",
                    f"{window}H MAE < {threshold:g}%",
                    window,
                    "single_mae",
                    lambda row, w=window, t=threshold: float(row[f"w{w}_mae"]) < t,
                )
            )

        for threshold in [8.0, 10.0, 15.0]:
            rules.append(
                WeakRule(
                    f"{prefix}_Range_gt_{threshold:g}",
                    f"{window}H range = MFE-MAE > {threshold:g}%",
                    window,
                    "single_range",
                    lambda row, w=window, t=threshold: float(row[f"w{w}_range"]) > t,
                )
            )

        for threshold in [0.0, -3.0, -5.0]:
            rules.append(
                WeakRule(
                    f"{prefix}_Score_lt_{threshold:g}",
                    f"{window}H score = MFE+close_return < {threshold:g}%",
                    window,
                    "single_score",
                    lambda row, w=window, t=threshold: float(row[f"w{w}_score"]) < t,
                )
            )

        combo_specs = [
            ("MFE2_Close-3", f"{window}H MFE < 2% and close_return < -3%", lambda row, w=window: float(row[f"w{w}_mfe"]) < 2 and float(row[f"w{w}_close_return"]) < -3),
            ("MFE1_Close0", f"{window}H MFE < 1% and close_return < 0%", lambda row, w=window: float(row[f"w{w}_mfe"]) < 1 and float(row[f"w{w}_close_return"]) < 0),
            ("MFE2_MAE-8", f"{window}H MFE < 2% and MAE < -8%", lambda row, w=window: float(row[f"w{w}_mfe"]) < 2 and float(row[f"w{w}_mae"]) < -8),
            ("MFE1_MAE-10", f"{window}H MFE < 1% and MAE < -10%", lambda row, w=window: float(row[f"w{w}_mfe"]) < 1 and float(row[f"w{w}_mae"]) < -10),
            ("MFE2_Close0_MAE-8", f"{window}H MFE < 2% and close_return < 0% and MAE < -8%", lambda row, w=window: float(row[f"w{w}_mfe"]) < 2 and float(row[f"w{w}_close_return"]) < 0 and float(row[f"w{w}_mae"]) < -8),
        ]
        for suffix, desc, fn in combo_specs:
            rules.append(WeakRule(f"{prefix}_{suffix}", desc, window, "combo", fn))

    return rules


def baseline_rule() -> WeakRule:
    return WeakRule("BASE_12H_MFE5_Close0", "No 4H/8H filter; current 12H MFE<5 and close<0", 0, "baseline", lambda row: False)


def has_12h_exit(row: pd.Series) -> bool:
    return bool(row.get("w12_available", False)) and float(row["w12_mfe"]) < 5.0 and float(row["w12_close_return"]) < 0.0


def simulate_rule(prepared: pd.DataFrame, kline_map: dict[str, pd.DataFrame], rule: WeakRule) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = prepared.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        symbol = str(row["symbol"])
        signal_time = int(row["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            skipped = skipped_open_position_trade(row, open_until)
            rows.append(skipped | {"rule_id": rule.rule_id, "rule_desc": rule.rule_desc, "window_h": rule.window_h, "rule_type": rule.rule_type})
            continue

        base = {
            "rule_id": rule.rule_id,
            "rule_desc": rule.rule_desc,
            "window_h": rule.window_h,
            "rule_type": rule.rule_type,
            "symbol": symbol,
            "rank": int(row["rank"]),
            "entry_time_ms": signal_time,
            "entry_time_utc": row["entry_time_utc"],
            "entry_time_bj": row["entry_time_bj"],
            "snapshot_hour_bj": row["snapshot_hour_bj"],
            "gain_24h": float(row["gain_24h"]),
            "gain_bucket": row.get("gain_bucket", ""),
            "month": row["month"],
            "volume_24h_ratio_7d": row.get("volume_24h_ratio_7d", np.nan),
            "volume_24h_ratio_7d_bucket": row.get("volume_24h_ratio_7d_bucket", ""),
            "entry_price": float(row["entry_price"]) if np.isfinite(row.get("entry_price", np.nan)) else np.nan,
        }
        if row.get("path_status") != "ok":
            rows.append(base | {"status": "skipped", "skip_reason": row.get("path_skip_reason", "path_not_ok")})
            continue

        weak_triggered = False
        if rule.window_h in WINDOWS and bool(row.get(f"w{rule.window_h}_available", False)):
            weak_triggered = bool(rule.fn(row))

        if weak_triggered:
            exit_time = int(row[f"w{rule.window_h}_exit_time_ms"])
            exit_price = float(row[f"w{rule.window_h}_exit_price"])
            exit_reason = f"extreme_weak_exit_{rule.window_h}h"
        elif has_12h_exit(row):
            exit_time = int(row["w12_exit_time_ms"])
            exit_price = float(row["w12_exit_price"])
            exit_reason = EARLY_REASON
        else:
            exit_time = int(row["exit_6d_time_ms"])
            exit_price = float(row["exit_6d_price"])
            exit_reason = row["exit_6d_reason"]

        if not np.isfinite(exit_price):
            rows.append(base | {"status": "skipped", "skip_reason": "missing_exit_price"})
            continue

        pnl_u, net_return_pct = calc_pnl(float(row["entry_price"]), exit_price)
        h1 = kline_map.get(symbol, pd.DataFrame())
        trade_path = path_slice(h1, signal_time, exit_time)
        mfe, mae, max_price, min_price = mfe_mae(trade_path, float(row["entry_price"]))
        rows.append(
            base
            | {
                "status": "completed",
                "skip_reason": "",
                "exit_time_ms": exit_time,
                "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time_bj": ms_to_bj_string(exit_time),
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "extreme_weak_triggered": weak_triggered,
                "holding_days": (exit_time - signal_time) / DAY_MS,
                "pnl_u": pnl_u,
                "net_return_pct": net_return_pct,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "max_price_during_trade": max_price,
                "min_price_during_trade": min_price,
                "w4_mfe": row.get("w4_mfe", np.nan),
                "w4_mae": row.get("w4_mae", np.nan),
                "w4_close_return": row.get("w4_close_return", np.nan),
                "w4_range": row.get("w4_range", np.nan),
                "w4_score": row.get("w4_score", np.nan),
                "w8_mfe": row.get("w8_mfe", np.nan),
                "w8_mae": row.get("w8_mae", np.nan),
                "w8_close_return": row.get("w8_close_return", np.nan),
                "w8_range": row.get("w8_range", np.nan),
                "w8_score": row.get("w8_score", np.nan),
                "w12_mfe": row.get("w12_mfe", np.nan),
                "w12_close_return": row.get("w12_close_return", np.nan),
            }
        )
        open_until_by_symbol[symbol] = exit_time
    return pd.DataFrame(rows)


def apply_leverage(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["leverage"] = np.nan
    out["leveraged_pnl_u"] = np.nan
    out["leveraged_return_pct"] = np.nan
    out["liquidated"] = False
    for idx in out.index[out["status"].eq("completed")]:
        row = out.loc[idx]
        leverage = leverage_for_current_main(row)
        pnl, ret, liq = leveraged_pnl(row, leverage)
        out.loc[idx, "leverage"] = leverage
        out.loc[idx, "leveraged_pnl_u"] = pnl
        out.loc[idx, "leveraged_return_pct"] = ret
        out.loc[idx, "liquidated"] = liq
    return out


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms")
    pnl = completed["leveraged_pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed["leveraged_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    weak = int(completed["extreme_weak_triggered"].fillna(False).sum()) if "extreme_weak_triggered" in completed else 0
    liq = int(completed["liquidated"].fillna(False).sum()) if "liquidated" in completed else 0
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "extreme_weak_count": weak,
        "extreme_weak_rate_pct": round(weak / len(completed) * 100, 2) if len(completed) else np.nan,
        "early_12h": int(completed["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in completed else 0,
        "liquidations": liq,
        "liquidation_rate_pct": round(liq / len(completed) * 100, 2) if len(completed) else np.nan,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "gross_profit_u": round(float(pnl[pnl > 0].sum()), 2) if len(pnl) else 0.0,
        "gross_loss_u": round(float(pnl[pnl < 0].sum()), 2) if len(pnl) else 0.0,
        "net_pnl_u": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "pf": round(float(profit_factor(pnl)), 2),
        "win_rate_pct": round(float(len(wins) / len(completed) * 100), 2) if len(completed) else np.nan,
        "avg_return_pct": round(float(ret.mean()), 2) if len(ret) else np.nan,
        "median_return_pct": round(float(ret.median()), 2) if len(ret) else np.nan,
        "max_drawdown_u": round(max_drawdown(pnl), 2),
        "best_trade_u": round(float(pnl.max()), 2) if len(pnl) else np.nan,
        "worst_trade_u": round(float(pnl.min()), 2) if len(pnl) else np.nan,
        "drop_top1_u": round(float(pnl.sum() - pnl.nlargest(1).sum()), 2) if len(pnl) >= 1 else np.nan,
        "drop_top3_u": round(float(pnl.sum() - pnl.nlargest(3).sum()), 2) if len(pnl) >= 3 else np.nan,
        "drop_top5_u": round(float(pnl.sum() - pnl.nlargest(5).sum()), 2) if len(pnl) >= 5 else np.nan,
    }


def monthly_stability(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, group in monthly.groupby("rule_id", sort=False):
        rows.append(
            {
                "rule_id": rid,
                "profitable_months": int((group["net_pnl_u"] > 0).sum()),
                "losing_months": int((group["net_pnl_u"] < 0).sum()),
                "worst_month_pnl_u": round(float(group["net_pnl_u"].min()), 2) if len(group) else np.nan,
                "best_month_pnl_u": round(float(group["net_pnl_u"].max()), 2) if len(group) else np.nan,
                "monthly_net_std": round(float(group["net_pnl_u"].std(ddof=0)), 2) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end_dt = latest_signal_end_dt()
    signal_end = int(end_dt.timestamp() * 1000)
    signal_start = signal_end - 180 * DAY_MS
    symbols = cached_symbols()
    kline_map = load_kline_map(symbols, signal_start - 7 * DAY_MS, signal_end + 8 * DAY_MS)
    signals = apply_current_main_filters(generate_signals(signal_start, signal_end, kline_map), kline_map)
    prepared = prepare_paths(signals, kline_map)

    rules = [baseline_rule()] + make_rules()
    summary_rows = []
    monthly_rows = []
    selected_trade_frames = []
    for index, rule in enumerate(rules, start=1):
        trades = apply_leverage(simulate_rule(prepared, kline_map, rule))
        row = {"rule_id": rule.rule_id, "rule_desc": rule.rule_desc, "window_h": rule.window_h, "rule_type": rule.rule_type, **summarize(trades)}
        summary_rows.append(row)
        for month, group in trades.groupby("month", sort=True):
            monthly_rows.append({"rule_id": rule.rule_id, "rule_desc": rule.rule_desc, "month": month, **summarize(group)})
        if rule.rule_id == "BASE_12H_MFE5_Close0":
            selected_trade_frames.append(trades.assign(selected_label="baseline"))
        if index % 10 == 0:
            print(f"processed {index}/{len(rules)}", flush=True)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    summary = summary.merge(monthly_stability(monthly), on="rule_id", how="left")
    baseline = summary[summary["rule_id"].eq("BASE_12H_MFE5_Close0")].iloc[0]
    for col in ["net_pnl_u", "pf", "win_rate_pct", "max_drawdown_u", "liquidations", "drop_top1_u", "drop_top3_u", "drop_top5_u"]:
        summary[f"delta_{col}"] = summary[col] - baseline[col]

    candidate = summary[
        (summary["rule_id"].ne("BASE_12H_MFE5_Close0"))
        & (summary["extreme_weak_rate_pct"].between(5, 20, inclusive="both"))
        & (summary["net_pnl_u"] >= float(baseline["net_pnl_u"]) * 0.90)
    ].copy()
    candidate["score"] = (
        candidate["drop_top5_u"].rank(ascending=True) * 1.5
        + candidate["max_drawdown_u"].rank(ascending=True)
        + candidate["pf"].rank(ascending=True)
        + candidate["liquidations"].rank(ascending=False)
        + candidate["net_pnl_u"].rank(ascending=True) * 0.5
        - (candidate["extreme_weak_rate_pct"] - 12).abs() * 0.2
    )
    shortlist = candidate.sort_values("score", ascending=False)

    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    shortlist.to_csv(OUT / f"{PREFIX}_shortlist.csv", index=False, encoding="utf-8-sig")
    prepared.to_csv(OUT / f"{PREFIX}_prepared_signals.csv", index=False, encoding="utf-8-sig")

    print("========== Top3 Extreme Weak Filter Research ==========")
    print(f"Signal end UTC: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Signal end BJ:  {ms_to_bj_string(signal_end)}")
    print(f"Rules tested: {len(summary) - 1} + baseline")
    print()
    cols = [
        "rule_id",
        "rule_desc",
        "trades",
        "extreme_weak_count",
        "extreme_weak_rate_pct",
        "early_12h",
        "liquidations",
        "net_pnl_u",
        "pf",
        "win_rate_pct",
        "median_return_pct",
        "max_drawdown_u",
        "drop_top1_u",
        "drop_top3_u",
        "drop_top5_u",
        "profitable_months",
        "losing_months",
        "worst_month_pnl_u",
    ]
    print("========== Baseline ==========")
    print(summary[summary["rule_id"].eq("BASE_12H_MFE5_Close0")][cols].to_string(index=False))
    print()
    print("========== Shortlist ==========")
    print(shortlist[cols + ["delta_net_pnl_u", "delta_pf", "delta_max_drawdown_u", "delta_liquidations", "delta_drop_top5_u"]].head(20).to_string(index=False))
    print()
    print(f"Wrote output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
