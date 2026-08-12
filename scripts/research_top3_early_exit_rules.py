from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (
    CACHE_DIR,
    DAY_MS,
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
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals
from scripts.test_main_strategy_hold_days_leverage import apply_current_main_filters


PREFIX = "top3_early_exit_rule_research"
LIQ_THRESHOLD = {2: -50.0, 3: -33.0, 5: -20.0}
WINDOWS = [8, 10, 12]
MFE_THRESHOLDS = [3.0, 5.0, 8.0]
CLOSE_THRESHOLDS = [0.0, -3.0, -5.0]
MAE_THRESHOLDS = [-3.0, -5.0, -8.0]
CONDITION_SETS = [
    ("MFE", ("mfe",)),
    ("Close", ("close",)),
    ("MAE", ("mae",)),
    ("MFE+Close", ("mfe", "close")),
    ("MFE+MAE", ("mfe", "mae")),
    ("Close+MAE", ("close", "mae")),
    ("MFE+Close+MAE", ("mfe", "close", "mae")),
]
DEFAULT_RULE_ID = "W12_MFE5_Close0_MAE-5_MFE+Close+MAE"


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def latest_common_open_time(symbols: list[str]) -> int:
    latest: list[int] = []
    for symbol in symbols:
        path = Path(CACHE_DIR) / f"{symbol}_1h.csv"
        try:
            tail = pd.read_csv(path, usecols=["open_time"]).tail(1)
        except Exception:
            continue
        if not tail.empty:
            latest.append(int(tail.iloc[0]["open_time"]))
    if not latest:
        raise RuntimeError("No cached 1H klines found.")
    return min(latest)


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
    if mae <= LIQ_THRESHOLD[leverage]:
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def prepare_signal_paths(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
            "exit_6d_time_ms": np.nan,
            "exit_6d_price": np.nan,
            "exit_6d_reason": "",
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

        full_target = entry_time + 6 * DAY_MS
        full_exit_time, full_exit_price, full_fallback = get_open_at_or_latest(h1, full_target, entry_time)
        if not np.isfinite(full_exit_price):
            rows.append(base | {"path_status": "skipped", "path_skip_reason": "missing_exit_price"})
            continue
        base["exit_6d_time_ms"] = int(full_exit_time)
        base["exit_6d_price"] = float(full_exit_price)
        base["exit_6d_reason"] = full_fallback or "fixed_6d"

        for window in WINDOWS:
            early_path = path_slice(h1, entry_time, entry_time + window * HOUR_MS - HOUR_MS)
            if early_path.empty:
                base[f"w{window}_available"] = False
                base[f"w{window}_mfe"] = np.nan
                base[f"w{window}_mae"] = np.nan
                base[f"w{window}_close_return"] = np.nan
                base[f"w{window}_exit_time_ms"] = np.nan
                base[f"w{window}_exit_price"] = np.nan
                base[f"w{window}_exit_reason"] = f"missing_{window}h_path"
                continue
            mfe, mae, _, _ = mfe_mae(early_path, entry_price)
            close_return = (float(early_path.iloc[-1]["close"]) / entry_price - 1.0) * 100.0
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, entry_time + window * HOUR_MS, entry_time)
            base[f"w{window}_available"] = bool(np.isfinite(exit_price))
            base[f"w{window}_mfe"] = float(mfe)
            base[f"w{window}_mae"] = float(mae)
            base[f"w{window}_close_return"] = float(close_return)
            base[f"w{window}_exit_time_ms"] = int(exit_time)
            base[f"w{window}_exit_price"] = float(exit_price) if np.isfinite(exit_price) else np.nan
            base[f"w{window}_exit_reason"] = fallback or f"early_exit_{window}h"
        rows.append(base)
    return add_gain_bucket(pd.DataFrame(rows))


def rule_id(window: int, mfe: float, close: float, mae: float, condition_name: str) -> str:
    return f"W{window}_MFE{mfe:g}_Close{close:g}_MAE{mae:g}_{condition_name}"


def rule_description(window: int, mfe: float, close: float, mae: float, condition_name: str) -> str:
    parts = []
    if "MFE" in condition_name:
        parts.append(f"{window}H MFE < {mfe:g}%")
    if "Close" in condition_name:
        parts.append(f"{window}H close_return < {close:g}%")
    if "MAE" in condition_name:
        parts.append(f"{window}H MAE < {mae:g}%")
    return " and ".join(parts)


def rule_trigger(row: pd.Series, window: int, mfe_t: float, close_t: float, mae_t: float, conditions: tuple[str, ...]) -> bool:
    if not bool(row.get(f"w{window}_available", False)):
        return False
    checks = []
    if "mfe" in conditions:
        checks.append(float(row[f"w{window}_mfe"]) < mfe_t)
    if "close" in conditions:
        checks.append(float(row[f"w{window}_close_return"]) < close_t)
    if "mae" in conditions:
        checks.append(float(row[f"w{window}_mae"]) < mae_t)
    return bool(checks) and all(checks)


def simulate_rule(prepared: pd.DataFrame, window: int, mfe_t: float, close_t: float, mae_t: float, condition_name: str, conditions: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = prepared.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    rid = rule_id(window, mfe_t, close_t, mae_t, condition_name)
    desc = rule_description(window, mfe_t, close_t, mae_t, condition_name)
    for _, row in ordered.iterrows():
        symbol = str(row["symbol"])
        signal_time = int(row["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            skipped = skipped_open_position_trade(row, open_until)
            skipped["rule_id"] = rid
            skipped["rule_desc"] = desc
            skipped["window_h"] = window
            skipped["condition_set"] = condition_name
            rows.append(skipped)
            continue
        base = {
            "rule_id": rid,
            "rule_desc": desc,
            "window_h": window,
            "mfe_threshold": mfe_t,
            "close_threshold": close_t,
            "mae_threshold": mae_t,
            "condition_set": condition_name,
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
            "entry_price": row.get("entry_price", np.nan),
        }
        if row.get("path_status") != "ok":
            rows.append(base | {"status": "skipped", "skip_reason": row.get("path_skip_reason", "path_not_ok")})
            continue
        triggered = rule_trigger(row, window, mfe_t, close_t, mae_t, conditions)
        if triggered:
            exit_time = int(row[f"w{window}_exit_time_ms"])
            exit_price = float(row[f"w{window}_exit_price"])
            exit_reason = f"early_exit_{window}h"
        else:
            exit_time = int(row["exit_6d_time_ms"])
            exit_price = float(row["exit_6d_price"])
            exit_reason = row["exit_6d_reason"]
        pnl_u, net_return_pct = calc_pnl(float(row["entry_price"]), exit_price)
        # For liquidation tests, use path to the actual exit time.
        # Reconstruct MFE/MAE from cached symbol h1 only for selected rows would be slow;
        # so approximate with available full-path fields by slicing per trade key in a compact local read.
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
                "early_triggered": triggered,
                "holding_days": (exit_time - signal_time) / DAY_MS,
                "pnl_u": pnl_u,
                "net_return_pct": net_return_pct,
            }
        )
        open_until_by_symbol[symbol] = exit_time
    return pd.DataFrame(rows)


def add_mfe_mae_for_trades(trades: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = trades.copy()
    out["mfe_pct"] = np.nan
    out["mae_pct"] = np.nan
    out["max_price_during_trade"] = np.nan
    out["min_price_during_trade"] = np.nan
    completed_idx = out.index[out["status"].eq("completed")]
    for idx in completed_idx:
        row = out.loc[idx]
        h1 = kline_map.get(str(row["symbol"]), pd.DataFrame())
        if h1.empty or pd.isna(row["entry_price"]) or pd.isna(row["exit_time_ms"]):
            continue
        path = path_slice(h1, int(row["entry_time_ms"]), int(row["exit_time_ms"]))
        mfe, mae, max_price, min_price = mfe_mae(path, float(row["entry_price"]))
        out.loc[idx, "mfe_pct"] = mfe
        out.loc[idx, "mae_pct"] = mae
        out.loc[idx, "max_price_during_trade"] = max_price
        out.loc[idx, "min_price_during_trade"] = min_price
    return out


def apply_leverage(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["leverage"] = np.nan
    out["leveraged_pnl_u"] = np.nan
    out["leveraged_return_pct"] = np.nan
    out["liquidated"] = False
    completed_idx = out.index[out["status"].eq("completed")]
    for idx in completed_idx:
        row = out.loc[idx]
        lev = leverage_for_current_main(row)
        pnl, ret, liq = leveraged_pnl(row, lev)
        out.loc[idx, "leverage"] = lev
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
    early = int(completed["early_triggered"].fillna(False).sum()) if "early_triggered" in completed else 0
    liq = int(completed["liquidated"].fillna(False).sum()) if "liquidated" in completed else 0
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "early_exit_count": early,
        "early_exit_rate_pct": round(early / len(completed) * 100, 2) if len(completed) else np.nan,
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
    for rid, g in monthly.groupby("rule_id", sort=False):
        rows.append(
            {
                "rule_id": rid,
                "profitable_months": int((g["net_pnl_u"] > 0).sum()),
                "losing_months": int((g["net_pnl_u"] < 0).sum()),
                "monthly_net_std": round(float(g["net_pnl_u"].std(ddof=0)), 2) if len(g) else np.nan,
                "worst_month_pnl_u": round(float(g["net_pnl_u"].min()), 2) if len(g) else np.nan,
                "best_month_pnl_u": round(float(g["net_pnl_u"].max()), 2) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = cached_symbols()
    signal_end = latest_common_open_time(symbols)
    signal_start = signal_end - 180 * DAY_MS
    kline_map = load_kline_map(symbols, signal_start - 7 * DAY_MS, signal_end + 8 * DAY_MS)
    signals = apply_current_main_filters(generate_signals(signal_start, signal_end, kline_map), kline_map)
    prepared = prepare_signal_paths(signals, kline_map)

    summary_rows = []
    monthly_rows = []
    candidate_trades: list[pd.DataFrame] = []
    rules = list(itertools.product(WINDOWS, MFE_THRESHOLDS, CLOSE_THRESHOLDS, MAE_THRESHOLDS, CONDITION_SETS))
    for i, (window, mfe_t, close_t, mae_t, cond_pair) in enumerate(rules, start=1):
        cond_name, conds = cond_pair
        trades = simulate_rule(prepared, window, mfe_t, close_t, mae_t, cond_name, conds)
        trades = add_mfe_mae_for_trades(trades, kline_map)
        trades = apply_leverage(trades)
        rid = rule_id(window, mfe_t, close_t, mae_t, cond_name)
        row = {
            "rule_id": rid,
            "rule_desc": rule_description(window, mfe_t, close_t, mae_t, cond_name),
            "window_h": window,
            "mfe_threshold": mfe_t,
            "close_threshold": close_t,
            "mae_threshold": mae_t,
            "condition_set": cond_name,
            **summarize(trades),
        }
        summary_rows.append(row)
        for month, g in trades.groupby("month", sort=True):
            monthly_rows.append({"rule_id": rid, "month": month, **summarize(g)})
        if rid == DEFAULT_RULE_ID:
            candidate_trades.append(trades.assign(candidate_label="default_12h"))
        if i % 50 == 0:
            print(f"processed {i}/{len(rules)}", flush=True)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    stability = monthly_stability(monthly)
    summary = summary.merge(stability, on="rule_id", how="left")
    default = summary[summary["rule_id"].eq(DEFAULT_RULE_ID)].iloc[0]
    for col in ["net_pnl_u", "pf", "win_rate_pct", "median_return_pct", "max_drawdown_u", "liquidations", "drop_top1_u", "drop_top3_u", "drop_top5_u"]:
        summary[f"delta_{col}"] = summary[col] - default[col]

    # Shortlist: avoid one-off tiny-trigger rules and prefer robustness over raw PnL.
    robust = summary[
        (summary["trades"] >= 200)
        & (summary["early_exit_count"] >= 20)
        & (summary["net_pnl_u"] >= float(default["net_pnl_u"]) * 0.85)
        & (summary["drop_top5_u"] >= float(default["drop_top5_u"]) - 250)
        & (summary["max_drawdown_u"] >= float(default["max_drawdown_u"]) - 250)
        & (summary["liquidations"] <= int(default["liquidations"]))
    ].copy()
    robust["score"] = (
        robust["pf"].rank(ascending=True)
        + robust["drop_top5_u"].rank(ascending=True)
        + robust["net_pnl_u"].rank(ascending=True) * 0.5
        + robust["liquidations"].rank(ascending=False) * 0.75
        + robust["max_drawdown_u"].rank(ascending=True) * 0.75
        + robust["profitable_months"].rank(ascending=True) * 0.5
    )
    shortlist = robust.sort_values("score", ascending=False).head(30)

    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    shortlist.to_csv(OUT / f"{PREFIX}_shortlist.csv", index=False, encoding="utf-8-sig")
    prepared.to_csv(OUT / f"{PREFIX}_prepared_signals.csv", index=False, encoding="utf-8-sig")

    print("========== Top3 Early Exit Rule Research ==========")
    print(f"Signal end UTC: {ms_to_utc(signal_end):%Y-%m-%d %H:%M:%S}")
    print(f"Signal end BJ:  {ms_to_bj_string(signal_end)}")
    print(f"Rules tested: {len(summary)}")
    print()
    print("========== Default Rule ==========")
    print(default[[
        "rule_id", "trades", "early_exit_count", "early_exit_rate_pct", "net_pnl_u", "pf", "win_rate_pct",
        "avg_return_pct", "median_return_pct", "max_drawdown_u", "liquidations", "liquidation_rate_pct",
        "drop_top1_u", "drop_top3_u", "drop_top5_u", "profitable_months", "losing_months", "worst_month_pnl_u",
    ]].to_string())
    print()
    print("========== Robust Shortlist Top 15 ==========")
    cols = [
        "rule_id", "rule_desc", "trades", "early_exit_count", "early_exit_rate_pct", "net_pnl_u", "pf",
        "win_rate_pct", "median_return_pct", "max_drawdown_u", "liquidations", "liquidation_rate_pct",
        "drop_top1_u", "drop_top3_u", "drop_top5_u", "profitable_months", "losing_months", "worst_month_pnl_u",
    ]
    print(shortlist[cols].head(15).to_string(index=False))
    print()
    print(f"Wrote output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
