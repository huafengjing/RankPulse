from __future__ import annotations

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


PREFIX = "top3_12h_plus_24h_close_exit"
LIQ_THRESHOLD = {2: -50.0, 3: -33.0, 5: -20.0}
EARLY_12H_REASON = "early_exit_12h_mfe_lt5_close_neg_mae_lt_minus5"
EARLY_24H_REASON = "early_exit_24h_close_neg"


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def latest_common_open_time(symbols: list[str]) -> int:
    latest = []
    for symbol in symbols:
        path = Path(CACHE_DIR) / f"{symbol}_1h.csv"
        try:
            tail = pd.read_csv(path, usecols=["open_time"]).tail(1)
        except Exception:
            continue
        if not tail.empty:
            latest.append(int(tail.iloc[0]["open_time"]))
    if not latest:
        raise RuntimeError("No cached kline latest timestamp found.")
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


def simulate_trade(signal: pd.Series, kline_map: dict[str, pd.DataFrame], add_24h_exit: bool) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    h1 = kline_map.get(symbol, pd.DataFrame())
    base = {
        "symbol": symbol,
        "rank": int(signal["rank"]),
        "entry_time_ms": entry_time,
        "entry_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_bj": ms_to_bj_string(entry_time),
        "snapshot_hour_bj": signal["snapshot_hour_bj"],
        "gain_24h": float(signal["gain_24h"]),
        "month": ms_to_bj_string(entry_time)[:7],
        "volume_24h_ratio_7d": signal.get("volume_24h_ratio_7d", np.nan),
        "volume_24h_ratio_7d_bucket": signal.get("volume_24h_ratio_7d_bucket", "missing"),
        "ma_structure_4h": signal.get("ma_structure_4h", "missing"),
        "distance_to_4h_ma7_pct": signal.get("distance_to_4h_ma7_pct", np.nan),
        "target_hold_days": 6,
        "add_24h_exit": add_24h_exit,
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

    first_12h = path_slice(h1, entry_time, entry_time + 12 * HOUR_MS - HOUR_MS)
    if first_12h.empty:
        return base | {"status": "skipped", "skip_reason": "missing_12h_path", "entry_price": entry_price}
    mfe12, mae12, _, _ = mfe_mae(first_12h, entry_price)
    close_return_12h = (float(first_12h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0

    first_24h = path_slice(h1, entry_time, entry_time + 24 * HOUR_MS - HOUR_MS)
    close_return_24h = np.nan
    if not first_24h.empty:
        close_return_24h = (float(first_24h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0

    if mfe12 < 5 and close_return_12h < 0 and mae12 < -5:
        exit_target = entry_time + 12 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or EARLY_12H_REASON
    elif add_24h_exit and not first_24h.empty and close_return_24h < 0:
        exit_target = entry_time + 24 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or EARLY_24H_REASON
    else:
        exit_target = entry_time + 6 * DAY_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or "fixed_6d"

    if not np.isfinite(exit_price):
        return base | {"status": "skipped", "skip_reason": "missing_exit_price", "entry_price": entry_price}

    pnl, net_return = calc_pnl(entry_price, exit_price)
    trade_path = path_slice(h1, entry_time, exit_time)
    mfe, mae, max_price, min_price = mfe_mae(trade_path, entry_price)
    return base | {
        "status": "completed",
        "skip_reason": "",
        "entry_price": entry_price,
        "exit_time_ms": exit_time,
        "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(exit_time),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": (exit_time - entry_time) / DAY_MS,
        "pnl_u": pnl,
        "net_return_pct": net_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "mfe_12h_pct": mfe12,
        "mae_12h_pct": mae12,
        "close_return_12h_pct": close_return_12h,
        "close_return_24h_pct": close_return_24h,
        "max_price_during_trade": max_price,
        "min_price_during_trade": min_price,
        "is_win": pnl > 0,
    }


def simulate_with_position_limit(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], add_24h_exit: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    for _, signal in ordered.iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["target_hold_days"] = 6
            row["add_24h_exit"] = add_24h_exit
            rows.append(row)
            continue
        trade = simulate_trade(signal, kline_map, add_24h_exit)
        rows.append(trade)
        if trade.get("status") == "completed":
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"]))
    return pd.DataFrame(rows)


def leverage_for_current_main(row: pd.Series) -> int:
    gain = float(row["gain_24h"])
    rank = int(row["rank"])
    if 0.10 <= gain < 0.20:
        return 3
    if 0.20 <= gain < 0.40:
        return 3 if rank == 2 else 5
    if 0.40 <= gain < 0.60:
        return 2
    raise ValueError(f"Unexpected traded row: gain={gain}, rank={rank}, symbol={row.get('symbol')}")


def leveraged_pnl(row: pd.Series, leverage: int) -> tuple[float, float, bool]:
    mae = float(row["mae_pct"])
    if mae <= LIQ_THRESHOLD[leverage]:
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def apply_leverage(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    completed_mask = out["status"].eq("completed")
    out["leverage"] = np.nan
    out.loc[completed_mask, "leverage"] = out[completed_mask].apply(leverage_for_current_main, axis=1)
    values = out[completed_mask].apply(lambda row: leveraged_pnl(row, int(row["leverage"])), axis=1)
    out["leveraged_pnl_u"] = np.nan
    out["leveraged_return_pct"] = np.nan
    out["liquidated"] = False
    out.loc[completed_mask, "leveraged_pnl_u"] = [value[0] for value in values]
    out.loc[completed_mask, "leveraged_return_pct"] = [value[1] for value in values]
    out.loc[completed_mask, "liquidated"] = [value[2] for value in values]
    out["entry_date_bj"] = pd.to_datetime(out["entry_time_bj"]).dt.strftime("%Y-%m-%d")
    return add_gain_bucket(out)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["leveraged_pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed["leveraged_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "early_12h": int(completed["exit_reason"].eq(EARLY_12H_REASON).sum()) if "exit_reason" in completed else 0,
        "early_24h": int(completed["exit_reason"].eq(EARLY_24H_REASON).sum()) if "exit_reason" in completed else 0,
        "liquidations": int(completed["liquidated"].sum()) if "liquidated" in completed else 0,
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = cached_symbols()
    signal_end = latest_common_open_time(symbols)
    signal_start = signal_end - 180 * DAY_MS
    kline_map = load_kline_map(symbols, signal_start - 7 * DAY_MS, signal_end + 8 * DAY_MS)
    signals = apply_current_main_filters(generate_signals(signal_start, signal_end, kline_map), kline_map)
    signals = add_gain_bucket(signals)

    rows = []
    for name, add_24h in [("base_12h_only", False), ("base_12h_plus_24h_close_neg", True)]:
        trades = simulate_with_position_limit(signals, kline_map, add_24h)
        trades = apply_leverage(trades)
        trades["version"] = name
        rows.append(trades)
    all_rows = pd.concat(rows, ignore_index=True)

    summary = pd.DataFrame([{"version": version, **summarize(group)} for version, group in all_rows.groupby("version", sort=True)])
    bucket = pd.DataFrame(
        [
            {"version": version, "gain_bucket": bucket, **summarize(group)}
            for (version, bucket), group in all_rows.groupby(["version", "gain_bucket"], sort=True, observed=True)
        ]
    )
    exit_reason = (
        all_rows[all_rows["status"].eq("completed")]
        .groupby(["version", "exit_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    daily = pd.DataFrame(
        [
            {"version": version, "entry_date_bj": date, **summarize(group)}
            for (version, date), group in all_rows.groupby(["version", "entry_date_bj"], sort=True)
        ]
    )

    completed = all_rows[all_rows["status"].eq("completed")].copy()
    completed["trade_key"] = (
        completed["symbol"].astype(str)
        + "|"
        + completed["entry_time_ms"].astype("int64").astype(str)
        + "|"
        + completed["rank"].astype("int64").astype(str)
    )
    keys_base = set(completed[completed["version"].eq("base_12h_only")]["trade_key"])
    keys_24 = set(completed[completed["version"].eq("base_12h_plus_24h_close_neg")]["trade_key"])
    trade_set_change = pd.DataFrame(
        [
            {"change_type": "only_base_12h", "count": len(keys_base - keys_24)},
            {"change_type": "only_plus_24h", "count": len(keys_24 - keys_base)},
            {"change_type": "both", "count": len(keys_base & keys_24)},
        ]
    )

    all_rows.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(OUT / f"{PREFIX}_signals.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(OUT / f"{PREFIX}_gain_bucket.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / f"{PREFIX}_daily.csv", index=False, encoding="utf-8-sig")
    exit_reason.to_csv(OUT / f"{PREFIX}_exit_reason.csv", index=False, encoding="utf-8-sig")
    trade_set_change.to_csv(OUT / f"{PREFIX}_trade_set_change.csv", index=False, encoding="utf-8-sig")

    print("========== Top3 12H + 24H Close<0 Exit ==========")
    print(f"Signal end UTC: {ms_to_utc(signal_end):%Y-%m-%d %H:%M:%S}")
    print(f"Signal end BJ:  {ms_to_bj_string(signal_end)}")
    print(summary.to_string(index=False))
    print()
    print("========== Gain Bucket ==========")
    print(bucket[["version", "gain_bucket", "trades", "early_12h", "early_24h", "liquidations", "net_pnl_u", "pf", "win_rate_pct", "median_return_pct", "max_drawdown_u"]].to_string(index=False))
    print()
    print("========== Exit Reasons ==========")
    print(exit_reason.to_string(index=False))
    print()
    print("========== Trade Set Change ==========")
    print(trade_set_change.to_string(index=False))
    print()
    print(f"Wrote output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
