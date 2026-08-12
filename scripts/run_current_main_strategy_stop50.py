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

from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    EARLY_REASON,
    FEE_RATE,
    HOUR_MS,
    OUT,
    get_open_at_or_latest,
    load_kline_map,
    mfe_mae,
    ms_to_bj_string,
    ms_to_utc,
    path_slice,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, generate_signals, latest_signal_end_dt
from scripts.run_current_main_strategy_2026_jan_jun import (
    HOLD_DAYS,
    PREFIX as BASE_PREFIX,
    SIGNAL_START_MS,
    apply_entry_rules,
    cache_common_end_ms,
    cached_symbols,
    calc_leveraged_pnl,
    current_close_at_or_before,
    gain_bucket,
    summarize,
)


PREFIX = f"{BASE_PREFIX}_stop50"
HARD_STOP_LOSS_U = BUY_NOTIONAL_U * 0.50


def stop_price_for_loss(entry_price: float, leverage: int, loss_u: float = HARD_STOP_LOSS_U) -> float:
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / entry_price
    target_exit_value = nominal - loss_u
    return target_exit_value / (qty * (1.0 - FEE_RATE))


def first_stop_hit(frame: pd.DataFrame, start_time: int, end_time: int, stop_price: float) -> tuple[int, float] | None:
    scoped = path_slice(frame, start_time, end_time)
    if scoped.empty:
        return None
    hit = scoped[scoped["low"].astype(float) <= stop_price]
    if hit.empty:
        return None
    row = hit.sort_values("open_time").iloc[0]
    return int(row["open_time"]), float(stop_price)


def simulate_trade_stop50(signal: pd.Series, kline_map: dict[str, pd.DataFrame], current_time: int) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    leverage = int(signal["leverage"])
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
    close_return_12h = (
        (float(first_12h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0
        if len(first_12h) >= 1
        else np.nan
    )

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

    stop_price = stop_price_for_loss(entry_price, leverage)
    stop_hit = first_stop_hit(h1, entry_time, int(exit_time), stop_price)
    if stop_hit is not None:
        exit_time, exit_price = stop_hit
        exit_reason = "hard_stop_50pct"
        status = "completed"

    trade_path = path_slice(h1, entry_time, int(exit_time))
    mfe, mae, max_price, min_price = mfe_mae(trade_path, entry_price)
    pnl, net_return = calc_leveraged_pnl(entry_price, exit_price, leverage)
    if exit_reason == "hard_stop_50pct":
        pnl = -HARD_STOP_LOSS_U
        net_return = -50.0

    return base | {
        "status": status,
        "skip_reason": "",
        "entry_price": entry_price,
        "exit_time_ms": exit_time,
        "exit_time_utc": ms_to_utc(int(exit_time)).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(int(exit_time)),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": (int(exit_time) - entry_time) / DAY_MS,
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
        "hard_stop_50pct": exit_reason == "hard_stop_50pct",
        "liquidated": False,
        "is_win": pnl > 0,
    }


def simulate_with_position_limit_stop50(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], current_time: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["leverage"] = int(signal["leverage"])
            row["gain_24h_bucket"] = signal.get("gain_24h_bucket", gain_bucket(float(signal["gain_24h"])))
            row["target_hold_days"] = HOLD_DAYS
            rows.append(row)
            continue
        trade = simulate_trade_stop50(signal, kline_map, current_time)
        rows.append(trade)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"]))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = cached_symbols()
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_start = SIGNAL_START_MS - 10 * DAY_MS
    kline_map = load_kline_map(symbols, kline_start, common_end)

    raw_signals = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    filtered_signals = apply_entry_rules(raw_signals, kline_map)
    trades = simulate_with_position_limit_stop50(filtered_signals, kline_map, common_end)

    summary = pd.DataFrame([summarize(trades)])
    summary.insert(0, "window_start_utc", ms_to_utc(SIGNAL_START_MS).strftime("%Y-%m-%d %H:%M:%S"))
    summary.insert(1, "window_end_utc", ms_to_utc(signal_end).strftime("%Y-%m-%d %H:%M:%S"))
    summary.insert(2, "cache_common_end_utc", ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"))

    monthly = pd.DataFrame([{"month": month, **summarize(group)} for month, group in trades.groupby("month", sort=True)])
    exit_reason = pd.DataFrame(
        [{"exit_reason": reason, **summarize(group)} for reason, group in trades[trades["status"].isin(["completed", "open_mark_to_market"])].groupby("exit_reason", sort=True)]
    )
    rank = pd.DataFrame(
        [{"rank": int(rank_value), "leverage": f"{int(leverage)}x", **summarize(group)} for (rank_value, leverage), group in trades.groupby(["rank", "leverage"], sort=True)]
    )

    filtered_signals.to_csv(OUT / f"{PREFIX}_signals.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    exit_reason.to_csv(OUT / f"{PREFIX}_exit_reason.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / f"{PREFIX}_rank_leverage.csv", index=False, encoding="utf-8-sig")

    print("========== Current Main Strategy + Hard Stop 50% ==========")
    print(summary.to_string(index=False))
    print()
    print(monthly.to_string(index=False))
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
