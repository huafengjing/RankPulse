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
    CACHE_DIR,
    DAY_MS,
    EARLY_REASON,
    FEE_RATE,
    HOUR_MS,
    OUT,
    add_entry_factors,
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


PREFIX = "current_main_strategy_hold4_8_leverage1_3"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}
LEVERAGE_LIQ = {1: None, 2: -50.0, 3: -33.0}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def apply_current_main_filters(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].lt(0.80)
        & signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)

    in_2040 = signals["gain_24h"].ge(0.20) & signals["gain_24h"].lt(0.40)
    pass_2040 = signals["volume_24h_ratio_7d"].ge(1.5) & signals["volume_24h_ratio_7d"].lt(5.0)

    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )

    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)

    return signals[
        ((~in_2040) | pass_2040)
        & ((~in_4060) | pass_4060)
        & (~in_6080)
    ].sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def simulate_trade(signal: pd.Series, kline_map: dict[str, pd.DataFrame], hold_days: int) -> dict[str, Any]:
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
        "target_hold_days": hold_days,
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

    if mfe12 < 5 and close_return_12h < 0:
        exit_target = entry_time + 12 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or EARLY_REASON
    else:
        exit_target = entry_time + hold_days * DAY_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or f"fixed_{hold_days}d"
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
        "max_price_during_trade": max_price,
        "min_price_during_trade": min_price,
        "is_win": pnl > 0,
    }


def simulate_with_position_limit(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], hold_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    for _, signal in ordered.iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["target_hold_days"] = hold_days
            rows.append(row)
            continue
        trade = simulate_trade(signal, kline_map, hold_days)
        rows.append(trade)
        if trade.get("status") == "completed":
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"]))
    return pd.DataFrame(rows)


def leveraged_pnl(row: pd.Series, leverage: int) -> tuple[float, float, bool]:
    mae = float(row["mae_pct"])
    if leverage > 1 and mae <= float(LEVERAGE_LIQ[leverage]):
        return -BUY_NOTIONAL_U, -100.0, True
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / float(row["entry_price"])
    exit_value = qty * float(row["exit_price"]) * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0, False


def summarize(group: pd.DataFrame, pnl_col: str, ret_col: str, liq_col: str | None = None) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed[pnl_col].astype(float) if not completed.empty else pd.Series(dtype=float)
    ret = completed[ret_col].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "signals": int(len(group)),
        "trades": int(len(completed)),
        "skipped": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "early_12h": int(completed["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in completed else 0,
        "liquidations": int(completed[liq_col].sum()) if liq_col else 0,
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
    end = int(latest_signal_end_dt().timestamp() * 1000)
    start = end - 180 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start - 7 * DAY_MS, end + 8 * DAY_MS)
    signals = apply_current_main_filters(generate_signals(start, end, kline_map), kline_map)

    all_trades = []
    for hold_days in [4, 5, 6, 7, 8]:
        trades = simulate_with_position_limit(signals, kline_map, hold_days)
        all_trades.append(trades)
    trades_df = pd.concat(all_trades, ignore_index=True)

    leveraged_rows = []
    completed = trades_df[trades_df["status"].eq("completed")].copy()
    for leverage in [1, 2, 3]:
        lev = completed.copy()
        values = lev.apply(lambda row: leveraged_pnl(row, leverage), axis=1)
        lev["leverage"] = leverage
        lev["leveraged_pnl_u"] = [v[0] for v in values]
        lev["leveraged_return_pct"] = [v[1] for v in values]
        lev["liquidated"] = [v[2] for v in values]
        leveraged_rows.append(lev)
    leveraged = pd.concat(leveraged_rows, ignore_index=True)

    summary_rows = []
    for (hold_days, leverage), group in leveraged.groupby(["target_hold_days", "leverage"], sort=True):
        summary_rows.append(
            {
                "hold_days": int(hold_days),
                "leverage": f"{int(leverage)}x",
                **summarize(group, "leveraged_pnl_u", "leveraged_return_pct", "liquidated"),
            }
        )
    summary = pd.DataFrame(summary_rows)

    monthly_rows = []
    for (hold_days, leverage, month), group in leveraged.groupby(["target_hold_days", "leverage", "month"], sort=True):
        monthly_rows.append(
            {
                "hold_days": int(hold_days),
                "leverage": f"{int(leverage)}x",
                "month": month,
                **summarize(group, "leveraged_pnl_u", "leveraged_return_pct", "liquidated"),
            }
        )
    monthly = pd.DataFrame(monthly_rows)

    trades_df.to_csv(OUT / f"{PREFIX}_base_trades.csv", index=False, encoding="utf-8-sig")
    leveraged.to_csv(OUT / f"{PREFIX}_leveraged_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8")

    print("========== Hold Days x Leverage Summary ==========")
    print(summary.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
