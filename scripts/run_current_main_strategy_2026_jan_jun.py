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
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, CACHE_DIR, generate_signals
from scripts.backtest_futures_top2_fixed_time import latest_signal_end_dt


PREFIX = "current_main_strategy_2026_jan_jun_vol55"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}
SIGNAL_START_MS = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)
HOLD_DAYS = 6
LIQUIDATION_THRESHOLDS_PCT = {2: -50.0, 3: -33.0, 5: -20.0}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def cache_common_end_ms(symbols: list[str]) -> int:
    max_times: list[int] = []
    for symbol in symbols:
        path = CACHE_DIR / f"{symbol}_1h.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=["open_time"])
        except Exception:
            continue
        if not frame.empty:
            max_times.append(int(pd.to_numeric(frame["open_time"], errors="coerce").max()))
    if not max_times:
        raise RuntimeError("No cached 1h klines found.")
    return int(pd.Series(max_times).median())


def gain_bucket(gain: float) -> str:
    if 0.10 <= gain < 0.20:
        return "10%-20%"
    if 0.20 <= gain < 0.40:
        return "20%-40%"
    if 0.40 <= gain < 0.60:
        return "40%-60%"
    if 0.60 <= gain < 0.80:
        return "60%-80%"
    if gain >= 0.80:
        return ">=80%"
    return "<10%"


def leverage_for_signal(row: pd.Series) -> int | None:
    gain = float(row["gain_24h"])
    rank = int(row["rank"])
    volume = float(row["volume_24h_ratio_7d"]) if pd.notna(row.get("volume_24h_ratio_7d", np.nan)) else math.nan
    if 0.10 <= gain < 0.20 and rank in {2, 3}:
        return 3
    if 0.20 <= gain < 0.40:
        if rank == 2 and 1.5 <= volume < 5.0:
            return 3
        if rank == 3 and 1.2 <= volume < 5.0:
            return 5
    if 0.40 <= gain < 0.60 and rank == 2 and 3.0 <= volume < 5.5:
        return 2
    return None


def apply_entry_rules(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["symbol"].astype(str).ne("RAVEUSDT")
        & signals["gain_24h"].ge(0.10)
        & signals["gain_24h"].lt(0.80)
    ].copy()
    signals = add_entry_factors(signals, kline_map)
    signals["leverage"] = signals.apply(leverage_for_signal, axis=1)
    signals = signals[signals["leverage"].notna()].copy()
    signals["leverage"] = signals["leverage"].astype(int)
    signals["gain_24h_bucket"] = signals["gain_24h"].astype(float).map(gain_bucket)
    return signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def calc_leveraged_pnl(entry_price: float, exit_price: float, leverage: int) -> tuple[float, float]:
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / entry_price
    exit_value = qty * exit_price * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return pnl, pnl / BUY_NOTIONAL_U * 100.0


def current_close_at_or_before(frame: pd.DataFrame, current_time: int) -> tuple[int, float]:
    available = frame[frame["open_time"] <= current_time].copy()
    if available.empty:
        return current_time, math.nan
    row = available.sort_values("open_time").iloc[-1]
    return int(row["open_time"]), float(row["close"])


def simulate_trade(signal: pd.Series, kline_map: dict[str, pd.DataFrame], current_time: int) -> dict[str, Any]:
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


def simulate_with_position_limit(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], current_time: int) -> pd.DataFrame:
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
        trade = simulate_trade(signal, kline_map, current_time)
        rows.append(trade)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            # Mark-to-market rows are still open at the cutoff, so they must
            # also block another signal on the same cutoff timestamp.
            lock_extra_ms = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + lock_extra_ms
    return pd.DataFrame(rows)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    evaluated = group[group["status"].isin(["completed", "open_mark_to_market"])].sort_values("entry_time_ms").copy()
    pnl = evaluated["pnl_u"].astype(float) if not evaluated.empty else pd.Series(dtype=float)
    ret = evaluated["net_return_pct"].astype(float) if not evaluated.empty else pd.Series(dtype=float)
    wins = evaluated[pnl > 0]
    losses = evaluated[pnl < 0]
    return {
        "signals": int(len(group)),
        "evaluated_positions": int(len(evaluated)),
        "closed_trades": int(group["status"].eq("completed").sum()) if "status" in group else 0,
        "open_mark_to_market": int(group["status"].eq("open_mark_to_market").sum()) if "status" in group else 0,
        "skipped": int(group["status"].eq("skipped").sum()) if "status" in group else 0,
        "extreme_weak_4h": int(evaluated["exit_reason"].eq("extreme_weak_4h").sum()) if "exit_reason" in evaluated else 0,
        "early_12h": int(evaluated["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in evaluated else 0,
        "liquidations": int(evaluated["liquidated"].fillna(False).sum()) if "liquidated" in evaluated else 0,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "gross_profit_u": round(float(pnl[pnl > 0].sum()), 2) if len(pnl) else 0.0,
        "gross_loss_u": round(float(pnl[pnl < 0].sum()), 2) if len(pnl) else 0.0,
        "net_pnl_u": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "pf": round(float(profit_factor(pnl)), 2),
        "win_rate_pct": round(float(len(wins) / len(evaluated) * 100), 2) if len(evaluated) else np.nan,
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
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_start = SIGNAL_START_MS - 10 * DAY_MS
    kline_end = common_end
    kline_map = load_kline_map(symbols, kline_start, kline_end)

    raw_signals = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    filtered_signals = apply_entry_rules(raw_signals, kline_map)
    trades = simulate_with_position_limit(filtered_signals, kline_map, common_end)

    summary = pd.DataFrame([summarize(trades)])
    summary.insert(0, "window_start_utc", ms_to_utc(SIGNAL_START_MS).strftime("%Y-%m-%d %H:%M:%S"))
    summary.insert(1, "window_end_utc", ms_to_utc(signal_end).strftime("%Y-%m-%d %H:%M:%S"))
    summary.insert(2, "cache_common_end_utc", ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"))

    monthly_rows = []
    for month, group in trades.groupby("month", sort=True):
        monthly_rows.append({"month": month, **summarize(group)})
    monthly = pd.DataFrame(monthly_rows)

    bucket_rows = []
    for bucket, group in trades.groupby("gain_24h_bucket", sort=True):
        bucket_rows.append({"gain_24h_bucket": bucket, **summarize(group)})
    bucket = pd.DataFrame(bucket_rows)

    rank_rows = []
    for (rank, leverage), group in trades.groupby(["rank", "leverage"], sort=True):
        rank_rows.append({"rank": int(rank), "leverage": f"{int(leverage)}x", **summarize(group)})
    rank = pd.DataFrame(rank_rows)

    exit_rows = []
    evaluated = trades[trades["status"].isin(["completed", "open_mark_to_market"])]
    for reason, group in evaluated.groupby("exit_reason", sort=True):
        exit_rows.append({"exit_reason": reason, **summarize(group)})
    exit_reason = pd.DataFrame(exit_rows)

    filtered_signals.to_csv(OUT / f"{PREFIX}_signals.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(OUT / f"{PREFIX}_gain_bucket.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / f"{PREFIX}_rank_leverage.csv", index=False, encoding="utf-8-sig")
    exit_reason.to_csv(OUT / f"{PREFIX}_exit_reason.csv", index=False, encoding="utf-8-sig")

    print("========== Current Main Strategy 2026 Jan-Jun Spec ==========")
    print(summary.to_string(index=False))
    print()
    print(monthly.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
