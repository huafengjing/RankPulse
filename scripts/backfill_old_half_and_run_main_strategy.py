from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_futures_top2_fixed_time import (
    BUY_NOTIONAL_U,
    CACHE_DIR,
    DAY_MS,
    FEE_RATE,
    HOUR_MS,
    SimpleBinanceFuturesClient,
    download_klines,
    generate_signals,
    get_futures_symbols,
    latest_signal_end_dt,
    ms_to_bj_string,
    ms_to_utc,
)


OUT = ROOT / "output"
INTERVAL = "1h"
FOUR_HOUR_MS = 4 * HOUR_MS
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}
MAIN_PREFIX = "main_strategy_1y_stability"
EARLY_REASON = "early_exit_12h_mfe_lt5_close_neg"


def read_cached_symbol(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for col in ["open_time", "close_time", "trade_count"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64").astype("int64")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["symbol"] = symbol
    frame["interval"] = INTERVAL
    return frame.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")


def write_cached_symbol(symbol: str, frame: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return
    frame = frame.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time").copy()
    frame["open_time_utc"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time_utc"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    keep = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
        "symbol",
        "interval",
        "open_time_utc",
        "close_time_utc",
    ]
    keep = [col for col in keep if col in frame.columns]
    frame[keep].to_csv(CACHE_DIR / f"{symbol}_1h.csv", index=False, encoding="utf-8-sig")


def missing_ranges(cached: pd.DataFrame, start_time: int, end_time: int) -> list[tuple[int, int]]:
    expected = pd.Index(range(start_time, end_time + 1, HOUR_MS), dtype="int64")
    if cached.empty:
        return [(start_time, end_time)]
    available = pd.Index(cached.loc[(cached["open_time"] >= start_time) & (cached["open_time"] <= end_time), "open_time"].astype("int64").unique())
    missing = expected.difference(available).sort_values()
    if missing.empty:
        return []

    ranges: list[tuple[int, int]] = []
    run_start = int(missing[0])
    prev = int(missing[0])
    for value in missing[1:]:
        value = int(value)
        if value == prev + HOUR_MS:
            prev = value
            continue
        ranges.append((run_start, prev))
        run_start = prev = value
    ranges.append((run_start, prev))
    return ranges


def backfill_symbols(symbols: list[str], start_time: int, end_time: int, sleep_seconds: float) -> pd.DataFrame:
    client = SimpleBinanceFuturesClient()
    rows: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols, start=1):
        cached = read_cached_symbol(symbol)
        gaps = missing_ranges(cached, start_time, end_time)
        downloaded_parts: list[pd.DataFrame] = []
        error = ""
        for gap_start, gap_end in gaps:
            try:
                part = download_klines(client, symbol, gap_start, gap_end, sleep_seconds=sleep_seconds)
                if not part.empty:
                    downloaded_parts.append(part)
            except Exception as exc:  # Keep progressing; failures are audited in CSV.
                error = repr(exc)
                break
        if downloaded_parts:
            combined = pd.concat([cached] + downloaded_parts, ignore_index=True) if not cached.empty else pd.concat(downloaded_parts, ignore_index=True)
            write_cached_symbol(symbol, combined)
            cached = combined.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")
        covered = cached[(cached["open_time"] >= start_time) & (cached["open_time"] <= end_time)] if not cached.empty else pd.DataFrame()
        rows.append(
            {
                "idx": index,
                "symbol": symbol,
                "gap_count": len(gaps),
                "downloaded_rows": int(sum(len(part) for part in downloaded_parts)),
                "covered_rows": int(len(covered)),
                "start_utc": ms_to_utc(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                "end_utc": ms_to_utc(end_time).strftime("%Y-%m-%d %H:%M:%S"),
                "error": error,
            }
        )
        if index % 25 == 0 or gaps or error:
            print(
                f"[{index}/{len(symbols)}] {symbol} gaps={len(gaps)} downloaded={sum(len(part) for part in downloaded_parts)} covered={len(covered)}"
                + (f" error={error}" if error else ""),
                flush=True,
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / f"{MAIN_PREFIX}_backfill_audit.csv", index=False, encoding="utf-8-sig")
    return audit


def load_kline_map(symbols: list[str], start_time: int, end_time: int) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = read_cached_symbol(symbol)
        if frame.empty:
            continue
        scoped = frame[(frame["open_time"] >= start_time) & (frame["open_time"] <= end_time)].copy()
        if scoped.empty:
            continue
        result[symbol] = scoped
    return result


def aggregate_4h(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.reset_index(drop=True).copy()
    work["bar_open_time"] = (work["open_time"] // FOUR_HOUR_MS) * FOUR_HOUR_MS
    grouped = work.sort_values("open_time").groupby("bar_open_time", sort=True)
    out = (
        grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"bar_open_time": "open_time"})
    )
    out["next_open_time"] = out["open_time"] + FOUR_HOUR_MS
    return out.set_index("open_time", drop=False).sort_index()


def bucket_ma_structure(close: float, ma7: float, ma21: float) -> str:
    if close > ma7 > ma21:
        return "close > MA7 > MA21"
    if close > ma7 and ma7 <= ma21:
        return "close > MA7 but MA7 <= MA21"
    if close <= ma7 and ma7 > ma21:
        return "close <= MA7 but MA7 > MA21"
    return "close <= MA7 <= MA21"


def bucket_volume(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 1:
        return "<1"
    if value < 1.5:
        return "1~1.5"
    if value < 3:
        return "1.5~3"
    if value < 5:
        return "3~5"
    if value < 10:
        return "5~10"
    return ">10"


def add_entry_factors(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    h4_cache: dict[str, pd.DataFrame] = {}
    for _, signal in signals.iterrows():
        symbol = str(signal["symbol"])
        entry_time = int(signal["signal_time"])
        h1 = kline_map.get(symbol, pd.DataFrame())
        if symbol not in h4_cache:
            h4_cache[symbol] = aggregate_4h(h1)
        h4 = h4_cache[symbol]
        last_4h_open = entry_time - FOUR_HOUR_MS
        factor = {
            "ma_structure_4h": "missing",
            "distance_to_4h_ma7_pct": np.nan,
            "volume_24h_ratio_7d": np.nan,
        }
        if not h4.empty:
            up_to = h4[h4["open_time"] <= last_4h_open].copy()
            if len(up_to) >= 21:
                close = float(up_to.iloc[-1]["close"])
                ma7 = float(up_to.tail(7)["close"].mean())
                ma21 = float(up_to.tail(21)["close"].mean())
                factor["ma_structure_4h"] = bucket_ma_structure(close, ma7, ma21)
                factor["distance_to_4h_ma7_pct"] = (close / ma7 - 1.0) * 100 if ma7 > 0 else np.nan
            recent_42 = h4[h4["open_time"] <= last_4h_open].tail(42)
            recent_6 = h4[h4["open_time"] <= last_4h_open].tail(6)
            if len(recent_42) == 42 and len(recent_6) == 6:
                avg_daily_volume_7d = float(recent_42["volume"].sum()) / 7.0
                volume_24h = float(recent_6["volume"].sum())
                factor["volume_24h_ratio_7d"] = volume_24h / avg_daily_volume_7d if avg_daily_volume_7d > 0 else np.nan
        factor["volume_24h_ratio_7d_bucket"] = bucket_volume(factor["volume_24h_ratio_7d"])
        rows.append(factor)
    return pd.concat([signals.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def get_open_at_or_latest(frame: pd.DataFrame, open_time: int, entry_time: int) -> tuple[int, float, str]:
    indexed = frame.set_index("open_time", drop=False)
    if open_time in indexed.index:
        row = indexed.loc[open_time]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return int(open_time), float(row["open"]), ""
    available = frame[frame["open_time"] >= entry_time]
    if available.empty:
        return int(open_time), np.nan, "data_missing"
    row = available.iloc[-1]
    return int(row["open_time"]), float(row["open"]), "data_latest_available"


def calc_pnl(entry_price: float, exit_price: float) -> tuple[float, float]:
    qty = BUY_NOTIONAL_U * (1.0 - FEE_RATE) / entry_price
    exit_value = qty * exit_price * (1.0 - FEE_RATE)
    pnl = exit_value - BUY_NOTIONAL_U
    return pnl, pnl / BUY_NOTIONAL_U * 100.0


def path_slice(frame: pd.DataFrame, start_time: int, end_time: int) -> pd.DataFrame:
    return frame[(frame["open_time"] >= start_time) & (frame["open_time"] <= end_time)].copy()


def mfe_mae(path: pd.DataFrame, entry_price: float) -> tuple[float, float, float, float]:
    if path.empty:
        return np.nan, np.nan, np.nan, np.nan
    max_price = float(path["high"].max())
    min_price = float(path["low"].min())
    return (max_price / entry_price - 1.0) * 100.0, (min_price / entry_price - 1.0) * 100.0, max_price, min_price


def simulate_main_trade(signal: pd.Series, kline_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
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
        exit_target = entry_time + 7 * DAY_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or "fixed_7d"

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


def skipped_open_position_trade(signal: pd.Series, open_until: int) -> dict[str, Any]:
    entry_time = int(signal["signal_time"])
    return {
        "symbol": str(signal["symbol"]),
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
        "status": "skipped",
        "skip_reason": "symbol_already_open",
        "blocking_position_exit_time_ms": int(open_until),
        "blocking_position_exit_time_utc": ms_to_utc(open_until).strftime("%Y-%m-%d %H:%M:%S"),
        "blocking_position_exit_time_bj": ms_to_bj_string(open_until),
    }


def simulate_main_trades_with_position_limit(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    for _, signal in ordered.iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            rows.append(skipped_open_position_trade(signal, open_until))
            continue
        trade = simulate_main_trade(signal, kline_map)
        rows.append(trade)
        if trade.get("status") == "completed":
            open_until_by_symbol[symbol] = int(trade["exit_time_ms"])
    return pd.DataFrame(rows)


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    return float((equity - equity.cummax()).min())


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "signals": int(len(group)),
        "completed_trades": int(len(completed)),
        "skipped_trades": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": float(len(wins) / len(completed)) if len(completed) else np.nan,
        "gross_profit_u": float(pnl[pnl > 0].sum()) if len(pnl) else 0.0,
        "gross_loss_u": float(pnl[pnl < 0].sum()) if len(pnl) else 0.0,
        "net_pnl_u": float(pnl.sum()) if len(pnl) else 0.0,
        "profit_factor": profit_factor(pnl),
        "avg_return_pct": float(returns.mean()) if len(returns) else np.nan,
        "median_return_pct": float(returns.median()) if len(returns) else np.nan,
        "max_drawdown_u": max_drawdown(pnl),
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
        "net_after_drop_top1_u": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "net_after_drop_top3_u": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "net_after_drop_top5_u": float(pnl.sum() - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
    }


def run_window(name: str, signal_start: int, signal_end: int, kline_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = generate_signals(signal_start, signal_end, kline_map)
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].lt(0.80)
        & signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)
    if "volume_24h_ratio_7d" not in signals.columns:
        signals["volume_24h_ratio_7d"] = np.nan
    if "ma_structure_4h" not in signals.columns:
        signals["ma_structure_4h"] = "missing"
    in_2040 = signals["gain_24h"].ge(0.20) & signals["gain_24h"].lt(0.40)
    pass_2040 = (
        signals["volume_24h_ratio_7d"].ge(1.5)
        & signals["volume_24h_ratio_7d"].lt(5.0)
    )
    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )
    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)
    signals = signals[
        ((~in_2040) | pass_2040)
        & ((~in_4060) | pass_4060)
        & (~in_6080)
    ].sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)

    trades = simulate_main_trades_with_position_limit(signals, kline_map)
    summary = pd.DataFrame([{"window": name, **summarize(trades)}])
    monthly_rows = []
    if not trades.empty:
        for month, group in trades.groupby("month", sort=True):
            monthly_rows.append({"window": name, "month": month, **summarize(group)})
    monthly = pd.DataFrame(monthly_rows)
    trades["window"] = name
    return trades, summary, monthly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="Only run strategy from existing local cache.")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--symbols-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = SimpleBinanceFuturesClient()
    symbols = get_futures_symbols(client)
    if args.symbols_limit:
        symbols = symbols[: args.symbols_limit]

    end_dt = latest_signal_end_dt()
    current_end = int(end_dt.timestamp() * 1000)
    old_start = current_end - 365 * DAY_MS
    old_end = current_end - 180 * DAY_MS
    recent_start = old_end
    recent_end = current_end
    kline_start = old_start - 7 * DAY_MS
    kline_end = current_end + 7 * DAY_MS

    print("========== One Year Main Strategy Stability ==========", flush=True)
    print(f"Symbols: {len(symbols)}", flush=True)
    print(f"Old signal window:    {ms_to_utc(old_start)} -> {ms_to_utc(old_end)}", flush=True)
    print(f"Recent signal window: {ms_to_utc(recent_start)} -> {ms_to_utc(recent_end)}", flush=True)
    print(f"Kline window:         {ms_to_utc(kline_start)} -> {ms_to_utc(kline_end)}", flush=True)

    if not args.skip_download:
        backfill_symbols(symbols, kline_start, kline_end, args.sleep)

    kline_map = load_kline_map(symbols, kline_start, kline_end)
    print(f"Loaded local kline symbols: {len(kline_map)}", flush=True)

    outputs = []
    summaries = []
    monthlies = []
    for name, start, end in [
        ("old_365d_to_180d", old_start, old_end),
        ("recent_180d", recent_start, recent_end),
        ("full_365d", old_start, recent_end),
    ]:
        trades, summary, monthly = run_window(name, start, end, kline_map)
        outputs.append(trades)
        summaries.append(summary)
        monthlies.append(monthly)
        print(f"\n{name}", flush=True)
        print(summary.to_string(index=False), flush=True)

    all_trades = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    all_monthly = pd.concat(monthlies, ignore_index=True) if monthlies else pd.DataFrame()

    all_trades.to_csv(OUT / f"{MAIN_PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(OUT / f"{MAIN_PREFIX}_summary.csv", index=False, encoding="utf-8-sig")
    all_monthly.to_csv(OUT / f"{MAIN_PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")
    print(f"\nWrote: {OUT / f'{MAIN_PREFIX}_summary.csv'}", flush=True)
    print(f"Wrote: {OUT / f'{MAIN_PREFIX}_monthly.csv'}", flush=True)
    print(f"Wrote: {OUT / f'{MAIN_PREFIX}_trades.csv'}", flush=True)


if __name__ == "__main__":
    main()
