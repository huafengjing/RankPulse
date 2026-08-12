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
    BUY_NOTIONAL_U,
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
    simulate_main_trade,
    simulate_main_trades_with_position_limit,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt


PREFIX = "main_strategy_test_24h_close_below_entry_exit"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}
EXIT_24H_REASON = "exit_24h_close_below_entry"


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


def simulate_trade_with_24h_close_exit(signal: pd.Series, kline_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    trade = simulate_main_trade(signal, kline_map)
    if trade.get("status") != "completed":
        return trade
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    h1 = kline_map.get(symbol, pd.DataFrame())
    if h1.empty:
        return trade
    indexed = h1.set_index("open_time", drop=False)
    if entry_time not in indexed.index:
        return trade
    entry_row = indexed.loc[entry_time]
    if isinstance(entry_row, pd.DataFrame):
        entry_row = entry_row.iloc[-1]
    entry_price = float(entry_row["open"])

    first_24h = path_slice(h1, entry_time, entry_time + 24 * HOUR_MS - HOUR_MS)
    if len(first_24h) < 24:
        return trade
    close_return_24h = (float(first_24h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0

    original_exit_time = int(float(trade["exit_time_ms"]))
    if close_return_24h < 0 and entry_time + 24 * HOUR_MS < original_exit_time:
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, entry_time + 24 * HOUR_MS, entry_time)
        if np.isfinite(exit_price):
            pnl, net_return = calc_pnl(entry_price, exit_price)
            trade_path = path_slice(h1, entry_time, exit_time)
            mfe, mae, max_price, min_price = mfe_mae(trade_path, entry_price)
            trade = trade | {
                "exit_time_ms": exit_time,
                "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time_bj": ms_to_bj_string(exit_time),
                "exit_price": exit_price,
                "exit_reason": fallback or EXIT_24H_REASON,
                "holding_days": (exit_time - entry_time) / DAY_MS,
                "pnl_u": pnl,
                "net_return_pct": net_return,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "max_price_during_trade": max_price,
                "min_price_during_trade": min_price,
                "is_win": pnl > 0,
            }
    trade["close_return_24h_pct"] = close_return_24h
    return trade


def simulate_with_position_limit(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], use_24h_exit: bool) -> pd.DataFrame:
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
        trade = simulate_trade_with_24h_close_exit(signal, kline_map) if use_24h_exit else simulate_main_trade(signal, kline_map)
        rows.append(trade)
        if trade.get("status") == "completed":
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"]))
    return pd.DataFrame(rows)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    return {
        "过滤后信号": int(len(group)),
        "完成交易": int(len(completed)),
        "持仓重复跳过": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "24H平仓": int(completed["exit_reason"].eq(EXIT_24H_REASON).sum()) if "exit_reason" in completed else 0,
        "12H早退": int(completed["exit_reason"].eq(EARLY_REASON).sum()) if "exit_reason" in completed else 0,
        "盈利单": int(len(wins)),
        "亏损单": int(len(losses)),
        "盈利总金额U": float(pnl[pnl > 0].sum()) if len(pnl) else 0.0,
        "亏损总金额U": float(pnl[pnl < 0].sum()) if len(pnl) else 0.0,
        "净收益U": float(pnl.sum()) if len(pnl) else 0.0,
        "PF": profit_factor(pnl),
        "胜率": float(len(wins) / len(completed)) if len(completed) else np.nan,
        "平均收益率": float(returns.mean()) if len(returns) else np.nan,
        "中位数收益率": float(returns.median()) if len(returns) else np.nan,
        "最大回撤U": max_drawdown(pnl),
        "最大单笔收益U": float(pnl.max()) if len(pnl) else np.nan,
        "最大单笔亏损U": float(pnl.min()) if len(pnl) else np.nan,
        "去最大1笔后U": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "去最大3笔后U": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "去最大5笔后U": float(pnl.sum() - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
    }


def add_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gain_24h_bucket"] = pd.cut(
        frame["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%"],
        right=False,
    )
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = int(latest_signal_end_dt().timestamp() * 1000)
    start = end - 180 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start - 7 * DAY_MS, end + 7 * DAY_MS)
    raw_signals = generate_signals(start, end, kline_map)
    signals = apply_current_main_filters(raw_signals, kline_map)

    outputs = []
    summaries = []
    for version, use_24h in [("current_main", False), ("add_24h_close_below_entry_exit", True)]:
        trades = add_bucket(simulate_with_position_limit(signals, kline_map, use_24h))
        trades["version"] = version
        outputs.append(trades)
        summaries.append({"version": version, **summarize(trades)})

    all_trades = pd.concat(outputs, ignore_index=True)
    summary = pd.DataFrame(summaries)
    all_trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")

    bucket_rows = []
    for (version, bucket), group in all_trades[all_trades["status"].eq("completed")].groupby(["version", "gain_24h_bucket"], observed=False):
        bucket_rows.append({"version": version, "bucket": str(bucket), **summarize(group)})
    pd.DataFrame(bucket_rows).to_csv(OUT / f"{PREFIX}_bucket_stats.csv", index=False, encoding="utf-8-sig")

    print("========== 24H Close Below Entry Exit Compare ==========")
    print(summary.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
