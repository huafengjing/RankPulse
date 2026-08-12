from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (  # noqa: E402
    DAY_MS,
    add_entry_factors,
    ms_to_utc,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt  # noqa: E402
from scripts.regime_adaptive_leverage_walkforward import simulate_trade_with_leverage  # noqa: E402
from scripts.run_current_main_strategy_2026_jan_jun import (  # noqa: E402
    cache_common_end_ms,
    cached_symbols,
    load_kline_map,
)


OUT_DIR = ROOT / "output" / "rank3_low_volume_bucket"
START_MS = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    evaluated = group[group["status"].isin(["completed", "open_mark_to_market"])].copy()
    pnl = pd.to_numeric(evaluated.get("pnl_u", pd.Series(dtype=float)), errors="coerce")
    ret = pd.to_numeric(evaluated.get("net_return_pct", pd.Series(dtype=float)), errors="coerce")
    gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    gross_loss_abs = abs(float(pnl[pnl < 0].sum())) if len(pnl) else 0.0
    equity = pnl.cumsum()
    max_drawdown = float((equity - equity.cummax()).min()) if len(pnl) else 0.0
    liquidations = int(evaluated["liquidated"].fillna(False).sum()) if "liquidated" in evaluated else 0
    return {
        "signals": int(len(group)),
        "evaluated": int(len(evaluated)),
        "closed": int(evaluated["status"].eq("completed").sum()) if len(evaluated) else 0,
        "open_mtm": int(evaluated["status"].eq("open_mark_to_market").sum()) if len(evaluated) else 0,
        "skipped": int(group["status"].eq("skipped").sum()) if "status" in group else 0,
        "net_pnl_u": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "gross_profit_u": round(gross_profit, 2),
        "gross_loss_u": round(-gross_loss_abs, 2),
        "pf": round(gross_profit / gross_loss_abs, 2) if gross_loss_abs else None,
        "win_rate_pct": round(float((pnl > 0).sum() / len(evaluated) * 100), 2) if len(evaluated) else None,
        "avg_return_pct": round(float(ret.mean()), 2) if len(ret) else None,
        "median_return_pct": round(float(ret.median()), 2) if len(ret) else None,
        "max_drawdown_u": round(max_drawdown, 2),
        "liquidations": liquidations,
        "best_trade_u": round(float(pnl.max()), 2) if len(pnl) else None,
        "worst_trade_u": round(float(pnl.min()), 2) if len(pnl) else None,
        "drop_top1_u": round(float(pnl.sum() - pnl.nlargest(1).sum()), 2) if len(pnl) >= 1 else None,
    }


def simulate_with_symbol_lock(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], current_time: int, leverage: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    for _, signal in ordered.iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["leverage"] = leverage
            row["status"] = "skipped"
            row["skip_reason"] = "symbol_already_open"
            rows.append(row)
            continue
        trade = simulate_trade_with_leverage(signal, kline_map, current_time, leverage)
        rows.append(trade)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            lock_extra_ms = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + lock_extra_ms
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-min", type=float, default=1.0)
    parser.add_argument("--volume-max", type=float, default=1.5)
    parser.add_argument("--out-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = cached_symbols()
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, START_MS - 10 * DAY_MS, common_end)
    raw = generate_signals(START_MS, signal_end, kline_map)
    base = raw[
        raw["snapshot_hour_bj"].isin(["00:00", "08:00"])
        & raw["rank"].eq(3)
        & raw["symbol"].astype(str).ne("RAVEUSDT")
        & raw["gain_24h"].ge(0.20)
        & raw["gain_24h"].lt(0.40)
    ].copy()
    base = add_entry_factors(base, kline_map)
    bucket = base[
        base["volume_24h_ratio_7d"].ge(args.volume_min)
        & base["volume_24h_ratio_7d"].lt(args.volume_max)
    ].copy()
    trades = simulate_with_symbol_lock(bucket, kline_map, common_end, leverage=5)

    summary = pd.DataFrame(
        [
            {
                "cutoff_utc": ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"),
                "strategy_window_start_utc": ms_to_utc(START_MS).strftime("%Y-%m-%d %H:%M:%S"),
                "bucket": f"Rank3 gain_24h 20%-40%, volume_24h_ratio_7d {args.volume_min}-{args.volume_max}, hypothetical 5x",
                **summarize(trades),
            }
        ]
    )
    monthly = pd.DataFrame([{"month": month, **summarize(group)} for month, group in trades.groupby("month", sort=True)])
    evaluated = trades[trades["status"].isin(["completed", "open_mark_to_market"])].copy()
    exit_reason = pd.DataFrame(
        [{"exit_reason": reason, **summarize(group)} for reason, group in evaluated.groupby("exit_reason", sort=True)]
    )
    detail_cols = [
        "entry_time_utc",
        "symbol",
        "rank",
        "gain_24h",
        "volume_24h_ratio_7d",
        "status",
        "entry_price",
        "exit_price",
        "exit_reason",
        "pnl_u",
        "net_return_pct",
        "mfe_pct",
        "mae_pct",
        "liquidated",
    ]
    details = evaluated[detail_cols].sort_values(["entry_time_utc", "symbol"]).copy() if len(evaluated) else pd.DataFrame(columns=detail_cols)

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    summary.to_csv(OUT_DIR / f"summary{suffix}.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / f"monthly{suffix}.csv", index=False, encoding="utf-8-sig")
    exit_reason.to_csv(OUT_DIR / f"exit_reason{suffix}.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT_DIR / f"trades{suffix}.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nMONTHLY")
    print(monthly.to_string(index=False))
    print("\nEXIT_REASON")
    print(exit_reason.to_string(index=False))
    print(f"\nfiles: {OUT_DIR}")


if __name__ == "__main__":
    main()
