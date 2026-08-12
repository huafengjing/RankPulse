from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.binance_client import BinanceFuturesClient
from src.data.cache import DataCache
from src.data.downloader import download_symbols_klines, load_cached_klines
from src.data.universe import eligible_symbols


OUT = ROOT / "outputs" / "scheme_b_365_daily_prefilter"
EXCLUDE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


def setup_logging() -> None:
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_dir / "scheme_b_download.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def daily_prefilter(
    daily: pd.DataFrame,
    start_ms: int,
    end_ms: int,
    breakout_min: float,
    breakout_max: float,
    min_candidate_days: int,
) -> pd.DataFrame:
    rows = []
    for symbol, group in daily.groupby("symbol", sort=False):
        g = group.sort_values("open_time").drop_duplicates("open_time").copy()
        g = g[(g["open_time"] >= start_ms - 30 * 24 * 60 * 60 * 1000) & (g["open_time"] <= end_ms)]
        if len(g) < 30:
            rows.append({"symbol": symbol, "candidate": False, "reason": "daily_history_lt_30", "best_pre_21d_range_pct": None, "best_abs_pre_21d_return": None})
            continue
        pre_high = g["high"].shift(1).rolling(21, min_periods=21).max()
        pre_low = g["low"].shift(1).rolling(21, min_periods=21).min()
        pre_close_21 = g["close"].shift(21)
        prev_close = g["close"].shift(1)
        g["pre_21d_range_pct_daily"] = pre_high / pre_low - 1
        g["pre_21d_return_daily"] = g["close"] / pre_close_21 - 1
        g["daily_high_vs_prev_close"] = g["high"] / prev_close - 1
        analysis = g[(g["open_time"] >= start_ms) & (g["open_time"] <= end_ms)].copy()
        strict = analysis[
            (analysis["pre_21d_range_pct_daily"] <= 0.65)
            & (analysis["pre_21d_return_daily"].abs() <= 0.30)
            & (analysis["daily_high_vs_prev_close"] >= breakout_min)
            & (analysis["daily_high_vs_prev_close"] <= breakout_max)
        ]
        candidate_days = int(len(strict))
        candidate = candidate_days >= min_candidate_days
        rows.append(
            {
                "symbol": symbol,
                "candidate": candidate,
                "reason": "daily_rulec_and_breakout_possible" if candidate else "no_daily_rulec_breakout_window",
                "candidate_days": candidate_days,
                "best_pre_21d_range_pct": float(analysis["pre_21d_range_pct_daily"].min()) if not analysis.empty else None,
                "best_abs_pre_21d_return": float(analysis["pre_21d_return_daily"].abs().min()) if not analysis.empty else None,
                "max_daily_high_vs_prev_close": float(analysis["daily_high_vs_prev_close"].max()) if not analysis.empty else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-downloads", type=int, default=0, help="For testing only; 0 means all candidates.")
    parser.add_argument("--breakout-min", type=float, default=0.15, help="Daily high vs previous close lower bound for a possible 5m Top10 momentum event.")
    parser.add_argument("--breakout-max", type=float, default=0.80, help="Daily high vs previous close upper bound to avoid extreme already-overheated days.")
    parser.add_argument("--min-candidate-days", type=int, default=1)
    parser.add_argument("--prefilter-only", action="store_true", help="Only generate daily candidate symbols; do not download 5m.")
    parser.add_argument("--skip-daily-download", action="store_true", help="Reuse existing 1d cache and only run prefilter / 5m candidate download.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging()
    client = BinanceFuturesClient()
    cache = DataCache(ROOT / "data", cache_format="parquet")
    exchange_info = client.exchange_info()
    symbols = eligible_symbols(exchange_info, tickers_24h=None, exclude_symbols=EXCLUDE_SYMBOLS)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.lookback_days + 22)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    analysis_start_ms = int((end - timedelta(days=args.lookback_days)).timestamp() * 1000)

    if args.skip_daily_download:
        daily_stats = {"completed": 0, "failed": 0}
        print("Step 1/3 skipping daily download; reusing existing 1d cache", flush=True)
    else:
        print(f"Step 1/3 downloading daily klines symbols={len(symbols)} lookback_days={args.lookback_days}+22", flush=True)
        daily_stats = download_symbols_klines(client, cache, symbols, "1d", start_ms, end_ms, OUT, args.sleep)
        print(f"Daily download completed {daily_stats}", flush=True)

    print("Step 2/3 daily prefiltering Rule C possible symbols...", flush=True)
    daily = load_cached_klines(ROOT / "data", "1d")
    daily = daily[daily["symbol"].isin(symbols)].copy()
    candidates = daily_prefilter(
        daily,
        analysis_start_ms,
        end_ms,
        breakout_min=args.breakout_min,
        breakout_max=args.breakout_max,
        min_candidate_days=args.min_candidate_days,
    )
    candidates.to_csv(OUT / "daily_rulec_candidate_symbols.csv", index=False)
    candidate_symbols = sorted(candidates.loc[candidates["candidate"] == True, "symbol"].tolist())
    if args.max_downloads > 0:
        candidate_symbols = candidate_symbols[: args.max_downloads]
    print(
        f"Daily prefilter candidates={len(candidate_symbols)} / {len(symbols)} "
        f"saved={OUT / 'daily_rulec_candidate_symbols.csv'}",
        flush=True,
    )
    if args.prefilter_only:
        summary = pd.DataFrame(
            [
                {
                    "lookback_days": args.lookback_days,
                    "all_symbols": len(symbols),
                    "candidate_symbols": len(candidate_symbols),
                    "daily_completed": daily_stats["completed"],
                    "daily_failed": daily_stats["failed"],
                    "breakout_min": args.breakout_min,
                    "breakout_max": args.breakout_max,
                    "min_candidate_days": args.min_candidate_days,
                    "analysis_start_utc": pd.to_datetime(analysis_start_ms, unit="ms", utc=True),
                    "end_utc": pd.to_datetime(end_ms, unit="ms", utc=True),
                    "status": "prefilter_only",
                }
            ]
        )
        summary.to_csv(OUT / "download_summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)
        return

    print("Step 3/3 downloading 5m klines for candidate symbols...", flush=True)
    fivem_start = int((end - timedelta(days=args.lookback_days + 22)).timestamp() * 1000)
    fivem_stats = download_symbols_klines(client, cache, candidate_symbols, "5m", fivem_start, end_ms, OUT, args.sleep)
    summary = pd.DataFrame(
        [
            {
                "lookback_days": args.lookback_days,
                "all_symbols": len(symbols),
                "candidate_symbols": len(candidate_symbols),
                "daily_completed": daily_stats["completed"],
                "daily_failed": daily_stats["failed"],
                "fivem_completed": fivem_stats["completed"],
                "fivem_failed": fivem_stats["failed"],
                "breakout_min": args.breakout_min,
                "breakout_max": args.breakout_max,
                "min_candidate_days": args.min_candidate_days,
                "analysis_start_utc": pd.to_datetime(analysis_start_ms, unit="ms", utc=True),
                "end_utc": pd.to_datetime(end_ms, unit="ms", utc=True),
                "status": "completed",
            }
        ]
    )
    summary.to_csv(OUT / "download_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
