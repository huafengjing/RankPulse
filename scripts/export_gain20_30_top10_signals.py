from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.downloader import INTERVAL_MS, load_cached_klines
from src.data.quality import check_klines
from src.research.ranking import build_rankings
from src.research.signals import identify_first_top_signals


def interval_to_minutes(interval: str) -> int:
    return int(INTERVAL_MS[interval] / 60_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--output-dir", default="outputs/half_year_gain20_30_signals")
    parser.add_argument("--min-gain", type=float, default=0.20)
    parser.add_argument("--max-gain", type=float, default=0.30)
    parser.add_argument("--save-rankings", action="store_true", help="Write full ranking snapshots. This is very large for half-year runs.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    interval = config["data"]["interval"]
    interval_minutes = interval_to_minutes(interval)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached klines...", flush=True)
    klines = load_cached_klines("data", interval)
    if klines.empty:
        raise RuntimeError("No cached klines found. Run download first.")
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

    end_ms = int(klines["open_time"].max())
    start_ms = end_ms - int(args.lookback_days) * 24 * 60 * 60 * 1000
    klines = klines[(klines["open_time"] >= start_ms) & (klines["open_time"] <= end_ms)].copy()
    print(
        f"Using cache range {pd.to_datetime(start_ms, unit='ms', utc=True)} "
        f"to {pd.to_datetime(end_ms, unit='ms', utc=True)} rows={len(klines):,}",
        flush=True,
    )

    quality = check_klines(klines, INTERVAL_MS[interval])
    quality.to_csv(out_dir / "data_quality.csv", index=False)

    print("Building rolling 24h rankings...", flush=True)
    rankings = build_rankings(klines, interval_minutes)
    if args.save_rankings:
        rankings.to_csv(out_dir / "ranking_snapshots.csv", index=False)
    print(f"Rankings rows={len(rankings):,}. Identifying first Top10 signals...", flush=True)

    signals = identify_first_top_signals(
        rankings,
        top_n=10,
        cooldown_days=config["strategy"]["cooldown_days"],
        observation_hours=config["strategy"]["observation_hours"],
    )
    signals = signals.sort_values("signal_time").reset_index(drop=True)
    signals.to_csv(out_dir / "signals_all_first_top10.csv", index=False)

    filtered = signals[
        (signals["rolling_24h_change_pct"] >= float(args.min_gain))
        & (signals["rolling_24h_change_pct"] < float(args.max_gain))
    ].copy()
    filtered["signal_month_utc"] = pd.to_datetime(filtered["signal_time_utc"], utc=True).dt.strftime("%Y-%m")
    filtered.to_csv(out_dir / "signals_gain20_30_first_top10.csv", index=False)

    monthly = filtered.groupby("signal_month_utc").size().reset_index(name="signal_count")
    monthly.to_csv(out_dir / "signals_gain20_30_monthly_counts.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "lookback_days": args.lookback_days,
                "interval": interval,
                "cache_start_utc": pd.to_datetime(start_ms, unit="ms", utc=True),
                "cache_end_utc": pd.to_datetime(end_ms, unit="ms", utc=True),
                "klines_rows": len(klines),
                "ranking_rows": len(rankings),
                "total_first_top10_signals": len(signals),
                "gain20_30_first_top10_signals": len(filtered),
                "unique_symbols_gain20_30": filtered["symbol"].nunique(),
                "min_gain_inclusive": args.min_gain,
                "max_gain_exclusive": args.max_gain,
            }
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print(monthly.to_string(index=False), flush=True)
    print(f"Wrote {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
