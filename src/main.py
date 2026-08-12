from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.engine import run_top10_immediate_backtest
from src.backtest.metrics import summarize_trades
from src.charts.plots import generate_basic_charts
from src.data.binance_client import BinanceFuturesClient
from src.data.cache import DataCache
from src.data.downloader import INTERVAL_MS, download_symbols_klines, load_cached_klines
from src.data.quality import check_klines
from src.data.universe import eligible_symbols
from src.reporting.report import write_markdown_report
from src.reporting.analyze_results import analyze_outputs
from src.research.features import add_ema20_and_vwap
from src.research.parameter_grid import load_parameter_grid
from src.research.ranking import build_rankings
from src.research.signals import identify_first_top_signals


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_dir / "run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def interval_to_minutes(interval: str) -> int:
    return int(INTERVAL_MS[interval] / 60_000)


def download_command(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(config["outputs"]["dir"])
    setup_logging(output_dir)
    data_cfg = config["data"]
    client = BinanceFuturesClient(base_url=data_cfg["base_url"], max_retries=data_cfg["max_retries"])
    cache = DataCache(cache_format=data_cfg["cache_format"])
    exchange_info = client.exchange_info()
    cache.write_json("exchange_info_latest.json", exchange_info)
    apply_volume_filter = bool(data_cfg.get("apply_volume_filter_on_download", False))
    if args.all_symbols:
        apply_volume_filter = False
    if args.apply_volume_filter:
        apply_volume_filter = True
    tickers = client.ticker_24hr() if apply_volume_filter else None
    symbols = eligible_symbols(
        exchange_info,
        tickers,
        data_cfg["exclude_symbols"],
        data_cfg["min_quote_volume"],
    )
    print(
        f"Download universe symbols={len(symbols)} volume_filter={apply_volume_filter} "
        f"unicode_symbols_supported=True",
        flush=True,
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(args.lookback_days or data_cfg["lookback_days"]) + 1)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    stats = download_symbols_klines(
        client,
        cache,
        symbols,
        data_cfg["interval"],
        start_ms,
        end_ms,
        output_dir,
        data_cfg["request_sleep_seconds"],
    )
    logging.getLogger(__name__).info("Download finished completed=%s failed=%s", stats["completed"], stats["failed"])


def backtest_command(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(config["outputs"]["dir"])
    setup_logging(output_dir)
    print("Backtest started: loading cached klines...", flush=True)
    interval = config["data"]["interval"]
    interval_minutes = interval_to_minutes(interval)
    klines = load_cached_klines("data", interval)
    if klines.empty:
        raise RuntimeError("No cached klines found. Run `python -m src.main download` first.")
    print(f"Loaded klines rows={len(klines):,}. Adding features and data quality checks...", flush=True)
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    klines = add_ema20_and_vwap(klines)
    quality = check_klines(klines, INTERVAL_MS[interval])
    output_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output_dir / "data_quality.csv", index=False)

    print("Building rolling 24h rankings...", flush=True)
    rankings = build_rankings(klines, interval_minutes)
    print(f"Built rankings rows={len(rankings):,}. Identifying first Top10 signals...", flush=True)
    signals = identify_first_top_signals(
        rankings,
        top_n=10,
        cooldown_days=config["strategy"]["cooldown_days"],
        observation_hours=config["strategy"]["observation_hours"],
    )
    signals.to_csv(output_dir / "signals.csv", index=False)
    print(f"Signals generated={len(signals):,}. Running parameter grid...", flush=True)
    grid = load_parameter_grid(args.grid)
    all_trades = []
    summaries = []
    for idx, params in enumerate(grid, start=1):
        parameter_set_id = f"ps_{idx:04d}"
        print(f"Running {parameter_set_id}/{len(grid):04d}: {params}", flush=True)
        full_params = {**config["data"], **config["strategy"], **params, "total_signals": len(signals)}
        trades = run_top10_immediate_backtest(
            signals,
            klines,
            rankings,
            parameter_set_id=parameter_set_id,
            strategy_variant=params["strategy_variant"],
            interval_minutes=interval_minutes,
            tp1_pct=float(params["tp1_pct"]),
            sl_pct=float(params["sl_pct"]),
            max_holding_hours=int(params["max_holding_hours"]),
            taker_fee_rate=config["strategy"]["taker_fee_rate"],
            slippage_rate=config["strategy"]["slippage_rate"],
        )
        all_trades.append(trades)
        summaries.append(summarize_trades(trades, parameter_set_id, full_params))
    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_out = pd.DataFrame(summaries)
    trades_out.to_csv(output_dir / "trades.csv", index=False)
    summary_out.to_csv(output_dir / "summary.csv", index=False)
    print(f"Wrote trades rows={len(trades_out):,} and summary rows={len(summary_out):,}. Generating charts/report...", flush=True)
    if not trades_out.empty:
        equity = trades_out.sort_values("entry_time_utc")[["entry_time_utc", "net_return_pct"]].copy()
        equity["equity"] = (1.0 + equity["net_return_pct"]).cumprod()
        equity.to_csv(output_dir / "equity_curve.csv", index=False)
    generate_basic_charts(trades_out, summary_out, output_dir / "charts")
    write_markdown_report(summary_out, output_dir / "reports" / "research_report.md")
    print(f"Backtest finished. Outputs written to {output_dir.resolve()}", flush=True)


def analyze_results_command(args: argparse.Namespace) -> None:
    text = analyze_outputs(args.output_dir)
    print(text, flush=True)
    print(f"\nAnalysis written to {Path(args.output_dir).resolve() / 'reports' / 'result_analysis.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--config", default="config/default.yaml")
    download.add_argument("--lookback-days", type=int)
    download.add_argument("--all-symbols", action="store_true", help="Download all TRADING USDT perpetual contracts without current 24h volume filtering.")
    download.add_argument("--apply-volume-filter", action="store_true", help="Apply current 24h ticker quote-volume filtering during download.")
    download.set_defaults(func=download_command)
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--config", default="config/default.yaml")
    backtest.add_argument("--grid", default="config/parameter_grid.yaml")
    backtest.set_defaults(func=backtest_command)
    analyze = sub.add_parser("analyze-results")
    analyze.add_argument("--output-dir", default="outputs")
    analyze.set_defaults(func=analyze_results_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
