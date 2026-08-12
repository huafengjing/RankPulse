from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_futures_top2_fixed_time import (
    calculate_group_stats,
    calculate_monthly_summary,
    calculate_summary,
    calculate_tail_dependency,
    final_conclusion,
    hold14_comparison_conclusion,
    simulate_trades_with_position_limit,
)


OUT = ROOT / "output"
SIGNALS_PATH = OUT / "futures_top2_fixed_time_signals.csv"
TRADES_PATH = OUT / "futures_top2_fixed_time_trades.csv"
SUMMARY_PATH = OUT / "futures_top2_fixed_time_summary.csv"
MONTHLY_PATH = OUT / "futures_top2_fixed_time_monthly.csv"
GROUP_PATH = OUT / "futures_top2_fixed_time_group_stats.csv"
TAIL_PATH = OUT / "futures_top2_fixed_time_tail_dependency.csv"
KLINE_DIR = ROOT / "data" / "futures_klines_1h"

SIGNAL_COLUMNS = [
    "signal_time_utc",
    "signal_time_bj",
    "snapshot_hour_bj",
    "symbol",
    "rank",
    "gain_24h",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Top2 fixed-time exit backtests from cached signals and cached 1H klines only."
    )
    parser.add_argument(
        "--holding-days",
        default="3,7,14",
        help="Comma-separated holding days. Default: 3,7,14",
    )
    parser.add_argument(
        "--signals",
        default=str(SIGNALS_PATH),
        help="Path to cached unique Top2 signals CSV.",
    )
    parser.add_argument(
        "--fallback-from-trades",
        action="store_true",
        help="If signals CSV is missing, rebuild unique signals from the existing trades CSV.",
    )
    return parser.parse_args()


def normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SIGNAL_COLUMNS if col not in signals.columns]
    if missing:
        raise ValueError(f"Missing signal columns: {missing}")
    signals = signals[SIGNAL_COLUMNS + [col for col in ["signal_time"] if col in signals.columns]].drop_duplicates().copy()
    if "signal_time" not in signals.columns:
        signal_times = pd.to_datetime(signals["signal_time_utc"], utc=True)
        signals["signal_time"] = signal_times.map(lambda value: int(value.timestamp() * 1000)).astype("int64")
    else:
        signals["signal_time"] = pd.to_numeric(signals["signal_time"], errors="coerce").astype("int64")
    signals["rank"] = pd.to_numeric(signals["rank"], errors="coerce").astype("int64")
    signals["gain_24h"] = pd.to_numeric(signals["gain_24h"], errors="coerce")
    return signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def load_signals(path: Path, fallback_from_trades: bool) -> pd.DataFrame:
    if path.exists():
        return normalize_signals(pd.read_csv(path))
    if fallback_from_trades and TRADES_PATH.exists():
        trades = pd.read_csv(TRADES_PATH)
        signals = normalize_signals(trades[SIGNAL_COLUMNS].drop_duplicates())
        path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(path, index=False)
        return signals
    raise FileNotFoundError(
        f"Missing signals CSV: {path}. Run scripts/backtest_futures_top2_fixed_time.py once to build signals, "
        "or pass --fallback-from-trades if the existing trades CSV contains the signal set you want."
    )


def load_1h_klines(symbols: list[str]) -> dict[str, pd.DataFrame]:
    kline_map: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = KLINE_DIR / f"{symbol}_1h.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        for col in ["open_time", "close_time", "trade_count"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64").astype("int64")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["symbol"] = symbol
        frame["interval"] = "1h"
        kline_map[symbol] = frame.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")
    return kline_map


def safe_to_csv(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        frame.to_csv(fallback, index=False)
        print(f"File locked, wrote fallback: {fallback}")
        return fallback


def main() -> None:
    args = parse_args()
    holding_days = [int(value.strip()) for value in args.holding_days.split(",") if value.strip()]
    signals = load_signals(Path(args.signals), args.fallback_from_trades)
    symbols = sorted(signals["symbol"].astype(str).unique().tolist())
    kline_map = load_1h_klines(symbols)
    missing_symbols = sorted(set(symbols) - set(kline_map))

    print("========== Top2 Exits From Cached Signals ==========")
    print("No Binance API calls. No downloads. No ranking rebuild.")
    print(f"Signals: {len(signals)}")
    print(f"Signal symbols: {len(symbols)}")
    print(f"Loaded local 1H symbols: {len(kline_map)}")
    print(f"Missing local 1H symbols: {len(missing_symbols)}")
    if missing_symbols:
        print("Missing symbols sample:", ", ".join(missing_symbols[:20]))
    print(f"Holding days: {holding_days}")

    trade_rows = []
    for days in holding_days:
        trade_rows.extend(simulate_trades_with_position_limit(signals, days, kline_map))
    trades = pd.DataFrame(trade_rows)
    summary = calculate_summary(trades)
    monthly = calculate_monthly_summary(trades)
    group_stats = calculate_group_stats(trades)
    tail = calculate_tail_dependency(trades)

    OUT.mkdir(parents=True, exist_ok=True)
    written = [
        safe_to_csv(trades, TRADES_PATH),
        safe_to_csv(summary, SUMMARY_PATH),
        safe_to_csv(monthly, MONTHLY_PATH),
        safe_to_csv(group_stats, GROUP_PATH),
        safe_to_csv(tail, TAIL_PATH),
    ]

    print("\n========== Overall Summary ==========")
    print(summary.to_string(index=False))
    if not summary.empty:
        print("\n".join(final_conclusion(summary, monthly, group_stats, tail)))
    if 14 in holding_days:
        print("\n".join(hold14_comparison_conclusion(summary, monthly, group_stats, tail)))
    print("\nWrote files:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
