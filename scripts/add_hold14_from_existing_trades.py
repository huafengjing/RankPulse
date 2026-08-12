from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_futures_top2_fixed_time import (
    calculate_group_stats,
    calculate_monthly_summary,
    calculate_summary,
    calculate_tail_dependency,
    hold14_comparison_conclusion,
    simulate_trades_with_position_limit,
)


TRADES_PATH = ROOT / "output" / "futures_top2_fixed_time_trades.csv"
SUMMARY_PATH = ROOT / "output" / "futures_top2_fixed_time_summary.csv"
MONTHLY_PATH = ROOT / "output" / "futures_top2_fixed_time_monthly.csv"
GROUP_PATH = ROOT / "output" / "futures_top2_fixed_time_group_stats.csv"
TAIL_PATH = ROOT / "output" / "futures_top2_fixed_time_tail_dependency.csv"
KLINE_DIR = ROOT / "data" / "futures_klines_1h"

SIGNAL_COLUMNS = [
    "signal_time_utc",
    "signal_time_bj",
    "snapshot_hour_bj",
    "symbol",
    "rank",
    "gain_24h",
]


def load_existing_signals(trades: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SIGNAL_COLUMNS if col not in trades.columns]
    if missing:
        raise ValueError(f"Missing signal columns in trades csv: {missing}")
    signals = trades[SIGNAL_COLUMNS].drop_duplicates().copy()
    signal_times = pd.to_datetime(signals["signal_time_utc"], utc=True)
    signals["signal_time"] = signal_times.map(lambda value: int(value.timestamp() * 1000)).astype("int64")
    signals["rank"] = pd.to_numeric(signals["rank"], errors="coerce").astype("int64")
    signals["gain_24h"] = pd.to_numeric(signals["gain_24h"], errors="coerce")
    signals = signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    return signals


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


def align_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for col in columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    extra_cols = [col for col in aligned.columns if col not in columns]
    return aligned[columns + extra_cols]


def main() -> None:
    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"Missing existing trades csv: {TRADES_PATH}")

    trades = pd.read_csv(TRADES_PATH)
    trades["holding_days"] = pd.to_numeric(trades["holding_days"], errors="coerce").astype("int64")
    signals = load_existing_signals(trades)
    symbols = sorted(signals["symbol"].astype(str).unique().tolist())
    kline_map = load_1h_klines(symbols)
    missing_symbols = sorted(set(symbols) - set(kline_map))

    print("========== Add Hold 14D From Existing Trades ==========")
    print(f"Existing trade rows: {len(trades)}")
    print(f"Raw unique signals: {len(signals)}")
    print(f"Unique signal symbols: {len(symbols)}")
    print(f"Loaded local 1H symbols: {len(kline_map)}")
    print(f"Missing local 1H symbols: {len(missing_symbols)}")
    if missing_symbols:
        print("Missing symbols sample:", ", ".join(missing_symbols[:20]))

    hold14 = pd.DataFrame(simulate_trades_with_position_limit(signals, 14, kline_map))
    base = trades[trades["holding_days"] != 14].copy()
    hold14 = align_columns(hold14, list(base.columns))
    combined = pd.concat([base, hold14], ignore_index=True, sort=False)
    combined["holding_days"] = pd.to_numeric(combined["holding_days"], errors="coerce").astype("int64")
    combined = combined.sort_values(["holding_days", "entry_time_utc", "rank", "symbol"]).reset_index(drop=True)

    summary = calculate_summary(combined)
    monthly = calculate_monthly_summary(combined)
    group_stats = calculate_group_stats(combined)
    tail = calculate_tail_dependency(combined)

    combined.to_csv(TRADES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    monthly.to_csv(MONTHLY_PATH, index=False)
    group_stats.to_csv(GROUP_PATH, index=False)
    tail.to_csv(TAIL_PATH, index=False)

    print("\n========== Updated Overall Summary ==========")
    print(summary.to_string(index=False))
    print("\n".join(hold14_comparison_conclusion(summary, monthly, group_stats, tail)))
    print(f"\nWrote updated files under: {ROOT / 'output'}")


if __name__ == "__main__":
    main()
