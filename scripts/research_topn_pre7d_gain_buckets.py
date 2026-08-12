from __future__ import annotations

import argparse
import math
import sys
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
    SIGNAL_DAYS,
    SNAPSHOT_UTC_HOURS,
    SimpleBinanceFuturesClient,
    calculate_drawdown,
    get_futures_symbols,
    latest_signal_end_dt,
    load_local_1h_cache,
    load_local_5m_as_1h,
    ms_to_bj_string,
    ms_to_utc,
    simulate_trades_with_position_limit,
)


OUT = ROOT / "output"
PRE7D_BUCKETS = [
    ("过去7天涨幅 <20%", -math.inf, 20.0),
    ("过去7天涨幅 20%-50%", 20.0, 50.0),
    ("过去7天涨幅 50%-80%", 50.0, 80.0),
    ("过去7天涨幅 80%-120%", 80.0, 120.0),
    ("过去7天涨幅 120%-200%", 120.0, 200.0),
    ("过去7天涨幅 200%+", 200.0, math.inf),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research TopN fixed-time 7D exits grouped by pre-signal 7D gain buckets."
    )
    parser.add_argument("--top-n", default="20,50", help="Comma-separated TopN values. Default: 20,50")
    parser.add_argument("--holding-days", type=int, default=7, help="Holding days. Default: 7")
    return parser.parse_args()


def make_indexed_kline_map(kline_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    indexed: dict[str, pd.DataFrame] = {}
    for symbol, frame in kline_map.items():
        if frame.empty:
            continue
        keep = frame.drop_duplicates(["open_time"]).sort_values("open_time").set_index("open_time", drop=False)
        indexed[symbol] = keep
    return indexed


def pre7d_gain_pct(symbol: str, signal_time: int, indexed_map: dict[str, pd.DataFrame]) -> float:
    frame = indexed_map.get(symbol)
    if frame is None or frame.empty:
        return np.nan
    current_open = signal_time - HOUR_MS
    previous_open = current_open - 7 * DAY_MS
    if current_open not in frame.index or previous_open not in frame.index:
        return np.nan
    current = frame.loc[current_open]
    previous = frame.loc[previous_open]
    if isinstance(current, pd.DataFrame):
        current = current.iloc[-1]
    if isinstance(previous, pd.DataFrame):
        previous = previous.iloc[-1]
    prev_close = float(previous["close"])
    curr_close = float(current["close"])
    if not np.isfinite(prev_close) or prev_close <= 0 or not np.isfinite(curr_close):
        return np.nan
    return (curr_close / prev_close - 1.0) * 100.0


def build_snapshot_rankings_fast(snapshot_time: int, indexed_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    current_open = snapshot_time - HOUR_MS
    close_24h_open = current_open - 24 * HOUR_MS
    rows: list[dict[str, Any]] = []
    for symbol, frame in indexed_map.items():
        if current_open not in frame.index or close_24h_open not in frame.index:
            continue
        current = frame.loc[current_open]
        previous = frame.loc[close_24h_open]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[-1]
        if isinstance(previous, pd.DataFrame):
            previous = previous.iloc[-1]
        prev_close = float(previous["close"])
        curr_close = float(current["close"])
        if not np.isfinite(prev_close) or prev_close <= 0 or not np.isfinite(curr_close):
            continue
        rows.append({"symbol": symbol, "gain_24h": curr_close / prev_close - 1.0})
    if not rows:
        return pd.DataFrame(columns=["symbol", "gain_24h", "rank"])
    ranked = pd.DataFrame(rows).sort_values(["gain_24h", "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def generate_topn_signals(
    signal_start: int,
    signal_end: int,
    indexed_map: dict[str, pd.DataFrame],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_dt = ms_to_utc(signal_start).floor("D")
    end_dt = ms_to_utc(signal_end).floor("D")
    for day in pd.date_range(start=start_dt, end=end_dt, freq="D", tz="UTC"):
        for hour in SNAPSHOT_UTC_HOURS:
            signal_time = int((day + pd.Timedelta(hours=hour)).timestamp() * 1000)
            if signal_time < signal_start or signal_time > signal_end:
                continue
            ranking = build_snapshot_rankings_fast(signal_time, indexed_map).head(top_n)
            snapshot_hour_bj = (ms_to_utc(signal_time) + pd.Timedelta(hours=8)).strftime("%H:%M")
            for _, item in ranking.iterrows():
                symbol = str(item["symbol"])
                rows.append(
                    {
                        "signal_time": signal_time,
                        "signal_time_utc": ms_to_utc(signal_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "signal_time_bj": ms_to_bj_string(signal_time),
                        "snapshot_hour_bj": snapshot_hour_bj,
                        "symbol": symbol,
                        "rank": int(item["rank"]),
                        "gain_24h": float(item["gain_24h"]),
                        "pre7d_gain_pct": pre7d_gain_pct(symbol, signal_time, indexed_map),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def add_pre7d_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    labels = [item[0] for item in PRE7D_BUCKETS]
    bins = [PRE7D_BUCKETS[0][1]] + [item[2] for item in PRE7D_BUCKETS]
    frame["pre7d_gain_bucket"] = pd.cut(
        frame["pre7d_gain_pct"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return frame


def profit_factor(pnl: pd.Series) -> float:
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def tail_net(group: pd.DataFrame, remove_count: int) -> float:
    if group.empty:
        return 0.0
    pnl = group.sort_values("pnl_u", ascending=False)["pnl_u"].reset_index(drop=True)
    if len(pnl) <= remove_count:
        return 0.0
    return float(pnl.iloc[remove_count:].sum())


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].copy()
    skipped = group[~group["status"].eq("completed")]
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    net_returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    gross_returns = completed["gross_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[completed["pnl_u"] > 0]
    losses = completed[completed["pnl_u"] < 0]
    total_deployed = len(completed) * BUY_NOTIONAL_U
    return {
        "raw_signals": int(len(group)),
        "completed_trades": int(len(completed)),
        "skipped_trades": int(len(skipped)),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": float(len(wins) / len(completed)) if len(completed) else np.nan,
        "total_net_pnl_u": float(pnl.sum()) if len(pnl) else 0.0,
        "profit_factor": profit_factor(pnl),
        "avg_net_return_pct": float(net_returns.mean()) if len(net_returns) else np.nan,
        "median_net_return_pct": float(net_returns.median()) if len(net_returns) else np.nan,
        "avg_gross_return_pct": float(gross_returns.mean()) if len(gross_returns) else np.nan,
        "median_gross_return_pct": float(gross_returns.median()) if len(gross_returns) else np.nan,
        "avg_win_pct": float(wins["net_return_pct"].mean()) if len(wins) else np.nan,
        "avg_loss_pct": float(losses["net_return_pct"].mean()) if len(losses) else np.nan,
        "max_win_pct": float(net_returns.max()) if len(net_returns) else np.nan,
        "max_loss_pct": float(net_returns.min()) if len(net_returns) else np.nan,
        "avg_pnl_u": float(pnl.mean()) if len(pnl) else np.nan,
        "max_win_u": float(pnl.max()) if len(pnl) else np.nan,
        "max_loss_u": float(pnl.min()) if len(pnl) else np.nan,
        "max_drawdown_u": calculate_drawdown(pnl),
        "max_drawdown_pct_on_total_deployed": float(calculate_drawdown(pnl) / total_deployed) if total_deployed else np.nan,
        "pre7d_gain_median_pct": float(completed["pre7d_gain_pct"].median()) if len(completed) else np.nan,
        "pre7d_gain_avg_pct": float(completed["pre7d_gain_pct"].mean()) if len(completed) else np.nan,
        "gain_24h_median_pct": float(completed["gain_24h"].median() * 100.0) if len(completed) else np.nan,
        "gain_24h_avg_pct": float(completed["gain_24h"].mean() * 100.0) if len(completed) else np.nan,
        "net_after_drop_top1_u": tail_net(completed, 1),
        "net_after_drop_top3_u": tail_net(completed, 3),
        "net_after_drop_top5_u": tail_net(completed, 5),
        "net_after_drop_top10_u": tail_net(completed, 10),
    }


def bucket_summary(trades: pd.DataFrame, top_n: int, holding_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, _, _ in PRE7D_BUCKETS:
        group = trades[trades["pre7d_gain_bucket"].astype(str).eq(label)]
        rows.append({"top_n": top_n, "holding_days": holding_days, "pre7d_gain_bucket": label} | summarize(group))
    missing = trades[trades["pre7d_gain_bucket"].isna()]
    if not missing.empty:
        rows.append({"top_n": top_n, "holding_days": holding_days, "pre7d_gain_bucket": "过去7天涨幅 缺失"} | summarize(missing))
    return pd.DataFrame(rows)


def monthly_summary(trades: pd.DataFrame, top_n: int, holding_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in trades.groupby("month", sort=True):
        rows.append({"top_n": top_n, "holding_days": holding_days, "month": month} | summarize(group))
    return pd.DataFrame(rows)


def load_research_klines(signal_start: int, signal_end: int) -> tuple[list[str], dict[str, pd.DataFrame]]:
    client = SimpleBinanceFuturesClient()
    symbols = get_futures_symbols(client)
    kline_start = signal_start - 8 * DAY_MS
    kline_map = load_local_1h_cache(symbols, kline_start, signal_end)
    try:
        missing = [symbol for symbol in symbols if symbol not in kline_map]
        kline_map.update(load_local_5m_as_1h(missing, kline_start, signal_end))
    except ImportError as exc:
        print(f"Local 5m cache unavailable: {exc}", flush=True)
    return symbols, kline_map


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    args = parse_args()
    top_ns = [int(value.strip()) for value in args.top_n.split(",") if value.strip()]
    holding_days = int(args.holding_days)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    signal_end_dt = latest_signal_end_dt(now)
    signal_start_dt = signal_end_dt - timedelta(days=SIGNAL_DAYS)
    signal_start = int(signal_start_dt.timestamp() * 1000)
    signal_end = int(signal_end_dt.timestamp() * 1000)

    print("========== TopN Pre-7D Gain Bucket Research ==========")
    print(f"Signal Start: {ms_to_utc(signal_start).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Signal End:   {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"TopN: {top_ns}")
    print(f"Holding Days: {holding_days}")
    print(f"Fee Assumption: buy {FEE_RATE:.3%}, sell {FEE_RATE:.3%}, no slippage, 1x, 100 USDT per completed entry")
    print("No Binance downloads. Local 1H/5M cache only.")

    symbols, kline_map = load_research_klines(signal_start, signal_end)
    print(f"Symbols Count: {len(symbols)}")
    print(f"Loaded local kline symbols: {len(kline_map)}")
    indexed_map = make_indexed_kline_map(kline_map)

    all_summary: list[pd.DataFrame] = []
    all_buckets: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    written: list[Path] = []

    max_top_n = max(top_ns)
    print(f"\nGenerating Top{max_top_n} signals once...", flush=True)
    max_signals = generate_topn_signals(signal_start, signal_end, indexed_map, max_top_n)
    print(f"Top{max_top_n} raw signals: {len(max_signals)}")

    for top_n in top_ns:
        print(f"\nPreparing Top{top_n} sample...", flush=True)
        signals = max_signals[max_signals["rank"].le(top_n)].copy()
        print(f"Top{top_n} raw signals: {len(signals)}")
        trade_rows = simulate_trades_with_position_limit(signals, holding_days, kline_map)
        trades = pd.DataFrame(trade_rows)
        signal_attrs = signals[["signal_time", "symbol", "rank", "pre7d_gain_pct"]].rename(
            columns={"signal_time": "signal_time_ms"}
        )
        trades = trades.merge(signal_attrs, on=["signal_time_ms", "symbol", "rank"], how="left")
        trades = add_pre7d_bucket(trades)
        summary = pd.DataFrame([{"top_n": top_n, "holding_days": holding_days} | summarize(trades)])
        buckets = bucket_summary(trades, top_n, holding_days)
        monthly = monthly_summary(trades, top_n, holding_days)

        prefix = OUT / f"futures_top{top_n}_7d_pre7d_gain"
        written.extend(
            [
                write_csv(signals, prefix.with_name(f"{prefix.name}_signals.csv")),
                write_csv(trades, prefix.with_name(f"{prefix.name}_trades.csv")),
                write_csv(summary, prefix.with_name(f"{prefix.name}_summary.csv")),
                write_csv(buckets, prefix.with_name(f"{prefix.name}_buckets.csv")),
                write_csv(monthly, prefix.with_name(f"{prefix.name}_monthly.csv")),
            ]
        )
        all_summary.append(summary)
        all_buckets.append(buckets)
        all_monthly.append(monthly)

    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_buckets = pd.concat(all_buckets, ignore_index=True)
    combined_monthly = pd.concat(all_monthly, ignore_index=True)
    written.extend(
        [
            write_csv(combined_summary, OUT / "futures_top20_top50_7d_pre7d_gain_summary.csv"),
            write_csv(combined_buckets, OUT / "futures_top20_top50_7d_pre7d_gain_buckets.csv"),
            write_csv(combined_monthly, OUT / "futures_top20_top50_7d_pre7d_gain_monthly.csv"),
        ]
    )

    print("\n========== Overall Summary ==========")
    print(combined_summary.to_string(index=False))
    print("\n========== Pre-7D Gain Buckets ==========")
    print(combined_buckets.to_string(index=False))
    print("\nWrote files:")
    for path in written:
        print(path)


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    main()
