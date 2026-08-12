from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import INTERVAL_MS, load_cached_klines
from src.data.quality import check_klines
from src.research.ranking import add_rolling_24h_change


DEFAULT_OUT = ROOT / "outputs" / "decline_top10_thresholds"
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
LIQUIDATION_UP_MOVE = 0.10


def interval_to_minutes(interval: str) -> int:
    return int(INTERVAL_MS[interval] / 60_000)


def build_decline_rankings(klines: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    frame = add_rolling_24h_change(klines, interval_minutes)
    frame = frame[frame["has_full_24h_history"]].copy()
    frame["decline_rank"] = frame.groupby("open_time")["rolling_24h_change_pct"].rank(method="first", ascending=True)
    frame["decline_rank"] = frame["decline_rank"].astype(int)
    frame["is_decline_top10"] = frame["decline_rank"] <= 10
    columns = [
        "open_time",
        "open_time_utc",
        "symbol",
        "decline_rank",
        "rolling_24h_change_pct",
        "close",
        "quote_volume",
        "is_decline_top10",
    ]
    return frame[columns].sort_values(["open_time", "decline_rank"]).reset_index(drop=True)


def identify_first_decline_top10_signals(
    rankings: pd.DataFrame,
    cooldown_days: int,
) -> pd.DataFrame:
    cooldown_ms = cooldown_days * 24 * 60 * 60 * 1000
    rows = []
    signal_id = 1
    for symbol, group in rankings.sort_values("open_time").groupby("symbol"):
        top_hits = group[group["is_decline_top10"]].copy()
        previous_time: int | None = None
        for _, row in top_hits.iterrows():
            now = int(row["open_time"])
            if previous_time is not None and now - previous_time <= cooldown_ms:
                previous_time = now
                continue
            rows.append(
                {
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "signal_time": now,
                    "signal_time_utc": row["open_time_utc"],
                    "decline_rank_at_signal": int(row["decline_rank"]),
                    "rolling_24h_change_pct_at_signal": float(row["rolling_24h_change_pct"]),
                    "close_at_signal": float(row["close"]),
                    "quote_volume_at_signal": float(row.get("quote_volume", 0)),
                    "cooldown_days": cooldown_days,
                }
            )
            signal_id += 1
            previous_time = now
    return pd.DataFrame(rows)


def scan_short_path(
    group: pd.DataFrame,
    signal_time: int,
    lookahead_hours: int,
) -> dict[str, object]:
    future_idx = group.index[group["open_time"] > signal_time]
    if len(future_idx) == 0:
        return {"evaluated": False}

    entry_idx = int(future_idx[0])
    entry = group.loc[entry_idx]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = entry_time + lookahead_hours * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    if path.empty:
        return {"evaluated": False}

    liquidation_price = entry_price * (1.0 + LIQUIDATION_UP_MOVE)
    result: dict[str, object] = {
        "evaluated": True,
        "entry_time": entry_time,
        "entry_time_utc": entry["open_time_utc"],
        "entry_price": entry_price,
        "liquidation_price": liquidation_price,
        "liquidated_before_any_threshold": False,
        "first_terminal_event": "lookahead_end",
        "first_terminal_time": int(path.iloc[-1]["open_time"]),
        "first_terminal_time_utc": path.iloc[-1]["open_time_utc"],
        "max_adverse_up_return": float(path["high"].max() / entry_price - 1.0),
        "max_favorable_down_return": float(1.0 - path["low"].min() / entry_price),
    }
    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        result[f"hit_down_{pct}_before_liquidation"] = False
        result[f"time_to_down_{pct}_minutes"] = pd.NA

    for _, bar in path.iterrows():
        bar_time = int(bar["open_time"])
        high = float(bar["high"])
        low = float(bar["low"])

        # Conservative same-candle rule for 10x short: liquidation wins if both
        # adverse +10% and a downside threshold print inside the same 5m candle.
        if high >= liquidation_price:
            result["liquidated_before_any_threshold"] = not any(
                bool(result[f"hit_down_{int(threshold * 100)}_before_liquidation"])
                for threshold in THRESHOLDS
            )
            result["first_terminal_event"] = "liquidation_up_10"
            result["first_terminal_time"] = bar_time
            result["first_terminal_time_utc"] = bar["open_time_utc"]
            return result

        for threshold in THRESHOLDS:
            pct = int(threshold * 100)
            key = f"hit_down_{pct}_before_liquidation"
            if not result[key] and low <= entry_price * (1.0 - threshold):
                result[key] = True
                result[f"time_to_down_{pct}_minutes"] = int((bar_time - entry_time) / 60_000)

    return result


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    total = len(detail)
    rows = []
    for threshold in THRESHOLDS:
        pct = int(threshold * 100)
        col = f"hit_down_{pct}_before_liquidation"
        count = int(detail[col].sum()) if total else 0
        rows.append(
            {
                "threshold": f"-{pct}%",
                "hit_count_before_plus10_liquidation": count,
                "hit_rate_before_plus10_liquidation": count / total if total else 0.0,
                "evaluated_signals": total,
            }
        )
    liq_count = int(detail["liquidated_before_any_threshold"].sum()) if total else 0
    rows.append(
        {
            "threshold": "first_liquidation_up_10",
            "hit_count_before_plus10_liquidation": liq_count,
            "hit_rate_before_plus10_liquidation": liq_count / total if total else 0.0,
            "evaluated_signals": total,
        }
    )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text_df.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).values.tolist()]
    return "\n".join([header, separator, *body])


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    lines = [
        "# Decline Top10 Threshold Research",
        "",
        "Signal: first entry into Binance USDT-M Futures rolling 24h decline Top10.",
        "Entry: next 5m candle open after the signal.",
        "Direction: 10x short stress test; adverse +10% from entry is treated as liquidation.",
        "Same-candle assumption: if a 5m candle touches both liquidation and downside target, liquidation is counted first.",
        "Fees/slippage: not applied, because this output is threshold path counting rather than PnL backtest.",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Threshold Summary", "", dataframe_to_markdown(summary), "", "## Monthly Signals", ""])
    lines.append(dataframe_to_markdown(monthly) if not monthly.empty else "No evaluated signals.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Counts are based on available cached 5m K-lines only.",
            "- Intrabar order is unknown; the same-candle liquidation-first rule is conservative for short entries.",
            "- This is not a profitability conclusion and does not include fees, slippage, funding, borrow constraints, or position sizing.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--lookahead-hours", type=int, default=240)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--save-rankings", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate report.md from existing CSV outputs.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    interval = config["data"]["interval"]
    interval_minutes = interval_to_minutes(interval)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        summary = pd.read_csv(out_dir / "threshold_summary.csv")
        monthly = pd.read_csv(out_dir / "monthly_signal_counts.csv")
        metadata = pd.read_csv(out_dir / "summary.csv").iloc[0].to_dict()
        write_report(out_dir, summary, monthly, metadata)
        print(f"Wrote {out_dir / 'report.md'}", flush=True)
        return

    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", interval)
    if klines.empty:
        raise RuntimeError("No cached klines found.")
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

    cache_end_ms = int(klines["open_time"].max())
    start_ms = cache_end_ms - int(args.lookback_days) * 24 * 60 * 60 * 1000
    klines = klines[(klines["open_time"] >= start_ms) & (klines["open_time"] <= cache_end_ms)].copy()
    klines = klines.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(
        f"Using range {pd.to_datetime(start_ms, unit='ms', utc=True)} to "
        f"{pd.to_datetime(cache_end_ms, unit='ms', utc=True)} rows={len(klines):,}",
        flush=True,
    )

    quality = check_klines(klines, INTERVAL_MS[interval])
    quality.to_csv(out_dir / "data_quality.csv", index=False)

    print("Building decline Top10 rankings...", flush=True)
    rankings = build_decline_rankings(klines, interval_minutes)
    if args.save_rankings:
        rankings.to_csv(out_dir / "decline_ranking_snapshots.csv", index=False)

    print("Identifying first decline Top10 signals...", flush=True)
    signals = identify_first_decline_top10_signals(
        rankings,
        cooldown_days=int(config["strategy"]["cooldown_days"]),
    ).sort_values("signal_time")
    signals.to_csv(out_dir / "signals_all_first_decline_top10.csv", index=False)

    kline_map = {symbol: group.reset_index(drop=True) for symbol, group in klines.groupby("symbol", sort=False)}
    rows = []
    for _, signal in signals.iterrows():
        group = kline_map.get(signal["symbol"])
        if group is None:
            continue
        scanned = scan_short_path(group, int(signal["signal_time"]), int(args.lookahead_hours))
        if not scanned["evaluated"]:
            continue
        rows.append({**signal.to_dict(), **scanned})

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "signal_threshold_detail.csv", index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "threshold_summary.csv", index=False)

    if detail.empty:
        monthly = pd.DataFrame(columns=["signal_month_utc", "evaluated_signals"])
    else:
        detail["signal_month_utc"] = pd.to_datetime(detail["signal_time_utc"], utc=True).dt.strftime("%Y-%m")
        monthly = detail.groupby("signal_month_utc").size().reset_index(name="evaluated_signals")
    monthly.to_csv(out_dir / "monthly_signal_counts.csv", index=False)

    metadata = {
        "lookback_days": int(args.lookback_days),
        "lookahead_hours": int(args.lookahead_hours),
        "interval": interval,
        "cache_start_utc": pd.to_datetime(start_ms, unit="ms", utc=True),
        "cache_end_utc": pd.to_datetime(cache_end_ms, unit="ms", utc=True),
        "klines_rows": len(klines),
        "ranking_rows": len(rankings),
        "first_decline_top10_signals": len(signals),
        "evaluated_signals": len(detail),
        "unique_symbols": detail["symbol"].nunique() if not detail.empty else 0,
        "cooldown_days": int(config["strategy"]["cooldown_days"]),
        "liquidation_assumption": "+10% adverse move from entry for 10x short",
    }
    pd.DataFrame([metadata]).to_csv(out_dir / "summary.csv", index=False)
    write_report(out_dir, summary, monthly, metadata)

    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
