from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, add_entry_factors, load_kline_map, max_drawdown, ms_to_utc, profit_factor
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS
from scripts.regime_adaptive_leverage_walkforward import simulate_trade_with_leverage
import scripts.regime_adaptive_leverage_walkforward as leverage_engine
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "rank1_long_ev_research"
TARGET_LEVERAGES = [1, 2, 3, 5]
MIN_TRADES_FOR_CANDIDATE = 8


def gain_bucket(value: float) -> str:
    if value < 0.10:
        return "<10%"
    if value < 0.20:
        return "10-20%"
    if value < 0.40:
        return "20-40%"
    if value < 0.60:
        return "40-60%"
    if value < 0.80:
        return "60-80%"
    return ">=80%"


def volume_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 1.0:
        return "<1"
    if value < 1.2:
        return "1-1.2"
    if value < 1.5:
        return "1.2-1.5"
    if value < 2.0:
        return "1.5-2"
    if value < 3.0:
        return "2-3"
    if value < 4.0:
        return "3-4"
    if value < 5.0:
        return "4-5"
    if value < 6.0:
        return "5-6"
    if value < 8.0:
        return "6-8"
    return ">=8"


def completed_or_open(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return pd.DataFrame()
    return frame[frame["status"].isin(["completed", "open_mark_to_market"])].copy()


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    evaluated = completed_or_open(frame).sort_values("entry_time_ms")
    pnl = pd.to_numeric(evaluated["pnl_u"], errors="coerce") if not evaluated.empty else pd.Series(dtype=float)
    ret = pd.to_numeric(evaluated["net_return_pct"], errors="coerce") if not evaluated.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    closed = frame[frame["status"].eq("completed")] if "status" in frame else pd.DataFrame()
    return {
        "signals": int(len(frame)),
        "evaluated": int(len(evaluated)),
        "closed": int(len(closed)),
        "open_mtm": int(frame["status"].eq("open_mark_to_market").sum()) if "status" in frame else 0,
        "skipped": int(frame["status"].eq("skipped").sum()) if "status" in frame else 0,
        "net_pnl_u": float(pnl.sum()) if len(pnl) else 0.0,
        "gross_profit_u": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss_u": float(losses.sum()) if len(losses) else 0.0,
        "pf": profit_factor(pnl),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else np.nan,
        "avg_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "max_drawdown_u": max_drawdown(pnl),
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
        "drop_top1_u": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "drop_top3_u": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "liquidations": int(evaluated["liquidated"].astype(bool).sum()) if "liquidated" in evaluated else 0,
    }


def positive_complete_months(frame: pd.DataFrame, cutoff_ms: int) -> tuple[int, int]:
    evaluated = completed_or_open(frame)
    if evaluated.empty:
        return 0, 0
    month_end = ms_to_utc(cutoff_ms).strftime("%Y-%m")
    rows = []
    for month, group in evaluated.groupby("month", sort=True):
        if month >= month_end:
            continue
        rows.append(float(pd.to_numeric(group["pnl_u"], errors="coerce").sum()))
    return sum(1 for value in rows if value > 0), len(rows)


def simulate_independent(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int, leverage: int, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        common = {
            "strategy_label": label,
            "research_leverage": leverage,
            "gain_bucket": signal["rank1_gain_bucket"],
            "volume_bucket": signal["rank1_volume_bucket"],
            "hour_bj": signal["snapshot_hour_bj"],
        }
        if open_until is not None and signal_time < open_until:
            rows.append(
                {
                    "symbol": symbol,
                    "rank": int(signal["rank"]),
                    "entry_time_ms": signal_time,
                    "entry_time_utc": ms_to_utc(signal_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "month": ms_to_utc(signal_time).strftime("%Y-%m"),
                    "status": "skipped",
                    "skip_reason": "symbol_already_open",
                    **common,
                }
            )
            continue
        trade = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, leverage)
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            extra = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + extra
    return pd.DataFrame(rows)


def add_summary(rows: list[dict[str, Any]], frame: pd.DataFrame, cutoff_ms: int, kind: str, label: str, leverage: int, **keys: Any) -> None:
    stats = summarize(frame)
    pos, total = positive_complete_months(frame, cutoff_ms)
    rows.append(
        {
            "kind": kind,
            "label": label,
            "leverage": leverage,
            **keys,
            **stats,
            "positive_complete_months": pos,
            "complete_months": total,
            "positive_month_ratio": float(pos / total) if total else np.nan,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leverage_engine.LIQUIDATION_THRESHOLDS_PCT[1] = -100.0

    symbols = [symbol for symbol in cached_symbols() if symbol not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)

    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    base = raw[
        raw["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & raw["rank"].eq(1)
        & raw["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    base = add_entry_factors(base, kline_map)
    base["rank1_gain_bucket"] = base["gain_24h"].astype(float).map(gain_bucket)
    base["rank1_volume_bucket"] = base["volume_24h_ratio_7d"].astype(float).map(volume_bucket)
    base = base.sort_values(["signal_time", "symbol"]).reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []

    for leverage in TARGET_LEVERAGES:
        all_trades = simulate_independent(base, kline_map, common_end, leverage, f"all_rank1_{leverage}x")
        trade_frames.append(all_trades)
        add_summary(summary_rows, all_trades, common_end, "overall", "all_rank1", leverage)

        for gain, group in base.groupby("rank1_gain_bucket", sort=True):
            label = f"gain={gain}"
            trades = simulate_independent(group, kline_map, common_end, leverage, label)
            trade_frames.append(trades)
            add_summary(summary_rows, trades, common_end, "gain", label, leverage, gain_bucket=gain)

        for volume, group in base.groupby("rank1_volume_bucket", sort=True):
            label = f"volume={volume}"
            trades = simulate_independent(group, kline_map, common_end, leverage, label)
            trade_frames.append(trades)
            add_summary(summary_rows, trades, common_end, "volume", label, leverage, volume_bucket=volume)

        for (gain, volume), group in base.groupby(["rank1_gain_bucket", "rank1_volume_bucket"], sort=True):
            label = f"gain={gain}|volume={volume}"
            trades = simulate_independent(group, kline_map, common_end, leverage, label)
            trade_frames.append(trades)
            add_summary(summary_rows, trades, common_end, "gain_volume", label, leverage, gain_bucket=gain, volume_bucket=volume)

        for (gain, volume, hour), group in base.groupby(["rank1_gain_bucket", "rank1_volume_bucket", "snapshot_hour_bj"], sort=True):
            label = f"gain={gain}|volume={volume}|hour={hour}"
            trades = simulate_independent(group, kline_map, common_end, leverage, label)
            trade_frames.append(trades)
            add_summary(summary_rows, trades, common_end, "gain_volume_hour", label, leverage, gain_bucket=gain, volume_bucket=volume, hour_bj=hour)

    summary = pd.DataFrame(summary_rows).sort_values(["kind", "leverage", "gain_bucket", "volume_bucket", "hour_bj"], na_position="last")
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    candidates = summary[
        summary["evaluated"].ge(MIN_TRADES_FOR_CANDIDATE)
        & summary["pf"].gt(1.0)
        & summary["net_pnl_u"].gt(0)
        & summary["drop_top1_u"].gt(0)
    ].copy()
    candidates["robust_score"] = (
        candidates["drop_top1_u"].fillna(-1e9)
        + candidates["drop_top3_u"].fillna(-1e9) * 0.25
        + candidates["positive_month_ratio"].fillna(0) * 100.0
    )
    candidates = candidates.sort_values(["robust_score", "pf", "net_pnl_u"], ascending=False)

    base.to_csv(OUT_DIR / "rank1_signals.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "rank1_bucket_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT_DIR / "rank1_top_candidates.csv", index=False, encoding="utf-8-sig")
    all_trades.to_csv(OUT_DIR / "rank1_all_research_trades.csv", index=False, encoding="utf-8-sig")

    run_info = {
        "cutoff_utc": ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"),
        "signal_end_utc": ms_to_utc(signal_end).strftime("%Y-%m-%d %H:%M:%S"),
        "rank1_signal_count": int(len(base)),
        "symbols": int(len(symbols)),
        "excluded_symbols": sorted(EXCLUDE_SYMBOLS | {"RAVEUSDT"}),
        "fee_slippage_exit_model": "same simulate_trade_with_leverage as current main strategy: 0.1% per side, no extra slippage, 6D default, 4H extreme weak, 12H weak, liquidation by leverage.",
    }
    pd.Series(run_info).to_json(OUT_DIR / "run_info.json", force_ascii=False, indent=2)

    print("output", OUT_DIR)
    print(pd.Series(run_info).to_string())
    print("\nOVERALL")
    print(summary[summary["kind"].eq("overall")][["leverage", "evaluated", "open_mtm", "net_pnl_u", "pf", "win_rate", "median_return_pct", "drop_top1_u", "drop_top3_u", "max_drawdown_u", "liquidations"]].round(3).to_string(index=False))
    print("\nTOP_CANDIDATES")
    show = candidates.head(20)
    if show.empty:
        print("NONE")
    else:
        print(show[["kind", "label", "leverage", "evaluated", "net_pnl_u", "pf", "win_rate", "median_return_pct", "drop_top1_u", "drop_top3_u", "positive_complete_months", "complete_months", "max_drawdown_u", "liquidations"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
