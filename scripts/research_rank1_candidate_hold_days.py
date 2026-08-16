from __future__ import annotations

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
import scripts.regime_adaptive_leverage_walkforward as leverage_engine
from scripts.regime_adaptive_leverage_walkforward import simulate_trade_with_leverage
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "rank1_candidate_hold_days"
LEVERAGE = 3
HOLD_OPTIONS = [
    ("24H", 1),
    ("48H", 2),
    ("72H", 3),
    ("4D", 4),
    ("5D", 5),
    ("6D", 6),
]
GAIN_BUCKETS = [("20-40%", 0.20, 0.40), ("40-60%", 0.40, 0.60)]
VR_BUCKETS = [("2-3", 2.0, 3.0), ("5-6", 5.0, 6.0)]


def bucket_label(value: float, buckets: list[tuple[str, float, float]]) -> str | None:
    for label, lower, upper in buckets:
        if lower <= value < upper:
            return label
    return None


def evaluated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return pd.DataFrame()
    return frame[frame["status"].isin(["completed", "open_mark_to_market"])].copy()


def summarize(frame: pd.DataFrame, raw_signals: int) -> dict[str, Any]:
    done = evaluated(frame).sort_values("entry_time_ms")
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liqs = done[done["liquidated"].astype(bool)] if len(done) and "liquidated" in done else pd.DataFrame()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    net = float(pnl.sum()) if len(pnl) else 0.0
    return {
        "raw_signals": int(raw_signals),
        "trades": int(len(done)),
        "conflict_skips": int(frame["status"].eq("skipped").sum()) if "status" in frame else 0,
        "open_mtm": int(frame["status"].eq("open_mark_to_market").sum()) if "status" in frame else 0,
        "net_pnl_u": net,
        "pf": profit_factor(pnl),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "avg_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "ex_top1_pnl_u": float(net - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "pnl_per_trade_u": float(net / len(pnl)) if len(pnl) else np.nan,
        "max_dd_u": max_drawdown(pnl),
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
        "liquidations": int(len(liqs)),
        "liq_rate": float(len(liqs) / len(done)) if len(done) else np.nan,
    }


def simulate_pool(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        common = {
            "pool": label,
            "research_leverage": LEVERAGE,
            "gain_bucket": signal["gain_bucket"],
            "vr_bucket": signal["vr_bucket"],
        }
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            rows.append(
                {
                    "symbol": symbol,
                    "rank": 1,
                    "entry_time_ms": signal_time,
                    "entry_time_utc": ms_to_utc(signal_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_time_bj": signal.get("snapshot_time_bj", ""),
                    "month": ms_to_utc(signal_time).strftime("%Y-%m"),
                    "status": "skipped",
                    "skip_reason": "symbol_already_open",
                    **common,
                }
            )
            continue
        trade = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, LEVERAGE)
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            extra = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + extra
    return pd.DataFrame(rows)


def monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    done = evaluated(trades)
    rows = []
    for month, group in done.groupby("month", sort=True):
        pnl = pd.to_numeric(group["pnl_u"], errors="coerce")
        rows.append(
            {
                "month": month,
                "trades": int(len(group)),
                "net_pnl_u": float(pnl.sum()),
                "liquidations": int(group["liquidated"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [symbol for symbol in cached_symbols() if symbol not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)

    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    signals = raw[
        raw["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & raw["rank"].eq(1)
        & raw["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)
    signals["gain_bucket"] = signals["gain_24h"].astype(float).map(lambda value: bucket_label(value, GAIN_BUCKETS))
    signals["vr_bucket"] = signals["volume_24h_ratio_7d"].astype(float).map(lambda value: bucket_label(value, VR_BUCKETS))
    scoped = signals[signals["gain_bucket"].notna() & signals["vr_bucket"].notna()].copy()
    scoped = scoped[~(scoped["gain_bucket"].eq("40-60%") & scoped["vr_bucket"].eq("5-6"))].copy()
    scoped = scoped.sort_values(["signal_time", "symbol"]).reset_index(drop=True)

    original_hold_days = leverage_engine.HOLD_DAYS
    rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    try:
        for label, hold_days in HOLD_OPTIONS:
            leverage_engine.HOLD_DAYS = hold_days
            trades = simulate_pool(scoped, kline_map, common_end, f"hold={label}")
            trade_frames.append(trades.assign(hold_option=label, hold_days=hold_days))
            rows.append({"hold_option": label, "hold_days": hold_days, **summarize(trades, len(scoped))})
            monthly = monthly_summary(trades)
            if not monthly.empty:
                monthly["hold_option"] = label
                monthly["hold_days"] = hold_days
                monthly_frames.append(monthly)
    finally:
        leverage_engine.HOLD_DAYS = original_hold_days

    summary = pd.DataFrame(rows)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()

    summary.to_csv(OUT_DIR / "hold_days_summary.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv(OUT_DIR / "hold_days_trade_details.csv", index=False, encoding="utf-8-sig")
    monthly_all.to_csv(OUT_DIR / "hold_days_monthly.csv", index=False, encoding="utf-8-sig")
    scoped.to_csv(OUT_DIR / "raw_candidate_signals.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Rank1 Candidate Hold Days Research",
        "",
        f"- Cutoff: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        "- Entry: Rank1, Gain 20-60, V/R 2-3 or 5-6, excluding 40-60/V/R 5-6, BJ 00:00 and 08:00.",
        "- Leverage: fixed 3x.",
        "- Only default max hold changes: 24H, 48H, 72H, 4D, 5D, 6D.",
        "- 4H extreme weak exit and 12H weak exit remain enabled.",
        "",
        summary.round(3).to_string(index=False),
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")

    print("output", OUT_DIR)
    print("cutoff_utc", ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"))
    print(summary[["hold_option", "trades", "net_pnl_u", "pf", "win_rate", "median_return_pct", "avg_return_pct", "ex_top1_pnl_u", "ex_top3_pnl_u", "pnl_per_trade_u", "max_dd_u", "liquidations", "liq_rate"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
