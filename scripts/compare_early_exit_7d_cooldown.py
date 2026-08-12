from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (
    CACHE_DIR,
    DAY_MS,
    EARLY_REASON,
    OUT,
    add_entry_factors,
    load_kline_map,
    max_drawdown,
    ms_to_bj_string,
    ms_to_utc,
    profit_factor,
    simulate_main_trade,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt


PREFIX = "main_strategy_early_exit_7d_cooldown_compare"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def apply_main_filters(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].lt(0.80)
        & signals["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)
    in_4060 = signals["gain_24h"].ge(0.40) & signals["gain_24h"].lt(0.60)
    pass_4060 = (
        signals["rank"].eq(2)
        & signals["volume_24h_ratio_7d"].ge(3.0)
        & signals["volume_24h_ratio_7d"].lt(6.0)
    )
    in_6080 = signals["gain_24h"].ge(0.60) & signals["gain_24h"].lt(0.80)
    return signals[
        ((~in_4060) | pass_4060)
        & (~in_6080)
    ].sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def simulate_with_cooldown(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], early_exit_cooldown_7d: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    for _, signal in ordered.iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            rows.append(skipped_open_position_trade(signal, open_until))
            continue
        trade = simulate_main_trade(signal, kline_map)
        rows.append(trade)
        if trade.get("status") == "completed":
            if early_exit_cooldown_7d and trade.get("exit_reason") == EARLY_REASON:
                open_until_by_symbol[symbol] = signal_time + 7 * DAY_MS
            else:
                open_until_by_symbol[symbol] = int(trade["exit_time_ms"])
    return pd.DataFrame(rows)


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"].eq("completed")].sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[pnl > 0]
    losses = completed[pnl < 0]
    early = completed[completed["exit_reason"].eq(EARLY_REASON)]
    return {
        "过滤后信号": int(len(group)),
        "完成交易": int(len(completed)),
        "持仓/冷却跳过": int((group["status"] != "completed").sum()) if "status" in group else 0,
        "早退交易": int(len(early)),
        "盈利单": int(len(wins)),
        "亏损单": int(len(losses)),
        "盈利总金额U": float(pnl[pnl > 0].sum()) if len(pnl) else 0.0,
        "亏损总金额U": float(pnl[pnl < 0].sum()) if len(pnl) else 0.0,
        "净收益U": float(pnl.sum()) if len(pnl) else 0.0,
        "PF": profit_factor(pnl),
        "胜率": float(len(wins) / len(completed)) if len(completed) else np.nan,
        "平均收益率": float(returns.mean()) if len(returns) else np.nan,
        "中位数收益率": float(returns.median()) if len(returns) else np.nan,
        "最大回撤U": max_drawdown(pnl),
        "最大单笔收益U": float(pnl.max()) if len(pnl) else np.nan,
        "最大单笔亏损U": float(pnl.min()) if len(pnl) else np.nan,
        "去最大1笔后U": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "去最大3笔后U": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "去最大5笔后U": float(pnl.sum() - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
    }


def add_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gain_24h_bucket"] = pd.cut(
        frame["gain_24h"],
        bins=[-np.inf, 0.10, 0.20, 0.40, 0.60, 0.80],
        labels=["<10%", "10%-20%", "20%-40%", "40%-60%", "60%-80%"],
        right=False,
    )
    return frame


def run_window(name: str, start: int, end: int, kline_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = apply_main_filters(generate_signals(start, end, kline_map), kline_map)
    before = add_bucket(simulate_with_cooldown(signals, kline_map, early_exit_cooldown_7d=False))
    after = add_bucket(simulate_with_cooldown(signals, kline_map, early_exit_cooldown_7d=True))
    before["版本"] = "before_no_early_exit_cooldown"
    after["版本"] = "after_early_exit_7d_cooldown"
    before["window"] = name
    after["window"] = name
    summary = pd.DataFrame(
        [
            {"window": name, "版本": "before_no_early_exit_cooldown", **summarize(before)},
            {"window": name, "版本": "after_early_exit_7d_cooldown", **summarize(after)},
        ]
    )
    return pd.concat([before, after], ignore_index=True), summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    end = int(latest_signal_end_dt().timestamp() * 1000)
    recent_start = end - 180 * DAY_MS
    full_start = end - 365 * DAY_MS
    kline_start = full_start - 7 * DAY_MS
    kline_end = end + 7 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), kline_start, kline_end)

    outputs = []
    summaries = []
    for name, start in [("recent_180d", recent_start), ("full_365d", full_start)]:
        trades, summary = run_window(name, start, end, kline_map)
        outputs.append(trades)
        summaries.append(summary)

    all_trades = pd.concat(outputs, ignore_index=True)
    all_summary = pd.concat(summaries, ignore_index=True)
    all_trades.to_csv(OUT / f"{PREFIX}_trades.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(OUT / f"{PREFIX}_summary.csv", index=False, encoding="utf-8-sig")

    monthly_rows = []
    for (window, version, month), group in all_trades.groupby(["window", "版本", "month"], sort=True):
        monthly_rows.append({"window": window, "版本": version, "month": month, **summarize(group)})
    pd.DataFrame(monthly_rows).to_csv(OUT / f"{PREFIX}_monthly.csv", index=False, encoding="utf-8-sig")

    print("========== Early Exit 7D Cooldown Compare ==========")
    print(all_summary.to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
