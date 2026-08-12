from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, add_entry_factors, ms_to_utc  # noqa: E402
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt  # noqa: E402
from scripts.bucket_b_rank3_regime_optimization import (  # noqa: E402
    EXCLUDE_SYMBOLS,
    IndicatorSpec,
    build_health_timeline,
    opportunity_sets,
)
from scripts.rank3_fast_recovery_vs_monthly_reset import (  # noqa: E402
    RecoverySpec,
    build_recovery_timeline,
    fast_recovery_action_timeline,
    precompute_outcomes,
    simulate_actions,
)
from scripts.regime_adaptive_leverage_walkforward import bucket_for_signal  # noqa: E402
from scripts.run_current_main_strategy_2026_jan_jun import (  # noqa: E402
    SIGNAL_START_MS,
    cache_common_end_ms,
    cached_symbols,
    gain_bucket,
    load_kline_map,
)


OUT_DIR = OUT / "rank3_b_volume_floor_compare_fr3"
MODEL_NAME = "FR_avg_return24_l3_gt_0_fr3_yr1"


def leverage_for_variant(row: pd.Series, rank3_b_volume_min: float) -> int | None:
    gain = float(row["gain_24h"])
    rank = int(row["rank"])
    volume = float(row["volume_24h_ratio_7d"]) if pd.notna(row.get("volume_24h_ratio_7d")) else math.nan
    if 0.10 <= gain < 0.20 and rank in {2, 3}:
        return 3
    if 0.20 <= gain < 0.40:
        if rank == 2 and 1.5 <= volume < 5.0:
            return 3
        if rank == 3 and rank3_b_volume_min <= volume < 5.0:
            return 5
    if 0.40 <= gain < 0.60 and rank == 2 and 3.0 <= volume < 5.5:
        return 2
    return None


def apply_entry_rules_variant(raw: pd.DataFrame, kline_map: dict[str, pd.DataFrame], rank3_b_volume_min: float) -> pd.DataFrame:
    signals = raw[
        raw["snapshot_hour_bj"].isin(["00:00", "08:00"])
        & raw["rank"].isin([2, 3])
        & raw["symbol"].astype(str).ne("RAVEUSDT")
        & raw["gain_24h"].ge(0.10)
        & raw["gain_24h"].lt(0.80)
    ].copy()
    signals = add_entry_factors(signals, kline_map)
    signals["leverage"] = signals.apply(lambda row: leverage_for_variant(row, rank3_b_volume_min), axis=1)
    signals = signals[signals["leverage"].notna()].copy()
    signals["leverage"] = signals["leverage"].astype(int)
    signals["gain_24h_bucket"] = signals["gain_24h"].astype(float).map(gain_bucket)
    signals["bucket"] = signals.apply(bucket_for_signal, axis=1)
    signals["signal_id"] = range(len(signals))
    return signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def profit_factor(pnl: pd.Series) -> float | None:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return None if gross_profit == 0 else math.inf
    return gross_profit / gross_loss


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    evaluated = group[group["status"].isin(["completed", "open_mark_to_market"])].sort_values("entry_time_ms").copy()
    pnl = pd.to_numeric(evaluated.get("pnl_u", pd.Series(dtype=float)), errors="coerce")
    ret = pd.to_numeric(evaluated.get("net_return_pct", pd.Series(dtype=float)), errors="coerce")
    equity = pnl.cumsum()
    max_drawdown = float((equity - equity.cummax()).min()) if len(pnl) else 0.0
    pf = profit_factor(pnl) if len(pnl) else None
    return {
        "signals": int(len(group)),
        "evaluated": int(len(evaluated)),
        "closed": int(evaluated["status"].eq("completed").sum()) if len(evaluated) else 0,
        "open_mtm": int(evaluated["status"].eq("open_mark_to_market").sum()) if len(evaluated) else 0,
        "skipped": int(group["status"].eq("skipped").sum()) if "status" in group else 0,
        "net_pnl_u": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "pf": round(float(pf), 2) if pf is not None and math.isfinite(pf) else pf,
        "win_rate_pct": round(float((pnl > 0).sum() / len(evaluated) * 100), 2) if len(evaluated) else None,
        "median_return_pct": round(float(ret.median()), 2) if len(ret) else None,
        "avg_return_pct": round(float(ret.mean()), 2) if len(ret) else None,
        "max_drawdown_u": round(max_drawdown, 2),
        "liquidations": int(evaluated["liquidated"].fillna(False).sum()) if "liquidated" in evaluated else 0,
        "best_trade_u": round(float(pnl.max()), 2) if len(pnl) else None,
        "worst_trade_u": round(float(pnl.min()), 2) if len(pnl) else None,
        "drop_top1_u": round(float(pnl.sum() - pnl.nlargest(1).sum()), 2) if len(pnl) >= 1 else None,
    }


def run_variant(raw: pd.DataFrame, kline_map: dict[str, pd.DataFrame], common_end: int, rank3_b_volume_min: float, label: str) -> pd.DataFrame:
    filtered = apply_entry_rules_variant(raw, kline_map, rank3_b_volume_min)
    signal_times = sorted(filtered["signal_time"].astype(int).unique())
    sets = opportunity_sets(raw, kline_map)
    d15_spec = IndicatorSpec("D_b_r3_decay_l15", "B_R3", "mean_decay48", "lower_bad", 15)
    d15 = build_health_timeline(signal_times, sets["B_R3"], d15_spec)
    recovery_spec = RecoverySpec("avg_return24_l3_gt_0", "avg_return24", 3, "gt_0")
    recovery = build_recovery_timeline(signal_times, sets["B_R3"], recovery_spec)
    action = fast_recovery_action_timeline(d15, recovery, "fr3", "yr1", MODEL_NAME)
    outcomes = precompute_outcomes(filtered, kline_map, common_end)
    trades = simulate_actions(filtered, outcomes, action, label)
    trades["rank3_b_volume_min"] = rank3_b_volume_min
    return trades


def key_frame(trades: pd.DataFrame, source: str) -> pd.DataFrame:
    evaluated = trades[trades["status"].isin(["completed", "open_mark_to_market"])].copy()
    evaluated["trade_key"] = (
        evaluated["entry_time_ms"].astype("int64").astype(str)
        + "|"
        + evaluated["symbol"].astype(str)
        + "|"
        + evaluated["rank"].astype("int64").astype(str)
    )
    evaluated["source"] = source
    return evaluated


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [symbol for symbol in cached_symbols() if symbol not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)
    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)

    current = run_variant(raw, kline_map, common_end, 1.5, "current_rank3_b_vol_1p5_5")
    variant = run_variant(raw, kline_map, common_end, 1.2, "variant_rank3_b_vol_1p2_5")
    all_trades = pd.concat([current, variant], ignore_index=True)

    summary = pd.DataFrame(
        [
            {"strategy": "current_rank3_b_vol_1p5_5", "cutoff_utc": ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"), **summarize(current)},
            {"strategy": "variant_rank3_b_vol_1p2_5", "cutoff_utc": ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"), **summarize(variant)},
        ]
    )
    monthly = pd.DataFrame(
        [
            {"strategy": strategy, "month": month, **summarize(group)}
            for strategy, sg in all_trades.groupby("candidate", sort=True)
            for month, group in sg.groupby("month", sort=True)
        ]
    )
    cur_keys = key_frame(current, "current")
    var_keys = key_frame(variant, "variant")
    added = var_keys[~var_keys["trade_key"].isin(set(cur_keys["trade_key"]))].copy()
    removed = cur_keys[~cur_keys["trade_key"].isin(set(var_keys["trade_key"]))].copy()
    affected_summary = pd.DataFrame(
        [
            {"change": "added_in_variant", **summarize(added)},
            {"change": "missing_vs_current", **summarize(removed)},
        ]
    )

    all_trades.to_csv(OUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly.csv", index=False, encoding="utf-8-sig")
    added.to_csv(OUT_DIR / "added_in_variant.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(OUT_DIR / "missing_vs_current.csv", index=False, encoding="utf-8-sig")
    affected_summary.to_csv(OUT_DIR / "affected_summary.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nMONTHLY")
    print(monthly.to_string(index=False))
    print("\nAFFECTED")
    print(affected_summary.to_string(index=False))
    print(f"\nfiles: {OUT_DIR}")


if __name__ == "__main__":
    main()
