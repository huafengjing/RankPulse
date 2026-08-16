from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.regime_adaptive_leverage_walkforward as leverage_engine
from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, add_entry_factors, load_kline_map, max_drawdown, ms_to_utc, profit_factor, skipped_open_position_trade
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS, IndicatorSpec, build_health_timeline, opportunity_sets
from scripts.rank3_fast_recovery_vs_monthly_reset import RecoverySpec, build_recovery_timeline, fast_recovery_action_timeline
from scripts.regime_adaptive_leverage_walkforward import bucket_for_signal, simulate_trade_with_leverage
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, apply_entry_rules, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "rank1_portfolio_variant_comparison"
BASELINE = "baseline_rank2_rank3"
RANK23_HOLD_DAYS = 6


@dataclass(frozen=True)
class Variant:
    name: str
    rank1_hold_days: int
    rank1_base_leverage: int
    adaptive_like_rank3: bool = False
    cell_overrides: dict[str, tuple[int, int]] | None = None


VARIANTS = [
    Variant(
        "rank1_mixed_3x5d_v23_5x2d_v56_4060v23",
        5,
        3,
        False,
        {
            "20-40 / V2-3": (3, 5),
            "20-40 / V5-6": (5, 2),
            "40-60 / V2-3": (5, 2),
        },
    ),
    Variant("rank1_3x_5d_current_candidate", 5, 3, False),
    Variant("rank1_5x_2d", 2, 5, False),
    Variant("rank1_5x_5d", 5, 5, False),
    Variant("rank1_5x_5d_fr3yr1_like_rank3", 5, 5, True),
]


def rank1_cell(row: pd.Series) -> str | None:
    gain = float(row["gain_24h"])
    vr = float(row["volume_24h_ratio_7d"]) if pd.notna(row.get("volume_24h_ratio_7d", np.nan)) else math.nan
    if 0.20 <= gain < 0.40 and 2.0 <= vr < 3.0:
        return "20-40 / V2-3"
    if 0.20 <= gain < 0.40 and 5.0 <= vr < 6.0:
        return "20-40 / V5-6"
    if 0.40 <= gain < 0.60 and 2.0 <= vr < 3.0:
        return "40-60 / V2-3"
    return None


def evaluated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return pd.DataFrame()
    return frame[frame["status"].isin(["completed", "open_mark_to_market"])].copy()


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    done = evaluated(frame)
    if not done.empty and all(col in done.columns for col in ["entry_time_ms", "rank", "symbol"]):
        done = done.sort_values(["entry_time_ms", "rank", "symbol"]).copy()
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liq = done["liquidated"].astype(bool) if len(done) and "liquidated" in done else pd.Series(dtype=bool)
    net = float(pnl.sum()) if len(pnl) else 0.0
    return {
        "signals": int(len(frame)),
        "trades": int(len(done)),
        "closed_trades": int(done["status"].eq("completed").sum()) if len(done) else 0,
        "open_mark_to_market": int(done["status"].eq("open_mark_to_market").sum()) if len(done) else 0,
        "skipped": int(frame["status"].eq("skipped").sum()) if "status" in frame else 0,
        "net_pnl_u": net,
        "pf": profit_factor(pnl),
        "win_rate": float((pnl > 0).sum() / len(pnl)) if len(pnl) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "avg_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "max_drawdown_u": max_drawdown(pnl),
        "liquidations": int(liq.sum()) if len(liq) else 0,
        "liq_rate": float(liq.sum() / len(done)) if len(done) else np.nan,
        "ex_top1_pnl_u": float(net - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "ex_top5_pnl_u": float(net - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
        "ex_top10_pnl_u": float(net - pnl.nlargest(10).sum()) if len(pnl) >= 10 else np.nan,
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
    }


def add_rank1_candidates(raw: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rank1 = raw[
        raw["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & raw["rank"].eq(1)
        & raw["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    rank1 = add_entry_factors(rank1, kline_map)
    rank1["rank1_cell"] = rank1.apply(rank1_cell, axis=1)
    rank1 = rank1[rank1["rank1_cell"].notna()].copy()
    rank1["strategy_component"] = "Rank1"
    rank1["bucket"] = "R1"
    return rank1


def build_fr3_yr1_actions(raw: pd.DataFrame, rank23: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signal_times = sorted(rank23["signal_time"].astype(int).unique())
    sets = opportunity_sets(raw, kline_map)
    d15 = build_health_timeline(signal_times, sets["B_R3"], IndicatorSpec("D_b_r3_decay_l15", "B_R3", "mean_decay48", "lower_bad", 15))
    recovery = build_recovery_timeline(signal_times, sets["B_R3"], RecoverySpec("avg_return24_l3_gt_0", "avg_return24", 3, "gt_0"))
    return fast_recovery_action_timeline(d15, recovery, "fr3", "yr1", "FR_avg_return24_l3_gt_0_fr3_yr1")


def rank23_leverage(signal: pd.Series, action: dict[str, Any]) -> int:
    bucket = bucket_for_signal(signal)
    if bucket == "B" and int(signal["rank"]) == 2:
        return int(action.get("r2_lev", 3))
    if bucket == "B" and int(signal["rank"]) == 3:
        return int(action.get("r3_lev", 5))
    return int(signal["leverage"])


def rank1_leverage(variant: Variant, action: dict[str, Any]) -> int:
    if variant.adaptive_like_rank3:
        return int(action.get("r3_lev", variant.rank1_base_leverage))
    return variant.rank1_base_leverage


def rank1_params(variant: Variant, signal: pd.Series, action: dict[str, Any]) -> tuple[int, int]:
    cell = str(signal.get("rank1_cell", ""))
    if variant.cell_overrides and cell in variant.cell_overrides:
        return variant.cell_overrides[cell]
    return rank1_leverage(variant, action), variant.rank1_hold_days


def precompute_outcomes(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int) -> dict[tuple[int, int, int], dict[str, Any]]:
    original_hold = leverage_engine.HOLD_DAYS
    outcomes: dict[tuple[int, int, int], dict[str, Any]] = {}
    rank1_hold_days = sorted({v.rank1_hold_days for v in VARIANTS} | {hold for v in VARIANTS for _, hold in (v.cell_overrides or {}).values()})
    rank1_leverages = sorted({1, 3, 5} | {lev for v in VARIANTS for lev, _ in (v.cell_overrides or {}).values()})
    try:
        for row in signals.itertuples(index=False):
            signal = pd.Series(row._asdict())
            sid = int(signal["signal_id"])
            if int(signal["rank"]) == 1:
                for hold_days in rank1_hold_days:
                    leverage_engine.HOLD_DAYS = hold_days
                    for lev in rank1_leverages:
                        outcomes[(sid, lev, hold_days)] = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, lev)
            else:
                leverage_engine.HOLD_DAYS = RANK23_HOLD_DAYS
                for lev in [1, 2, 3, 5]:
                    outcomes[(sid, lev, RANK23_HOLD_DAYS)] = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, lev)
    finally:
        leverage_engine.HOLD_DAYS = original_hold
    return outcomes


def replay_portfolio(
    strategy: str,
    signals: pd.DataFrame,
    outcomes: dict[tuple[int, int, int], dict[str, Any]],
    actions: pd.DataFrame,
    variant: Variant | None,
) -> pd.DataFrame:
    action_by_time = actions.set_index("signal_time").to_dict("index") if not actions.empty else {}
    rows: list[dict[str, Any]] = []
    open_by_symbol: dict[str, dict[str, Any]] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        rank = int(signal["rank"])
        if rank == 1 and variant is None:
            continue
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        action = action_by_time.get(signal_time, {})
        if rank == 1:
            assert variant is not None
            lev, hold_days = rank1_params(variant, signal, action)
            component = "Rank1"
            bucket = "R1"
            original_leverage = lev
        else:
            hold_days = RANK23_HOLD_DAYS
            lev = rank23_leverage(signal, action)
            component = f"Rank{rank}"
            bucket = bucket_for_signal(signal)
            original_leverage = int(signal["leverage"])
        common = {
            "strategy": strategy,
            "strategy_component": component,
            "rank1_cell": signal.get("rank1_cell", ""),
            "bucket": bucket,
            "target_hold_days": hold_days,
            "original_leverage": original_leverage,
            "adaptive_leverage": lev,
            "regime_state": action.get("strategy_state", "NA" if rank == 1 else "GREEN"),
            "base_state": action.get("base_state", "NA" if rank == 1 else "GREEN"),
            "recovery_signal": bool(action.get("recovery_signal", False)),
            "rank1_adaptive_like_rank3": bool(variant.adaptive_like_rank3) if variant else False,
        }
        open_info = open_by_symbol.get(symbol)
        if open_info is not None and signal_time < int(open_info["open_until"]):
            row = skipped_open_position_trade(signal, int(open_info["open_until"]))
            row["leverage"] = lev
            rows.append(
                row
                | common
                | {
                    "status": "skipped",
                    "skip_reason": "symbol_already_open",
                    "blocking_component": open_info["component"],
                    "blocking_rank": open_info["rank"],
                    "blocking_entry_time_ms": open_info["entry_time_ms"],
                    "blocking_entry_time_utc": ms_to_utc(int(open_info["entry_time_ms"])).strftime("%Y-%m-%d %H:%M:%S"),
                    "blocking_pnl_u": open_info.get("pnl_u", np.nan),
                }
            )
            continue
        trade = outcomes[(int(signal["signal_id"]), lev, hold_days)].copy()
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            lock_extra_ms = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_by_symbol[symbol] = {
                "open_until": int(float(trade["exit_time_ms"])) + lock_extra_ms,
                "component": component,
                "rank": rank,
                "entry_time_ms": signal_time,
                "pnl_u": trade.get("pnl_u", np.nan),
            }
    return pd.DataFrame(rows)


def displaced_pnl(baseline: pd.DataFrame, variant_trades: pd.DataFrame) -> float:
    base_done = evaluated(baseline)
    base_map = {
        (str(r.symbol), int(r.entry_time_ms), int(r.rank)): float(r.pnl_u)
        for r in base_done[base_done["rank"].isin([2, 3])].itertuples(index=False)
    }
    skipped = variant_trades[
        variant_trades["status"].eq("skipped")
        & variant_trades["rank"].isin([2, 3])
        & variant_trades.get("blocking_component", pd.Series(dtype=str)).eq("Rank1")
    ].copy()
    return float(sum(base_map.get((str(r.symbol), int(r.entry_time_ms), int(r.rank)), 0.0) for r in skipped.itertuples(index=False)))


def build_outputs(trades_by_strategy: dict[str, pd.DataFrame], cutoff_ms: int, signal_end: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = trades_by_strategy[BASELINE]
    base_summary = summarize(baseline)
    summary_rows = []
    monthly_rows = []
    rank1_rows = []
    cell_rows = []
    rank_rows = []
    displaced_rows = []
    all_trades = []
    for strategy, trades in trades_by_strategy.items():
        all_trades.append(trades)
        s = summarize(trades)
        r1 = evaluated(trades)
        r1 = r1[r1["rank"].eq(1)].copy()
        r1_s = summarize(r1)
        disp = displaced_pnl(baseline, trades) if strategy != BASELINE else 0.0
        summary_rows.append(
            {
                "strategy": strategy,
                **s,
                "delta_vs_baseline_u": s["net_pnl_u"] - base_summary["net_pnl_u"],
                "rank1_pnl_u": r1_s["net_pnl_u"],
                "rank1_trades": r1_s["trades"],
                "rank1_liquidations": r1_s["liquidations"],
                "lost_rank23_pnl_due_to_rank1_occupancy": disp,
                "rank1_minus_displaced_u": r1_s["net_pnl_u"] - disp,
            }
        )
        if strategy != BASELINE:
            displaced_rows.append(
                {
                    "strategy": strategy,
                    "rank1_pnl_gained": r1_s["net_pnl_u"],
                    "lost_rank23_pnl_due_to_rank1_occupancy": disp,
                    "portfolio_incremental_value": s["net_pnl_u"] - base_summary["net_pnl_u"],
                    "rank1_minus_displaced_u": r1_s["net_pnl_u"] - disp,
                }
            )
        done = evaluated(trades)
        for month, group in done.groupby("month", sort=True):
            monthly_rows.append({"strategy": strategy, "month": month, **summarize(group), "status_note": "INCOMPLETE / OOS SHADOW" if month == "2026-08" else ""})
        for rank, group in done.groupby("rank", sort=True):
            rank_rows.append({"strategy": strategy, "rank": int(rank), **summarize(group)})
        if strategy != BASELINE:
            rank1_rows.append({"strategy": strategy, **r1_s})
            for cell, group in r1.groupby("rank1_cell", sort=True):
                cell_rows.append({"strategy": strategy, "cell": cell, **summarize(group)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "variant_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rank1_rows).to_csv(OUT_DIR / "rank1_summary_by_variant.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cell_rows).to_csv(OUT_DIR / "rank1_cell_by_variant.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / "rank_results_by_variant.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(displaced_rows).to_csv(OUT_DIR / "displaced_by_variant.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_trades, ignore_index=True).to_csv(OUT_DIR / "trade_details_all_variants.csv", index=False, encoding="utf-8-sig")

    view_cols = [
        "strategy",
        "trades",
        "net_pnl_u",
        "delta_vs_baseline_u",
        "pf",
        "win_rate",
        "median_return_pct",
        "max_drawdown_u",
        "liquidations",
        "liq_rate",
        "ex_top1_pnl_u",
        "ex_top3_pnl_u",
        "ex_top5_pnl_u",
        "ex_top10_pnl_u",
        "rank1_pnl_u",
        "lost_rank23_pnl_due_to_rank1_occupancy",
    ]
    lines = [
        "# Rank1 Portfolio Variant Comparison",
        "",
        f"- Cutoff: {ms_to_utc(cutoff_ms).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "- Rank1 entries: Rank1 20-40/V2-3, 20-40/V5-6, 40-60/V2-3 only.",
        "- Portfolio replay: same-symbol lock active; Rank2/3 keep current FR3/YR1 and 6D hold.",
        "- Adaptive Rank1 variant uses current FR3/YR1 Rank3 leverage at each timestamp.",
        "- Fees: 0.1% per side; slippage 0.",
        "",
        summary[view_cols].round(4).to_string(index=False),
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")
    print("output", OUT_DIR)
    print(summary[view_cols].round(4).to_string(index=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [s for s in cached_symbols() if s not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)
    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)

    rank23 = apply_entry_rules(raw, kline_map).copy()
    rank23["strategy_component"] = rank23["rank"].map(lambda r: f"Rank{int(r)}")
    rank23["bucket"] = rank23.apply(bucket_for_signal, axis=1)
    rank23["rank1_cell"] = ""

    rank1 = add_rank1_candidates(raw, kline_map)
    combined = pd.concat([rank1, rank23], ignore_index=True, sort=False)
    combined = combined.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    combined["signal_id"] = combined.index.astype(int)

    actions = build_fr3_yr1_actions(raw, rank23, kline_map)
    actions.to_csv(OUT_DIR / "fr3_yr1_action_timeline.csv", index=False, encoding="utf-8-sig")
    outcomes = precompute_outcomes(combined, kline_map, common_end)

    trades_by_strategy: dict[str, pd.DataFrame] = {
        BASELINE: replay_portfolio(BASELINE, combined[combined["rank"].isin([2, 3])].copy(), outcomes, actions, None)
    }
    for variant in VARIANTS:
        trades_by_strategy[variant.name] = replay_portfolio(variant.name, combined, outcomes, actions, variant)
    build_outputs(trades_by_strategy, common_end, signal_end)


if __name__ == "__main__":
    main()
