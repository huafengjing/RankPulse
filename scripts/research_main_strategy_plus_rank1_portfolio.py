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

import scripts.regime_adaptive_leverage_walkforward as leverage_engine
from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, add_entry_factors, load_kline_map, max_drawdown, ms_to_utc, profit_factor, skipped_open_position_trade
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS, IndicatorSpec, build_health_timeline, opportunity_sets
from scripts.rank3_fast_recovery_vs_monthly_reset import RecoverySpec, build_recovery_timeline, fast_recovery_action_timeline
from scripts.regime_adaptive_leverage_walkforward import bucket_for_signal, simulate_trade_with_leverage
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, apply_entry_rules, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "main_strategy_plus_rank1_optimization"
BASELINE = "当前主策略"
WITH_RANK1 = "当前主策略 + Rank1"
RANK1_LEVERAGE = 3
RANK1_HOLD_DAYS = 5
RANK23_HOLD_DAYS = 6
FR3_YR1_NAME = "FR_avg_return24_l3_gt_0_fr3_yr1"


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
    done = evaluated(frame).sort_values(["entry_time_ms", "rank", "symbol"]).copy()
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
    rank1["leverage"] = RANK1_LEVERAGE
    rank1["bucket"] = "R1"
    rank1["target_hold_days"] = RANK1_HOLD_DAYS
    return rank1


def build_fr3_yr1_actions(raw: pd.DataFrame, rank23: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signal_times = sorted(rank23["signal_time"].astype(int).unique())
    sets = opportunity_sets(raw, kline_map)
    d15_spec = IndicatorSpec("D_b_r3_decay_l15", "B_R3", "mean_decay48", "lower_bad", 15)
    d15 = build_health_timeline(signal_times, sets["B_R3"], d15_spec)
    recovery = build_recovery_timeline(signal_times, sets["B_R3"], RecoverySpec("avg_return24_l3_gt_0", "avg_return24", 3, "gt_0"))
    return fast_recovery_action_timeline(d15, recovery, "fr3", "yr1", FR3_YR1_NAME)


def precompute_outcomes(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int) -> dict[tuple[int, int, int], dict[str, Any]]:
    original_hold = leverage_engine.HOLD_DAYS
    outcomes: dict[tuple[int, int, int], dict[str, Any]] = {}
    try:
        for row in signals.itertuples(index=False):
            signal = pd.Series(row._asdict())
            sid = int(signal["signal_id"])
            if int(signal["rank"]) == 1:
                leverage_engine.HOLD_DAYS = RANK1_HOLD_DAYS
                outcomes[(sid, RANK1_LEVERAGE, RANK1_HOLD_DAYS)] = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, RANK1_LEVERAGE)
            else:
                leverage_engine.HOLD_DAYS = RANK23_HOLD_DAYS
                for lev in [1, 2, 3, 5]:
                    outcomes[(sid, lev, RANK23_HOLD_DAYS)] = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, lev)
    finally:
        leverage_engine.HOLD_DAYS = original_hold
    return outcomes


def adaptive_leverage(signal: pd.Series, action: dict[str, Any]) -> int:
    bucket = bucket_for_signal(signal)
    if bucket == "B" and int(signal["rank"]) == 2:
        return int(action.get("r2_lev", 3))
    if bucket == "B" and int(signal["rank"]) == 3:
        return int(action.get("r3_lev", 5))
    return int(signal["leverage"])


def replay_portfolio(
    name: str,
    signals: pd.DataFrame,
    outcomes: dict[tuple[int, int, int], dict[str, Any]],
    actions: pd.DataFrame,
    include_rank1: bool,
) -> pd.DataFrame:
    action_by_time = actions.set_index("signal_time").to_dict("index") if not actions.empty else {}
    rows: list[dict[str, Any]] = []
    open_by_symbol: dict[str, dict[str, Any]] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        rank = int(signal["rank"])
        if rank == 1 and not include_rank1:
            continue
        action = action_by_time.get(signal_time, {})
        is_rank1 = rank == 1
        hold_days = RANK1_HOLD_DAYS if is_rank1 else RANK23_HOLD_DAYS
        lev = RANK1_LEVERAGE if is_rank1 else adaptive_leverage(signal, action)
        component = "Rank1" if is_rank1 else f"Rank{rank}"
        common = {
            "strategy": name,
            "strategy_component": component,
            "rank1_cell": signal.get("rank1_cell", ""),
            "bucket": "R1" if is_rank1 else bucket_for_signal(signal),
            "target_hold_days": hold_days,
            "original_leverage": int(signal["leverage"]),
            "adaptive_leverage": lev,
            "regime_state": action.get("strategy_state", "NA" if is_rank1 else "GREEN"),
            "base_state": action.get("base_state", "NA" if is_rank1 else "GREEN"),
            "recovery_signal": bool(action.get("recovery_signal", False)),
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
        row = trade | common
        rows.append(row)
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


def monthly_comparison(baseline: pd.DataFrame, with_rank1: pd.DataFrame) -> pd.DataFrame:
    months = sorted(set(evaluated(baseline)["month"].dropna().astype(str)) | set(evaluated(with_rank1)["month"].dropna().astype(str)))
    rows = []
    for month in months:
        b = evaluated(baseline[baseline["month"].eq(month)])
        w = evaluated(with_rank1[with_rank1["month"].eq(month)])
        r1 = w[w["rank"].eq(1)]
        b_pnl = float(pd.to_numeric(b["pnl_u"], errors="coerce").sum()) if len(b) else 0.0
        w_pnl = float(pd.to_numeric(w["pnl_u"], errors="coerce").sum()) if len(w) else 0.0
        rows.append(
            {
                "Month": month,
                "Baseline": b_pnl,
                "+Rank1": w_pnl,
                "Delta": w_pnl - b_pnl,
                "Rank1 Contribution": float(pd.to_numeric(r1["pnl_u"], errors="coerce").sum()) if len(r1) else 0.0,
                "Status": "INCOMPLETE / OOS SHADOW" if month == "2026-08" else "",
            }
        )
    return pd.DataFrame(rows)


def rank_results(frame: pd.DataFrame, strategy: str) -> list[dict[str, Any]]:
    rows = []
    done = evaluated(frame)
    for rank, group in done.groupby("rank", sort=True):
        s = summarize(group)
        rows.append({"strategy": strategy, "rank": int(rank), "trade_count": s["trades"], "pnl_u": s["net_pnl_u"], "pf": s["pf"]})
    return rows


def collision_audit(baseline: pd.DataFrame, with_rank1: pd.DataFrame) -> pd.DataFrame:
    base_done = evaluated(baseline)
    base_map = {
        (str(r.symbol), int(r.entry_time_ms), int(r.rank)): r
        for r in base_done[base_done["rank"].isin([2, 3])].itertuples(index=False)
    }
    new_done = evaluated(with_rank1)
    new_keys = {(str(r.symbol), int(r.entry_time_ms), int(r.rank)) for r in new_done[new_done["rank"].isin([2, 3])].itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    skipped = with_rank1[
        with_rank1["status"].eq("skipped")
        & with_rank1["rank"].isin([2, 3])
        & with_rank1.get("blocking_component", pd.Series(dtype=str)).eq("Rank1")
    ].copy()
    for r in skipped.itertuples(index=False):
        key = (str(r.symbol), int(r.entry_time_ms), int(r.rank))
        b = base_map.get(key)
        rows.append(
            {
                "case": "A/B_rank1_occupancy_displaced_rank23",
                "symbol": r.symbol,
                "Rank1 entry": getattr(r, "blocking_entry_time_utc", ""),
                "later Rank2/Rank3 signal": r.entry_time_utc,
                "Baseline是否交易": b is not None,
                "New Strategy是否交易": key in new_keys,
                "skip reason": r.skip_reason,
                "Baseline PnL": float(getattr(b, "pnl_u", np.nan)) if b is not None else np.nan,
                "Rank1 PnL": float(getattr(r, "blocking_pnl_u", np.nan)),
                "net portfolio impact": float(getattr(r, "blocking_pnl_u", 0.0)) - (float(getattr(b, "pnl_u", 0.0)) if b is not None else 0.0),
            }
        )
    rank1_done = new_done[new_done["rank"].eq(1)].copy()
    if not rank1_done.empty:
        for r in new_done[new_done["rank"].isin([2, 3])].itertuples(index=False):
            prior = rank1_done[
                rank1_done["symbol"].eq(r.symbol)
                & (rank1_done["entry_time_ms"] < int(r.entry_time_ms))
                & (rank1_done["exit_time_ms"] <= int(r.entry_time_ms))
            ]
            if prior.empty:
                continue
            p = prior.sort_values("exit_time_ms").iloc[-1]
            rows.append(
                {
                    "case": "C_rank1_released_then_rank23_traded",
                    "symbol": r.symbol,
                    "Rank1 entry": p["entry_time_utc"],
                    "later Rank2/Rank3 signal": r.entry_time_utc,
                    "Baseline是否交易": (str(r.symbol), int(r.entry_time_ms), int(r.rank)) in base_map,
                    "New Strategy是否交易": True,
                    "skip reason": "",
                    "Baseline PnL": float(getattr(base_map.get((str(r.symbol), int(r.entry_time_ms), int(r.rank))), "pnl_u", np.nan)),
                    "Rank1 PnL": float(p["pnl_u"]),
                    "net portfolio impact": float(r.pnl_u),
                }
            )
    return pd.DataFrame(rows)


def chinese_comparison(baseline: pd.DataFrame, with_rank1: pd.DataFrame) -> pd.DataFrame:
    b = summarize(baseline)
    w = summarize(with_rank1)
    mapping = [
        ("总净收益", "net_pnl_u"),
        ("PF", "pf"),
        ("胜率", "win_rate"),
        ("中位收益率", "median_return_pct"),
        ("平均收益率", "avg_return_pct"),
        ("最大回撤", "max_drawdown_u"),
        ("爆仓数", "liquidations"),
        ("爆仓率", "liq_rate"),
        ("交易笔数", "trades"),
        ("去 Top1 后收益", "ex_top1_pnl_u"),
        ("去 Top3 后收益", "ex_top3_pnl_u"),
    ]
    rows = []
    for label, key in mapping:
        rows.append({"指标": label, "当前主策略": b[key], "当前主策略 + Rank1": w[key], "变化": w[key] - b[key]})
    return pd.DataFrame(rows)


def write_outputs(baseline: pd.DataFrame, with_rank1: pd.DataFrame, cutoff_ms: int, signal_end: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(OUT_DIR / "trade_details_baseline.csv", index=False, encoding="utf-8-sig")
    with_rank1.to_csv(OUT_DIR / "trade_details_with_rank1.csv", index=False, encoding="utf-8-sig")
    comparison = chinese_comparison(baseline, with_rank1)
    comparison.to_csv(OUT_DIR / "baseline_vs_rank1.csv", index=False, encoding="utf-8-sig")
    monthly = monthly_comparison(baseline, with_rank1)
    monthly.to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")

    r1_done = evaluated(with_rank1)
    r1_done = r1_done[r1_done["rank"].eq(1)].copy()
    cell_rows = []
    for cell, group in r1_done.groupby("rank1_cell", sort=True):
        cell_rows.append({"cell": cell, **summarize(group)})
    pd.DataFrame(cell_rows).to_csv(OUT_DIR / "rank1_cell_results.csv", index=False, encoding="utf-8-sig")

    rank_rows = rank_results(baseline, BASELINE) + rank_results(with_rank1, WITH_RANK1)
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / "rank_results_before_after.csv", index=False, encoding="utf-8-sig")

    audit = collision_audit(baseline, with_rank1)
    audit.to_csv(OUT_DIR / "rank_collision_audit.csv", index=False, encoding="utf-8-sig")
    displaced = audit[audit["case"].eq("A/B_rank1_occupancy_displaced_rank23")].copy() if not audit.empty else pd.DataFrame()
    displaced.to_csv(OUT_DIR / "displaced_trade_analysis.csv", index=False, encoding="utf-8-sig")

    b_sum = summarize(baseline)
    w_sum = summarize(with_rank1)
    displaced_pnl = float(pd.to_numeric(displaced.get("Baseline PnL", pd.Series(dtype=float)), errors="coerce").sum()) if not displaced.empty else 0.0
    rank1_pnl = float(pd.to_numeric(r1_done["pnl_u"], errors="coerce").sum()) if len(r1_done) else 0.0
    incremental = pd.DataFrame(
        [
            {
                "Baseline Total PnL": b_sum["net_pnl_u"],
                "NewStrategy Total PnL": w_sum["net_pnl_u"],
                "Portfolio Incremental Value": w_sum["net_pnl_u"] - b_sum["net_pnl_u"],
                "Rank1 PnL gained": rank1_pnl,
                "Lost Rank2/3 PnL due to Rank1 occupancy": displaced_pnl,
                "Rank1 gained - displaced Rank2/3 PnL": rank1_pnl - displaced_pnl,
            }
        ]
    )
    incremental.to_csv(OUT_DIR / "portfolio_incremental_value.csv", index=False, encoding="utf-8-sig")

    top_tail = pd.DataFrame(
        [
            {"strategy": BASELINE, **{k: b_sum[k] for k in ["net_pnl_u", "ex_top1_pnl_u", "ex_top3_pnl_u", "ex_top5_pnl_u", "ex_top10_pnl_u"]}},
            {"strategy": WITH_RANK1, **{k: w_sum[k] for k in ["net_pnl_u", "ex_top1_pnl_u", "ex_top3_pnl_u", "ex_top5_pnl_u", "ex_top10_pnl_u"]}},
        ]
    )
    top_tail.to_csv(OUT_DIR / "top_tail_comparison.csv", index=False, encoding="utf-8-sig")

    july = monthly[monthly["Month"].eq("2026-07")]
    july_rank1 = r1_done[r1_done["month"].eq("2026-07")]
    judgement = "OBSERVE"
    if w_sum["net_pnl_u"] < b_sum["net_pnl_u"] or w_sum["pf"] < b_sum["pf"] * 0.95:
        judgement = "REJECT"
    elif (
        w_sum["net_pnl_u"] > b_sum["net_pnl_u"]
        and w_sum["pf"] >= b_sum["pf"]
        and w_sum["max_drawdown_u"] >= b_sum["max_drawdown_u"] * 1.15
        and w_sum["liquidations"] <= b_sum["liquidations"] + 3
        and rank1_pnl - displaced_pnl > 0
    ):
        judgement = "SHADOW CANDIDATE"

    lines = [
        "# Rank1 加入主策略组合回测",
        "",
        f"- Cutoff: {ms_to_utc(cutoff_ms).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "- Universe: `BTWUSDT` excluded due stale cache, `RAVEUSDT` filtered by strategy, `VELVETUSDT` retained.",
        "- Baseline: 当前 FR3/YR1 主策略，Rank2/3 默认 6D。",
        "- Rank1: 20-40/V2-3, 20-40/V5-6, 40-60/V2-3; fixed 3x; max hold 5D; same early exits.",
        "- Fee/slippage: 0.1% per side, slippage 0.",
        "- MAX_OPEN_POSITIONS: no active global cap was found in current research replay code; same-symbol lock is active.",
        "",
        "## Core Comparison",
        "",
        comparison.round(4).to_string(index=False),
        "",
        "## Portfolio Incremental Value",
        "",
        incremental.round(4).to_string(index=False),
        "",
        "## July Check",
        "",
        july.round(4).to_string(index=False) if not july.empty else "(no July rows)",
        f"- July Rank1 contribution: {float(pd.to_numeric(july_rank1['pnl_u'], errors='coerce').sum()) if len(july_rank1) else 0.0:.2f}U",
        f"- July Rank1 liquidations: {int(july_rank1['liquidated'].astype(bool).sum()) if len(july_rank1) else 0}",
        "",
        "## Final Judgment",
        "",
        judgement,
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")
    print("output", OUT_DIR)
    print(comparison.round(4).to_string(index=False))
    print()
    print(incremental.round(4).to_string(index=False))
    print()
    print("judgment", judgement)


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
    rank23["target_hold_days"] = RANK23_HOLD_DAYS

    rank1 = add_rank1_candidates(raw, kline_map)
    combined = pd.concat([rank1, rank23], ignore_index=True, sort=False)
    combined = combined.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    combined["signal_id"] = combined.index.astype(int)

    actions = build_fr3_yr1_actions(raw, rank23, kline_map)
    actions.to_csv(OUT_DIR / "fr3_yr1_action_timeline.csv", index=False, encoding="utf-8-sig")
    outcomes = precompute_outcomes(combined, kline_map, common_end)
    baseline = replay_portfolio(BASELINE, combined[combined["rank"].isin([2, 3])].copy(), outcomes, actions, include_rank1=False)
    with_rank1 = replay_portfolio(WITH_RANK1, combined, outcomes, actions, include_rank1=True)
    write_outputs(baseline, with_rank1, common_end, signal_end)


if __name__ == "__main__":
    main()
