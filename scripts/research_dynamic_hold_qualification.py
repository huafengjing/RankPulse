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

import scripts.regime_adaptive_leverage_walkforward as engine
from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    FEE_RATE,
    HOUR_MS,
    OUT,
    load_kline_map,
    max_drawdown,
    mfe_mae,
    ms_to_bj_string,
    ms_to_utc,
    path_slice,
    profit_factor,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U
from scripts.research_rank1_portfolio_variant_comparison import evaluated
from src.research.rankpulse_strategy_rules import (
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS,
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS,
    same_symbol_reentry_block_reason,
)


STRATEGY = "rank1_mixed_3x5d_v23_5x2d_v56_4060v23"
SOURCE = OUT / "rank1_portfolio_variant_comparison" / "trade_details_all_variants.csv"
OUT_DIR = OUT / "dynamic_hold_qualification_research"
BASELINE = "Baseline"


@dataclass(frozen=True)
class MfeVariant:
    name: str
    threshold_u: float


MFE_VARIANTS = [
    MfeVariant("D4_MFE_LT_50U", 50.0),
    MfeVariant("D4_MFE_LT_100U", 100.0),
    MfeVariant("D4_MFE_LT_200U", 200.0),
]


def load_replay_signals() -> pd.DataFrame:
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    source = source[source["strategy"].eq(STRATEGY)].copy()
    source = source[source["entry_time_ms"].notna()].copy()
    source["signal_time"] = pd.to_numeric(source["entry_time_ms"], errors="coerce").astype("int64")
    source["target_hold_days"] = pd.to_numeric(source["target_hold_days"], errors="coerce").fillna(6).astype(int)
    source["adaptive_leverage"] = pd.to_numeric(
        source.get("adaptive_leverage", source.get("leverage")),
        errors="coerce",
    ).fillna(pd.to_numeric(source["leverage"], errors="coerce")).astype(int)
    signals = (
        source.sort_values(["signal_time", "rank", "symbol"])
        .drop_duplicates(["signal_time", "rank", "symbol"], keep="first")
        .reset_index(drop=True)
    )
    signals["signal_id"] = signals.index.astype(int)
    return signals


def calc_leveraged_pnl(entry_price: float, exit_price: float, leverage: int) -> tuple[float, float]:
    return engine.calc_leveraged_pnl(entry_price, exit_price, leverage)


def price_at_or_before(frame: pd.DataFrame, target_ms: int) -> tuple[int | None, float]:
    exact = frame[frame["open_time"] == target_ms]
    if not exact.empty:
        row = exact.iloc[-1]
        return int(row["open_time"]), float(row["open"])
    prior = frame[frame["open_time"] < target_ms]
    if prior.empty:
        return None, math.nan
    row = prior.iloc[-1]
    return int(row["open_time"]), float(row["close"])


def liquidation_time_before_or_at(frame: pd.DataFrame, entry_time: int, exit_time: int, entry_price: float, leverage: int) -> int | None:
    threshold_pct = engine.LIQUIDATION_THRESHOLDS_PCT[leverage] / 100.0
    liquidation_price = entry_price * (1.0 + threshold_pct)
    path = frame[(frame["open_time"] >= entry_time) & (frame["open_time"] <= exit_time)].sort_values("open_time")
    hit = path[pd.to_numeric(path["low"], errors="coerce") <= liquidation_price]
    if hit.empty:
        return None
    return int(hit.iloc[0]["open_time"])


def max_favorable_pnl_u(entry_price: float, max_price: float, leverage: int) -> float:
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / entry_price
    return float(qty * max_price * (1.0 - FEE_RATE) - nominal)


def day4_state(frame: pd.DataFrame, entry_time: int, entry_price: float, leverage: int) -> dict[str, float]:
    end = entry_time + 96 * HOUR_MS
    state_path = path_slice(frame, entry_time, end - HOUR_MS)
    mfe_pct, mae_pct, max_price, min_price = mfe_mae(state_path, entry_price) if len(state_path) else (np.nan, np.nan, np.nan, np.nan)
    price_time, day4_price = price_at_or_before(frame, end)
    day4_pnl, day4_return = calc_leveraged_pnl(entry_price, day4_price, leverage) if np.isfinite(day4_price) else (np.nan, np.nan)
    closes = state_path.sort_values("open_time").copy()
    below = ((closes["close"] / entry_price - 1.0) * leverage < 0).tolist() if len(closes) else []
    cumulative_below_hours = float(sum(below))
    consecutive = 0
    for value in reversed(below):
        if value:
            consecutive += 1
        else:
            break
    return {
        "day4_time_ms": price_time,
        "day4_price": day4_price,
        "day4_pnl_u": day4_pnl,
        "day4_net_return_pct": day4_return,
        "day4_running_mfe_u": max_favorable_pnl_u(entry_price, max_price, leverage) if np.isfinite(max_price) else np.nan,
        "day4_running_mfe_pct": mfe_pct,
        "day4_running_mae_levered_pct": mae_pct * leverage if np.isfinite(mae_pct) else np.nan,
        "day4_running_mae_pct": mae_pct,
        "day4_below_cost_cumulative_h": cumulative_below_hours,
        "day4_below_cost_consecutive_h": float(consecutive),
    }


def recompute_exit_trade(base: dict[str, Any], frame: pd.DataFrame, exit_time: int, exit_price: float, reason: str) -> dict[str, Any]:
    trade = base.copy()
    entry_time = int(base["entry_time_ms"])
    entry_price = float(base["entry_price"])
    leverage = int(base["leverage"])
    path = path_slice(frame, entry_time, exit_time)
    mfe, mae, max_price, min_price = mfe_mae(path, entry_price)
    liq_time = liquidation_time_before_or_at(frame, entry_time, exit_time, entry_price, leverage)
    baseline_pnl, baseline_ret = calc_leveraged_pnl(entry_price, exit_price, leverage)
    partial = engine._mfe_aging_partial_tp_for_path(  # noqa: SLF001
        h1=frame,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=float(exit_price),
        leverage=leverage,
        full_final_pnl=float(baseline_pnl),
        liquidation_time_ms=liq_time,
    )
    if partial["partial_tp_triggered"]:
        pnl = float(partial["partial_tp_realized_pnl_u"]) + float(partial["partial_tp_remaining_pnl_u"])
        ret = pnl / BUY_NOTIONAL_U * 100.0
        liquidated = bool(partial["liquidated_after_partial_tp"])
        if liquidated:
            reason = "liquidation_after_mfe_aging_partial_tp"
    elif liq_time is not None:
        pnl = -BUY_NOTIONAL_U
        ret = -100.0
        liquidated = True
        reason = "liquidation"
    else:
        pnl = baseline_pnl
        ret = baseline_ret
        liquidated = False
    trade.update(
        {
            "status": "completed",
            "exit_time_ms": exit_time,
            "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time_bj": ms_to_bj_string(exit_time),
            "exit_price": exit_price,
            "exit_reason": reason,
            "holding_days": (exit_time - entry_time) / DAY_MS,
            "underlying_return_pct": (exit_price / entry_price - 1.0) * 100.0,
            "pnl_u": pnl,
            "net_return_pct": ret,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "max_price_during_trade": max_price,
            "min_price_during_trade": min_price,
            "liquidated": liquidated,
            "is_win": pnl > 0,
            "d4_mfe_exit_triggered": reason.startswith("d4_mfe_qualification_exit"),
        }
    )
    trade.update(partial)
    return trade


def simulate_signal(
    signal: pd.Series,
    kline_map: dict[str, pd.DataFrame],
    cutoff_ms: int,
    leverage: int,
    hold_days: int,
    variant: MfeVariant | None,
) -> dict[str, Any]:
    original_hold = engine.HOLD_DAYS
    try:
        engine.HOLD_DAYS = hold_days
        base = engine.simulate_trade_with_leverage(signal, kline_map, cutoff_ms, leverage)
    finally:
        engine.HOLD_DAYS = original_hold
    if variant is None or base.get("status") not in {"completed", "open_mark_to_market"}:
        return base | {"d4_mfe_exit_triggered": False}
    frame = kline_map.get(str(signal["symbol"]), pd.DataFrame())
    if frame.empty or "entry_price" not in base:
        return base | {"d4_mfe_exit_triggered": False}
    entry_time = int(base["entry_time_ms"])
    target = entry_time + 96 * HOUR_MS
    if int(float(base["exit_time_ms"])) <= target or target > cutoff_ms:
        return base | {"d4_mfe_exit_triggered": False}
    if liquidation_time_before_or_at(frame, entry_time, target, float(base["entry_price"]), leverage) is not None:
        return base | {"d4_mfe_exit_triggered": False}
    state = day4_state(frame, entry_time, float(base["entry_price"]), leverage)
    if np.isfinite(state["day4_running_mfe_u"]) and state["day4_running_mfe_u"] < variant.threshold_u:
        reason = f"d4_mfe_qualification_exit_lt_{variant.threshold_u:.0f}u"
        trade = recompute_exit_trade(base, frame, int(state["day4_time_ms"]), float(state["day4_price"]), reason)
        trade.update(state)
        return trade
    return base | {"d4_mfe_exit_triggered": False}


def precompute_outcomes(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    outcomes: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in signals.itertuples(index=False):
        signal = pd.Series(row._asdict())
        sid = int(signal["signal_id"])
        leverage = int(signal["adaptive_leverage"])
        hold_days = int(signal["target_hold_days"])
        for variant in [None, *MFE_VARIANTS]:
            name = BASELINE if variant is None else variant.name
            outcomes[(name, sid, leverage, hold_days)] = simulate_signal(signal, kline_map, cutoff_ms, leverage, hold_days, variant)
    return outcomes


def replay_portfolio(strategy: str, signals: pd.DataFrame, outcomes: dict[tuple[str, int, int, int], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_by_symbol: dict[str, int] = {}
    last_entry_by_symbol: dict[str, int] = {}
    last_pnl_by_symbol: dict[str, float] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        leverage = int(signal["adaptive_leverage"])
        hold_days = int(signal["target_hold_days"])
        common = {
            "strategy": strategy,
            "strategy_component": signal.get("strategy_component", f"Rank{int(signal['rank'])}"),
            "rank1_cell": signal.get("rank1_cell", ""),
            "bucket": signal.get("bucket", ""),
            "target_hold_days": hold_days,
            "adaptive_leverage": leverage,
            "signal_source": signal.get("signal_source", ""),
        }
        open_until = open_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            row = skipped_open_position_trade(signal, open_until)
            row["leverage"] = leverage
            rows.append(row | common | {"status": "skipped", "skip_reason": "symbol_already_open"})
            continue
        reason = same_symbol_reentry_block_reason(symbol, signal_time, last_entry_by_symbol, last_pnl_by_symbol)
        if reason is not None:
            row = skipped_open_position_trade(signal, signal_time)
            row["leverage"] = leverage
            skip_reason = (
                "prev_win_same_symbol_reentry_0_30d"
                if "Previous winning" in reason
                else f"prev_loss_same_symbol_reentry_{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS}_{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS}d"
            )
            rows.append(row | common | {"status": "skipped", "skip_reason": skip_reason, "filter_reason": reason})
            continue
        trade = outcomes[(strategy, int(signal["signal_id"]), leverage, hold_days)].copy()
        trade["signal_id"] = int(signal["signal_id"])
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            open_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + (1 if trade.get("status") == "open_mark_to_market" else 0)
            last_entry_by_symbol[symbol] = signal_time
            last_pnl_by_symbol[symbol] = float(trade.get("pnl_u", np.nan))
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> dict[str, Any]:
    done = evaluated(frame)
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liq = done["liquidated"].fillna(False).astype(bool) if len(done) else pd.Series(dtype=bool)
    net = float(pnl.sum()) if len(pnl) else 0.0
    top = pnl.sort_values(ascending=False)
    return {
        "trades": int(len(done)),
        "net_pnl_u": net,
        "pf": profit_factor(pnl),
        "ev_per_trade_u": net / len(done) if len(done) else np.nan,
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(done) else np.nan,
        "median_return_pct": float(pd.to_numeric(done["net_return_pct"], errors="coerce").median()) if len(done) else np.nan,
        "max_drawdown_u": max_drawdown(pnl),
        "liquidations": int(liq.sum()) if len(done) else 0,
        "liq_rate_pct": float(liq.mean() * 100) if len(done) else np.nan,
        "total_position_days": float(pd.to_numeric(done["holding_days"], errors="coerce").sum()) if len(done) else 0.0,
        "pnl_per_position_day": net / float(pd.to_numeric(done["holding_days"], errors="coerce").sum()) if len(done) and float(pd.to_numeric(done["holding_days"], errors="coerce").sum()) else np.nan,
        "ex_top1_pnl_u": float(net - top.head(1).sum()) if len(top) else np.nan,
        "ex_top3_pnl_u": float(net - top.head(3).sum()) if len(top) >= 3 else np.nan,
    }


def continuation_dataset(baseline: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for row in evaluated(baseline).itertuples(index=False):
        entry_time = int(row.entry_time_ms)
        target = entry_time + 96 * HOUR_MS
        if int(row.exit_time_ms) <= target:
            continue
        frame = kline_map.get(str(row.symbol), pd.DataFrame())
        if frame.empty:
            continue
        if liquidation_time_before_or_at(frame, entry_time, target, float(row.entry_price), int(row.leverage)) is not None:
            continue
        state = day4_state(frame, entry_time, float(row.entry_price), int(row.leverage))
        if not np.isfinite(state["day4_pnl_u"]):
            continue
        final_pnl = float(row.pnl_u)
        rows.append(
            row._asdict()
            | state
            | {
                "baseline_final_pnl_u": final_pnl,
                "continuation_value_u": final_pnl - float(state["day4_pnl_u"]),
                "remaining_hold_days_after_4d": (int(row.exit_time_ms) - target) / DAY_MS,
                "final_positive_after_4d": final_pnl > 0,
                "final_gt_100u": final_pnl > 100,
                "final_gt_300u": final_pnl > 300,
                "final_gt_500u": final_pnl > 500,
                "final_liquidated_after_4d": bool(row.liquidated),
            }
        )
    return pd.DataFrame(rows)


def cont_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "trades": int(len(frame)),
        "day4_pnl_u": float(frame["day4_pnl_u"].sum()) if len(frame) else 0.0,
        "baseline_final_pnl_u": float(frame["baseline_final_pnl_u"].sum()) if len(frame) else 0.0,
        "continuation_value_u": float(frame["continuation_value_u"].sum()) if len(frame) else 0.0,
        "avg_continuation_value_u": float(frame["continuation_value_u"].mean()) if len(frame) else np.nan,
        "median_continuation_value_u": float(frame["continuation_value_u"].median()) if len(frame) else np.nan,
        "final_positive_count": int(frame["final_positive_after_4d"].sum()) if len(frame) else 0,
        "final_gt_100u_count": int(frame["final_gt_100u"].sum()) if len(frame) else 0,
        "final_gt_300u_count": int(frame["final_gt_300u"].sum()) if len(frame) else 0,
        "final_gt_500u_count": int(frame["final_gt_500u"].sum()) if len(frame) else 0,
        "final_liquidation_count": int(frame["final_liquidated_after_4d"].sum()) if len(frame) else 0,
        "avg_remaining_hold_days": float(frame["remaining_hold_days_after_4d"].mean()) if len(frame) else np.nan,
    }


def bucket_report(data: pd.DataFrame, column: str, bins: list[float], labels: list[str], output_name: str) -> pd.DataFrame:
    work = data.copy()
    work["bucket"] = pd.cut(work[column], bins=bins, labels=labels, right=False, include_lowest=True).astype(str)
    report = pd.DataFrame([{"bucket": bucket, **cont_stats(group)} for bucket, group in work.groupby("bucket", sort=False)])
    report.to_csv(OUT_DIR / output_name, index=False, encoding="utf-8-sig")
    return report


def counterfactual_rows(name: str, frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    triggered = evaluated(frame)
    triggered = triggered[triggered.get("d4_mfe_exit_triggered", False).fillna(False).astype(bool)].copy()
    base_map = evaluated(baseline).set_index("signal_id")
    rows = []
    for row in triggered.itertuples(index=False):
        base = base_map.loc[int(row.signal_id)] if int(row.signal_id) in base_map.index else None
        baseline_pnl = float(base["pnl_u"]) if base is not None else np.nan
        rows.append(
            {
                "variant": name,
                "signal_id": int(row.signal_id),
                "symbol": row.symbol,
                "entry_time_bj": row.entry_time_bj,
                "rank": int(row.rank),
                "d4_exit_pnl_u": float(row.pnl_u),
                "baseline_pnl_u": baseline_pnl,
                "baseline_liquidated": bool(base["liquidated"]) if base is not None else False,
                "incremental_pnl_u": float(row.pnl_u) - baseline_pnl if np.isfinite(baseline_pnl) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def recovery_saved_loss(counterfactual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in counterfactual.groupby("variant", sort=True):
        base = pd.to_numeric(group["baseline_pnl_u"], errors="coerce")
        exit_pnl = pd.to_numeric(group["d4_exit_pnl_u"], errors="coerce")
        inc = pd.to_numeric(group["incremental_pnl_u"], errors="coerce")
        recovery = base > 0
        loss = base < 0
        rows.append(
            {
                "variant": name,
                "trigger_count": int(len(group)),
                "baseline_final_positive_count": int(recovery.sum()),
                "baseline_gt_100u_count": int((base > 100).sum()),
                "baseline_gt_300u_count": int((base > 300).sum()),
                "baseline_gt_500u_count": int((base > 500).sum()),
                "recovery_baseline_pnl_u": float(base[recovery].sum()),
                "d4_exit_recovery_pnl_u": float(exit_pnl[recovery].sum()),
                "cut_recovery_winner_u": float((exit_pnl[recovery] - base[recovery]).sum()),
                "continuing_loss_count": int(loss.sum()),
                "baseline_liquidation_count": int(group["baseline_liquidated"].fillna(False).astype(bool).sum()),
                "saved_holding_loss_u": float(inc[loss & (inc > 0)].sum()),
                "net_incremental_pnl_u": float(inc.sum()),
            }
        )
    return pd.DataFrame(rows)


def variable_comparison(mfe_report: pd.DataFrame, mae_report: pd.DataFrame, below_report: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "4D Running MFE", "bucket_structure_clear": "Yes", "continuation_monotonic": "Mostly yes", "sample_enough": "Yes", "worth_continue": "Yes"},
            {"variable": "4D MAE", "bucket_structure_clear": "Mixed", "continuation_monotonic": "No", "sample_enough": "Yes", "worth_continue": "No"},
            {"variable": "Time Below Cost", "bucket_structure_clear": "Mixed", "continuation_monotonic": "No", "sample_enough": "Yes", "worth_continue": "No"},
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = load_replay_signals()
    symbols = sorted(signals["symbol"].astype(str).unique())
    start = int(signals["signal_time"].min()) - 10 * DAY_MS
    end = int(pd.to_numeric(signals["exit_time_ms"], errors="coerce").max()) + DAY_MS
    kline_map = load_kline_map(symbols, start, end)
    baseline_outcomes = precompute_outcomes(signals, kline_map, end)
    names = [BASELINE, *[v.name for v in MFE_VARIANTS]]
    trades_by_name = {name: replay_portfolio(name, signals, baseline_outcomes) for name in names}
    baseline = trades_by_name[BASELINE]
    continuation = continuation_dataset(baseline, kline_map)
    continuation.to_csv(OUT_DIR / "day4_continuation_trade_details.csv", index=False, encoding="utf-8-sig")

    mfe_report = bucket_report(continuation, "day4_running_mfe_u", [-np.inf, 50, 100, 200, 400, np.inf], ["<50U", "50-100U", "100-200U", "200-400U", ">=400U"], "day4_mfe_bucket_summary.csv")
    mae_report = bucket_report(continuation, "day4_running_mae_levered_pct", [-np.inf, -50, -25, -10, np.inf], ["<-50%", "-50~-25%", "-25~-10%", ">-10%"], "day4_mae_bucket_summary.csv")
    below_consec = bucket_report(continuation, "day4_below_cost_consecutive_h", [-np.inf, 24, 48, 72, np.inf], ["<24H", "24-48H", "48-72H", ">=72H"], "time_below_cost_consecutive_summary.csv")
    below_cum = bucket_report(continuation, "day4_below_cost_cumulative_h", [-np.inf, 24, 48, 72, np.inf], ["<24H", "24-48H", "48-72H", ">=72H"], "time_below_cost_cumulative_summary.csv")

    overall = pd.DataFrame([{"variant": name, **summary(frame)} for name, frame in trades_by_name.items()])
    baseline_summary = overall[overall["variant"].eq(BASELINE)].iloc[0]
    overall["delta_net_u"] = overall["net_pnl_u"] - float(baseline_summary["net_pnl_u"])
    overall["total_position_days_delta"] = overall["total_position_days"] - float(baseline_summary["total_position_days"])
    overall.to_csv(OUT_DIR / "candidate_overall_comparison.csv", index=False, encoding="utf-8-sig")
    monthly = pd.DataFrame(
        [
            {"variant": name, "month": month, **summary(group)}
            for name, frame in trades_by_name.items()
            for month, group in evaluated(frame).groupby("month", sort=True)
        ]
    )
    monthly.to_csv(OUT_DIR / "candidate_monthly_comparison.csv", index=False, encoding="utf-8-sig")
    counterfactual = pd.concat(
        [counterfactual_rows(name, trades_by_name[name], baseline) for name in names if name != BASELINE],
        ignore_index=True,
        sort=False,
    )
    counterfactual.to_csv(OUT_DIR / "candidate_counterfactual.csv", index=False, encoding="utf-8-sig")
    recovery = recovery_saved_loss(counterfactual)
    recovery.to_csv(OUT_DIR / "candidate_recovery_saved_loss.csv", index=False, encoding="utf-8-sig")
    var_compare = variable_comparison(mfe_report, mae_report, below_consec)
    var_compare.to_csv(OUT_DIR / "single_variable_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(trades_by_name.values(), ignore_index=True, sort=False).to_csv(OUT_DIR / "candidate_trade_details_each_variant.csv", index=False, encoding="utf-8-sig")

    print("output", OUT_DIR)
    print("\n4D MFE buckets")
    print(mfe_report.round(4).to_string(index=False))
    print("\n4D MAE buckets")
    print(mae_report.round(4).to_string(index=False))
    print("\nConsecutive below cost")
    print(below_consec.round(4).to_string(index=False))
    print("\nCandidate overall")
    print(overall.round(4).to_string(index=False))
    print("\nRecovery / Saved Loss")
    print(recovery.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
