from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import research_rank1_portfolio_variant_comparison as main
from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    FEE_RATE,
    HOUR_MS,
    OUT,
    max_drawdown,
    ms_to_bj_string,
    ms_to_utc,
    path_slice,
    profit_factor,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, latest_signal_end_dt
from scripts.regime_adaptive_leverage_walkforward import (
    LIQUIDATION_THRESHOLDS_PCT,
    bucket_for_signal,
    simulate_trade_with_leverage,
)
from src.research.rankpulse_strategy_rules import (
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS,
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS,
    same_symbol_reentry_block_reason,
)


OUT_DIR = OUT / "mfe_aging_partial_tp_research"
BASELINE = "Baseline"
ACTIVATION_MFE_U = [300.0, 400.0]
TP_RATIOS = [0.30, 0.40, 0.50]
AGING_MS = 24 * HOUR_MS
RANK23_HOLD_DAYS = main.RANK23_HOLD_DAYS
BIG_WINNER_THRESHOLDS = [100.0, 300.0, 500.0, 1000.0]


@dataclass(frozen=True)
class MfeAgingVariant:
    name: str
    activation_mfe_u: float
    tp_ratio: float


VARIANTS = [
    MfeAgingVariant(
        name=f"MFE{int(threshold)}_TP{int(tp * 100)}",
        activation_mfe_u=threshold,
        tp_ratio=tp,
    )
    for threshold in ACTIVATION_MFE_U
    for tp in TP_RATIOS
]


def calc_leveraged_pnl(entry_price: float, exit_price: float, leverage: int, margin_usdt: float = BUY_NOTIONAL_U) -> float:
    nominal = margin_usdt * leverage
    qty = nominal * (1.0 - FEE_RATE) / entry_price
    exit_value = qty * exit_price * (1.0 - FEE_RATE)
    return float(exit_value - nominal)


def liquidation_time_before_or_at(
    frame: pd.DataFrame,
    entry_time: int,
    exit_time: int,
    entry_price: float,
    leverage: int,
) -> int | None:
    liq_pct = LIQUIDATION_THRESHOLDS_PCT[leverage] / 100.0
    liq_price = entry_price * (1.0 + liq_pct)
    path = frame[(frame["open_time"] >= entry_time) & (frame["open_time"] <= exit_time)].sort_values("open_time")
    hit = path[pd.to_numeric(path["low"], errors="coerce") <= liq_price]
    if hit.empty:
        return None
    return int(hit.iloc[0]["open_time"])


def find_mfe_aging_trigger(
    frame: pd.DataFrame,
    entry_time: int,
    exit_time: int,
    entry_price: float,
    leverage: int,
    activation_mfe_u: float,
) -> dict[str, Any] | None:
    path = frame[(frame["open_time"] >= entry_time) & (frame["open_time"] <= exit_time)].sort_values("open_time")
    running_mfe_u = -np.inf
    last_mfe_high_time = entry_time
    last_mfe_high_price = entry_price

    for row in path.itertuples(index=False):
        open_time = int(row.open_time)
        high = float(row.high)
        close = float(row.close)
        current_mfe_u = calc_leveraged_pnl(entry_price, high, leverage)
        if current_mfe_u > running_mfe_u:
            running_mfe_u = current_mfe_u
            last_mfe_high_time = open_time
            last_mfe_high_price = high
            continue

        if running_mfe_u >= activation_mfe_u and open_time - last_mfe_high_time >= AGING_MS:
            return {
                "partial_tp_time_ms": open_time,
                "partial_tp_time_utc": ms_to_utc(open_time).strftime("%Y-%m-%d %H:%M:%S"),
                "partial_tp_time_bj": ms_to_bj_string(open_time),
                "partial_tp_price": close,
                "trigger_running_mfe_u": float(running_mfe_u),
                "trigger_holding_days": (open_time - entry_time) / DAY_MS,
                "last_mfe_high_time_ms": last_mfe_high_time,
                "last_mfe_high_time_utc": ms_to_utc(last_mfe_high_time).strftime("%Y-%m-%d %H:%M:%S"),
                "last_mfe_high_time_bj": ms_to_bj_string(last_mfe_high_time),
                "last_mfe_high_price": float(last_mfe_high_price),
            }
    return None


def apply_mfe_aging_partial_tp(
    baseline: dict[str, Any],
    signal: pd.Series,
    kline_map: dict[str, pd.DataFrame],
    variant: MfeAgingVariant,
) -> dict[str, Any]:
    trade = baseline.copy()
    if trade.get("status") not in {"completed", "open_mark_to_market"}:
        return trade | no_partial_fields(variant)

    symbol = str(signal["symbol"])
    frame = kline_map.get(symbol, pd.DataFrame())
    if frame.empty:
        return trade | no_partial_fields(variant)

    entry_time = int(trade["entry_time_ms"])
    exit_time = int(float(trade["exit_time_ms"]))
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    leverage = int(trade["leverage"])

    trigger = find_mfe_aging_trigger(frame, entry_time, exit_time, entry_price, leverage, variant.activation_mfe_u)
    if trigger is None:
        return trade | no_partial_fields(variant)

    liq_time = liquidation_time_before_or_at(frame, entry_time, exit_time, entry_price, leverage)
    if liq_time is not None and liq_time <= int(trigger["partial_tp_time_ms"]):
        return trade | no_partial_fields(variant)

    tp_ratio = variant.tp_ratio
    tp_pnl = calc_leveraged_pnl(entry_price, float(trigger["partial_tp_price"]), leverage) * tp_ratio
    full_final_pnl = calc_leveraged_pnl(entry_price, exit_price, leverage)
    remaining_ratio = 1.0 - tp_ratio

    liquidated_after_partial = liq_time is not None and liq_time > int(trigger["partial_tp_time_ms"])
    if liquidated_after_partial:
        remaining_pnl = -BUY_NOTIONAL_U * remaining_ratio
        final_pnl = tp_pnl + remaining_pnl
        exit_reason = "liquidation_after_mfe_aging_partial_tp"
        liquidated = True
    else:
        remaining_pnl = full_final_pnl * remaining_ratio
        final_pnl = tp_pnl + remaining_pnl
        exit_reason = trade.get("exit_reason", "")
        liquidated = bool(trade.get("liquidated", False))

    trade["pnl_u"] = float(final_pnl)
    trade["net_return_pct"] = float(final_pnl / BUY_NOTIONAL_U * 100.0)
    trade["is_win"] = final_pnl > 0
    trade["liquidated"] = liquidated
    trade["exit_reason"] = exit_reason
    trade["partial_tp_triggered"] = True
    trade["partial_tp_ratio"] = tp_ratio
    trade["partial_tp_realized_pnl_u"] = float(tp_pnl)
    trade["partial_tp_remaining_pnl_u"] = float(remaining_pnl)
    trade["baseline_pnl_u"] = float(baseline.get("pnl_u", np.nan))
    trade["incremental_pnl_u"] = float(final_pnl - float(baseline.get("pnl_u", 0.0)))
    trade["liquidated_after_partial_tp"] = liquidated_after_partial
    trade.update(trigger)
    return trade


def no_partial_fields(variant: MfeAgingVariant) -> dict[str, Any]:
    return {
        "partial_tp_triggered": False,
        "partial_tp_ratio": variant.tp_ratio,
        "activation_mfe_u": variant.activation_mfe_u,
        "partial_tp_realized_pnl_u": 0.0,
        "partial_tp_remaining_pnl_u": np.nan,
        "baseline_pnl_u": np.nan,
        "incremental_pnl_u": 0.0,
        "liquidated_after_partial_tp": False,
        "partial_tp_time_ms": np.nan,
        "partial_tp_time_utc": "",
        "partial_tp_time_bj": "",
        "partial_tp_price": np.nan,
        "trigger_running_mfe_u": np.nan,
        "trigger_holding_days": np.nan,
        "last_mfe_high_time_ms": np.nan,
        "last_mfe_high_time_utc": "",
        "last_mfe_high_time_bj": "",
        "last_mfe_high_price": np.nan,
    }


def precompute_outcomes(
    signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    cutoff_ms: int,
    actions: pd.DataFrame,
    rank1_variant: main.Variant,
) -> tuple[dict[tuple[int, int, int], dict[str, Any]], dict[tuple[str, int, int, int], dict[str, Any]]]:
    original_hold = main.leverage_engine.HOLD_DAYS
    baseline: dict[tuple[int, int, int], dict[str, Any]] = {}
    variants: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    action_by_time = actions.set_index("signal_time").to_dict("index") if not actions.empty else {}
    try:
        for row in signals.itertuples(index=False):
            signal = pd.Series(row._asdict())
            sid = int(signal["signal_id"])
            action = action_by_time.get(int(signal["signal_time"]), {})
            if int(signal["rank"]) == 1:
                pairs = [main.rank1_params(rank1_variant, signal, action)]
            else:
                lev = int(signal["adaptive_leverage"]) if "adaptive_leverage" in signal and pd.notna(signal["adaptive_leverage"]) else main.rank23_leverage(signal, action)
                pairs = [(lev, RANK23_HOLD_DAYS)]
            for lev, hold_days in pairs:
                main.leverage_engine.HOLD_DAYS = hold_days
                base = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, lev)
                baseline[(sid, lev, hold_days)] = base
                signal_with_hold = signal.copy()
                signal_with_hold["target_hold_days"] = hold_days
                for variant in VARIANTS:
                    variants[(variant.name, sid, lev, hold_days)] = apply_mfe_aging_partial_tp(
                        base,
                        signal_with_hold,
                        kline_map,
                        variant,
                    )
            if sid and sid % 100 == 0:
                print(f"precomputed {sid}/{len(signals)}", flush=True)
    finally:
        main.leverage_engine.HOLD_DAYS = original_hold
    return baseline, variants


def replay_portfolio(
    strategy: str,
    signals: pd.DataFrame,
    outcomes: dict[tuple[int, int, int], dict[str, Any]] | dict[tuple[str, int, int, int], dict[str, Any]],
    actions: pd.DataFrame,
    rank1_variant: main.Variant,
    variant_name: str | None,
) -> pd.DataFrame:
    action_by_time = actions.set_index("signal_time").to_dict("index") if not actions.empty else {}
    rows: list[dict[str, Any]] = []
    open_by_symbol: dict[str, dict[str, Any]] = {}
    last_entry_by_symbol: dict[str, int] = {}
    last_pnl_by_symbol: dict[str, float] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        rank = int(signal["rank"])
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        action = action_by_time.get(signal_time, {})
        if rank == 1:
            lev, hold_days = main.rank1_params(rank1_variant, signal, action)
            component = "Rank1"
            bucket = "R1"
            original_leverage = lev
        else:
            hold_days = RANK23_HOLD_DAYS
            lev = int(signal["adaptive_leverage"]) if "adaptive_leverage" in signal and pd.notna(signal["adaptive_leverage"]) else main.rank23_leverage(signal, action)
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
            "signal_source": signal.get("signal_source", "reconstructed_1h_close"),
        }
        open_info = open_by_symbol.get(symbol)
        if open_info is not None and signal_time < int(open_info["open_until"]):
            row = skipped_open_position_trade(signal, int(open_info["open_until"]))
            row["leverage"] = lev
            rows.append(row | common | {"status": "skipped", "skip_reason": "symbol_already_open"})
            continue
        reason = same_symbol_reentry_block_reason(symbol, signal_time, last_entry_by_symbol, last_pnl_by_symbol)
        if reason is not None:
            row = skipped_open_position_trade(signal, signal_time)
            row["leverage"] = lev
            skip_reason = (
                "prev_win_same_symbol_reentry_0_30d"
                if "Previous winning" in reason
                else f"prev_loss_same_symbol_reentry_{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS}_{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS}d"
            )
            rows.append(
                row
                | common
                | {
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "filter_reason": reason,
                    "days_since_prev_trade": (signal_time - last_entry_by_symbol[symbol]) / DAY_MS,
                    "prev_pnl_u": last_pnl_by_symbol.get(symbol, np.nan),
                }
            )
            continue

        key = (int(signal["signal_id"]), lev, hold_days)
        trade = outcomes[key].copy() if variant_name is None else outcomes[(variant_name, *key)].copy()  # type: ignore[index]
        trade["signal_id"] = int(signal["signal_id"])
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            open_by_symbol[symbol] = {
                "open_until": int(float(trade["exit_time_ms"])) + (1 if trade.get("status") == "open_mark_to_market" else 0),
                "rank": rank,
                "entry_time_ms": signal_time,
                "pnl_u": trade.get("pnl_u", np.nan),
            }
            last_entry_by_symbol[symbol] = signal_time
            last_pnl_by_symbol[symbol] = float(trade.get("pnl_u", np.nan))
    return pd.DataFrame(rows)


def evaluated(frame: pd.DataFrame) -> pd.DataFrame:
    return main.evaluated(frame).copy()


def summary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    done = evaluated(frame)
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liq = done["liquidated"].astype(bool) if len(done) and "liquidated" in done else pd.Series(dtype=bool)
    net = float(pnl.sum()) if len(pnl) else 0.0
    top = pnl.sort_values(ascending=False)
    mfe = pd.to_numeric(done.get("max_price_during_trade", np.nan), errors="coerce")
    entry = pd.to_numeric(done.get("entry_price", np.nan), errors="coerce")
    lev = pd.to_numeric(done.get("leverage", np.nan), errors="coerce")
    max_fav = ((mfe / entry - 1.0) * lev * BUY_NOTIONAL_U).replace([np.inf, -np.inf], np.nan)
    denom = float(max_fav[max_fav > 0].sum())
    return {
        "trades": int(len(done)),
        "net_pnl_u": net,
        "pf": profit_factor(pnl),
        "ev_per_trade_u": net / len(done) if len(done) else np.nan,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "avg_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "max_drawdown_u": max_drawdown(pnl),
        "liquidations": int(liq.sum()) if len(liq) else 0,
        "liq_rate": float(liq.mean()) if len(liq) else np.nan,
        "ex_top1_pnl_u": float(net - top.head(1).sum()) if len(top) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - top.head(3).sum()) if len(top) >= 3 else np.nan,
        "mfe_capture": net / denom if denom else np.nan,
    }


def partial_tp_triggered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "partial_tp_triggered" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["partial_tp_triggered"].fillna(False).astype(bool)].copy()


def build_raw_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], int, int]:
    source_path = OUT / "rank1_portfolio_variant_comparison" / "trade_details_all_variants.csv"
    source = pd.read_csv(source_path)
    source = source[source["strategy"].eq("rank1_mixed_3x5d_v23_5x2d_v56_4060v23")].copy()
    source = source[source["status"].isin(["completed", "open_mark_to_market", "skipped"])].copy()
    source = source[source["entry_time_ms"].notna()].copy()
    source["signal_time"] = pd.to_numeric(source["entry_time_ms"], errors="coerce").astype("int64")
    source["snapshot_hour_bj"] = source.get("snapshot_hour_bj", "").fillna("")
    source["adaptive_leverage"] = pd.to_numeric(source.get("adaptive_leverage", source.get("leverage")), errors="coerce")
    combined = source.drop_duplicates(["signal_time", "rank", "symbol"], keep="first").copy()
    combined = combined.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    combined["signal_id"] = combined.index.astype(int)
    symbols = sorted(combined["symbol"].astype(str).unique().tolist())
    start = int(combined["signal_time"].min()) - 10 * DAY_MS
    common_end = int(max(pd.to_numeric(source.get("exit_time_ms"), errors="coerce").max(), combined["signal_time"].max()))
    signal_end = int(combined["signal_time"].max())
    kline_map = main.load_kline_map(symbols, start, common_end)
    actions = pd.DataFrame()
    return combined, actions, kline_map, common_end, signal_end


def partial_summary(name: str, trades: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    done = evaluated(trades)
    base_done = evaluated(baseline)
    triggered = partial_tp_triggered_frame(done)
    base_by_signal = base_done.set_index("signal_id")
    base_rows = base_by_signal.loc[triggered["signal_id"]] if len(triggered) else pd.DataFrame()
    candidate_pnl = pd.to_numeric(triggered["pnl_u"], errors="coerce") if len(triggered) else pd.Series(dtype=float)
    baseline_pnl = pd.to_numeric(base_rows["pnl_u"], errors="coerce") if len(base_rows) else pd.Series(dtype=float)
    bs = summary_metrics(baseline)
    cs = summary_metrics(trades)
    return {
        "variant": name,
        "triggered_trades": int(len(triggered)),
        "trigger_rate": float(len(triggered) / len(done)) if len(done) else np.nan,
        "avg_trigger_running_mfe_u": float(pd.to_numeric(triggered.get("trigger_running_mfe_u", pd.Series(dtype=float)), errors="coerce").mean()) if len(triggered) else np.nan,
        "avg_trigger_holding_days": float(pd.to_numeric(triggered.get("trigger_holding_days", pd.Series(dtype=float)), errors="coerce").mean()) if len(triggered) else np.nan,
        "partial_tp_realized_pnl_u": float(pd.to_numeric(triggered.get("partial_tp_realized_pnl_u", pd.Series(dtype=float)), errors="coerce").sum()) if len(triggered) else 0.0,
        "triggered_baseline_pnl_u": float(baseline_pnl.sum()) if len(baseline_pnl) else 0.0,
        "triggered_candidate_pnl_u": float(candidate_pnl.sum()) if len(candidate_pnl) else 0.0,
        "triggered_delta_u": float(candidate_pnl.sum() - baseline_pnl.sum()) if len(candidate_pnl) else 0.0,
        "net_pnl_delta_u": cs["net_pnl_u"] - bs["net_pnl_u"],
        "dd_delta_u": cs["max_drawdown_u"] - bs["max_drawdown_u"],
        "liquidation_delta": cs["liquidations"] - bs["liquidations"],
        "mfe_capture_delta": cs["mfe_capture"] - bs["mfe_capture"],
    }


def big_winner_damage(name: str, trades: pd.DataFrame, baseline: pd.DataFrame) -> list[dict[str, Any]]:
    done = evaluated(trades)
    base_done = evaluated(baseline)
    merged = base_done[["signal_id", "symbol", "rank", "month", "pnl_u"]].merge(
        done[["signal_id", "pnl_u", "partial_tp_triggered"]],
        on="signal_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )
    rows = []
    for threshold in BIG_WINNER_THRESHOLDS:
        scoped = merged[(merged["pnl_u_baseline"] > threshold) & (merged["partial_tp_triggered"].astype(bool))]
        rows.append(
            {
                "variant": name,
                "winner_threshold_u": threshold,
                "triggered_big_winner_count": int(len(scoped)),
                "baseline_total_pnl_u": float(scoped["pnl_u_baseline"].sum()) if len(scoped) else 0.0,
                "candidate_total_pnl_u": float(scoped["pnl_u_candidate"].sum()) if len(scoped) else 0.0,
                "right_tail_profit_loss_u": float((scoped["pnl_u_baseline"] - scoped["pnl_u_candidate"]).sum()) if len(scoped) else 0.0,
            }
        )
    return rows


def top10_winner_audit(trades_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = evaluated(trades_by_variant[BASELINE]).copy()
    top = base.sort_values("pnl_u", ascending=False).head(10)
    rows = []
    for variant_name, frame in trades_by_variant.items():
        if variant_name == BASELINE:
            continue
        done = evaluated(frame).set_index("signal_id")
        for item in top.itertuples(index=False):
            cand = done.loc[int(item.signal_id)]
            rows.append(
                {
                    "variant": variant_name,
                    "symbol": item.symbol,
                    "entry_time_bj": item.entry_time_bj,
                    "rank": int(item.rank),
                    "running_mfe_u": float((item.max_price_during_trade / item.entry_price - 1.0) * item.leverage * BUY_NOTIONAL_U),
                    "max_mfe_time_bj": str(cand.get("last_mfe_high_time_bj", "")),
                    "partial_tp_triggered": bool(cand.get("partial_tp_triggered", False)),
                    "partial_tp_time_bj": str(cand.get("partial_tp_time_bj", "")),
                    "baseline_final_pnl_u": float(item.pnl_u),
                    "candidate_final_pnl_u": float(cand["pnl_u"]),
                    "pnl_delta_u": float(cand["pnl_u"] - item.pnl_u),
                }
            )
    return pd.DataFrame(rows)


def write_final(
    overall: pd.DataFrame,
    partial: pd.DataFrame,
    big_damage: pd.DataFrame,
    signal_end: int,
    common_end: int,
) -> None:
    base = overall[overall["variant"].eq(BASELINE)].iloc[0]
    candidates = overall[~overall["variant"].eq(BASELINE)].copy()
    candidates["net_delta"] = candidates["net_pnl_u"] - float(base["net_pnl_u"])
    candidates["pf_delta"] = candidates["pf"] - float(base["pf"])
    candidates["ev_delta"] = candidates["ev_per_trade_u"] - float(base["ev_per_trade_u"])
    best = candidates.sort_values(["net_delta", "pf_delta"], ascending=False).iloc[0]
    improved = candidates[(candidates["net_delta"] > 0) & (candidates["pf_delta"] > 0) & (candidates["ev_delta"] > 0)]
    damage_500 = big_damage[big_damage["winner_threshold_u"].eq(500.0)].groupby("variant")["right_tail_profit_loss_u"].sum()
    damage_1000 = big_damage[big_damage["winner_threshold_u"].eq(1000.0)].groupby("variant")["right_tail_profit_loss_u"].sum()
    judgment = "OBSERVE" if len(improved) >= 2 else "REJECT"
    if len(improved) >= 4 and float(damage_500.get(best["variant"], 0.0)) <= 0:
        judgment = "SHADOW CANDIDATE"
    lines = [
        "# MFE Aging Partial TP Research",
        "",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Cache common end: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "- Baseline: current RankPulse main strategy replay.",
        "- Trigger: Running MFE >= 300U/400U and no new high for >=24H; one partial TP.",
        "",
        "## Final Judgment",
        "",
        judgment,
        "",
        f"- Best net variant: {best['variant']} ({best['net_delta']:+.2f}U vs Baseline).",
        f"- Best net PF/EV delta: {best['pf_delta']:+.4f} / {best['ev_delta']:+.2f}U per trade.",
        f"- Best net >500U winner damage: {float(damage_500.get(best['variant'], 0.0)):.2f}U.",
        f"- Best net >1000U winner damage: {float(damage_1000.get(best['variant'], 0.0)):.2f}U.",
        f"- Parameter points improving Net/PF/EV together: {len(improved)} of {len(candidates)}.",
        "",
        "## Overall",
        "",
        overall.round(4).to_string(index=False),
        "",
        "## MFE Aging Summary",
        "",
        partial.round(4).to_string(index=False),
        "",
        "## Big Winner Damage",
        "",
        big_damage.round(4).to_string(index=False),
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")


def main_run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building inputs", flush=True)
    signals, actions, kline_map, common_end, signal_end = build_raw_inputs()
    print(f"signals {len(signals)}", flush=True)
    rank1_variant = [item for item in main.VARIANTS if item.name == "rank1_mixed_3x5d_v23_5x2d_v56_4060v23"][0]
    print("precomputing outcomes", flush=True)
    baseline_outcomes, variant_outcomes = precompute_outcomes(signals, kline_map, common_end, actions, rank1_variant)
    print("replaying baseline/candidates", flush=True)
    trades_by_variant: dict[str, pd.DataFrame] = {
        BASELINE: replay_portfolio(BASELINE, signals, baseline_outcomes, actions, rank1_variant, None)
    }
    for variant in VARIANTS:
        trades_by_variant[variant.name] = replay_portfolio(variant.name, signals, variant_outcomes, actions, rank1_variant, variant.name)

    overall = pd.DataFrame([{"variant": name, **summary_metrics(frame)} for name, frame in trades_by_variant.items()])
    overall.to_csv(OUT_DIR / "overall_comparison.csv", index=False, encoding="utf-8-sig")

    base_done = evaluated(trades_by_variant[BASELINE])
    monthly_rows = []
    rank_rows = []
    for name, frame in trades_by_variant.items():
        done = evaluated(frame)
        for month in sorted(set(base_done["month"].dropna().astype(str)) | set(done["month"].dropna().astype(str))):
            base_m = base_done[base_done["month"].eq(month)]
            cand_m = done[done["month"].eq(month)]
            monthly_rows.append(
                {
                    "variant": name,
                    "month": month,
                    "baseline_pnl_u": summary_metrics(base_m)["net_pnl_u"],
                    "candidate_pnl_u": summary_metrics(cand_m)["net_pnl_u"],
                    "delta_u": summary_metrics(cand_m)["net_pnl_u"] - summary_metrics(base_m)["net_pnl_u"],
                }
            )
        for rank in [1, 2, 3]:
            base_r = base_done[base_done["rank"].eq(rank)]
            cand_r = done[done["rank"].eq(rank)]
            bs = summary_metrics(base_r)
            cs = summary_metrics(cand_r)
            triggered = partial_tp_triggered_frame(cand_r)
            rank_rows.append(
                {
                    "variant": name,
                    "rank": rank,
                    "triggered_count": int(len(triggered)) if name != BASELINE else 0,
                    "pnl_delta_u": cs["net_pnl_u"] - bs["net_pnl_u"],
                    "pf_delta": cs["pf"] - bs["pf"],
                    "dd_delta_u": cs["max_drawdown_u"] - bs["max_drawdown_u"],
                    "mfe_capture_delta": cs["mfe_capture"] - bs["mfe_capture"],
                    "big_winner_damage_u": 0.0,
                    "liquidation_delta": cs["liquidations"] - bs["liquidations"],
                }
            )

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT_DIR / "rank_comparison.csv", index=False, encoding="utf-8-sig")

    partial = pd.DataFrame([partial_summary(name, frame, trades_by_variant[BASELINE]) for name, frame in trades_by_variant.items() if name != BASELINE])
    partial.to_csv(OUT_DIR / "mfe_aging_summary.csv", index=False, encoding="utf-8-sig")

    big_rows: list[dict[str, Any]] = []
    for name, frame in trades_by_variant.items():
        if name != BASELINE:
            big_rows.extend(big_winner_damage(name, frame, trades_by_variant[BASELINE]))
    big_damage = pd.DataFrame(big_rows)
    big_damage.to_csv(OUT_DIR / "big_winner_damage.csv", index=False, encoding="utf-8-sig")

    top10 = top10_winner_audit(trades_by_variant)
    top10.to_csv(OUT_DIR / "baseline_top10_winner_audit.csv", index=False, encoding="utf-8-sig")

    details = pd.concat(trades_by_variant.values(), ignore_index=True, sort=False)
    details.to_csv(OUT_DIR / "trade_details_each_variant.csv", index=False, encoding="utf-8-sig")
    partial_tp_triggered_frame(details).to_csv(OUT_DIR / "mfe_aging_counterfactual.csv", index=False, encoding="utf-8-sig")

    write_final(overall, partial, big_damage, signal_end, common_end)
    print("output", OUT_DIR)
    print(overall.round(4).to_string(index=False))
    print()
    print(partial.round(4).to_string(index=False))


if __name__ == "__main__":
    main_run()
