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
from scripts import research_rank1_portfolio_variant_comparison as main
from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    EARLY_REASON,
    FEE_RATE,
    HOUR_MS,
    OUT,
    get_open_at_or_latest,
    max_drawdown,
    mfe_mae,
    ms_to_bj_string,
    ms_to_utc,
    path_slice,
    profit_factor,
    skipped_open_position_trade,
)
from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, latest_signal_end_dt
from scripts.regime_adaptive_leverage_walkforward import bucket_for_signal
from src.research.rankpulse_strategy_rules import (
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS,
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS,
    same_symbol_reentry_block_reason,
)


OUT_DIR = OUT / "24h_weak_exit_research"
BASELINE = "Baseline"
THRESHOLDS = {
    "MFE5": 5.0,
    "MFE8": 8.0,
    "MFE10": 10.0,
    "MFE15": 15.0,
}
RANK23_HOLD_DAYS = main.RANK23_HOLD_DAYS
LIQUIDATION_THRESHOLDS_PCT = {1: -100.0, 2: -50.0, 3: -33.0, 5: -20.0}
BIG_WINNER_THRESHOLDS = [50.0, 100.0, 300.0, 500.0]


def calc_leveraged_pnl(entry_price: float, exit_price: float, leverage: int) -> tuple[float, float]:
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / entry_price
    exit_value = qty * exit_price * (1.0 - FEE_RATE)
    pnl = exit_value - nominal
    return float(pnl), float(pnl / BUY_NOTIONAL_U * 100.0)


def current_close_at_or_before(frame: pd.DataFrame, current_time: int) -> tuple[int, float]:
    available = frame[frame["open_time"] <= current_time].copy()
    if available.empty:
        return current_time, math.nan
    row = available.sort_values("open_time").iloc[-1]
    return int(row["open_time"]), float(row["close"])


def max_favorable_pnl_u(entry_price: float, max_price: float, leverage: int) -> float:
    nominal = BUY_NOTIONAL_U * leverage
    qty = nominal * (1.0 - FEE_RATE) / entry_price
    return float(qty * max_price * (1.0 - FEE_RATE) - nominal)


def simulate_trade(
    signal: pd.Series,
    kline_map: dict[str, pd.DataFrame],
    current_time: int,
    leverage: int,
    threshold: float | None,
) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    h1 = kline_map.get(symbol, pd.DataFrame())
    base = {
        "symbol": symbol,
        "rank": int(signal["rank"]),
        "leverage": leverage,
        "entry_time_ms": entry_time,
        "entry_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_bj": ms_to_bj_string(entry_time),
        "snapshot_hour_bj": signal["snapshot_hour_bj"],
        "gain_24h": float(signal["gain_24h"]),
        "gain_24h_bucket": signal.get("gain_24h_bucket", main.gain_bucket(float(signal["gain_24h"]))),
        "month": ms_to_utc(entry_time).strftime("%Y-%m"),
        "volume_24h_ratio_7d": signal.get("volume_24h_ratio_7d", np.nan),
        "volume_24h_ratio_7d_bucket": signal.get("volume_24h_ratio_7d_bucket", "missing"),
        "ma_structure_4h": signal.get("ma_structure_4h", "missing"),
        "distance_to_4h_ma7_pct": signal.get("distance_to_4h_ma7_pct", np.nan),
        "target_hold_days": int(signal.get("target_hold_days", RANK23_HOLD_DAYS)),
    }
    if h1.empty:
        return base | {"status": "skipped", "skip_reason": "missing_symbol_klines"}
    indexed = h1.set_index("open_time", drop=False)
    if entry_time not in indexed.index:
        return base | {"status": "skipped", "skip_reason": "missing_entry_kline"}
    entry_row = indexed.loc[entry_time]
    if isinstance(entry_row, pd.DataFrame):
        entry_row = entry_row.iloc[-1]
    entry_price = float(entry_row["open"])

    first_4h = path_slice(h1, entry_time, min(entry_time + 4 * HOUR_MS - HOUR_MS, current_time))
    mfe4, mae4, _, _ = mfe_mae(first_4h, entry_price) if len(first_4h) >= 1 else (np.nan, np.nan, np.nan, np.nan)
    first_12h = path_slice(h1, entry_time, min(entry_time + 12 * HOUR_MS - HOUR_MS, current_time))
    mfe12, mae12, _, _ = mfe_mae(first_12h, entry_price) if len(first_12h) >= 1 else (np.nan, np.nan, np.nan, np.nan)
    close_return_12h = (
        (float(first_12h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0
        if len(first_12h) >= 1
        else np.nan
    )
    first_24h = path_slice(h1, entry_time, min(entry_time + 24 * HOUR_MS - HOUR_MS, current_time))
    mfe24, mae24, _, _ = mfe_mae(first_24h, entry_price) if len(first_24h) >= 1 else (np.nan, np.nan, np.nan, np.nan)
    close_return_24h = (
        (float(first_24h.iloc[-1]["close"]) / entry_price - 1.0) * 100.0
        if len(first_24h) >= 1
        else np.nan
    )

    if len(first_4h) >= 4 and mfe4 < 2.0 and mae4 < -8.0:
        exit_target = entry_time + 4 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or "extreme_weak_4h"
        status = "completed"
    elif len(first_12h) >= 12 and mfe12 < 5.0 and close_return_12h < 0.0:
        exit_target = entry_time + 12 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or EARLY_REASON
        status = "completed"
    elif threshold is not None and len(first_24h) >= 24 and mfe24 < threshold and close_return_24h < 0.0:
        exit_target = entry_time + 24 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        exit_reason = fallback or f"weak_24h_mfe_lt{threshold:g}_close_neg"
        status = "completed"
    else:
        hold_days = int(signal.get("target_hold_days", RANK23_HOLD_DAYS))
        exit_target = entry_time + hold_days * DAY_MS
        if exit_target <= current_time:
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
            exit_reason = fallback or f"fixed_{hold_days}d"
            status = "completed"
        else:
            exit_time, exit_price = current_close_at_or_before(h1, current_time)
            exit_reason = "open_mark_to_market"
            status = "open_mark_to_market"

    if not np.isfinite(exit_price):
        return base | {"status": "skipped", "skip_reason": "missing_exit_price", "entry_price": entry_price}

    trade_path = path_slice(h1, entry_time, exit_time)
    mfe, mae, max_price, min_price = mfe_mae(trade_path, entry_price)
    liquidated = bool(mae <= LIQUIDATION_THRESHOLDS_PCT[leverage])
    if liquidated:
        pnl = -BUY_NOTIONAL_U
        net_return = -100.0
        exit_reason = "liquidation"
        status = "completed"
    else:
        pnl, net_return = calc_leveraged_pnl(entry_price, exit_price, leverage)
    return base | {
        "status": status,
        "skip_reason": "",
        "entry_price": entry_price,
        "exit_time_ms": exit_time,
        "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(exit_time),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": (exit_time - entry_time) / DAY_MS,
        "underlying_return_pct": (float(exit_price) / entry_price - 1.0) * 100.0,
        "pnl_u": pnl,
        "net_return_pct": net_return,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "mfe_4h_pct": mfe4,
        "mae_4h_pct": mae4,
        "mfe_12h_pct": mfe12,
        "mae_12h_pct": mae12,
        "close_return_12h_pct": close_return_12h,
        "mfe_24h_pct": mfe24,
        "mae_24h_pct": mae24,
        "close_return_24h_pct": close_return_24h,
        "max_price_during_trade": max_price,
        "min_price_during_trade": min_price,
        "max_favorable_pnl_u": max_favorable_pnl_u(entry_price, max_price, leverage),
        "liquidated": liquidated,
        "is_win": pnl > 0,
    }


def precompute_outcomes(
    signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    cutoff_ms: int,
) -> tuple[dict[tuple[int, int, int], dict[str, Any]], dict[tuple[str, int, int, int], dict[str, Any]]]:
    baseline: dict[tuple[int, int, int], dict[str, Any]] = {}
    variants: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    rank1_hold_days = sorted({v.rank1_hold_days for v in main.VARIANTS} | {hold for v in main.VARIANTS for _, hold in (v.cell_overrides or {}).values()})
    rank1_leverages = sorted({1, 3, 5} | {lev for v in main.VARIANTS for lev, _ in (v.cell_overrides or {}).values()})
    for row in signals.itertuples(index=False):
        raw = row._asdict()
        signal = pd.Series(raw)
        sid = int(signal["signal_id"])
        if int(signal["rank"]) == 1:
            pairs = [(lev, hold) for hold in rank1_hold_days for lev in rank1_leverages]
        else:
            pairs = [(lev, RANK23_HOLD_DAYS) for lev in [1, 2, 3, 5]]
        for lev, hold_days in pairs:
            signal_with_hold = signal.copy()
            signal_with_hold["target_hold_days"] = hold_days
            baseline[(sid, lev, hold_days)] = simulate_trade(signal_with_hold, kline_map, cutoff_ms, lev, None)
            for name, threshold in THRESHOLDS.items():
                variants[(name, sid, lev, hold_days)] = simulate_trade(signal_with_hold, kline_map, cutoff_ms, lev, threshold)
    return baseline, variants


def replay_portfolio(
    strategy: str,
    signals: pd.DataFrame,
    outcomes: dict[tuple[int, int, int], dict[str, Any]] | dict[tuple[str, int, int, int], dict[str, Any]],
    actions: pd.DataFrame,
    variant: main.Variant,
    threshold_name: str | None,
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
            lev, hold_days = main.rank1_params(variant, signal, action)
            component = "Rank1"
            bucket = "R1"
            original_leverage = lev
        else:
            hold_days = RANK23_HOLD_DAYS
            lev = main.rank23_leverage(signal, action)
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
        if threshold_name is None:
            trade = outcomes[key].copy()  # type: ignore[index]
        else:
            trade = outcomes[(threshold_name, *key)].copy()  # type: ignore[index]
        trade["signal_id"] = int(signal["signal_id"])
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
            last_entry_by_symbol[symbol] = signal_time
            last_pnl_by_symbol[symbol] = float(trade.get("pnl_u", np.nan))
    return pd.DataFrame(rows)


def summary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    done = main.evaluated(frame)
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liq = done["liquidated"].astype(bool) if len(done) and "liquidated" in done else pd.Series(dtype=bool)
    net = float(pnl.sum()) if len(pnl) else 0.0
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
        "ex_top1_pnl_u": float(net - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
    }


def counterfactual_for_variant(
    name: str,
    trades: pd.DataFrame,
    baseline_outcomes: dict[tuple[int, int, int], dict[str, Any]],
) -> pd.DataFrame:
    exits = main.evaluated(trades)
    exits = exits[exits["exit_reason"].astype(str).eq(f"weak_24h_mfe_lt{THRESHOLDS[name]:g}_close_neg")].copy()
    rows: list[dict[str, Any]] = []
    for trade in exits.itertuples(index=False):
        baseline = baseline_outcomes[(int(trade.signal_id), int(trade.leverage), int(trade.target_hold_days))]
        baseline_mfe_u = float(baseline.get("max_favorable_pnl_u", np.nan))
        rows.append(
            {
                "variant": name,
                "symbol": trade.symbol,
                "entry_time": trade.entry_time_bj,
                "rank": int(trade.rank),
                "bucket": trade.bucket,
                "leverage": int(trade.leverage),
                "MFE24": float(getattr(trade, "mfe_24h_pct", np.nan)),
                "MAE24": float(getattr(trade, "mae_24h_pct", np.nan)),
                "close_return_24h": float(getattr(trade, "close_return_24h_pct", np.nan)),
                "24H_exit_pnl": float(trade.pnl_u),
                "baseline_exit_time": baseline.get("exit_time_bj", ""),
                "baseline_exit_reason": baseline.get("exit_reason", ""),
                "baseline_pnl": float(baseline.get("pnl_u", np.nan)),
                "baseline_mfe_u": baseline_mfe_u,
                "baseline_mfe_return_pct": baseline_mfe_u,
                "incremental_pnl": float(trade.pnl_u) - float(baseline.get("pnl_u", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def build_raw_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], int, int]:
    symbols = [s for s in main.cached_symbols() if s not in main.EXCLUDE_SYMBOLS]
    common_end = main.cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = main.load_kline_map(symbols, main.SIGNAL_START_MS - 10 * DAY_MS, common_end)
    reconstructed = main.generate_signals(main.SIGNAL_START_MS, signal_end, kline_map)
    live = main.load_live_event_signals(
        main.event_log_paths(None),
        main.SIGNAL_START_MS,
        signal_end,
        include_opened_fallback=True,
    )
    raw = main.combine_signal_sources(reconstructed, live, "hybrid")
    rank23 = main.apply_entry_rules_preserving(raw, kline_map).copy()
    rank23["strategy_component"] = rank23["rank"].map(lambda r: f"Rank{int(r)}")
    rank23["bucket"] = rank23.apply(bucket_for_signal, axis=1)
    rank23["rank1_cell"] = ""
    rank1 = main.add_rank1_candidates(raw, kline_map)
    combined = pd.concat([rank1, rank23], ignore_index=True, sort=False)
    combined = combined.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)
    combined["signal_id"] = combined.index.astype(int)
    actions = main.build_fr3_yr1_actions(raw, rank23, kline_map)
    return combined, actions, kline_map, common_end, signal_end


def main_run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined, actions, kline_map, common_end, signal_end = build_raw_inputs()
    variant = [item for item in main.VARIANTS if item.name == "rank1_mixed_3x5d_v23_5x2d_v56_4060v23"][0]
    baseline_outcomes, variant_outcomes = precompute_outcomes(combined, kline_map, common_end)

    trades_by_variant: dict[str, pd.DataFrame] = {
        BASELINE: replay_portfolio(BASELINE, combined, baseline_outcomes, actions, variant, None)
    }
    for name in THRESHOLDS:
        trades_by_variant[name] = replay_portfolio(name, combined, variant_outcomes, actions, variant, name)

    overall = pd.DataFrame([{"variant": name, **summary_metrics(frame)} for name, frame in trades_by_variant.items()])
    overall.to_csv(OUT_DIR / "overall_comparison.csv", index=False, encoding="utf-8-sig")

    monthly_rows = []
    rank_rows = []
    for name, frame in trades_by_variant.items():
        done = main.evaluated(frame)
        for month, group in done.groupby("month", sort=True):
            monthly_rows.append({"variant": name, "month": month, **summary_metrics(group)})
        for rank, group in done.groupby("rank", sort=True):
            s = summary_metrics(group)
            base_rank = summary_metrics(main.evaluated(trades_by_variant[BASELINE]).query("rank == @rank")) if name != BASELINE else s
            rank_rows.append(
                {
                    "variant": name,
                    "rank": int(rank),
                    **s,
                    "net_pnl_delta_u": s["net_pnl_u"] - base_rank["net_pnl_u"],
                    "pf_delta": s["pf"] - base_rank["pf"],
                    "liquidations_reduced": base_rank["liquidations"] - s["liquidations"],
                }
            )
    pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")

    counterfactual = pd.concat(
        [counterfactual_for_variant(name, trades_by_variant[name], baseline_outcomes) for name in THRESHOLDS],
        ignore_index=True,
        sort=False,
    )
    counterfactual.to_csv(OUT_DIR / "24h_exit_counterfactual.csv", index=False, encoding="utf-8-sig")

    exit_rows = []
    missed_rows = []
    for name in THRESHOLDS:
        cf = counterfactual[counterfactual["variant"].eq(name)].copy()
        pnl = pd.to_numeric(cf["24H_exit_pnl"], errors="coerce") if len(cf) else pd.Series(dtype=float)
        base_pnl = pd.to_numeric(cf["baseline_pnl"], errors="coerce") if len(cf) else pd.Series(dtype=float)
        inc = pd.to_numeric(cf["incremental_pnl"], errors="coerce") if len(cf) else pd.Series(dtype=float)
        saved_loss = float(inc[(base_pnl < 0) & (inc > 0)].sum()) if len(cf) else 0.0
        cut_winner = float((-inc[(base_pnl > 0) & (inc < 0)]).sum()) if len(cf) else 0.0
        row = {
            "variant": name,
            "trigger_count": int(len(cf)),
            "exit_avg_pnl_u": float(pnl.mean()) if len(pnl) else np.nan,
            "exit_median_pnl_u": float(pnl.median()) if len(pnl) else np.nan,
            "baseline_hold_pnl_u": float(base_pnl.sum()) if len(base_pnl) else 0.0,
            "24h_exit_pnl_u": float(pnl.sum()) if len(pnl) else 0.0,
            "saved_loss_u": saved_loss,
            "cut_winner_u": cut_winner,
            "net_incremental_pnl_u": float(inc.sum()) if len(inc) else 0.0,
        }
        for threshold in BIG_WINNER_THRESHOLDS:
            big = cf[cf["baseline_mfe_return_pct"] > threshold].copy()
            loss = float((big["baseline_pnl"] - big["24H_exit_pnl"]).clip(lower=0).sum()) if len(big) else 0.0
            row[f"missed_gt_{int(threshold)}pct_count"] = int(len(big))
            row[f"missed_gt_{int(threshold)}pct_loss_u"] = loss
            for _, item in big.iterrows():
                missed_rows.append(
                    {
                        "variant": name,
                        "threshold_pct": threshold,
                        "symbol": item["symbol"],
                        "entry_time": item["entry_time"],
                        "rank": item["rank"],
                        "bucket": item["bucket"],
                        "leverage": item["leverage"],
                        "baseline_mfe_return_pct": item["baseline_mfe_return_pct"],
                        "24H_exit_pnl": item["24H_exit_pnl"],
                        "baseline_pnl": item["baseline_pnl"],
                        "lost_pnl_u": max(0.0, item["baseline_pnl"] - item["24H_exit_pnl"]),
                    }
                )
        exit_rows.append(row)
    exit_summary = pd.DataFrame(exit_rows)
    exit_summary.to_csv(OUT_DIR / "24h_exit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missed_rows).to_csv(OUT_DIR / "missed_big_winners.csv", index=False, encoding="utf-8-sig")

    rank_df = pd.DataFrame(rank_rows)
    missed_gt100 = counterfactual[counterfactual["baseline_mfe_return_pct"] > 100].groupby(["variant", "rank"]).size().rename("missed_gt_100pct_count").reset_index()
    rank_df = rank_df.merge(missed_gt100, on=["variant", "rank"], how="left")
    rank_df["missed_gt_100pct_count"] = rank_df["missed_gt_100pct_count"].fillna(0).astype(int)
    rank_df.to_csv(OUT_DIR / "rank_comparison.csv", index=False, encoding="utf-8-sig")

    pd.concat(trades_by_variant.values(), ignore_index=True, sort=False).to_csv(
        OUT_DIR / "trade_details_each_variant.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# 24H Weak Exit Research",
        "",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Cache common end: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "- Baseline: current RankPulse main strategy replay.",
        "- Variable: 24H MFE < X and 24H close_return < 0, X in 5/8/10/15.",
        "- Exit priority: 4H extreme weak, 12H weak, 24H weak, fixed hold.",
        "",
        "## Overall",
        "",
        overall.round(4).to_string(index=False),
        "",
        "## 24H Exit Summary",
        "",
        exit_summary.round(4).to_string(index=False),
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")
    print("output", OUT_DIR)
    print(overall.round(4).to_string(index=False))
    print()
    print(exit_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main_run()
