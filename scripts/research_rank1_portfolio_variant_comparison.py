from __future__ import annotations

import argparse
import json
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
from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, add_entry_factors, backfill_symbols, bucket_volume, load_kline_map, max_drawdown, ms_to_utc, profit_factor, skipped_open_position_trade
from scripts.backtest_futures_top2_fixed_time import SimpleBinanceFuturesClient, generate_signals, get_futures_symbols, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS, IndicatorSpec, build_health_timeline, opportunity_sets
from scripts.rank3_fast_recovery_vs_monthly_reset import RecoverySpec, build_recovery_timeline, fast_recovery_action_timeline
from scripts.regime_adaptive_leverage_walkforward import bucket_for_signal, simulate_trade_with_leverage
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, cache_common_end_ms, cached_symbols, gain_bucket, leverage_for_signal
from src.research.rankpulse_strategy_rules import (
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS,
    PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS,
    same_symbol_reentry_block_reason,
)


OUT_DIR = OUT / "rank1_portfolio_variant_comparison"
DEFAULT_SNAPSHOT_LOGS = [
    ROOT / "data" / "live" / "production" / "events.jsonl",
]
FACTOR_COLUMNS = [
    "ma_structure_4h",
    "distance_to_4h_ma7_pct",
    "volume_24h_ratio_7d",
    "volume_24h_ratio_7d_bucket",
]
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
    ex_rave = done[~done["symbol"].astype(str).eq("RAVEUSDT")].copy() if len(done) else done
    ex_rave_pnl = pd.to_numeric(ex_rave["pnl_u"], errors="coerce") if len(ex_rave) else pd.Series(dtype=float)
    ex_rave_ret = pd.to_numeric(ex_rave["net_return_pct"], errors="coerce") if len(ex_rave) else pd.Series(dtype=float)
    ex_rave_liq = ex_rave["liquidated"].astype(bool) if len(ex_rave) and "liquidated" in ex_rave else pd.Series(dtype=bool)
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
        "ex_rave_trades": int(len(ex_rave)),
        "ex_rave_net_pnl_u": float(ex_rave_pnl.sum()) if len(ex_rave_pnl) else 0.0,
        "ex_rave_pf": profit_factor(ex_rave_pnl),
        "ex_rave_win_rate": float((ex_rave_pnl > 0).sum() / len(ex_rave_pnl)) if len(ex_rave_pnl) else np.nan,
        "ex_rave_median_return_pct": float(ex_rave_ret.median()) if len(ex_rave_ret) else np.nan,
        "ex_rave_liquidations": int(ex_rave_liq.sum()) if len(ex_rave_liq) else 0,
        "ex_rave_liq_rate": float(ex_rave_liq.sum() / len(ex_rave)) if len(ex_rave) else np.nan,
        "rave_pnl_u": float(net - ex_rave_pnl.sum()) if len(done) else 0.0,
        "rave_trades": int(done["symbol"].astype(str).eq("RAVEUSDT").sum()) if len(done) else 0,
        "ex_top1_pnl_u": float(net - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "ex_top5_pnl_u": float(net - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
        "ex_top10_pnl_u": float(net - pnl.nlargest(10).sum()) if len(pnl) >= 10 else np.nan,
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
    }



def snapshot_hour_bj(signal_time_ms: int) -> str:
    return (ms_to_utc(signal_time_ms) + pd.Timedelta(hours=8)).strftime("%H:%M")


def canonical_signal_time_ms(timestamp_ms: int) -> int:
    return int(timestamp_ms) // 60_000 * 60_000


def event_log_paths(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(path) for path in paths]
    return DEFAULT_SNAPSHOT_LOGS


def load_live_event_signals(
    paths: list[Path],
    signal_start: int,
    signal_end: int,
    include_opened_fallback: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_name = str(event.get("event", ""))
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                if event_name.endswith("_signal_snapshot"):
                    snapshot_time = payload.get("signal_time_ms")
                    snapshot_rows = payload.get("rows") or payload.get("snapshot_rows") or []
                    if not isinstance(snapshot_rows, list):
                        continue
                    for item in snapshot_rows:
                        if not isinstance(item, dict):
                            continue
                        signal_time = item.get("signal_time_ms", snapshot_time)
                        if signal_time is None:
                            continue
                        signal_time = canonical_signal_time_ms(int(signal_time))
                        if signal_time < signal_start or signal_time > signal_end:
                            continue
                        symbol = str(item.get("symbol", "")).upper()
                        if not symbol:
                            continue
                        rows.append(
                            {
                                "signal_time": signal_time,
                                "signal_time_utc": ms_to_utc(signal_time).strftime("%Y-%m-%d %H:%M:%S"),
                                "signal_time_bj": (ms_to_utc(signal_time) + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                                "snapshot_hour_bj": snapshot_hour_bj(signal_time),
                                "symbol": symbol,
                                "rank": int(item.get("rank")),
                                "gain_24h": float(item.get("gain_24h")),
                                "volume_24h_ratio_7d": item.get("volume_24h_ratio_7d"),
                                "live_price": item.get("price"),
                                "live_passed": item.get("passed"),
                                "live_filter_reason": item.get("filter_reason"),
                                "live_leverage": item.get("leverage"),
                                "signal_source": event_name,
                                "source_priority": 0,
                            }
                        )
                elif include_opened_fallback and event_name in {"live_opened", "testnet_opened", "paper_opened"}:
                    entry_time = payload.get("entry_time_ms")
                    if entry_time is None:
                        continue
                    signal_time = canonical_signal_time_ms(int(entry_time))
                    if signal_time < signal_start or signal_time > signal_end:
                        continue
                    symbol = str(payload.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    rows.append(
                        {
                            "signal_time": signal_time,
                            "signal_time_utc": ms_to_utc(signal_time).strftime("%Y-%m-%d %H:%M:%S"),
                            "signal_time_bj": (ms_to_utc(signal_time) + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                            "snapshot_hour_bj": snapshot_hour_bj(signal_time),
                            "symbol": symbol,
                            "rank": int(payload.get("rank")),
                            "gain_24h": float(payload.get("gain_24h")),
                            "volume_24h_ratio_7d": np.nan,
                            "live_price": payload.get("entry_price"),
                            "live_passed": True,
                            "live_filter_reason": "opened_event_fallback",
                            "live_leverage": payload.get("leverage"),
                            "signal_source": event_name,
                            "source_priority": 1,
                        }
                    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["volume_24h_ratio_7d"] = pd.to_numeric(frame["volume_24h_ratio_7d"], errors="coerce")
    frame["gain_24h"] = pd.to_numeric(frame["gain_24h"], errors="coerce")
    frame = frame.dropna(subset=["signal_time", "symbol", "rank", "gain_24h"])
    frame = frame.sort_values(["signal_time", "source_priority", "rank", "symbol"])
    return frame.drop_duplicates(["signal_time", "symbol"], keep="first").reset_index(drop=True)


def combine_signal_sources(reconstructed: pd.DataFrame, live: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "reconstructed":
        out = reconstructed.copy()
        out["signal_source"] = "reconstructed_1h_close"
        return out
    if source == "live_snapshots":
        return live.copy()
    if live.empty:
        out = reconstructed.copy()
        out["signal_source"] = "reconstructed_1h_close"
        return out
    live_times = set(live["signal_time"].astype(int).unique())
    historical = reconstructed[~reconstructed["signal_time"].astype(int).isin(live_times)].copy()
    historical["signal_source"] = "reconstructed_1h_close"
    return pd.concat([historical, live], ignore_index=True, sort=False).sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def add_entry_factors_preserving(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    preserved_volume = pd.to_numeric(signals.get("volume_24h_ratio_7d", pd.Series(np.nan, index=signals.index)), errors="coerce")
    base = signals.drop(columns=[col for col in FACTOR_COLUMNS if col in signals.columns]).copy()
    enriched = add_entry_factors(base, kline_map)
    live_volume = preserved_volume.reset_index(drop=True)
    enriched_volume = pd.to_numeric(enriched["volume_24h_ratio_7d"], errors="coerce")
    enriched["volume_24h_ratio_7d"] = live_volume.where(live_volume.notna(), enriched_volume)
    enriched["volume_24h_ratio_7d_bucket"] = enriched["volume_24h_ratio_7d"].map(bucket_volume)
    return enriched


def apply_entry_rules_preserving(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    signals = signals[
        signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & signals["rank"].isin([2, 3])
        & signals["gain_24h"].ge(0.10)
        & signals["gain_24h"].lt(0.80)
    ].copy()
    signals = add_entry_factors_preserving(signals, kline_map)
    signals["leverage"] = signals.apply(leverage_for_signal, axis=1)
    if "live_leverage" in signals.columns:
        live_leverage = pd.to_numeric(signals["live_leverage"], errors="coerce")
        signals["leverage"] = live_leverage.where(live_leverage.notna(), signals["leverage"])
    signals = signals[signals["leverage"].notna()].copy()
    signals["leverage"] = signals["leverage"].astype(int)
    signals["gain_24h_bucket"] = signals["gain_24h"].astype(float).map(gain_bucket)
    return signals.sort_values(["signal_time", "rank", "symbol"]).reset_index(drop=True)


def refresh_missing_current_symbols(symbols: list[str], target_ms: int, sleep_seconds: float) -> list[str]:
    client = SimpleBinanceFuturesClient()
    current_symbols = get_futures_symbols(client)
    cached = set(cached_symbols())
    missing = sorted(set(current_symbols) - cached)
    if missing:
        print(f"Refreshing missing current symbols: {len(missing)}", flush=True)
        backfill_symbols(missing, SIGNAL_START_MS - 10 * DAY_MS, target_ms, sleep_seconds)
    return sorted(set(cached_symbols()) | set(current_symbols) | set(symbols))


def add_rank1_candidates(raw: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rank1 = raw[
        raw["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & raw["rank"].eq(1)
    ].copy()
    rank1 = add_entry_factors_preserving(rank1, kline_map)
    rank1["rank1_cell"] = rank1.apply(rank1_cell, axis=1)
    rank1 = rank1[rank1["rank1_cell"].notna()].copy()
    rank1["strategy_component"] = "Rank1"
    rank1["bucket"] = "R1"
    return rank1


def build_fr3_yr1_actions(raw: pd.DataFrame, rank23: pd.DataFrame, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signal_times = sorted(rank23["signal_time"].astype(int).unique())
    regime_raw = raw.drop(columns=[col for col in FACTOR_COLUMNS if col in raw.columns]).copy()
    sets = opportunity_sets(regime_raw, kline_map)
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
    last_entry_by_symbol: dict[str, int] = {}
    last_pnl_by_symbol: dict[str, float] = {}
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
            "signal_source": signal.get("signal_source", "reconstructed_1h_close"),
            "live_price": signal.get("live_price", np.nan),
            "live_passed": signal.get("live_passed", np.nan),
            "live_filter_reason": signal.get("live_filter_reason", ""),
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
        reentry_reason = same_symbol_reentry_block_reason(
            symbol,
            signal_time,
            last_entry_by_symbol,
            last_pnl_by_symbol,
        )
        if reentry_reason is not None:
            row = skipped_open_position_trade(signal, signal_time)
            row["leverage"] = lev
            days_since_prev_trade = (signal_time - last_entry_by_symbol[symbol]) / DAY_MS
            skip_reason = (
                "prev_win_same_symbol_reentry_0_30d"
                if "Previous winning" in reentry_reason
                else (
                    f"prev_loss_same_symbol_reentry_"
                    f"{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS}_"
                    f"{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS}d"
                )
            )
            rows.append(
                row
                | common
                | {
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "filter_reason": reentry_reason,
                    "days_since_prev_trade": days_since_prev_trade,
                    "prev_pnl_u": last_pnl_by_symbol.get(symbol, np.nan),
                    "blocking_entry_time_ms": last_entry_by_symbol[symbol],
                    "blocking_entry_time_utc": ms_to_utc(last_entry_by_symbol[symbol]).strftime("%Y-%m-%d %H:%M:%S"),
                    "blocking_component": "same_symbol_last_trade",
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
            last_entry_by_symbol[symbol] = signal_time
            last_pnl_by_symbol[symbol] = float(trade.get("pnl_u", np.nan))
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
        "ex_rave_net_pnl_u",
        "rave_pnl_u",
        "rave_trades",
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
        "- RAVEUSDT is included in gross results; Ex-RAVE metrics are printed separately for optimization stability.",
        f"- Signal source: {next(iter(trades_by_strategy.values())).get('signal_source', pd.Series(['unknown'])).dropna().astype(str).unique().tolist() if trades_by_strategy else []}.",
        "- Portfolio replay: same-symbol lock active; Rank2/3 keep current FR3/YR1 and 6D hold.",
        "- Adaptive Rank1 variant uses current FR3/YR1 Rank3 leverage at each timestamp.",
        "- Fees: 0.1% per side; slippage 0.",
        "",
        summary[view_cols].round(4).to_string(index=False),
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")
    print("output", OUT_DIR)
    print(summary[view_cols].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signal-source",
        choices=["reconstructed", "live_snapshots", "hybrid"],
        default="hybrid",
        help="reconstructed=1H close rebuilt rankings; live_snapshots=event log only; hybrid=live event windows override rebuilt history.",
    )
    parser.add_argument("--snapshot-log", action="append", help="Path to events.jsonl containing *_signal_snapshot or *_opened events. May be repeated.")
    parser.add_argument("--no-opened-fallback", action="store_true", help="Do not use *_opened events when signal snapshots are absent.")
    parser.add_argument("--refresh-universe", action="store_true", help="Fetch current Binance USDT perpetual symbols and backfill missing local 1H caches.")
    parser.add_argument("--sleep", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_end = int(latest_signal_end_dt().timestamp() * 1000)
    symbols = cached_symbols()
    if args.refresh_universe:
        symbols = refresh_missing_current_symbols(symbols, target_end, args.sleep)
    symbols = [s for s in symbols if s not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(target_end, common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)
    reconstructed = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    live = load_live_event_signals(
        event_log_paths(args.snapshot_log),
        SIGNAL_START_MS,
        signal_end,
        include_opened_fallback=not args.no_opened_fallback,
    )
    raw = combine_signal_sources(reconstructed, live, args.signal_source)
    raw.to_csv(OUT_DIR / "raw_signals_selected_source.csv", index=False, encoding="utf-8-sig")

    rank23 = apply_entry_rules_preserving(raw, kline_map).copy()
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
