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
    HOUR_MS,
    OUT,
    get_open_at_or_latest,
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
OUT_DIR = OUT / "weak_exit_48h_4d_research"
BASELINE = "Baseline"
BIG_WINNER_THRESHOLDS = [100.0, 300.0, 500.0, 1000.0]


@dataclass(frozen=True)
class Variant:
    name: str
    checks: tuple[tuple[str, int, float], ...]


VARIANTS = [
    Variant("48H_LRET_LT_0", (("48h", 48, 0.00),)),
    Variant("48H_LRET_LT_10", (("48h", 48, 0.10),)),
    Variant("4D_LRET_LT_-10", (("4d", 96, -0.10),)),
    Variant("4D_LRET_LT_-25", (("4d", 96, -0.25),)),
    Variant("4D_LRET_LT_-50", (("4d", 96, -0.50),)),
    Variant("COMBO_48H_LT_10_OR_4D_LT_-10", (("48h", 48, 0.10), ("4d", 96, -0.10))),
]


def calc_leveraged_pnl(entry_price: float, exit_price: float, leverage: int) -> tuple[float, float]:
    return engine.calc_leveraged_pnl(entry_price, exit_price, leverage)


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


def condition_return_at(frame: pd.DataFrame, entry_time: int, entry_price: float, hours: int) -> tuple[int | None, float, float]:
    target = entry_time + hours * HOUR_MS
    exact = frame[frame["open_time"] == target]
    if not exact.empty:
        row = exact.iloc[-1]
        return int(row["open_time"]), float(row["open"]), float(row["open"] / entry_price - 1.0)
    prior = frame[frame["open_time"] < target]
    if prior.empty:
        return None, math.nan, math.nan
    row = prior.iloc[-1]
    return int(row["open_time"]), float(row["close"]), float(row["close"] / entry_price - 1.0)


def liquidation_time_before_or_at(frame: pd.DataFrame, entry_time: int, exit_time: int, entry_price: float, leverage: int) -> int | None:
    threshold_pct = engine.LIQUIDATION_THRESHOLDS_PCT[leverage] / 100.0
    liquidation_price = entry_price * (1.0 + threshold_pct)
    path = frame[(frame["open_time"] >= entry_time) & (frame["open_time"] <= exit_time)].sort_values("open_time")
    hit = path[pd.to_numeric(path["low"], errors="coerce") <= liquidation_price]
    if hit.empty:
        return None
    return int(hit.iloc[0]["open_time"])


def recompute_exit_trade(
    base: dict[str, Any],
    frame: pd.DataFrame,
    exit_time: int,
    exit_price: float,
    reason: str,
) -> dict[str, Any]:
    trade = base.copy()
    entry_time = int(base["entry_time_ms"])
    entry_price = float(base["entry_price"])
    leverage = int(base["leverage"])
    path = path_slice(frame, entry_time, exit_time)
    mfe, mae, max_price, min_price = mfe_mae(path, entry_price)
    liq_time = liquidation_time_before_or_at(frame, entry_time, exit_time, entry_price, leverage)
    baseline_pnl, baseline_ret = calc_leveraged_pnl(entry_price, exit_price, leverage)
    partial = engine._mfe_aging_partial_tp_for_path(  # noqa: SLF001 - research uses production simulator helper.
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
            "mid_exit_triggered": reason.startswith("weak_48h") or reason.startswith("weak_4d"),
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
    variant: Variant | None,
) -> dict[str, Any]:
    original_hold = engine.HOLD_DAYS
    try:
        engine.HOLD_DAYS = hold_days
        base = engine.simulate_trade_with_leverage(signal, kline_map, cutoff_ms, leverage)
    finally:
        engine.HOLD_DAYS = original_hold
    if variant is None or base.get("status") not in {"completed", "open_mark_to_market"}:
        return base | {"mid_exit_triggered": False}
    symbol = str(signal["symbol"])
    frame = kline_map.get(symbol, pd.DataFrame())
    if frame.empty or "entry_price" not in base:
        return base | {"mid_exit_triggered": False}
    entry_time = int(base["entry_time_ms"])
    entry_price = float(base["entry_price"])
    base_exit_time = int(float(base["exit_time_ms"]))
    for label, hours, threshold in variant.checks:
        target = entry_time + hours * HOUR_MS
        if base_exit_time <= target or target > cutoff_ms:
            continue
        price_time, price, ret = condition_return_at(frame, entry_time, entry_price, hours)
        if price_time is None or not np.isfinite(price):
            continue
        levered = ret * leverage
        if levered < threshold:
            reason = f"weak_{label}_levered_return_lt_{threshold:.0%}"
            trade = recompute_exit_trade(base, frame, price_time, price, reason)
            trade[f"levered_return_{label}_at_exit_check"] = levered
            trade[f"underlying_return_{label}_at_exit_check"] = ret
            return trade
    return base | {"mid_exit_triggered": False}


def precompute_outcomes(
    signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    cutoff_ms: int,
) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    outcomes: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    variants: list[Variant | None] = [None, *VARIANTS]
    for row in signals.itertuples(index=False):
        signal = pd.Series(row._asdict())
        sid = int(signal["signal_id"])
        leverage = int(signal["adaptive_leverage"])
        hold_days = int(signal["target_hold_days"])
        for variant in variants:
            name = BASELINE if variant is None else variant.name
            outcomes[(name, sid, leverage, hold_days)] = simulate_signal(
                signal,
                kline_map,
                cutoff_ms,
                leverage,
                hold_days,
                variant,
            )
    return outcomes


def replay_portfolio(strategy: str, signals: pd.DataFrame, outcomes: dict[tuple[str, int, int, int], dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_by_symbol: dict[str, dict[str, Any]] = {}
    last_entry_by_symbol: dict[str, int] = {}
    last_pnl_by_symbol: dict[str, float] = {}
    for _, signal in signals.sort_values(["signal_time", "rank", "symbol"]).iterrows():
        signal_time = int(signal["signal_time"])
        symbol = str(signal["symbol"])
        rank = int(signal["rank"])
        leverage = int(signal["adaptive_leverage"])
        hold_days = int(signal["target_hold_days"])
        common = {
            "strategy": strategy,
            "strategy_component": signal.get("strategy_component", f"Rank{rank}"),
            "rank1_cell": signal.get("rank1_cell", ""),
            "bucket": signal.get("bucket", ""),
            "target_hold_days": hold_days,
            "original_leverage": signal.get("original_leverage", leverage),
            "adaptive_leverage": leverage,
            "regime_state": signal.get("regime_state", ""),
            "base_state": signal.get("base_state", ""),
            "recovery_signal": signal.get("recovery_signal", False),
            "signal_source": signal.get("signal_source", ""),
        }
        open_info = open_by_symbol.get(symbol)
        if open_info is not None and signal_time < int(open_info["open_until"]):
            row = skipped_open_position_trade(signal, int(open_info["open_until"]))
            row["leverage"] = leverage
            rows.append(row | common | {"status": "skipped", "skip_reason": "symbol_already_open"})
            continue
        reentry_reason = same_symbol_reentry_block_reason(symbol, signal_time, last_entry_by_symbol, last_pnl_by_symbol)
        if reentry_reason is not None:
            row = skipped_open_position_trade(signal, signal_time)
            row["leverage"] = leverage
            skip_reason = (
                "prev_win_same_symbol_reentry_0_30d"
                if "Previous winning" in reentry_reason
                else f"prev_loss_same_symbol_reentry_{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS}_{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS}d"
            )
            rows.append(row | common | {"status": "skipped", "skip_reason": skip_reason, "filter_reason": reentry_reason})
            continue
        trade = outcomes[(strategy, int(signal["signal_id"]), leverage, hold_days)].copy()
        trade["signal_id"] = int(signal["signal_id"])
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            lock_extra_ms = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_by_symbol[symbol] = {
                "open_until": int(float(trade["exit_time_ms"])) + lock_extra_ms,
                "entry_time_ms": signal_time,
                "pnl_u": trade.get("pnl_u", np.nan),
            }
            last_entry_by_symbol[symbol] = signal_time
            last_pnl_by_symbol[symbol] = float(trade.get("pnl_u", np.nan))
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> dict[str, Any]:
    done = evaluated(frame)
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liq = done["liquidated"].fillna(False).astype(bool) if len(done) else pd.Series(dtype=bool)
    net = float(pnl.sum()) if len(pnl) else 0.0
    top = pnl.sort_values(ascending=False)
    return {
        "trades": int(len(done)),
        "net_pnl_u": net,
        "pf": profit_factor(pnl),
        "ev_per_trade_u": net / len(done) if len(done) else np.nan,
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(done) else np.nan,
        "median_return_pct": float(ret.median()) if len(done) else np.nan,
        "max_drawdown_u": max_drawdown(pnl),
        "liquidations": int(liq.sum()) if len(done) else 0,
        "liq_rate_pct": float(liq.mean() * 100) if len(done) else np.nan,
        "ex_top1_pnl_u": float(net - top.head(1).sum()) if len(top) else np.nan,
        "ex_top3_pnl_u": float(net - top.head(3).sum()) if len(top) >= 3 else np.nan,
    }


def counterfactual_rows(variant_name: str, frame: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    done = evaluated(frame)
    triggered = done[done.get("mid_exit_triggered", False).fillna(False).astype(bool)].copy()
    base_map = evaluated(baseline).set_index("signal_id")
    rows = []
    for row in triggered.itertuples(index=False):
        base = base_map.loc[int(row.signal_id)] if int(row.signal_id) in base_map.index else None
        baseline_pnl = float(base["pnl_u"]) if base is not None else np.nan
        rows.append(
            {
                "variant": variant_name,
                "signal_id": int(row.signal_id),
                "symbol": row.symbol,
                "entry_time_bj": row.entry_time_bj,
                "rank": int(row.rank),
                "leverage": int(row.leverage),
                "exit_reason": row.exit_reason,
                "mid_exit_pnl_u": float(row.pnl_u),
                "baseline_exit_time_bj": base["exit_time_bj"] if base is not None else "",
                "baseline_exit_reason": base["exit_reason"] if base is not None else "",
                "baseline_pnl_u": baseline_pnl,
                "incremental_pnl_u": float(row.pnl_u) - baseline_pnl if np.isfinite(baseline_pnl) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = load_replay_signals()
    symbols = sorted(signals["symbol"].astype(str).unique())
    start = int(signals["signal_time"].min()) - 10 * DAY_MS
    end = int(pd.to_numeric(signals["exit_time_ms"], errors="coerce").max()) + DAY_MS
    kline_map = load_kline_map(symbols, start, end)
    outcomes = precompute_outcomes(signals, kline_map, end)

    names = [BASELINE, *[v.name for v in VARIANTS]]
    trades_by_name = {name: replay_portfolio(name, signals, outcomes) for name in names}
    overall = pd.DataFrame([{"variant": name, **summary(frame)} for name, frame in trades_by_name.items()])
    monthly = pd.DataFrame(
        [
            {"variant": name, "month": month, **summary(group)}
            for name, frame in trades_by_name.items()
            for month, group in evaluated(frame).groupby("month", sort=True)
        ]
    )
    exits = pd.DataFrame(
        [
            {"variant": name, "exit_reason": reason, **summary(group)}
            for name, frame in trades_by_name.items()
            for reason, group in evaluated(frame).groupby("exit_reason", sort=True)
        ]
    )
    counterfactual = pd.concat(
        [counterfactual_rows(name, trades_by_name[name], trades_by_name[BASELINE]) for name in names if name != BASELINE],
        ignore_index=True,
        sort=False,
    )

    big_rows = []
    base_done = evaluated(trades_by_name[BASELINE]).set_index("signal_id")
    for name in names:
        if name == BASELINE:
            continue
        cf = counterfactual[counterfactual["variant"].eq(name)].copy()
        for threshold in BIG_WINNER_THRESHOLDS:
            scoped = cf[cf["baseline_pnl_u"] > threshold].copy()
            big_rows.append(
                {
                    "variant": name,
                    "baseline_winner_gt_u": threshold,
                    "triggered_count": int(len(scoped)),
                    "baseline_pnl_u": float(scoped["baseline_pnl_u"].sum()) if len(scoped) else 0.0,
                    "candidate_pnl_u": float(scoped["mid_exit_pnl_u"].sum()) if len(scoped) else 0.0,
                    "right_tail_loss_u": float((scoped["mid_exit_pnl_u"] - scoped["baseline_pnl_u"]).sum()) if len(scoped) else 0.0,
                }
            )

    all_trades = pd.concat(trades_by_name.values(), ignore_index=True, sort=False)
    overall.to_csv(OUT_DIR / "overall_comparison.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_comparison.csv", index=False, encoding="utf-8-sig")
    exits.to_csv(OUT_DIR / "exit_reason_summary.csv", index=False, encoding="utf-8-sig")
    counterfactual.to_csv(OUT_DIR / "mid_exit_counterfactual.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(big_rows).to_csv(OUT_DIR / "missed_big_winners.csv", index=False, encoding="utf-8-sig")
    all_trades.to_csv(OUT_DIR / "trade_details_each_variant.csv", index=False, encoding="utf-8-sig")

    print("output", OUT_DIR)
    print("\nOverall")
    print(overall.round(4).to_string(index=False))
    print("\nExit counts")
    print(exits[exits["variant"].ne(BASELINE)].round(4).to_string(index=False))
    print("\nMissed big winners")
    print(pd.DataFrame(big_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
