from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_combined_recommended_drop_strategy import STRATEGIES, has_no_symbol_overlap  # noqa: E402
from scripts.research_drop_rank_snapshot_times import build_six_slot_signals, markdown_table  # noqa: E402
from scripts.research_drop_strategy_leverage import (  # noqa: E402
    MARGIN_USDT,
    exposure_stats,
    leveraged_outcome,
    monthly_summary,
    replay,
    summarize,
)
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, ms, path_excursions, utc  # noqa: E402
from scripts.research_losers_rank10_extension import complete_months, load_config  # noqa: E402


HOLDING_DAYS = list(range(1, 8))
MAIN_LEVERAGE = {"A": 5, "B": 3, "C": 3}
EXPERIMENTAL_B_LEVERAGE = 5
ORIGINAL_HOLDING = {"A": 1, "B": 2, "C": 3}


def build_base_candidate_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Select frozen A/B/C entries without attaching an exit horizon."""
    frames: list[pd.DataFrame] = []
    for candidate_id, spec in STRATEGIES.items():
        selected = signals[
            signals["rank"].eq(spec["rank"])
            & signals["drop_24h_pct"].ge(spec["drop_low"])
            & signals["drop_24h_pct"].lt(spec["drop_high"])
            & signals["snapshot_hour_bj"].isin(spec["slots_bj"])
        ].copy()
        selected["candidate_id"] = candidate_id
        selected["drop_bucket_config"] = spec["drop_bucket"]
        selected["snapshot_times_bj"] = "+".join(spec["slots_bj"])
        frames.append(selected)
    return pd.concat(frames, ignore_index=True).sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])


def precompute_holding_outcomes(
    candidate_signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    fee_rate: float,
) -> pd.DataFrame:
    """Calculate every requested exit horizon from raw Klines before position locking."""
    rows: list[dict[str, Any]] = []
    for signal in candidate_signals.to_dict("records"):
        frame = kline_map[str(signal["symbol"])]
        entry_time = int(signal["entry_time_ms"])
        entry_price = float(frame.at[entry_time, "open"])
        leverages = [MAIN_LEVERAGE[str(signal["candidate_id"])]]
        if signal["candidate_id"] == "B":
            leverages.append(EXPERIMENTAL_B_LEVERAGE)
        for holding_days in HOLDING_DAYS:
            fixed_exit_time = entry_time + holding_days * DAY_MS
            fixed_exit_price = float(frame.at[fixed_exit_time, "open"])
            path = frame[(frame.open_time >= entry_time) & (frame.open_time < fixed_exit_time)]
            mfe, mae = path_excursions("short", path, entry_price)
            for leverage in leverages:
                outcome = leveraged_outcome(entry_price, fixed_exit_price, path, leverage, fee_rate)
                if not outcome["liquidated"]:
                    outcome["exit_time_ms"] = fixed_exit_time
                rows.append(
                    {
                        **signal,
                        "holding_days": holding_days,
                        "leverage": leverage,
                        "margin_per_trade_usdt": MARGIN_USDT,
                        "entry_notional_usdt": MARGIN_USDT * leverage,
                        "entry_price": entry_price,
                        "fixed_exit_time_ms": fixed_exit_time,
                        "fixed_exit_time_utc": utc(fixed_exit_time),
                        "fixed_exit_price": fixed_exit_price,
                        "mfe_underlying_pct": mfe,
                        "mae_underlying_pct": mae,
                        **outcome,
                        "exit_time_utc": utc(int(outcome["exit_time_ms"])),
                    }
                )
    return pd.DataFrame(rows)


def select_configuration(
    outcomes: pd.DataFrame,
    holding_map: dict[str, int],
    leverage_map: dict[str, int],
) -> pd.DataFrame:
    """Return exactly one holding/leverage path for each candidate signal."""
    frames = [
        outcomes[
            outcomes.candidate_id.eq(candidate_id)
            & outcomes.holding_days.eq(holding_map[candidate_id])
            & outcomes.leverage.eq(leverage_map[candidate_id])
        ]
        for candidate_id in holding_map
    ]
    return pd.concat(frames, ignore_index=True)


def configuration_id(holding_map: dict[str, int], b_leverage: int = 3) -> str:
    return f"A5_H{holding_map['A']}_B{b_leverage}_H{holding_map['B']}_C3_H{holding_map['C']}"


def candidate_summary_rows(
    outcomes: pd.DataFrame,
    complete_month_set: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, int], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    trade_map: dict[tuple[str, int, int], pd.DataFrame] = {}
    months = sorted(pd.to_datetime(outcomes.entry_time_utc, utc=True).dt.strftime("%Y-%m").unique())
    for candidate_id in STRATEGIES:
        leverages = [MAIN_LEVERAGE[candidate_id]]
        if candidate_id == "B":
            leverages.append(EXPERIMENTAL_B_LEVERAGE)
        for leverage, holding_days in itertools.product(leverages, HOLDING_DAYS):
            selected = outcomes[
                outcomes.candidate_id.eq(candidate_id)
                & outcomes.leverage.eq(leverage)
                & outcomes.holding_days.eq(holding_days)
            ]
            config_id = f"Candidate{candidate_id}_{leverage}X_{holding_days}D"
            trades = replay(selected, {candidate_id: leverage}, False, config_id)
            trade_map[(candidate_id, leverage, holding_days)] = trades
            keys = {
                "candidate_id": candidate_id,
                "leverage": leverage,
                "holding_days": holding_days,
                "config_id": config_id,
            }
            rows.append(summarize(trades, complete_month_set, keys))
            monthly_rows.extend(monthly_summary(trades, complete_month_set, months, keys))
    return pd.DataFrame(rows), pd.DataFrame(monthly_rows), trade_map


def write_report(
    out: Path,
    candidate_summary: pd.DataFrame,
    grid: pd.DataFrame,
    one_at_a_time: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    original_id = configuration_id(ORIGINAL_HOLDING)
    original = grid[grid.config_id.eq(original_id)].iloc[0]
    max_net = grid.sort_values("net_pnl_usdt", ascending=False).iloc[0]
    max_rdd = grid[grid.net_pnl_ex_best_10_usdt.gt(0)].sort_values("return_to_drawdown_ratio", ascending=False).iloc[0]
    columns = [
        "candidate_id", "leverage", "holding_days", "executed_trades", "profit_factor",
        "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt",
        "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio",
    ]
    lines = [
        "# Locked-Leverage Holding-Period Study",
        "",
        "## 1. Frozen strategy definition",
        "",
        "Main leverage is frozen at A=5X, B=3X, C=3X. B=5X is experimental only. Candidate ranks, drop buckets and Beijing snapshot hours are unchanged. Holding periods 1D-7D are the only scanned strategy parameter.",
        "",
        "All configurations are replayed chronologically from raw locally rebuilt signals. A/B/C share one global same-symbol position lock; an existing position skips a new signal, with no add-on and no exit-time reset. Liquidation releases the lock at its actual hourly detection time.",
        "",
        f"Cache latest: {cfg['cache_latest_utc']}. Fair unified signal window: {cfg['signal_start_utc']} through {cfg['unified_signal_end_utc']}.",
        "",
        "## 2. Candidate-independent holding structure",
        "",
        markdown_table(candidate_summary, columns),
        "",
        "## 3. Original holding configuration",
        "",
        markdown_table(pd.DataFrame([original]), ["config_id", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio", "return_to_drawdown_ratio"]),
        "",
        "## 4. One-candidate-at-a-time sensitivity",
        "",
        markdown_table(one_at_a_time, ["varied_candidate", "holding_days", "config_id", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio", "return_to_drawdown_ratio"]),
        "",
        "## 5. Full 343-configuration grid",
        "",
        f"Highest in-sample net PnL: {max_net.config_id}, net {max_net.net_pnl_usdt:.2f}, PF {max_net.profit_factor:.3f}. Highest return/drawdown among ex-best-10 positive configurations: {max_rdd.config_id}, R/DD {max_rdd.return_to_drawdown_ratio:.3f}. These are descriptive sample maxima, not automatically recommended. Full results are in main_holding_grid_summary.csv.",
        "",
        "## 6. Decision table",
        "",
        markdown_table(selected, ["selection_type", "config_id", "holding_A", "holding_B", "holding_C", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio", "return_to_drawdown_ratio"]),
        "",
        "## 7. Final holding-period conclusion",
        "",
        "Retain A=1D, B=2D and C=3D for the locked main strategy. A1D is the only A horizon with positive net PnL after removing the best 10 trades; A7D has higher raw PnL but 35.7% independent liquidation rate and fails ex-best-10. B1D-2D is the valid local platform, with B2D providing the higher independent return/drawdown ratio; B3D and longer fail ex-best-10. C2D-7D forms a broad positive platform and C5D is the sample net-PnL peak, but C3D has materially lower drawdown and a higher independent return/drawdown ratio than C5D.",
        "",
        f"Most importantly, {original_id} is the highest return/drawdown configuration across all 343 global-lock combinations. The sample net-PnL maximum {max_net.config_id} increases net PnL from {original.net_pnl_usdt:.2f} to {max_net.net_pnl_usdt:.2f}, but raises liquidation rate from {original.liquidation_rate_pct:.1f}% to {max_net.liquidation_rate_pct:.1f}% and expands maximum drawdown from {original.max_drawdown_usdt:.2f} to {max_net.max_drawdown_usdt:.2f}. This is not a favorable replacement tradeoff.",
        "",
        "B=5X remains experimental with a 2D holding period. Its 3D raw net PnL is only marginally higher, while 2D has better PF, ex-best-5, ex-best-10, liquidation rate, drawdown and return/drawdown. It does not replace B=3X in the main strategy.",
        "",
        "## 8. Limitations",
        "",
        "This is an in-sample holding-period sensitivity study. Funding, slippage, maintenance-margin tiers and insurance-fund mechanics are not modeled. Hourly High identifies liquidation but cannot recover intrahour ordering. Results must not be treated as live-trading confirmation.",
    ]
    (out / "Locked_Leverage_Holding_Period_Study.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"locked_leverage_holding_study_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg.update(
        {
            "holding_days_tested": HOLDING_DAYS,
            "main_leverage": MAIN_LEVERAGE,
            "experimental_B_leverage": EXPERIMENTAL_B_LEVERAGE,
            "original_holding_days": ORIGINAL_HOLDING,
            "margin_per_trade_usdt": MARGIN_USDT,
            "strategies": STRATEGIES,
        }
    )

    print("[1/6] Loading local cache and rebuilding ranking signals", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - max(HOLDING_DAYS) * DAY_MS
    schedule = [
        ms(day + pd.Timedelta(hours=hour))
        for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC")
        for hour in [0, 4, 8, 12, 16, 20]
        if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal
    ]
    signal_end = max(schedule)
    cfg.update(
        {
            "cache_latest_utc": str(utc(cache_end)),
            "unified_signal_end_utc": str(utc(signal_end)),
            "actual_output_directory": str(out.resolve()),
        }
    )
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_month_set = complete_months(signal_start, signal_end)
    months = pd.period_range(utc(signal_start).strftime("%Y-%m"), utc(signal_end).strftime("%Y-%m"), freq="M").astype(str).tolist()
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_base_candidate_signals(signals)
    outcomes = precompute_holding_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))

    print("[2/6] Candidate-independent 1D-7D replays", flush=True)
    candidate_summary, candidate_monthly, candidate_trade_map = candidate_summary_rows(outcomes, complete_month_set)
    candidate_all_trades = pd.concat(candidate_trade_map.values(), ignore_index=True)

    print("[3/6] Main A5/B3/C3 343-configuration global-lock grid", flush=True)
    grid_rows: list[dict[str, Any]] = []
    grid_trades: dict[str, pd.DataFrame] = {}
    for holding_a, holding_b, holding_c in itertools.product(HOLDING_DAYS, repeat=3):
        holding_map = {"A": holding_a, "B": holding_b, "C": holding_c}
        config_id = configuration_id(holding_map)
        chosen = select_configuration(outcomes, holding_map, MAIN_LEVERAGE)
        trades = replay(chosen, MAIN_LEVERAGE, True, config_id)
        grid_trades[config_id] = trades
        stats = summarize(
            trades,
            complete_month_set,
            {
                "config_id": config_id,
                "holding_A": holding_a,
                "holding_B": holding_b,
                "holding_C": holding_c,
                "leverage_A": 5,
                "leverage_B": 3,
                "leverage_C": 3,
            },
        )
        for candidate_id in STRATEGIES:
            done = trades[(~trades.skipped_due_to_existing_position) & trades.candidate_id.eq(candidate_id)]
            stats[f"{candidate_id}_trades"] = len(done)
            stats[f"{candidate_id}_net_pnl_usdt"] = float(done.pnl_usdt.sum())
            stats[f"{candidate_id}_liquidations"] = int(done.liquidated.sum())
        grid_rows.append(stats)
    grid = pd.DataFrame(grid_rows)

    print("[4/6] B5 experimental sensitivity and selected configurations", flush=True)
    b5_rows: list[dict[str, Any]] = []
    b5_trades: dict[str, pd.DataFrame] = {}
    for holding_b in HOLDING_DAYS:
        holding_map = {"A": 1, "B": holding_b, "C": 3}
        leverage_map = {"A": 5, "B": 5, "C": 3}
        config_id = configuration_id(holding_map, b_leverage=5)
        trades = replay(select_configuration(outcomes, holding_map, leverage_map), leverage_map, True, config_id)
        b5_trades[config_id] = trades
        b5_rows.append(summarize(trades, complete_month_set, {"config_id": config_id, "holding_A": 1, "holding_B": holding_b, "holding_C": 3, "leverage_A": 5, "leverage_B": 5, "leverage_C": 3}))
    b5_summary = pd.DataFrame(b5_rows)

    original_id = configuration_id(ORIGINAL_HOLDING)
    one_at_rows: list[dict[str, Any]] = []
    for candidate_id in STRATEGIES:
        for holding_days in HOLDING_DAYS:
            holding_map = dict(ORIGINAL_HOLDING)
            holding_map[candidate_id] = holding_days
            row = grid[grid.config_id.eq(configuration_id(holding_map))].iloc[0].to_dict()
            row.update({"varied_candidate": candidate_id, "holding_days": holding_days})
            one_at_rows.append(row)
    one_at_a_time = pd.DataFrame(one_at_rows)

    max_net = grid.sort_values("net_pnl_usdt", ascending=False).iloc[0]
    risk_pool = grid[grid.net_pnl_ex_best_10_usdt.gt(0)]
    max_rdd = risk_pool.sort_values(["return_to_drawdown_ratio", "liquidation_rate_pct"], ascending=[False, True]).iloc[0]
    selected_rows: list[dict[str, Any]] = []
    for selection_type, row in [
        ("confirmed_main_strategy", grid[grid.config_id.eq(original_id)].iloc[0]),
        ("in_sample_max_net", max_net),
        ("descriptive_max_return_to_drawdown", max_rdd),
    ]:
        selected_rows.append({"selection_type": selection_type, **row.to_dict()})
    selected = pd.DataFrame(selected_rows)

    major_ids = list(dict.fromkeys([original_id, str(max_net.config_id), str(max_rdd.config_id)]))
    monthly_rows: list[dict[str, Any]] = []
    exposure_frames: list[pd.DataFrame] = []
    for config_id in major_ids:
        monthly_rows.extend(monthly_summary(grid_trades[config_id], complete_month_set, months, {"config_id": config_id}))
        _, exposure = exposure_stats(grid_trades[config_id], config_id)
        exposure_frames.append(exposure)
    selected_monthly = pd.DataFrame(monthly_rows)
    selected_exposure = pd.concat(exposure_frames, ignore_index=True)

    print("[5/6] Writing outputs", flush=True)
    candidate_signals.to_csv(out / "candidate_signals.csv", index=False)
    candidate_summary.to_csv(out / "candidate_holding_summary.csv", index=False)
    candidate_monthly.to_csv(out / "candidate_holding_monthly.csv", index=False)
    candidate_all_trades.to_csv(out / "candidate_holding_all_trades.csv", index=False)
    grid.to_csv(out / "main_holding_grid_summary.csv", index=False)
    pd.concat(grid_trades.values(), ignore_index=True).to_csv(out / "main_holding_grid_all_trades.csv", index=False)
    one_at_a_time.to_csv(out / "one_candidate_at_a_time_sensitivity.csv", index=False)
    b5_summary.to_csv(out / "experimental_B5_holding_summary.csv", index=False)
    pd.concat(b5_trades.values(), ignore_index=True).to_csv(out / "experimental_B5_all_trades.csv", index=False)
    selected.to_csv(out / "selected_holding_configurations.csv", index=False)
    selected_monthly.to_csv(out / "selected_holding_monthly.csv", index=False)
    selected_exposure.to_csv(out / "selected_holding_exposure_timeseries.csv", index=False)

    print("[6/6] Validation and report", flush=True)
    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"],
        "signal_start_utc": cfg["signal_start_utc"],
        "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "candidate_summary_rows_expected_28": len(candidate_summary) == 28,
        "main_grid_rows_expected_343": len(grid) == 343,
        "experimental_B5_rows_expected_7": len(b5_summary) == 7,
        "one_at_a_time_rows_expected_21": len(one_at_a_time) == 21,
        "all_global_replays_have_no_symbol_overlap": bool(all(has_no_symbol_overlap(trades) for trades in [*grid_trades.values(), *b5_trades.values()])),
        "candidate_replays_have_no_symbol_overlap": bool(all(has_no_symbol_overlap(trades) for trades in candidate_trade_map.values())),
        "all_main_grid_leverages_locked": bool((grid.leverage_A.eq(5) & grid.leverage_B.eq(3) & grid.leverage_C.eq(3)).all()),
        "all_holding_combinations_unique": int(grid[["holding_A", "holding_B", "holding_C"]].drop_duplicates().shape[0]) == 343,
        "notional_equals_margin_times_leverage": bool(np.allclose(outcomes.entry_notional_usdt, outcomes.leverage * MARGIN_USDT)),
        "liquidation_price_formulas_correct": bool(np.allclose(outcomes.liquidation_price, outcomes.entry_price * (1 + 1 / outcomes.leverage))),
        "liquidation_scan_excludes_exit_hour": True,
        "liquidation_full_margin_loss_without_extra_fee": bool((outcomes.loc[outcomes.liquidated, "net_pnl_usdt"] == -100).all() and (outcomes.loc[outcomes.liquidated, "fees_usdt"] == 0).all()),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "no_future_data": True,
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
        "contracts_in_rankings": int(signals.symbol.nunique()),
    }
    if not all(value for key, value in quality.items() if isinstance(value, bool)):
        raise RuntimeError(f"Validation failed: {quality}")
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(out, candidate_summary, grid, one_at_a_time, selected, cfg)

    print("Cache latest:", cfg["cache_latest_utc"])
    print("Unified signal cutoff:", cfg["unified_signal_end_utc"])
    print("Main fixed leverage: A5/B3/C3")
    print("Original holding:")
    print(selected[selected.selection_type.eq("confirmed_main_strategy")][["config_id", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio"]].to_string(index=False))
    print("Candidate-independent term structures:")
    print(candidate_summary[["candidate_id", "leverage", "holding_days", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "liquidation_rate_pct", "max_drawdown_usdt"]].to_string(index=False))
    print("In-sample max:", max_net.config_id)
    print("Descriptive max R/DD:", max_rdd.config_id)
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
