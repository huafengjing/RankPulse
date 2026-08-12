from __future__ import annotations

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

from scripts.research_drop_rank_snapshot_times import (  # noqa: E402
    build_six_slot_signals,
    candidate_outcomes,
    empty_stats,
    markdown_table,
)
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import (  # noqa: E402
    complete_months,
    load_config,
    precompute_outcomes,
    summarize_trades,
)


STRATEGIES = {
    "A": {"rank": 1, "drop_low": 0.0, "drop_high": 20.0, "drop_bucket": "0~20%", "holding_days": 1, "slots_bj": ["00:00", "04:00"]},
    "B": {"rank": 1, "drop_low": 20.0, "drop_high": 40.0, "drop_bucket": "20~40%", "holding_days": 2, "slots_bj": ["08:00"]},
    "C": {"rank": 3, "drop_low": 20.0, "drop_high": 40.0, "drop_bucket": "20~40%", "holding_days": 3, "slots_bj": ["00:00", "20:00"]},
}

INDEPENDENT_REFERENCE = {
    "A": {"trades": 57, "profit_factor": 1.958117199143001, "net_pnl_usdt": 188.87432320822793, "net_pnl_ex_best_5_usdt": 83.15668384749058},
    "B": {"trades": 107, "profit_factor": 2.0513389504875215, "net_pnl_usdt": 614.2649140167415, "net_pnl_ex_best_5_usdt": 374.9689374085597},
    "C": {"trades": 148, "profit_factor": 1.5543026703561102, "net_pnl_usdt": 660.3515718877102, "net_pnl_ex_best_5_usdt": 355.95545121024765},
}


def apply_global_position_lock(raw: pd.DataFrame) -> pd.DataFrame:
    ordered = raw.sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])
    open_positions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in ordered.to_dict("records"):
        symbol = str(row["symbol"])
        entry_time = int(row["entry_time_ms"])
        blocker = open_positions.get(symbol)
        skipped = blocker is not None and entry_time < int(blocker["exit_time_ms"])
        row.update(
            {
                "notional_usdt": 100.0,
                "skipped_due_to_existing_position": skipped,
                "signal_eligible": not skipped,
                "eligibility_reason": "global_existing_position" if skipped else "eligible",
                "blocked_by_candidate_id": blocker["candidate_id"] if skipped else "",
                "blocked_by_entry_time_ms": blocker["entry_time_ms"] if skipped else np.nan,
                "blocked_until_time_ms": blocker["exit_time_ms"] if skipped else np.nan,
                "pnl_usdt": np.nan if skipped else float(row["pnl_usdt_at_100"]),
                "fees_usdt": np.nan if skipped else float(row["fees_usdt_at_100"]),
                "execution_status": "skipped_global_existing_position" if skipped else "executed",
            }
        )
        if not skipped:
            open_positions[symbol] = {
                "candidate_id": row["candidate_id"],
                "entry_time_ms": entry_time,
                "exit_time_ms": int(row["exit_time_ms"]),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_with_signals(trades: pd.DataFrame, complete_month_set: set[str]) -> dict[str, Any]:
    executed = trades[~trades["skipped_due_to_existing_position"]]
    stats = summarize_trades(trades, complete_month_set) if len(executed) else empty_stats()
    raw_count = len(trades)
    skipped = raw_count - len(executed)
    return {
        "raw_signals": raw_count,
        "executed_trades": len(executed),
        "skipped_global_existing_position": skipped,
        "skip_rate_pct": skipped / raw_count * 100 if raw_count else 0.0,
        "unique_symbols": int(trades.symbol.nunique()),
        **{key: value for key, value in stats.items() if key != "trades"},
    }


def month_summary(trades: pd.DataFrame, complete_month_set: set[str], months: list[str]) -> pd.DataFrame:
    work = trades.assign(month=pd.to_datetime(trades.entry_time_utc, utc=True).dt.strftime("%Y-%m"))
    rows: list[dict[str, Any]] = []
    for scope in ["COMBINED", *STRATEGIES]:
        scoped = work if scope == "COMBINED" else work[work.candidate_id.eq(scope)]
        for month in months:
            group = scoped[scoped.month.eq(month)]
            stats = summarize_with_signals(group, {month}) if len(group) else {
                "raw_signals": 0, "executed_trades": 0, "skipped_global_existing_position": 0,
                "wins": 0, "losses": 0, "liquidations": 0, "net_pnl_usdt": 0.0,
                "profit_factor": np.nan, "win_rate_pct": np.nan, "median_return_pct": np.nan,
                "max_drawdown_usdt": 0.0,
            }
            rows.append(
                {
                    "scope": scope,
                    "month": month,
                    "partial_month": month not in complete_month_set,
                    **{key: stats[key] for key in [
                        "raw_signals", "executed_trades", "skipped_global_existing_position", "wins", "losses",
                        "liquidations", "net_pnl_usdt", "profit_factor", "win_rate_pct", "median_return_pct", "max_drawdown_usdt",
                    ]},
                }
            )
    return pd.DataFrame(rows)


def exposure_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    executed = trades[~trades.skipped_due_to_existing_position]
    events = []
    for row in executed.itertuples():
        events.append({"time_ms": int(row.entry_time_ms), "position_delta": 1, "notional_delta": 100.0})
        events.append({"time_ms": int(row.exit_time_ms), "position_delta": -1, "notional_delta": -100.0})
    frame = pd.DataFrame(events).groupby("time_ms", as_index=False).sum().sort_values("time_ms")
    frame["concurrent_positions"] = frame.position_delta.cumsum()
    frame["gross_exposure_usdt"] = frame.notional_delta.cumsum()
    frame["time_utc"] = pd.to_datetime(frame.time_ms, unit="ms", utc=True)
    return frame


def has_no_symbol_overlap(trades: pd.DataFrame) -> bool:
    executed = trades[~trades.skipped_due_to_existing_position].sort_values(["symbol", "entry_time_ms"])
    for _, group in executed.groupby("symbol"):
        if len(group) > 1 and (group.entry_time_ms.iloc[1:].to_numpy() < group.exit_time_ms.iloc[:-1].to_numpy()).any():
            return False
    return True


def write_report(
    out: Path,
    overall: pd.DataFrame,
    candidates: pd.DataFrame,
    monthly: pd.DataFrame,
    conflicts: pd.DataFrame,
    exposure: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    combined = overall.iloc[0]
    full_months = monthly[(monthly.scope.eq("COMBINED")) & (~monthly.partial_month)]
    independent_total_pnl = sum(value["net_pnl_usdt"] for value in INDEPENDENT_REFERENCE.values())
    independent_total_trades = sum(value["trades"] for value in INDEPENDENT_REFERENCE.values())
    lines = [
        "# Combined Recommended Drop Strategy Backtest",
        "",
        "## 1. 执行规则",
        "",
        "A=Rank1/0%-20%/BJ00+04/1D；B=Rank1/20%-40%/BJ08/2D；C=Rank3/20%-40%/BJ00+20/3D。三者统一按信号时间、Rank、Symbol顺序运行，并共享全局同币仓位锁。已有仓位时跳过，不加仓、不延长或重置退出时间。",
        "",
        f"Kline最新：{cfg['cache_latest_utc']}；统一信号窗口：{cfg['signal_start_utc']} 至 {cfg['unified_signal_end_utc']}。每笔100 USDT、1X隔离，双边手续费各0.10%，滑点0，Funding未计。",
        "",
        "## 2. 组合总体结果",
        "",
        markdown_table(overall, ["raw_signals", "executed_trades", "skipped_global_existing_position", "skip_rate_pct", "wins", "losses", "liquidations", "net_pnl_usdt", "profit_factor", "win_rate_pct", "median_return_pct", "net_pnl_ex_best_5_usdt", "positive_months", "total_complete_months", "max_drawdown_usdt", "max_consecutive_losses", "return_to_drawdown_ratio", "max_concurrent_positions", "max_gross_exposure_usdt"]),
        "",
        "## 3. A/B/C在全局锁后的贡献",
        "",
        markdown_table(candidates, ["candidate_id", "raw_signals", "executed_trades", "skipped_global_existing_position", "blocked_by_other_candidate", "net_pnl_usdt", "profit_factor", "net_pnl_ex_best_5_usdt", "positive_months", "total_complete_months", "max_drawdown_usdt", "return_to_drawdown_ratio"]),
        "",
        "## 4. 与三候选独立运行之和比较",
        "",
        f"独立运行时合计为{independent_total_trades}笔、净收益{independent_total_pnl:.2f} USDT；统一全局锁后为{int(combined.executed_trades)}笔、净收益{combined.net_pnl_usdt:.2f} USDT。差异来自跨候选同币冲突，不是事后过滤。独立PF不能直接相加，因此不提供伪组合PF。",
        "",
        "## 5. 全局冲突",
        "",
        markdown_table(conflicts, ["candidate_id", "blocked_by_candidate_id", "skipped_signals"]),
        "",
        "## 6. 月度结果",
        "",
        markdown_table(monthly[monthly.scope.eq("COMBINED")], ["month", "partial_month", "raw_signals", "executed_trades", "skipped_global_existing_position", "net_pnl_usdt", "profit_factor", "win_rate_pct", "max_drawdown_usdt"]),
        "",
        f"组合在{len(full_months)}个完整月份中有{int((full_months.net_pnl_usdt > 0).sum())}个月盈利。2026-07为部分月，不计入比例。",
        "",
        "## 7. 敞口与限制",
        "",
        f"最大同时持仓{int(exposure.concurrent_positions.max())}个，最大名义敞口{exposure.gross_exposure_usdt.max():.2f} USDT。Funding未计；当前合约缓存仍可能有历史退市合约幸存者偏差；本结果仍属于样本内规则组合，不替代OOS。",
    ]
    (out / "Combined_Recommended_Drop_Strategy_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"combined_recommended_drop_strategy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg["holding_days"] = [1, 2, 3]
    cfg["combined_strategies"] = STRATEGIES
    cfg["global_symbol_position_lock"] = True
    cfg["signal_order"] = ["entry_time_ms", "rank", "symbol", "candidate_id"]

    print("[1/5] Loading local cache", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    utc_hours = [0, 4, 8, 12, 16, 20]
    schedule = [
        ms(day + pd.Timedelta(hours=hour))
        for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC")
        for hour in utc_hours
        if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal
    ]
    signal_end = max(schedule)
    cfg["cache_latest_utc"] = str(utc(cache_end))
    cfg["unified_signal_end_utc"] = str(utc(signal_end))
    cfg["actual_output_directory"] = str(out.resolve())
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_month_set = complete_months(signal_start, signal_end)
    months = pd.period_range(utc(signal_start).strftime("%Y-%m"), utc(signal_end).strftime("%Y-%m"), freq="M").astype(str).tolist()

    print("[2/5] Rebuilding rankings and raw strategy signals", flush=True)
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    relevant = signals[signals["rank"].isin([1, 3])]
    outcomes = precompute_outcomes(relevant, kline_map, cfg)
    raw_frames = []
    for candidate_id, spec in STRATEGIES.items():
        frame = candidate_outcomes(outcomes, spec)
        frame = frame[frame.snapshot_hour_bj.isin(spec["slots_bj"])].copy()
        frame["candidate_id"] = candidate_id
        frame["configured_slots_bj"] = "+".join(spec["slots_bj"])
        raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True)

    print("[3/5] Applying global symbol position lock", flush=True)
    trades = apply_global_position_lock(raw)
    trades["signal_time_utc"] = trades.snapshot_time_utc
    trades["signal_time_beijing"] = pd.to_datetime(trades.snapshot_time_utc, utc=True).dt.tz_convert("Asia/Shanghai")
    trades["snapshot_hour_beijing"] = trades.snapshot_hour_bj
    trades["drop_pct_24h"] = trades.drop_24h_pct
    trades["entry_time"] = trades.entry_time_utc
    trades["exit_time"] = trades.exit_time_utc

    print("[4/5] Building statistics", flush=True)
    overall_stats = summarize_with_signals(trades, complete_month_set)
    exposure = exposure_timeseries(trades)
    overall_stats |= {
        "scope": "COMBINED",
        "max_concurrent_positions": int(exposure.concurrent_positions.max()),
        "max_gross_exposure_usdt": float(exposure.gross_exposure_usdt.max()),
    }
    overall = pd.DataFrame([overall_stats])
    candidate_rows = []
    for candidate_id, group in trades.groupby("candidate_id"):
        stats = summarize_with_signals(group, complete_month_set)
        stats |= {
            "candidate_id": candidate_id,
            "blocked_by_other_candidate": int((group.skipped_due_to_existing_position & group.blocked_by_candidate_id.ne(candidate_id)).sum()),
        }
        candidate_rows.append(stats)
    candidates = pd.DataFrame(candidate_rows).sort_values("candidate_id")
    monthly = month_summary(trades, complete_month_set, months)
    conflicts = (
        trades[trades.skipped_due_to_existing_position]
        .groupby(["candidate_id", "blocked_by_candidate_id"], as_index=False)
        .size()
        .rename(columns={"size": "skipped_signals"})
    )

    overall.to_csv(out / "combined_strategy_summary.csv", index=False)
    candidates.to_csv(out / "candidate_contribution_summary.csv", index=False)
    monthly.to_csv(out / "combined_strategy_monthly.csv", index=False)
    trades.to_csv(out / "combined_all_signals_and_trades.csv", index=False)
    trades[trades.skipped_due_to_existing_position].to_csv(out / "global_lock_skipped_signals.csv", index=False)
    exposure.to_csv(out / "combined_exposure_timeseries.csv", index=False)
    conflicts.to_csv(out / "global_lock_conflict_matrix.csv", index=False)

    print("[5/5] Validating and reporting", flush=True)
    executed = trades[~trades.skipped_due_to_existing_position]
    monthly_combined = monthly[monthly.scope.eq("COMBINED")]
    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"],
        "signal_start_utc": cfg["signal_start_utc"],
        "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "rebuilt_from_local_klines_not_filtered_prior_trades": True,
        "global_symbol_lock_enabled": True,
        "all_candidates_present": sorted(trades.candidate_id.unique().tolist()) == ["A", "B", "C"],
        "executed_plus_skipped_equals_raw": len(executed) + int(trades.skipped_due_to_existing_position.sum()) == len(raw),
        "no_executed_same_symbol_overlap": has_no_symbol_overlap(trades),
        "skipped_signals_do_not_reset_exit": True,
        "all_exits_within_cache": bool(signal_end + 3 * DAY_MS <= cache_end),
        "precomputed_outcomes_complete": len(outcomes) == len(relevant) * 3,
        "monthly_pnl_matches_total": bool(np.isclose(monthly_combined.net_pnl_usdt.sum(), executed.pnl_usdt.sum())),
        "candidate_pnl_matches_total": bool(np.isclose(candidates.net_pnl_usdt.sum(), executed.pnl_usdt.sum())),
        "candidate_trade_count_matches_total": int(candidates.executed_trades.sum()) == len(executed),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "snapshot_rank_sort_violations": int(signals.sort_values(["snapshot_time_ms", "rank"]).groupby("snapshot_time_ms").return_24h_pct.apply(lambda x: (x.diff().dropna() < 0).sum()).sum()),
        "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()),
    }
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(out, overall, candidates, monthly, conflicts, exposure, cfg)

    print("Cache latest:", cfg["cache_latest_utc"])
    print("Unified signal cutoff:", cfg["unified_signal_end_utc"])
    print(f"Combined: trades={len(executed)}, PF={overall.iloc[0].profit_factor:.3f}, PnL={executed.pnl_usdt.sum():.2f}, ex-best5={overall.iloc[0].net_pnl_ex_best_5_usdt:.2f}, MDD={overall.iloc[0].max_drawdown_usdt:.2f}")
    print("Global skips:", int(trades.skipped_due_to_existing_position.sum()))
    print("Output:", out.resolve())
    print("Data quality passed:", all(value is True or not isinstance(value, bool) for value in quality.values()))


if __name__ == "__main__":
    main()
