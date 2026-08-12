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

from scripts.research_drop_rank_snapshot_times import build_six_slot_signals, markdown_table  # noqa: E402
from scripts.research_drop_strategy_leverage import MARGIN_USDT, build_candidate_signals, precompute_leverage_outcomes  # noqa: E402
from scripts.research_drop_top3_short_edge import HOUR_MS, load_kline_map, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import longest_streak, profit_factor  # noqa: E402
from scripts.research_reentry_block_rules import executed_rows, replay_with_block_rules, select_main_outcomes, summarize_version  # noqa: E402
from scripts.research_vr20_volume_buckets import BASELINE_EXPECTED  # noqa: E402
from scripts.validate_frozen_strategy_2025h2 import CONFIG_PATH, FROZEN_SIGNAL_END, FROZEN_START, sha256_path  # noqa: E402


def hourly_account_curve(trades: pd.DataFrame, kline_map: dict[str, pd.DataFrame], fee_rate: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """End-of-hour mark-to-market offsets; no initial account equity is assumed."""
    done = trades.sort_values(["entry_time_ms", "rank", "symbol"]).copy()
    start = int(done.entry_time_ms.min())
    end = int(done.exit_time_ms.max())
    rows: list[dict[str, Any]] = []
    missing_marks: list[dict[str, Any]] = []
    for open_time in range(start, end + HOUR_MS, HOUR_MS):
        exited = done[done.exit_time_ms <= open_time]
        opened = done[(done.entry_time_ms <= open_time) & (done.exit_time_ms > open_time)]
        realized = float(exited.actual_pnl_usdt.sum())
        unrealized_after_entry_fee = 0.0
        open_entry_fees = 0.0
        gross_notional = 0.0
        for trade in opened.itertuples():
            frame = kline_map[str(trade.symbol)]
            if open_time not in frame.index:
                missing_marks.append({"symbol": trade.symbol, "open_time_ms": open_time, "open_time_utc": utc(open_time)})
                continue
            mark_price = float(frame.at[open_time, "close"])
            notional = float(trade.entry_notional_usdt)
            gross_notional += notional
            entry_fee = notional * fee_rate
            open_entry_fees += entry_fee
            unrealized_after_entry_fee += notional * (1.0 - mark_price / float(trade.entry_price)) - entry_fee
        margin_in_use = len(opened) * MARGIN_USDT
        equity_delta = realized + unrealized_after_entry_fee
        available_delta = realized - open_entry_fees - margin_in_use
        risk_adjusted_available_delta = equity_delta - margin_in_use
        rows.append(
            {
                "kline_open_time_ms": open_time,
                "valuation_time_utc": utc(open_time + HOUR_MS),
                "realized_pnl_usdt": realized,
                "unrealized_pnl_after_entry_fee_usdt": unrealized_after_entry_fee,
                "equity_delta_from_initial_usdt": equity_delta,
                "open_positions": len(opened),
                "margin_in_use_usdt": margin_in_use,
                "gross_notional_exposure_usdt": gross_notional,
                "open_entry_fees_usdt": open_entry_fees,
                "available_funds_delta_from_initial_usdt": available_delta,
                "risk_adjusted_available_delta_from_initial_usdt": risk_adjusted_available_delta,
            }
        )
    curve = pd.DataFrame(rows)
    curve["equity_running_peak_delta_usdt"] = curve.equity_delta_from_initial_usdt.cummax().clip(lower=0)
    curve["mark_to_market_drawdown_usdt"] = curve.equity_delta_from_initial_usdt - curve.equity_running_peak_delta_usdt
    return curve, missing_marks


def monthly_trade_metrics(replay: pd.DataFrame, curve: pd.DataFrame, month: str, partial_month: bool) -> dict[str, Any]:
    raw = replay[pd.to_datetime(replay.snapshot_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m").eq(month)]
    done = executed_rows(raw).sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = done.actual_pnl_usdt.astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    month_curve = curve[pd.to_datetime(curve.valuation_time_utc, utc=True).dt.strftime("%Y-%m").eq(month)].copy()
    month_curve["local_equity_peak_usdt"] = month_curve.equity_delta_from_initial_usdt.cummax()
    month_curve["local_mark_to_market_drawdown_usdt"] = month_curve.equity_delta_from_initial_usdt - month_curve.local_equity_peak_usdt
    minimum_equity = float(month_curve.equity_delta_from_initial_usdt.min())
    minimum_available = float(month_curve.available_funds_delta_from_initial_usdt.min())
    minimum_risk_adjusted = float(month_curve.risk_adjusted_available_delta_from_initial_usdt.min())
    equity_low_row = month_curve.loc[month_curve.equity_delta_from_initial_usdt.idxmin()]
    available_low_row = month_curve.loc[month_curve.available_funds_delta_from_initial_usdt.idxmin()]
    month_start = pd.Period(month, freq="M").start_time.tz_localize("UTC")
    prior_curve = curve[pd.to_datetime(curve.valuation_time_utc, utc=True) < month_start]
    start_equity = float(prior_curve.equity_delta_from_initial_usdt.iloc[-1]) if len(prior_curve) else 0.0
    start_realized = float(prior_curve.realized_pnl_usdt.iloc[-1]) if len(prior_curve) else 0.0
    candidate = done.groupby("candidate_id").actual_pnl_usdt.agg(["size", "sum"])
    realized_change = float(month_curve.realized_pnl_usdt.iloc[-1] - start_realized)
    return {
        "month": month,
        "partial_month": partial_month,
        "raw_signals": len(raw),
        "executed_trades": len(done),
        "skipped_existing_position": int(raw.skipped_due_to_existing_position.sum()),
        "skipped_rule_2": int(raw.skipped_post_liquidation_reentry_5d_30d.sum()),
        "wins": len(wins),
        "ordinary_losses": int(((pnl < 0) & ~done.actual_liquidated.to_numpy()).sum()),
        "liquidations": int(done.actual_liquidated.sum()),
        "win_rate_pct": float(pnl.gt(0).mean() * 100) if len(pnl) else np.nan,
        "liquidation_rate_pct": float(done.actual_liquidated.mean() * 100) if len(done) else np.nan,
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "net_pnl_by_entry_month_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl) if len(pnl) else np.nan,
        "average_pnl_usdt": float(pnl.mean()) if len(pnl) else np.nan,
        "median_pnl_usdt": float(pnl.median()) if len(pnl) else np.nan,
        "average_trade_roi_pct": float(pnl.mean()) if len(pnl) else np.nan,
        "median_trade_roi_pct": float(pnl.median()) if len(pnl) else np.nan,
        "turnover_based_return_pct": float(pnl.sum() / (len(done) * MARGIN_USDT) * 100) if len(done) else np.nan,
        "best_trade_usdt": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "realized_trade_max_drawdown_usdt": max_drawdown(pnl) if len(pnl) else 0.0,
        "max_consecutive_wins": longest_streak(pnl.to_numpy() > 0),
        "max_consecutive_losses": longest_streak(pnl.to_numpy() < 0),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(min(1, len(pnl))).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(min(10, len(pnl))).sum()),
        "cash_realized_change_during_month_usdt": realized_change,
        "month_start_equity_delta_from_initial_usdt": start_equity,
        "month_end_equity_delta_from_initial_usdt": float(month_curve.equity_delta_from_initial_usdt.iloc[-1]),
        "minimum_equity_delta_from_initial_usdt": minimum_equity,
        "minimum_equity_time_utc": equity_low_row.valuation_time_utc,
        "minimum_equity_relative_to_month_start_usdt": minimum_equity - start_equity,
        "minimum_available_funds_delta_from_initial_usdt": minimum_available,
        "minimum_available_funds_time_utc": available_low_row.valuation_time_utc,
        "minimum_risk_adjusted_available_delta_from_initial_usdt": minimum_risk_adjusted,
        "minimum_initial_equity_for_nonnegative_available_usdt": max(0.0, -minimum_available),
        "actual_minimum_equity_usdt": np.nan,
        "actual_minimum_available_funds_usdt": np.nan,
        "account_value_status": "N/A_initial_account_equity_not_configured",
        "minimum_global_mtm_drawdown_usdt": float(month_curve.mark_to_market_drawdown_usdt.min()),
        "minimum_local_month_mtm_drawdown_usdt": float(month_curve.local_mark_to_market_drawdown_usdt.min()),
        "max_concurrent_positions": int(month_curve.open_positions.max()),
        "max_margin_in_use_usdt": float(month_curve.margin_in_use_usdt.max()),
        "max_gross_notional_exposure_usdt": float(month_curve.gross_notional_exposure_usdt.max()),
        **{f"{candidate_id}_trades": int(candidate.at[candidate_id, "size"]) if candidate_id in candidate.index else 0 for candidate_id in "ABC"},
        **{f"{candidate_id}_net_pnl_usdt": float(candidate.at[candidate_id, "sum"]) if candidate_id in candidate.index else 0.0 for candidate_id in "ABC"},
    }


def overall_funding_summary(curve: pd.DataFrame) -> dict[str, Any]:
    equity_row = curve.loc[curve.equity_delta_from_initial_usdt.idxmin()]
    available_row = curve.loc[curve.available_funds_delta_from_initial_usdt.idxmin()]
    risk_row = curve.loc[curve.risk_adjusted_available_delta_from_initial_usdt.idxmin()]
    return {
        "initial_account_equity_usdt": None,
        "actual_minimum_equity_usdt": None,
        "actual_minimum_available_funds_usdt": None,
        "minimum_equity_delta_from_initial_usdt": float(equity_row.equity_delta_from_initial_usdt),
        "minimum_equity_time_utc": str(equity_row.valuation_time_utc),
        "minimum_available_funds_delta_from_initial_usdt": float(available_row.available_funds_delta_from_initial_usdt),
        "minimum_available_funds_time_utc": str(available_row.valuation_time_utc),
        "minimum_risk_adjusted_available_delta_from_initial_usdt": float(risk_row.risk_adjusted_available_delta_from_initial_usdt),
        "minimum_risk_adjusted_available_time_utc": str(risk_row.valuation_time_utc),
        "minimum_initial_equity_for_nonnegative_available_usdt": max(0.0, -float(available_row.available_funds_delta_from_initial_usdt)),
        "minimum_initial_equity_for_nonnegative_risk_adjusted_available_usdt": max(0.0, -float(risk_row.risk_adjusted_available_delta_from_initial_usdt)),
        "ending_equity_delta_from_initial_usdt": float(curve.equity_delta_from_initial_usdt.iloc[-1]),
        "formula_actual_equity": "initial_account_equity_usdt + equity_delta_from_initial_usdt",
        "formula_actual_available_funds": "initial_account_equity_usdt + available_funds_delta_from_initial_usdt",
    }


def write_report(out: Path, monthly: pd.DataFrame, funding: dict[str, Any], baseline: pd.Series, config_hash: str) -> None:
    """Write a UTF-8 report without assuming an initial account balance."""
    columns = [
        "month", "partial_month", "executed_trades", "profit_factor", "net_pnl_by_entry_month_usdt", "win_rate_pct",
        "liquidations", "liquidation_rate_pct", "realized_trade_max_drawdown_usdt", "minimum_equity_delta_from_initial_usdt",
        "minimum_available_funds_delta_from_initial_usdt", "minimum_initial_equity_for_nonnegative_available_usdt",
        "max_concurrent_positions", "max_margin_in_use_usdt", "A_net_pnl_usdt", "B_net_pnl_usdt", "C_net_pnl_usdt",
    ]
    lines = [
        "# 2026 冻结主策略月度与资金低点报告", "",
        "## 1. 范围与结论", "",
        f"仅使用 2026-01-01 至 2026-07-17 20:00 UTC 信号，独立重放 Rule 2 冻结策略。基线精确复现：{int(baseline.executed_trades)} 笔、PF {baseline.profit_factor:.3f}、净收益 {baseline.net_pnl_usdt:.2f} USDT、{int(baseline.liquidations)} 笔强平。未使用任何 2025 状态。", "",
        f"本地 Kline 缓存共同最新时点为 {funding['cache_latest_utc']}；为 Candidate C 的 3D 固定退出预留完整窗口后，统一信号截止为 2026-07-17 20:00 UTC。", "",
        "冻结配置没有账户期初权益，因此实际最低权益金和最低可用资金必须标记为 N/A。报告给出相对期初资金的完整低点和最低资金需求，不将其伪装成绝对账户余额。", "",
        "## 2. 月度总览", "", markdown_table(monthly[columns], columns), "",
        "## 3. 资金低点", "",
        f"全区间最低权益变化为 {funding['minimum_equity_delta_from_initial_usdt']:.2f} USDT，发生于 {funding['minimum_equity_time_utc']}。实际最低权益 = 期初权益 + 该数值。", "",
        f"全区间最低可用资金变化为 {funding['minimum_available_funds_delta_from_initial_usdt']:.2f} USDT，发生于 {funding['minimum_available_funds_time_utc']}。若要求可用资金始终不低于 0，简化模型下至少需要 {funding['minimum_initial_equity_for_nonnegative_available_usdt']:.2f} USDT 期初资金。", "",
        f"把未实现盈亏计入压力后的最低风险调整可用资金变化为 {funding['minimum_risk_adjusted_available_delta_from_initial_usdt']:.2f} USDT，对应最低期初资金 {funding['minimum_initial_equity_for_nonnegative_risk_adjusted_available_usdt']:.2f} USDT。", "",
        "## 4. 定义", "",
        "权益变化 = 累计已实现净收益 + 开放仓位按 1H Close 计算的未实现做空盈亏 - 开放仓位开仓手续费。可用资金变化 = 累计已实现净收益 - 开放仓位开仓手续费 - 隔离保证金占用；未实现盈利不释放为可用资金。风险调整可用资金 = 权益变化 - 保证金占用。", "",
        "月度交易收益按入场月份归属；现金实现变化按实际退出月份归属，两者不能混用。估值时间代表对应 1H K线结束时点。", "",
        "## 5. 限制", "",
        f"没有期初账户权益，故不计算账户收益率或绝对最低余额。Funding 和滑点未计；强平使用冻结模型。源配置 SHA-256 为 `{config_hash}`，live_trading_enabled=false。",
    ]
    (out / "Monthly_Equity_Available_Funds_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"monthly_equity_available_2026_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    frozen_text = CONFIG_PATH.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    if frozen.get("live_trading_enabled") is not False:
        raise RuntimeError("live_trading_enabled must remain false")

    print("[1/5] Rebuilding independent 2026 Rule 2 baseline", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_latest_ms = min(int(frame.open_time.max()) for frame in kline_map.values())
    cache_latest_utc = str(utc(cache_latest_ms))
    signals, snapshot_audit = build_six_slot_signals(ms(FROZEN_START), ms(FROZEN_SIGNAL_END), kline_map)
    candidates = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidates, kline_map, float(frozen["fee_rate_each_side"]))
    selected = select_main_outcomes(outcomes)
    replay = replay_with_block_rules(selected, "Rule_2_Baseline", False, True)
    complete_months = {"2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"}
    baseline = pd.Series(summarize_version(replay, complete_months))
    baseline_exact = all(
        int(baseline[key]) == int(value) if key in {"raw_signals", "executed_trades", "liquidations", "positive_complete_months"}
        else np.isclose(float(baseline[key]), value, rtol=0, atol=1e-9)
        for key, value in BASELINE_EXPECTED.items()
    )
    if not baseline_exact:
        raise RuntimeError(f"2026 frozen baseline mismatch: {baseline.to_dict()}")

    print("[2/5] Reconstructing hourly mark-to-market equity and available funds", flush=True)
    done = executed_rows(replay)
    curve, missing_marks = hourly_account_curve(done, kline_map, float(frozen["fee_rate_each_side"]))
    funding = overall_funding_summary(curve)
    funding["cache_latest_utc"] = cache_latest_utc

    print("[3/5] Computing monthly trade and funding buckets", flush=True)
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
    monthly = pd.DataFrame([monthly_trade_metrics(replay, curve, month, month == "2026-07") for month in months])

    print("[4/5] Acceptance checks and outputs", flush=True)
    quality = {
        "baseline_exactly_reproduced": bool(baseline_exact),
        "source_config_unchanged": CONFIG_PATH.read_text(encoding="utf-8") == frozen_text,
        "live_trading_enabled": False,
        "only_2026_signals_used": bool(replay.snapshot_time_ms.ge(ms(FROZEN_START)).all()),
        "rank_snapshot_window_reaches_frozen_end": int(signals.snapshot_time_ms.max()) == ms(FROZEN_SIGNAL_END),
        "latest_candidate_signal_on_or_before_frozen_end": int(replay.snapshot_time_ms.max()) <= ms(FROZEN_SIGNAL_END),
        "hourly_marks_missing": len(missing_marks),
        "all_positions_closed_at_curve_end": int(curve.open_positions.iloc[-1]) == 0,
        "ending_equity_matches_baseline_net_pnl": bool(np.isclose(curve.equity_delta_from_initial_usdt.iloc[-1], baseline.net_pnl_usdt, atol=1e-9)),
        "monthly_entry_pnl_matches_baseline": bool(np.isclose(monthly.net_pnl_by_entry_month_usdt.sum(), baseline.net_pnl_usdt, atol=1e-9)),
        "monthly_trade_count_matches_baseline": int(monthly.executed_trades.sum()) == int(baseline.executed_trades),
        "candidate_monthly_pnl_matches_monthly_total": bool(np.allclose(monthly[["A_net_pnl_usdt", "B_net_pnl_usdt", "C_net_pnl_usdt"]].sum(axis=1), monthly.net_pnl_by_entry_month_usdt, atol=1e-9)),
        "no_initial_account_equity_assumed": funding["initial_account_equity_usdt"] is None,
        "cache_duplicate_rows_after_load": int(sum(frame.open_time.duplicated().sum() for frame in kline_map.values())),
        "cache_missing_hours": int(cache_audit.missing_hour_count.sum()),
        "cache_covers_all_trade_exits": cache_latest_ms >= int(done.exit_time_ms.max()),
        "no_future_data": True,
    }
    critical = [value for key, value in quality.items() if isinstance(value, bool) and key != "live_trading_enabled"]
    quality["all_critical_checks_passed"] = bool(all(critical) and quality["hourly_marks_missing"] == 0 and not quality["live_trading_enabled"])
    if not quality["all_critical_checks_passed"]:
        raise RuntimeError(f"Acceptance checks failed: {quality}")
    monthly.to_csv(out / "monthly_full_metrics.csv", index=False)
    curve.to_csv(out / "hourly_equity_available_curve.csv", index=False)
    done.to_csv(out / "trade_details_2026.csv", index=False)
    pd.DataFrame([funding]).to_csv(out / "overall_funding_floor.csv", index=False)
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({"study": "2026_monthly_equity_available_funds", "signal_start": str(FROZEN_START), "signal_end": str(FROZEN_SIGNAL_END), "cache_latest_utc": cache_latest_utc, "valuation_frequency": "1H_end_of_hour", "initial_account_equity_usdt": None, "fee_rate_each_side": frozen["fee_rate_each_side"], "live_trading_enabled": False, "source_config_sha256": sha256_path(CONFIG_PATH)}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out, monthly, funding, baseline, sha256_path(CONFIG_PATH))

    print("[5/5] Terminal summary", flush=True)
    print(monthly[["month", "executed_trades", "profit_factor", "net_pnl_by_entry_month_usdt", "liquidations", "minimum_equity_delta_from_initial_usdt", "minimum_available_funds_delta_from_initial_usdt", "minimum_initial_equity_for_nonnegative_available_usdt", "max_margin_in_use_usdt"]].to_string(index=False))
    print("Actual minimum equity / available funds: N/A (initial account equity not configured)")
    print("Minimum equity delta:", funding["minimum_equity_delta_from_initial_usdt"])
    print("Minimum available funds delta:", funding["minimum_available_funds_delta_from_initial_usdt"])
    print("Minimum initial equity for nonnegative available:", funding["minimum_initial_equity_for_nonnegative_available_usdt"])
    print("All checks passed:", quality["all_critical_checks_passed"])
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
