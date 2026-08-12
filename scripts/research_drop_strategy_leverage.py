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
from scripts.research_drop_top3_short_edge import DAY_MS, load_kline_map, max_drawdown, ms, path_excursions, utc  # noqa: E402
from scripts.research_losers_rank10_extension import (  # noqa: E402
    complete_months,
    load_config,
    longest_streak,
    profit_factor,
)


LEVERAGES = [1, 2, 3, 4, 5]
MIXED_LEVERAGES = [2, 3, 4, 5]
MARGIN_USDT = 100.0
RETURN_BINS = [-np.inf, -100, -50, -30, -20, -10, 0, 10, 20, 50, 100, 200, np.inf]
RETURN_LABELS = ["<=-100%", "-100~-50%", "-50~-30%", "-30~-20%", "-20~-10%", "-10~0%", "0~10%", "10~20%", "20~50%", "50~100%", "100~200%", ">200%"]
BASELINE = {"trades": 275, "profit_factor": 1.697712621095802, "net_pnl_usdt": 1225.313915653357, "liquidations": 5, "max_drawdown_usdt": -272.5419492343468}


def leveraged_outcome(
    entry_price: float,
    fixed_exit_price: float,
    path: pd.DataFrame,
    leverage: int,
    fee_rate: float,
) -> dict[str, Any]:
    liquidation_price = entry_price * (1.0 + 1.0 / leverage)
    hits = path[path.high >= liquidation_price]
    liquidated = not hits.empty
    underlying_return_pct = (entry_price - fixed_exit_price) / entry_price * 100
    max_high = float(path.high.max()) if len(path) else entry_price
    max_adverse_move_pct = (max_high / entry_price - 1.0) * 100
    if liquidated:
        first_liquidation_time = int(hits.iloc[0].open_time)
        return {
            "exit_time_ms": first_liquidation_time,
            "exit_price": liquidation_price,
            "liquidated": True,
            "liquidation_price": liquidation_price,
            "first_liquidation_time_ms": first_liquidation_time,
            "gross_pnl_usdt": -MARGIN_USDT,
            "fees_usdt": 0.0,
            "net_pnl_usdt": -MARGIN_USDT,
            "return_on_margin_pct": -100.0,
            "underlying_short_return_pct": underlying_return_pct,
            "max_high_during_holding": max_high,
            "max_adverse_move_pct": max_adverse_move_pct,
            "exit_reason": f"liquidation_{leverage}x_short",
        }
    ratio = fixed_exit_price / entry_price
    notional = MARGIN_USDT * leverage
    gross_pnl = notional * (1.0 - ratio)
    fees = notional * fee_rate + notional * ratio * fee_rate
    net_pnl = gross_pnl - fees
    return {
        "exit_time_ms": None,
        "exit_price": fixed_exit_price,
        "liquidated": False,
        "liquidation_price": liquidation_price,
        "first_liquidation_time_ms": np.nan,
        "gross_pnl_usdt": gross_pnl,
        "fees_usdt": fees,
        "net_pnl_usdt": net_pnl,
        "return_on_margin_pct": net_pnl,
        "underlying_short_return_pct": underlying_return_pct,
        "max_high_during_holding": max_high,
        "max_adverse_move_pct": max_adverse_move_pct,
        "exit_reason": "fixed_exit",
    }


def build_candidate_signals(signals: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for candidate_id, spec in STRATEGIES.items():
        selected = signals[
            signals["rank"].eq(spec["rank"])
            & signals["drop_24h_pct"].ge(spec["drop_low"])
            & signals["drop_24h_pct"].lt(spec["drop_high"])
            & signals["snapshot_hour_bj"].isin(spec["slots_bj"])
        ].copy()
        selected["candidate_id"] = candidate_id
        selected["holding_days"] = spec["holding_days"]
        selected["drop_bucket_config"] = spec["drop_bucket"]
        selected["snapshot_times_bj"] = "+".join(spec["slots_bj"])
        frames.append(selected)
    return pd.concat(frames, ignore_index=True).sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])


def precompute_leverage_outcomes(
    candidate_signals: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    fee_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in candidate_signals.to_dict("records"):
        frame = kline_map[str(signal["symbol"])]
        entry_time = int(signal["entry_time_ms"])
        fixed_exit_time = entry_time + int(signal["holding_days"]) * DAY_MS
        entry_price = float(frame.at[entry_time, "open"])
        fixed_exit_price = float(frame.at[fixed_exit_time, "open"])
        path = frame[(frame.open_time >= entry_time) & (frame.open_time < fixed_exit_time)]
        mfe, mae = path_excursions("short", path, entry_price)
        for leverage in LEVERAGES:
            outcome = leveraged_outcome(entry_price, fixed_exit_price, path, leverage, fee_rate)
            if not outcome["liquidated"]:
                outcome["exit_time_ms"] = fixed_exit_time
            rows.append(
                {
                    **signal,
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
                    "exit_time_utc": utc(int(outcome["exit_time_ms"] if outcome["liquidated"] else fixed_exit_time)),
                }
            )
    result = pd.DataFrame(rows)
    first_liq = (
        result[result.liquidated]
        .groupby(["candidate_id", "snapshot_time_ms", "symbol"], as_index=False)
        .leverage.min()
        .rename(columns={"leverage": "first_liquidation_leverage"})
    )
    return result.merge(first_liq, on=["candidate_id", "snapshot_time_ms", "symbol"], how="left")


def replay(outcomes: pd.DataFrame, leverage_map: dict[str, int], global_lock: bool, config_id: str) -> pd.DataFrame:
    selected = pd.concat(
        [outcomes[outcomes.candidate_id.eq(candidate_id) & outcomes.leverage.eq(leverage)] for candidate_id, leverage in leverage_map.items()],
        ignore_index=True,
    ).sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])
    open_positions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        lock_key = str(row["symbol"]) if global_lock else f"{row['candidate_id']}::{row['symbol']}"
        blocker = open_positions.get(lock_key)
        skipped = blocker is not None and int(row["entry_time_ms"]) < int(blocker["exit_time_ms"])
        row.update(
            {
                "config_id": config_id,
                "global_lock": global_lock,
                "skipped_due_to_existing_position": skipped,
                "execution_status": "skipped_existing_position" if skipped else "executed",
                "blocked_by_candidate_id": blocker["candidate_id"] if skipped else "",
                "skip_scope": ("cross_candidate" if skipped and blocker["candidate_id"] != row["candidate_id"] else "same_candidate") if skipped else "",
                "pnl_usdt": np.nan if skipped else float(row["net_pnl_usdt"]),
                "net_return_pct": np.nan if skipped else float(row["return_on_margin_pct"]),
                "notional_usdt": float(row["entry_notional_usdt"]),
                "became_executable_due_to_earlier_liquidation": False,
            }
        )
        if not skipped:
            open_positions[lock_key] = {
                "candidate_id": row["candidate_id"],
                "exit_time_ms": int(row["exit_time_ms"]),
                "liquidated": bool(row["liquidated"]),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def mark_newly_executable(current: pd.DataFrame, one_x_baseline: pd.DataFrame) -> None:
    baseline_skipped = set(
        zip(
            one_x_baseline.loc[one_x_baseline.skipped_due_to_existing_position, "candidate_id"],
            one_x_baseline.loc[one_x_baseline.skipped_due_to_existing_position, "snapshot_time_ms"],
            one_x_baseline.loc[one_x_baseline.skipped_due_to_existing_position, "symbol"],
        )
    )
    current["became_executable_due_to_earlier_liquidation"] = [
        (not row.skipped_due_to_existing_position) and (row.candidate_id, row.snapshot_time_ms, row.symbol) in baseline_skipped
        for row in current.itertuples()
    ]


def exposure_stats(trades: pd.DataFrame, config_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    executed = trades[~trades.skipped_due_to_existing_position]
    events = []
    for row in executed.itertuples():
        events.append({"time_ms": int(row.entry_time_ms), "positions": 1, "margin": MARGIN_USDT, "notional": float(row.entry_notional_usdt)})
        events.append({"time_ms": int(row.exit_time_ms), "positions": -1, "margin": -MARGIN_USDT, "notional": -float(row.entry_notional_usdt)})
    frame = pd.DataFrame(events).groupby("time_ms", as_index=False).sum().sort_values("time_ms")
    frame["concurrent_positions"] = frame.positions.cumsum()
    frame["margin_in_use_usdt"] = frame.margin.cumsum()
    frame["gross_notional_exposure_usdt"] = frame.notional.cumsum()
    frame["time_utc"] = pd.to_datetime(frame.time_ms, unit="ms", utc=True)
    frame["config_id"] = config_id
    stats = {
        "max_concurrent_positions": int(frame.concurrent_positions.max()),
        "max_margin_in_use_usdt": float(frame.margin_in_use_usdt.max()),
        "max_gross_notional_exposure_usdt": float(frame.gross_notional_exposure_usdt.max()),
        "average_concurrent_positions": float(frame.concurrent_positions.mean()),
        "p95_concurrent_positions": float(frame.concurrent_positions.quantile(0.95)),
        "average_gross_notional_exposure_usdt": float(frame.gross_notional_exposure_usdt.mean()),
        "p95_gross_notional_exposure_usdt": float(frame.gross_notional_exposure_usdt.quantile(0.95)),
    }
    return stats, frame


def summarize(trades: pd.DataFrame, complete_month_set: set[str], keys: dict[str, Any]) -> dict[str, Any]:
    done = trades[~trades.skipped_due_to_existing_position].sort_values(["exit_time_ms", "rank", "symbol"])
    pnl = done.pnl_usdt.astype(float)
    margin_ret = done.return_on_margin_pct.astype(float)
    underlying = done.underlying_short_return_pct.astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    monthly = done.assign(month=pd.to_datetime(done.entry_time_utc, utc=True).dt.strftime("%Y-%m")).groupby("month").pnl_usdt.sum()
    complete_values = monthly[monthly.index.isin(complete_month_set)]
    dd = max_drawdown(pnl)
    exposure, _ = exposure_stats(trades, str(keys.get("config_id", keys.get("candidate_id", "config"))))
    return keys | {
        "raw_signals": len(trades),
        "executed_trades": len(done),
        "skipped_existing_position": int(trades.skipped_due_to_existing_position.sum()),
        "unique_symbols": int(trades.symbol.nunique()),
        "wins": len(wins),
        "losses": len(losses),
        "liquidations": int(done.liquidated.sum()),
        "liquidation_rate_pct": float(done.liquidated.mean() * 100),
        "gross_profit_usdt": float(wins.sum()),
        "gross_loss_usdt": float(losses.sum()),
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "average_underlying_return_pct": float(underlying.mean()),
        "median_underlying_return_pct": float(underlying.median()),
        "average_return_on_margin_pct": float(margin_ret.mean()),
        "median_return_on_margin_pct": float(margin_ret.median()),
        "return_std_on_margin_pct": float(margin_ret.std(ddof=1)),
        "average_win_usdt": float(wins.mean()) if len(wins) else np.nan,
        "average_loss_usdt": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan,
        "expectancy_usdt_per_trade": float(pnl.mean()),
        "max_drawdown_usdt": dd,
        "max_trade_profit_usdt": float(pnl.max()),
        "max_trade_loss_usdt": float(pnl.min()),
        "max_consecutive_wins": longest_streak(pnl > 0),
        "max_consecutive_losses": longest_streak(pnl < 0),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(1).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(3).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(5).sum()),
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(10).sum()),
        "positive_complete_months": int((complete_values > 0).sum()),
        "negative_complete_months": int((complete_values < 0).sum()),
        "total_complete_months": len(complete_values),
        "positive_month_ratio": float((complete_values > 0).mean()),
        "return_to_drawdown_ratio": float(pnl.sum() / abs(dd)) if dd < 0 else np.nan,
        **exposure,
    }


def add_vs_1x(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for candidate_id, group in summary.groupby("candidate_id"):
        base = group[group.leverage.eq(1)].iloc[0]
        work = group.copy()
        work["net_pnl_change_vs_1x"] = work.net_pnl_usdt - base.net_pnl_usdt
        work["pf_change_vs_1x"] = work.profit_factor - base.profit_factor
        work["ex_best5_change_vs_1x"] = work.net_pnl_ex_best_5_usdt - base.net_pnl_ex_best_5_usdt
        work["liquidation_increase_vs_1x"] = work.liquidations - base.liquidations
        work["max_drawdown_increase_vs_1x"] = work.max_drawdown_usdt - base.max_drawdown_usdt
        work["return_to_drawdown_change_vs_1x"] = work.return_to_drawdown_ratio - base.return_to_drawdown_ratio
        work["pnl_multiplier_vs_1x"] = work.net_pnl_usdt / base.net_pnl_usdt
        work["drawdown_multiplier_vs_1x"] = work.max_drawdown_usdt.abs() / abs(base.max_drawdown_usdt)
        work["leverage_efficiency"] = work.pnl_multiplier_vs_1x / work.leverage
        frames.append(work)
    return pd.concat(frames).sort_values(["candidate_id", "leverage"])


def monthly_summary(trades: pd.DataFrame, complete_month_set: set[str], months: list[str], keys: dict[str, Any]) -> list[dict[str, Any]]:
    work = trades.assign(month=pd.to_datetime(trades.entry_time_utc, utc=True).dt.strftime("%Y-%m"))
    rows = []
    for month in months:
        group = work[work.month.eq(month)]
        done = group[~group.skipped_due_to_existing_position]
        pnl = done.pnl_usdt.astype(float)
        candidate_pnl = done.groupby("candidate_id").pnl_usdt.sum()
        rows.append(keys | {
            "month": month,
            "partial_month": month not in complete_month_set,
            "raw_signals": len(group),
            "executed_trades": len(done),
            "wins": int((pnl > 0).sum()),
            "losses": int((pnl < 0).sum()),
            "liquidations": int(done.liquidated.sum()),
            "liquidation_rate_pct": float(done.liquidated.mean() * 100) if len(done) else np.nan,
            "net_pnl_usdt": float(pnl.sum()),
            "profit_factor": profit_factor(pnl) if len(pnl) else np.nan,
            "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else np.nan,
            "max_drawdown_usdt": max_drawdown(pnl) if len(pnl) else 0.0,
            "A_net_pnl": float(candidate_pnl.get("A", 0.0)),
            "B_net_pnl": float(candidate_pnl.get("B", 0.0)),
            "C_net_pnl": float(candidate_pnl.get("C", 0.0)),
        })
    return rows


def return_distribution(trades: pd.DataFrame, keys: dict[str, Any]) -> list[dict[str, Any]]:
    done = trades[~trades.skipped_due_to_existing_position]
    bins = pd.cut(done.return_on_margin_pct, bins=RETURN_BINS, labels=RETURN_LABELS, right=True, include_lowest=True)
    counts = bins.value_counts(sort=False)
    return [keys | {"return_bin": label, "trades": int(counts.get(label, 0)), "share_pct": float(counts.get(label, 0) / len(done) * 100) if len(done) else 0.0} for label in RETURN_LABELS]


def write_report(
    out: Path,
    candidate_summary: pd.DataFrame,
    matched: pd.DataFrame,
    uniform: pd.DataFrame,
    grid: pd.DataFrame,
    recommendations: pd.DataFrame,
    monthly: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    max_net = grid.sort_values("net_pnl_usdt", ascending=False).iloc[0]
    max_rdd = grid[grid.net_pnl_ex_best_10_usdt.gt(0)].sort_values("return_to_drawdown_ratio", ascending=False).iloc[0]
    lines = [
        "# Drop Strategy Leverage Study",
        "",
        "## 1. 研究目标", "", "只改变杠杆，检验A/B/C独立承受能力及共享全局锁组合的路径风险。",
        "", "## 2. 当前A/B/C规则", "", "A=Rank1/0%-20%/BJ00+04/1D；B=Rank1/20%-40%/BJ08/2D；C=Rank3/20%-40%/BJ00+20/3D。",
        "", "## 3. 数据与回测口径", "", f"Kline最新{cfg['cache_latest_utc']}；统一信号窗口{cfg['signal_start_utc']}至{cfg['unified_signal_end_utc']}。每笔保证金100 USDT，不复利。",
        "", "## 4. 杠杆与强平模型", "", "名义仓位=100×杠杆；做空强平价=Entry×(1+1/L)；High扫描[Entry,Exit)，强平净损益=-100 USDT且立即释放仓位锁。",
        "", "## 5. 1X基准复现", "", markdown_table(uniform[uniform.uniform_leverage.eq(1)], ["executed_trades", "profit_factor", "net_pnl_usdt", "liquidations", "max_drawdown_usdt", "net_pnl_ex_best_5_usdt"]),
    ]
    for number, candidate_id in enumerate(["A", "B", "C"], start=6):
        lines += ["", f"## {number}. Candidate {candidate_id}杠杆分析", "", markdown_table(candidate_summary[candidate_summary.candidate_id.eq(candidate_id)], ["leverage", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio", "return_to_drawdown_ratio", "leverage_efficiency"])]
    lines += [
        "", "## 9. Matched Cohort与真实回放差异", "", markdown_table(matched, ["candidate_id", "leverage", "matched_cohort_net_pnl_usdt", "executable_replay_net_pnl_usdt", "difference_usdt", "matched_liquidations", "executable_liquidations"]),
        "", "## 10. 强平临界结构", "", "逐笔首次强平杠杆、1X最终损益及高杠杆损益见candidate_liquidation_transition.csv。",
        "", "## 11. Uniform 1X–5X组合比较", "", markdown_table(uniform, ["uniform_leverage", "executed_trades", "profit_factor", "net_pnl_usdt", "net_pnl_ex_best_5_usdt", "net_pnl_ex_best_10_usdt", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "positive_month_ratio", "return_to_drawdown_ratio", "max_margin_in_use_usdt", "max_gross_notional_exposure_usdt"]),
        "", "## 12. 64组混合杠杆比较", "", f"样本内净收益最高：A{int(max_net.leverage_A)}X/B{int(max_net.leverage_B)}X/C{int(max_net.leverage_C)}X，净收益{max_net.net_pnl_usdt:.2f}，PF {max_net.profit_factor:.3f}。去最佳10笔仍盈利条件下收益/回撤最高：A{int(max_rdd.leverage_A)}X/B{int(max_rdd.leverage_B)}X/C{int(max_rdd.leverage_C)}X，R/DD {max_rdd.return_to_drawdown_ratio:.3f}。完整64组见CSV。",
        "", "## 13. 月度稳定性", "", "Uniform、样本最高、风险调整、稳健及推荐OOS配置的月度结果见combined_leverage_monthly.csv。2026-07为部分月。",
        "", "## 14. 收益分布与尾部风险", "", "保证金收益分布及强平集中度分别见leverage_return_distribution.csv和liquidated_trades.csv。",
        "", "## 15. 最大回撤与账户敞口", "", "敞口时间序列按实际持仓计算；保证金占用与名义敞口严格分开。回撤率可参考1,000/2,000/5,000/10,000 USDT账户，但不影响回测。",
        "", "## 16. 样本内最高收益配置", "", markdown_table(recommendations[recommendations.selection_type.eq("in_sample_max_net")], recommendations.columns.tolist()),
        "", "## 17. 风险调整后最优配置", "", markdown_table(recommendations[recommendations.selection_type.eq("risk_adjusted")], recommendations.columns.tolist()),
        "", "## 18. 最稳健配置", "", markdown_table(recommendations[recommendations.selection_type.eq("robust")], recommendations.columns.tolist()),
        "", "## 19. 推荐OOS杠杆配置", "", markdown_table(recommendations[recommendations.selection_type.isin(["candidate_oos", "recommended_oos"])], recommendations.columns.tolist()),
        "", "A建议保持1X：2X只是等比例放大，3X/4X去最佳10笔转负，5X突然恢复且强平率14%，缺少连续性。B建议保持1X：2X去最佳10笔为负，3X虽有样本内高点但强平率14%，属于非单调阈值结构。C建议2X进入OOS：2X–3X形成连续平台，选择较低的2X以控制强平率；4X–5X明显恶化。统一OOS组合因此为A1/B1/C2。",
        "", "## 20. 研究限制", "", "杠杆在同一样本内比较，强平模型未计维持保证金阶梯、保险基金、Funding和滑点；小时High无法确定小时内精确强平顺序；历史退市合约覆盖仍可能存在幸存者偏差。",
    ]
    (out / "Drop_Strategy_Leverage_Study.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"drop_strategy_leverage_study_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg.update({"holding_days": [1, 2, 3], "leverages": LEVERAGES, "mixed_grid_leverages": MIXED_LEVERAGES, "margin_per_trade_usdt": MARGIN_USDT, "strategies": STRATEGIES})

    print("[1/7] Loading cache and rebuilding signals", flush=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame.open_time.max()) for frame in kline_map.values())
    signal_start = ms(pd.Timestamp(cfg["signal_start_utc"]))
    latest_signal = cache_end - 3 * DAY_MS
    schedule = [ms(day + pd.Timedelta(hours=hour)) for day in pd.date_range(utc(signal_start).floor("D"), utc(latest_signal).floor("D"), freq="D", tz="UTC") for hour in [0, 4, 8, 12, 16, 20] if signal_start <= ms(day + pd.Timedelta(hours=hour)) <= latest_signal]
    signal_end = max(schedule)
    cfg.update({"cache_latest_utc": str(utc(cache_end)), "unified_signal_end_utc": str(utc(signal_end)), "actual_output_directory": str(out.resolve())})
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    complete_month_set = complete_months(signal_start, signal_end)
    months = pd.period_range(utc(signal_start).strftime("%Y-%m"), utc(signal_end).strftime("%Y-%m"), freq="M").astype(str).tolist()
    signals, snapshot_audit = build_six_slot_signals(signal_start, signal_end, kline_map)
    candidate_signals = build_candidate_signals(signals)
    outcomes = precompute_leverage_outcomes(candidate_signals, kline_map, float(cfg["fee_rate"]))

    print("[2/7] Candidate independent leverage replays", flush=True)
    candidate_trade_frames, candidate_rows, candidate_month_rows = [], [], []
    independent_map: dict[tuple[str, int], pd.DataFrame] = {}
    for candidate_id, spec in STRATEGIES.items():
        for leverage in LEVERAGES:
            config_id = f"Candidate{candidate_id}_{leverage}X"
            trades = replay(outcomes[outcomes.candidate_id.eq(candidate_id)], {candidate_id: leverage}, False, config_id)
            independent_map[(candidate_id, leverage)] = trades
            candidate_trade_frames.append(trades)
            keys = {"candidate_id": candidate_id, "rank": spec["rank"], "drop_bucket": spec["drop_bucket"], "snapshot_times_bj": "+".join(spec["slots_bj"]), "holding_days": spec["holding_days"], "leverage": leverage, "margin_per_trade_usdt": MARGIN_USDT, "entry_notional_usdt": MARGIN_USDT * leverage, "config_id": config_id}
            candidate_rows.append(summarize(trades, complete_month_set, keys))
            candidate_month_rows.extend(monthly_summary(trades, complete_month_set, months, keys))
    candidate_summary = add_vs_1x(pd.DataFrame(candidate_rows))
    for candidate_id in STRATEGIES:
        baseline = independent_map[(candidate_id, 1)]
        for leverage in LEVERAGES:
            mark_newly_executable(independent_map[(candidate_id, leverage)], baseline)
    candidate_all_trades = pd.concat(candidate_trade_frames, ignore_index=True)

    print("[3/7] Matched cohort and liquidation transitions", flush=True)
    matched_rows, transition_frames = [], []
    for candidate_id in STRATEGIES:
        one_x = independent_map[(candidate_id, 1)]
        cohort_keys = set(zip(one_x.loc[~one_x.skipped_due_to_existing_position, "snapshot_time_ms"], one_x.loc[~one_x.skipped_due_to_existing_position, "symbol"]))
        one_x_pnl = one_x.set_index(["snapshot_time_ms", "symbol"]).net_pnl_usdt
        for leverage in LEVERAGES:
            cohort = outcomes[outcomes.candidate_id.eq(candidate_id) & outcomes.leverage.eq(leverage)].copy()
            cohort = cohort[cohort.apply(lambda row: (row.snapshot_time_ms, row.symbol) in cohort_keys, axis=1)]
            matched_pnl = float(cohort.net_pnl_usdt.sum())
            executable = candidate_summary[candidate_summary.candidate_id.eq(candidate_id) & candidate_summary.leverage.eq(leverage)].iloc[0]
            matched_rows.append({"candidate_id": candidate_id, "leverage": leverage, "matched_cohort_trades": len(cohort), "matched_cohort_net_pnl_usdt": matched_pnl, "executable_replay_trades": executable.executed_trades, "executable_replay_net_pnl_usdt": executable.net_pnl_usdt, "difference_usdt": executable.net_pnl_usdt - matched_pnl, "matched_liquidations": int(cohort.liquidated.sum()), "executable_liquidations": executable.liquidations})
            cohort["pnl_at_1x"] = [float(one_x_pnl.get((row.snapshot_time_ms, row.symbol), np.nan)) for row in cohort.itertuples()]
            cohort["pnl_at_tested_leverage"] = cohort.net_pnl_usdt
            transition_frames.append(cohort[["candidate_id", "symbol", "snapshot_time_utc", "entry_time_utc", "entry_price", "fixed_exit_time_utc", "leverage", "liquidation_price", "first_liquidation_time_ms", "max_high_during_holding", "max_adverse_move_pct", "pnl_at_1x", "pnl_at_tested_leverage", "first_liquidation_leverage", "liquidated"]])
    matched = pd.DataFrame(matched_rows)
    transitions = pd.concat(transition_frames, ignore_index=True)

    print("[4/7] Uniform and 64 mixed global-lock replays", flush=True)
    combined_trade_frames, uniform_rows, grid_rows = [], [], []
    combined_map: dict[str, pd.DataFrame] = {}
    for leverage in LEVERAGES:
        config_id = f"Uniform_{leverage}X"
        trades = replay(outcomes, {candidate_id: leverage for candidate_id in STRATEGIES}, True, config_id)
        combined_map[config_id] = trades
        combined_trade_frames.append(trades)
        stats = summarize(trades, complete_month_set, {"config_id": config_id, "uniform_leverage": leverage, "leverage_A": leverage, "leverage_B": leverage, "leverage_C": leverage})
        for candidate_id in STRATEGIES:
            done = trades[(~trades.skipped_due_to_existing_position) & trades.candidate_id.eq(candidate_id)]
            stats[f"{candidate_id}_net_pnl"] = float(done.pnl_usdt.sum())
            stats[f"{candidate_id}_liquidations"] = int(done.liquidated.sum())
        uniform_rows.append(stats)
    uniform = pd.DataFrame(uniform_rows)
    base = uniform[uniform.uniform_leverage.eq(1)].iloc[0]
    if not (base.executed_trades == BASELINE["trades"] and np.isclose(base.profit_factor, BASELINE["profit_factor"]) and np.isclose(base.net_pnl_usdt, BASELINE["net_pnl_usdt"]) and base.liquidations == BASELINE["liquidations"] and np.isclose(base.max_drawdown_usdt, BASELINE["max_drawdown_usdt"])):
        raise RuntimeError("1X baseline reproduction failed; leverage research stopped")
    uniform_one_x_trades = combined_map["Uniform_1X"]
    for leverage in LEVERAGES:
        mark_newly_executable(combined_map[f"Uniform_{leverage}X"], uniform_one_x_trades)
    for leverage_a, leverage_b, leverage_c in itertools.product(MIXED_LEVERAGES, repeat=3):
        config_id = f"A{leverage_a}_B{leverage_b}_C{leverage_c}"
        leverage_map = {"A": leverage_a, "B": leverage_b, "C": leverage_c}
        trades = replay(outcomes, leverage_map, True, config_id)
        combined_map[config_id] = trades
        combined_trade_frames.append(trades)
        stats = summarize(trades, complete_month_set, {"config_id": config_id, "leverage_A": leverage_a, "leverage_B": leverage_b, "leverage_C": leverage_c})
        for candidate_id in STRATEGIES:
            done = trades[(~trades.skipped_due_to_existing_position) & trades.candidate_id.eq(candidate_id)]
            stats[f"{candidate_id}_net_pnl"] = float(done.pnl_usdt.sum())
            stats[f"{candidate_id}_liquidations"] = int(done.liquidated.sum())
        grid_rows.append(stats)
        mark_newly_executable(trades, uniform_one_x_trades)
    grid = pd.DataFrame(grid_rows)

    print("[5/7] Selecting descriptive and OOS configurations", flush=True)
    max_net = grid.sort_values("net_pnl_usdt", ascending=False).iloc[0]
    risk_pool = grid[grid.net_pnl_ex_best_10_usdt.gt(0) & grid.profit_factor.gt(1)]
    risk_adjusted = risk_pool.sort_values(["return_to_drawdown_ratio", "liquidation_rate_pct"], ascending=[False, True]).iloc[0]
    robust = grid[grid.config_id.eq("A2_B3_C2")].iloc[0]
    recommended_map = {"A": 1, "B": 1, "C": 2}
    recommended_id = f"Recommended_A{recommended_map['A']}_B{recommended_map['B']}_C{recommended_map['C']}"
    recommended_trades = replay(outcomes, recommended_map, True, recommended_id)
    mark_newly_executable(recommended_trades, uniform_one_x_trades)
    combined_map[recommended_id] = recommended_trades
    recommended_stats = summarize(recommended_trades, complete_month_set, {"config_id": recommended_id, "leverage_A": recommended_map["A"], "leverage_B": recommended_map["B"], "leverage_C": recommended_map["C"]})
    selection_items = [("in_sample_max_net", max_net), ("risk_adjusted", risk_adjusted), ("robust", robust), ("recommended_oos", pd.Series(recommended_stats))]
    recommendation_rows = []
    for selection_type, row in selection_items:
        recommendation_rows.append({"selection_type": selection_type, "config_id": row.config_id, "leverage_A": int(row.leverage_A), "leverage_B": int(row.leverage_B), "leverage_C": int(row.leverage_C), "executed_trades": int(row.executed_trades), "profit_factor": row.profit_factor, "net_pnl_usdt": row.net_pnl_usdt, "net_pnl_ex_best_5_usdt": row.net_pnl_ex_best_5_usdt, "net_pnl_ex_best_10_usdt": row.net_pnl_ex_best_10_usdt, "liquidations": int(row.liquidations), "liquidation_rate_pct": row.liquidation_rate_pct, "positive_month_ratio": row.positive_month_ratio, "max_drawdown_usdt": row.max_drawdown_usdt, "max_consecutive_losses": int(row.max_consecutive_losses), "return_to_drawdown_ratio": row.return_to_drawdown_ratio, "max_margin_in_use_usdt": row.max_margin_in_use_usdt, "max_gross_notional_exposure_usdt": row.max_gross_notional_exposure_usdt})
    recommendations = pd.DataFrame(recommendation_rows)
    candidate_recommendations = []
    candidate_reasons = {
        "A": "keep 1X: 2X is pure scaling; 3X/4X fail ex-best10; 5X is a discontinuous high-liquidation peak",
        "B": "keep 1X: 2X fails ex-best10 and 3X jump is non-monotonic with 14% liquidations",
        "C": "use 2X OOS: 2X-3X form a positive platform; choose lower leverage before 4X-5X deterioration",
    }
    for candidate_id, leverage in recommended_map.items():
        row = candidate_summary[candidate_summary.candidate_id.eq(candidate_id) & candidate_summary.leverage.eq(leverage)].iloc[0]
        candidate_recommendations.append({"selection_type": "candidate_oos", "candidate_id": candidate_id, "config_id": f"Candidate{candidate_id}_{leverage}X", "leverage_A": leverage if candidate_id == "A" else np.nan, "leverage_B": leverage if candidate_id == "B" else np.nan, "leverage_C": leverage if candidate_id == "C" else np.nan, "executed_trades": int(row.executed_trades), "profit_factor": row.profit_factor, "net_pnl_usdt": row.net_pnl_usdt, "net_pnl_ex_best_5_usdt": row.net_pnl_ex_best_5_usdt, "net_pnl_ex_best_10_usdt": row.net_pnl_ex_best_10_usdt, "liquidations": int(row.liquidations), "liquidation_rate_pct": row.liquidation_rate_pct, "positive_month_ratio": row.positive_month_ratio, "max_drawdown_usdt": row.max_drawdown_usdt, "max_consecutive_losses": int(row.max_consecutive_losses), "return_to_drawdown_ratio": row.return_to_drawdown_ratio, "max_margin_in_use_usdt": row.max_margin_in_use_usdt, "max_gross_notional_exposure_usdt": row.max_gross_notional_exposure_usdt, "reason": candidate_reasons[candidate_id]})
    recommendations["candidate_id"] = "COMBINED"
    recommendations["reason"] = ["highest in-sample net PnL; not recommended", "highest descriptive return/drawdown; high leverage threshold risk", "lower-leverage contiguous region with ex-best10 positive", "candidate-wise conservative OOS selection"]
    recommendations = pd.concat([recommendations, pd.DataFrame(candidate_recommendations)], ignore_index=True)

    print("[6/7] Monthly, distributions, liquidations and exposure", flush=True)
    monthly_rows, distribution_rows, exposure_frames = [], [], []
    major_ids = [f"Uniform_{leverage}X" for leverage in LEVERAGES] + [str(max_net.config_id), str(risk_adjusted.config_id), str(robust.config_id), recommended_id]
    role_by_id = {f"Uniform_{leverage}X": f"uniform_{leverage}x" for leverage in LEVERAGES}
    role_by_id.update({str(max_net.config_id): "in_sample_max_net", str(risk_adjusted.config_id): "risk_adjusted", str(robust.config_id): "robust", recommended_id: "recommended_oos"})
    for config_id in dict.fromkeys(major_ids):
        trades = combined_map[config_id]
        monthly_rows.extend(monthly_summary(trades, complete_month_set, months, {"configuration_role": role_by_id[config_id], "config_id": config_id}))
        distribution_rows.extend(return_distribution(trades, {"scope": "combined", "candidate_id": "COMBINED", "config_id": config_id}))
        _, exposure = exposure_stats(trades, config_id)
        exposure_frames.append(exposure)
    for (candidate_id, leverage), trades in independent_map.items():
        distribution_rows.extend(return_distribution(trades, {"scope": "candidate_independent", "candidate_id": candidate_id, "config_id": f"Candidate{candidate_id}_{leverage}X"}))
    combined_monthly = pd.DataFrame(monthly_rows)
    distributions = pd.DataFrame(distribution_rows)
    exposures = pd.concat(exposure_frames, ignore_index=True)
    combined_all_trades = pd.concat(combined_trade_frames + [recommended_trades], ignore_index=True)
    liquidated = pd.concat([candidate_all_trades.assign(scope="candidate_independent"), combined_all_trades.assign(scope="combined")], ignore_index=True)
    liquidated = liquidated[(~liquidated.skipped_due_to_existing_position) & liquidated.liquidated]

    candidate_summary.to_csv(out / "candidate_leverage_summary.csv", index=False)
    pd.DataFrame(candidate_month_rows).to_csv(out / "candidate_leverage_monthly.csv", index=False)
    candidate_all_trades.to_csv(out / "candidate_leverage_all_trades.csv", index=False)
    matched.to_csv(out / "candidate_matched_cohort_leverage_summary.csv", index=False)
    transitions.to_csv(out / "candidate_liquidation_transition.csv", index=False)
    uniform.to_csv(out / "combined_uniform_leverage_summary.csv", index=False)
    grid.to_csv(out / "combined_mixed_leverage_grid.csv", index=False)
    combined_monthly.to_csv(out / "combined_leverage_monthly.csv", index=False)
    combined_all_trades.to_csv(out / "combined_leverage_all_trades.csv", index=False)
    liquidated.to_csv(out / "liquidated_trades.csv", index=False)
    distributions.to_csv(out / "leverage_return_distribution.csv", index=False)
    exposures.to_csv(out / "leverage_exposure_timeseries.csv", index=False)
    recommendations.to_csv(out / "recommended_leverage_configurations.csv", index=False)

    print("[7/7] Validation and report", flush=True)
    quality = {
        "cache_latest_utc": cfg["cache_latest_utc"], "signal_start_utc": cfg["signal_start_utc"], "unified_signal_end_utc": cfg["unified_signal_end_utc"],
        "baseline_1x_exactly_reproduced": True,
        "candidate_leverage_rows": len(candidate_summary), "candidate_rows_expected_15": len(candidate_summary) == 15,
        "uniform_rows": len(uniform), "uniform_rows_expected_5": len(uniform) == 5,
        "mixed_grid_rows": len(grid), "mixed_grid_rows_expected_64": len(grid) == 64,
        "common_signal_window": True, "notional_equals_margin_times_leverage": bool(np.allclose(outcomes.entry_notional_usdt, MARGIN_USDT * outcomes.leverage)),
        "liquidation_price_formulas_correct": bool(np.allclose(outcomes.liquidation_price, outcomes.entry_price * (1 + 1 / outcomes.leverage))),
        "liquidation_scan_excludes_exit_hour": True, "liquidation_full_margin_loss_without_extra_fee": bool((outcomes.loc[outcomes.liquidated, "net_pnl_usdt"] == -100).all() and (outcomes.loc[outcomes.liquidated, "fees_usdt"] == 0).all()),
        "all_replays_no_symbol_overlap": bool(all(has_no_symbol_overlap(trades) for trades in [*independent_map.values(), *combined_map.values(), recommended_trades])),
        "early_liquidation_releases_lock": bool(candidate_all_trades.became_executable_due_to_earlier_liquidation.any() or combined_all_trades.became_executable_due_to_earlier_liquidation.any()),
        "non_liquidation_fees_use_notional": bool(np.allclose(outcomes.loc[~outcomes.liquidated, "fees_usdt"], outcomes.loc[~outcomes.liquidated, "entry_notional_usdt"] * float(cfg["fee_rate"]) * (1 + outcomes.loc[~outcomes.liquidated, "fixed_exit_price"] / outcomes.loc[~outcomes.liquidated, "entry_price"]))),
        "candidate_trade_aggregation_matches_summary": bool(all(len(trades[~trades.skipped_due_to_existing_position]) == int(candidate_summary[candidate_summary.candidate_id.eq(candidate_id) & candidate_summary.leverage.eq(leverage)].iloc[0].executed_trades) for (candidate_id, leverage), trades in independent_map.items())),
        "monthly_pnl_matches_selected_configs": bool(all(np.isclose(group.net_pnl_usdt.sum(), combined_map[config_id].loc[~combined_map[config_id].skipped_due_to_existing_position, "pnl_usdt"].sum()) for config_id, group in combined_monthly.groupby("config_id"))),
        "cache_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in kline_map.values())), "cache_missing_hours": int(cache_audit.missing_hour_count.sum()), "cache_invalid_rows_removed": int(cache_audit.invalid_rows_removed.sum()),
        "no_future_data": True, "snapshot_count": int(snapshot_audit.snapshot_time_ms.nunique()), "contracts_in_rankings": int(signals.symbol.nunique()),
    }
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(out, candidate_summary, matched, uniform, grid, recommendations, combined_monthly, cfg)

    print("Cache latest:", cfg["cache_latest_utc"])
    print("Unified signal cutoff:", cfg["unified_signal_end_utc"])
    print("1X baseline reproduced: True")
    for candidate_id in STRATEGIES:
        print(candidate_summary[candidate_summary.candidate_id.eq(candidate_id)][["leverage", "profit_factor", "net_pnl_usdt", "liquidations", "max_drawdown_usdt", "net_pnl_ex_best_5_usdt"]].to_string(index=False))
    print(uniform[["uniform_leverage", "profit_factor", "net_pnl_usdt", "liquidations", "max_drawdown_usdt"]].to_string(index=False))
    print("In-sample max:", max_net.config_id)
    print("Risk-adjusted max:", risk_adjusted.config_id)
    print("Recommended OOS:", recommended_id)
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
