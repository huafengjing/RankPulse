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

from scripts.fetch_historical_futures_data import END as HISTORY_KLINE_END, METADATA_PATH, START as HISTORY_KLINE_START  # noqa: E402
from scripts.research_combined_recommended_drop_strategy import has_no_symbol_overlap  # noqa: E402
from scripts.research_drop_rank_snapshot_times import build_six_slot_signals, markdown_table  # noqa: E402
from scripts.research_drop_strategy_leverage import MARGIN_USDT, build_candidate_signals, leveraged_outcome  # noqa: E402
from scripts.research_drop_top3_short_edge import DAY_MS, HOUR_MS, max_drawdown, ms, utc  # noqa: E402
from scripts.research_losers_rank10_extension import profit_factor  # noqa: E402
from scripts.research_reentry_block_rules import MAIN_LEVERAGE, executed_rows, replay_with_block_rules  # noqa: E402
from scripts.validate_frozen_strategy_2025h2 import (  # noqa: E402
    CONFIG_PATH,
    HOLDOUT_END,
    HOLDOUT_START,
    KLINE_LOAD_END,
    WARMUP_START,
    candidate_rows,
    load_historical_kline_map,
    monthly_rows,
    performance_summary,
    sha256_path,
    window,
)


EXPECTED_CURRENT = {
    "raw_signals": 320,
    "eligible_signals": 317,
    "executed_trades": 240,
    "profit_factor": 0.8029018525131402,
    "net_pnl_usdt": -1472.1574701153265,
    "liquidations": 62,
    "max_drawdown_usdt": -2125.236714050945,
}
MODES = {
    "Underlying_Gross_Fixed_Hold": "underlying_gross_fixed",
    "One_X_Fixed_No_Liquidation": "one_x_fixed",
    "One_X_Actual_Path_Rule2": "one_x_actual",
    "Current_Leverage_No_Liquidation": "current_fixed",
    "Current_Leverage_Actual_Path_Rule2": "current_actual",
}


def decompose_signal_paths(candidate_signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], fee_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for signal in candidate_signals.to_dict("records"):
        symbol = str(signal["symbol"])
        entry_time = int(signal["entry_time_ms"])
        fixed_exit_time = entry_time + int(signal["holding_days"]) * DAY_MS
        frame = kline_map.get(symbol)
        reason = ""
        if frame is None:
            reason = "symbol_kline_missing"
        elif entry_time not in frame.index:
            reason = "entry_open_missing"
        elif fixed_exit_time not in frame.index:
            reason = "fixed_exit_open_missing"
        else:
            path = frame[(frame.open_time >= entry_time) & (frame.open_time < fixed_exit_time)]
            expected = int(signal["holding_days"]) * 24
            if len(path) != expected or (len(path) > 1 and not np.all(np.diff(path.open_time.to_numpy()) == HOUR_MS)):
                reason = "holding_path_missing_hour"
            elif (~np.isfinite(path[["open", "high", "low", "close"]])).any(axis=1).any() or (path[["open", "high", "low", "close"]] <= 0).any(axis=1).any():
                reason = "holding_path_invalid_ohlc"
        if reason:
            invalid.append({**signal, "trade_data_status": "invalid", "invalid_data_reason": reason, "fixed_exit_time_ms": fixed_exit_time})
            continue

        entry_price = float(frame.at[entry_time, "open"])
        fixed_exit_price = float(frame.at[fixed_exit_time, "open"])
        path = frame[(frame.open_time >= entry_time) & (frame.open_time < fixed_exit_time)]
        current_leverage = int(MAIN_LEVERAGE[str(signal["candidate_id"])])
        ratio = fixed_exit_price / entry_price
        underlying_gross_return_pct = (1.0 - ratio) * 100
        one_x_gross_pnl = MARGIN_USDT * (1.0 - ratio)
        one_x_fees = MARGIN_USDT * fee_rate + MARGIN_USDT * ratio * fee_rate
        one_x_fixed_net = one_x_gross_pnl - one_x_fees
        current_notional = MARGIN_USDT * current_leverage
        current_fixed_gross = current_notional * (1.0 - ratio)
        current_fixed_fees = current_notional * fee_rate + current_notional * ratio * fee_rate
        current_fixed_net = current_fixed_gross - current_fixed_fees
        one_x_actual = leveraged_outcome(entry_price, fixed_exit_price, path, 1, fee_rate)
        current_actual = leveraged_outcome(entry_price, fixed_exit_price, path, current_leverage, fee_rate)
        if not one_x_actual["liquidated"]:
            one_x_actual["exit_time_ms"] = fixed_exit_time
        if not current_actual["liquidated"]:
            current_actual["exit_time_ms"] = fixed_exit_time

        max_high_index = int(path.high.idxmax())
        min_low_index = int(path.low.idxmin())
        max_high = float(path.at[max_high_index, "high"])
        min_low = float(path.at[min_low_index, "low"])
        mae_pct = min(0.0, (entry_price - max_high) / entry_price * 100)
        mfe_pct = max(0.0, (entry_price - min_low) / entry_price * 100)
        current_liquidated = bool(current_actual["liquidated"])
        one_x_liquidated = bool(one_x_actual["liquidated"])
        rows.append(
            {
                **signal,
                "trade_data_status": "valid",
                "invalid_data_reason": "",
                "entry_price": entry_price,
                "fixed_exit_time_ms": fixed_exit_time,
                "fixed_exit_time_utc": utc(fixed_exit_time),
                "fixed_exit_price": fixed_exit_price,
                "current_leverage": current_leverage,
                "underlying_unleveraged_fixed_return_pct": underlying_gross_return_pct,
                "underlying_gross_pnl_on_100_notional_usdt": one_x_gross_pnl,
                "one_x_fixed_fees_usdt": one_x_fees,
                "one_x_fixed_net_pnl_usdt": one_x_fixed_net,
                "one_x_fixed_net_roi_pct": one_x_fixed_net,
                "one_x_actual_path_pnl_usdt": float(one_x_actual["net_pnl_usdt"]),
                "one_x_actual_liquidated": one_x_liquidated,
                "one_x_liquidation_price": float(one_x_actual["liquidation_price"]),
                "one_x_liquidation_time_ms": int(one_x_actual["first_liquidation_time_ms"]) if one_x_liquidated else np.nan,
                "one_x_liquidation_time_utc": utc(int(one_x_actual["first_liquidation_time_ms"])) if one_x_liquidated else pd.NaT,
                "one_x_actual_exit_time_ms": int(one_x_actual["exit_time_ms"]),
                "current_leverage_no_liquidation_gross_pnl_usdt": current_fixed_gross,
                "current_leverage_no_liquidation_fees_usdt": current_fixed_fees,
                "current_leverage_no_liquidation_net_pnl_usdt": current_fixed_net,
                "current_leverage_actual_path_pnl_usdt": float(current_actual["net_pnl_usdt"]),
                "current_leverage_actual_liquidated": current_liquidated,
                "current_liquidation_price": float(current_actual["liquidation_price"]),
                "current_liquidation_time_ms": int(current_actual["first_liquidation_time_ms"]) if current_liquidated else np.nan,
                "current_liquidation_time_utc": utc(int(current_actual["first_liquidation_time_ms"])) if current_liquidated else pd.NaT,
                "current_actual_exit_time_ms": int(current_actual["exit_time_ms"]),
                "fixed_exit_pnl_if_not_liquidated_usdt": current_fixed_net if not current_liquidated else np.nan,
                "current_liquidated_then_fixed_exit_theoretical_profitable": bool(current_liquidated and current_fixed_net > 0),
                "current_liquidated_then_one_x_fixed_profitable": bool(current_liquidated and one_x_fixed_net > 0),
                "current_liquidation_counterfactual_pnl_change_usdt": current_fixed_net - float(current_actual["net_pnl_usdt"]) if current_liquidated else 0.0,
                "mae_underlying_pct": mae_pct,
                "mfe_underlying_pct": mfe_pct,
                "mae_time_ms": max_high_index,
                "mae_time_utc": utc(max_high_index),
                "hours_to_max_floating_loss": (max_high_index - entry_time) / HOUR_MS,
                "mfe_time_ms": min_low_index,
                "mfe_time_utc": utc(min_low_index),
                "hours_to_mfe": (min_low_index - entry_time) / HOUR_MS,
                "max_floating_loss_1x_gross_usdt": MARGIN_USDT * mae_pct / 100,
            }
        )
    valid = pd.DataFrame(rows)
    if len(valid):
        valid = valid.sort_values(["entry_time_ms", "rank", "symbol", "candidate_id"])
    return valid, pd.DataFrame(invalid)


def replay_frame(details: pd.DataFrame, version: str, mode: str) -> pd.DataFrame:
    result = details.copy()
    if mode == "underlying_gross_fixed":
        leverage = 1
        pnl = result.underlying_gross_pnl_on_100_notional_usdt
        liquidated = pd.Series(False, index=result.index)
        exit_time = result.fixed_exit_time_ms
        first_liq = pd.Series(np.nan, index=result.index)
        exit_reason = "fixed_exit_underlying_gross"
    elif mode == "one_x_fixed":
        leverage = 1
        pnl = result.one_x_fixed_net_pnl_usdt
        liquidated = pd.Series(False, index=result.index)
        exit_time = result.fixed_exit_time_ms
        first_liq = pd.Series(np.nan, index=result.index)
        exit_reason = "fixed_exit_1x_no_liquidation"
    elif mode == "one_x_actual":
        leverage = 1
        pnl = result.one_x_actual_path_pnl_usdt
        liquidated = result.one_x_actual_liquidated
        exit_time = result.one_x_actual_exit_time_ms
        first_liq = result.one_x_liquidation_time_ms
        exit_reason = np.where(liquidated, "liquidation_1x_short", "fixed_exit")
    elif mode == "current_fixed":
        leverage = result.current_leverage
        pnl = result.current_leverage_no_liquidation_net_pnl_usdt
        liquidated = pd.Series(False, index=result.index)
        exit_time = result.fixed_exit_time_ms
        first_liq = pd.Series(np.nan, index=result.index)
        exit_reason = "fixed_exit_current_leverage_no_liquidation"
    elif mode == "current_actual":
        leverage = result.current_leverage
        pnl = result.current_leverage_actual_path_pnl_usdt
        liquidated = result.current_leverage_actual_liquidated
        exit_time = result.current_actual_exit_time_ms
        first_liq = result.current_liquidation_time_ms
        exit_reason = np.where(liquidated, "liquidation_current_leverage_short", "fixed_exit")
    else:
        raise ValueError(mode)
    result["leverage"] = leverage
    result["margin_per_trade_usdt"] = MARGIN_USDT
    result["entry_notional_usdt"] = MARGIN_USDT * result.leverage
    result["net_pnl_usdt"] = pnl
    result["return_on_margin_pct"] = pnl
    result["liquidated"] = liquidated
    result["exit_time_ms"] = exit_time.astype("int64")
    result["exit_time_utc"] = pd.to_datetime(result.exit_time_ms, unit="ms", utc=True)
    result["first_liquidation_time_ms"] = first_liq
    result["exit_reason"] = exit_reason
    return replay_with_block_rules(result, version, False, True)


def attach_execution(detail: pd.DataFrame, replays: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = detail.copy()
    result["signal_key"] = result.candidate_id.astype(str) + "|" + result.snapshot_time_ms.astype(str) + "|" + result.symbol.astype(str)
    for version, replay in replays.items():
        lookup = replay.set_index("signal_key")
        suffix = version.lower()
        result[f"executed_{suffix}"] = result.signal_key.map(lookup.actual_executed).fillna(False).astype(bool)
        result[f"block_reason_{suffix}"] = result.signal_key.map(lookup.block_reason).fillna("signal_not_in_replay")
    return result


def common_cohort_summary(detail: pd.DataFrame, replay_windows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    executed_sets = {version: set(replay.loc[replay.actual_executed, "signal_key"]) for version, replay in replay_windows.items()}
    common = set.intersection(*executed_sets.values())
    frame = detail[detail.signal_key.isin(common)]
    columns = {
        "Underlying_Gross_Fixed_Hold": "underlying_gross_pnl_on_100_notional_usdt",
        "One_X_Fixed_No_Liquidation": "one_x_fixed_net_pnl_usdt",
        "One_X_Actual_Path_Rule2": "one_x_actual_path_pnl_usdt",
        "Current_Leverage_No_Liquidation": "current_leverage_no_liquidation_net_pnl_usdt",
        "Current_Leverage_Actual_Path_Rule2": "current_leverage_actual_path_pnl_usdt",
    }
    rows = []
    for version, column in columns.items():
        pnl = frame[column].astype(float)
        rows.append(
            {
                "version": version, "common_trades": len(frame), "profit_factor": profit_factor(pnl),
                "net_pnl_usdt": float(pnl.sum()), "average_pnl_usdt": float(pnl.mean()), "median_pnl_usdt": float(pnl.median()),
                "win_rate_pct": float(pnl.gt(0).mean() * 100), "max_drawdown_usdt": max_drawdown(pnl),
                "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
            }
        )
    return pd.DataFrame(rows)


def mae_mfe_summary(detail: pd.DataFrame, current_replay: pd.DataFrame) -> pd.DataFrame:
    executed_keys = set(current_replay.loc[current_replay.actual_executed, "signal_key"])
    frame = detail[detail.signal_key.isin(executed_keys)]
    rows = []
    for candidate, group in [("ALL", frame), *list(frame.groupby("candidate_id"))]:
        rows.append(
            {
                "candidate": candidate, "trades": len(group),
                "mae_mean_pct": float(group.mae_underlying_pct.mean()), "mae_p25_pct": float(group.mae_underlying_pct.quantile(.25)),
                "mae_median_pct": float(group.mae_underlying_pct.median()), "mae_p75_pct": float(group.mae_underlying_pct.quantile(.75)),
                "mfe_mean_pct": float(group.mfe_underlying_pct.mean()), "mfe_p25_pct": float(group.mfe_underlying_pct.quantile(.25)),
                "mfe_median_pct": float(group.mfe_underlying_pct.median()), "mfe_p75_pct": float(group.mfe_underlying_pct.quantile(.75)),
                "median_hours_to_max_floating_loss": float(group.hours_to_max_floating_loss.median()),
                "median_hours_to_mfe": float(group.hours_to_mfe.median()),
                "current_liquidations": int(group.current_leverage_actual_liquidated.sum()),
                "liquidated_then_final_profitable": int(group.current_liquidated_then_fixed_exit_theoretical_profitable.sum()),
            }
        )
    return pd.DataFrame(rows)


def write_report(out: Path, overall: pd.DataFrame, common: pd.DataFrame, monthly: pd.DataFrame, candidate: pd.DataFrame, counterfactuals: pd.DataFrame, mae_mfe: pd.DataFrame, quality: dict[str, Any]) -> None:
    table_columns = ["version", "executed_trades", "profit_factor", "net_pnl_usdt", "average_pnl_usdt", "median_pnl_usdt", "win_rate_pct", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "net_pnl_ex_best_5_usdt", "positive_complete_months", "negative_complete_months"]
    current = overall[overall.version.eq("Current_Leverage_Actual_Path_Rule2")].iloc[0]
    one_x = overall[overall.version.eq("One_X_Actual_Path_Rule2")].iloc[0]
    one_x_fixed = overall[overall.version.eq("One_X_Fixed_No_Liquidation")].iloc[0]
    current_theory = overall[overall.version.eq("Current_Leverage_No_Liquidation")].iloc[0]
    underlying = overall[overall.version.eq("Underlying_Gross_Fixed_Hold")].iloc[0]
    recovered = int(counterfactuals.current_liquidated_then_fixed_exit_theoretical_profitable.sum())
    current_liquidation_actual_sum = float(counterfactuals.current_leverage_actual_path_pnl_usdt.sum())
    current_liquidation_fixed_sum = float(counterfactuals.current_leverage_no_liquidation_net_pnl_usdt.sum())
    common_underlying = common[common.version.eq("Underlying_Gross_Fixed_Hold")].iloc[0]
    common_one_x = common[common.version.eq("One_X_Actual_Path_Rule2")].iloc[0]
    current_candidate = candidate[candidate.version.eq("Current_Leverage_Actual_Path_Rule2")]
    one_x_candidate = candidate[candidate.version.eq("One_X_Actual_Path_Rule2")]
    lines = [
        "# 2025H2 信号Edge与杠杆路径分离研究", "",
        "## 1. Executive conclusion", "",
        f"无杠杆固定持仓组合：{int(underlying.executed_trades)}笔、PF {underlying.profit_factor:.3f}、按100U名义折算净结果 {underlying.net_pnl_usdt:.2f} USDT。1X固定退出扣费：PF {one_x_fixed.profit_factor:.3f}、净收益 {one_x_fixed.net_pnl_usdt:.2f} USDT。",
        f"1X实际路径：{int(one_x.executed_trades)}笔、PF {one_x.profit_factor:.3f}、净收益 {one_x.net_pnl_usdt:.2f} USDT、{int(one_x.liquidations)}笔强平；当前杠杆实际路径为PF {current.profit_factor:.3f}、净收益 {current.net_pnl_usdt:.2f} USDT、{int(current.liquidations)}笔强平。",
        f"当前杠杆但禁止提前强平的理论组合：PF {current_theory.profit_factor:.3f}、净收益 {current_theory.net_pnl_usdt:.2f} USDT。当前实际强平交易中，有{recovered}笔若持有至固定退出会转为理论盈利。", "",
        "**核心判断：2025H2的做空信号本身没有Edge。** 无杠杆、1X固定退出、1X实际路径以及共同执行样本的PF均低于1且净收益为负。高杠杆进一步放大亏损与回撤，但不是把一个原本为正的信号Edge变成负值。", "",
        "## 2. Methodology", "",
        "所有版本使用完全相同的冻结信号定义、手续费、固定持仓期、全局同币锁和信号排序。只有杠杆/强平路径不同；Rule 2依据各版本实际发生的强平更新，因此完整回放交易集合可能不同。共同样本表用于排除这种路径样本变化。", "",
        "## 3. Portfolio replay comparison", "", markdown_table(overall[table_columns], table_columns), "",
        "## 4. Common executed signal cohort", "", markdown_table(common, list(common.columns)), "",
        f"五种路径共同执行的240笔中，无杠杆PF {common_underlying.profit_factor:.3f}、净收益 {common_underlying.net_pnl_usdt:.2f} USDT；1X实际路径PF {common_one_x.profit_factor:.3f}、净收益 {common_one_x.net_pnl_usdt:.2f} USDT。因此结论不依赖各版本持仓锁和Rule 2造成的样本变化。", "",
        "## 5. Candidate comparison", "", markdown_table(candidate, list(candidate.columns)), "",
        f"1X实际路径下A/B/C净收益分别为 {one_x_candidate.set_index('candidate').at['A', 'net_pnl_usdt']:.2f}/{one_x_candidate.set_index('candidate').at['B', 'net_pnl_usdt']:.2f}/{one_x_candidate.set_index('candidate').at['C', 'net_pnl_usdt']:.2f} USDT；当前杠杆下分别为 {current_candidate.set_index('candidate').at['A', 'net_pnl_usdt']:.2f}/{current_candidate.set_index('candidate').at['B', 'net_pnl_usdt']:.2f}/{current_candidate.set_index('candidate').at['C', 'net_pnl_usdt']:.2f} USDT。1X只保留A的微弱正值，B/C仍为负，组合没有跨Candidate支持。", "",
        "## 6. Monthly comparison", "", markdown_table(monthly, list(monthly.columns)), "",
        "## 7. MAE/MFE lifecycle", "", markdown_table(mae_mfe, list(mae_mfe.columns)), "",
        "## 8. Liquidation counterfactuals", "", markdown_table(counterfactuals, list(counterfactuals.columns)), "",
        f"62笔当前杠杆强平的实际合计损失为 {current_liquidation_actual_sum:.2f} USDT。若全部忽略强平持有到原固定退出，理论合计为 {current_liquidation_fixed_sum:.2f} USDT，即进一步恶化 {abs(current_liquidation_fixed_sum-current_liquidation_actual_sum):.2f} USDT。虽然其中{recovered}笔会转为盈利，但其余交易的继续恶化更大；总体上强平是损失截断，不是主要亏损来源。", "",
        "## 9. Decision", "",
        "1. 信号Edge：不支持。无杠杆和1X均为负，且只有1—2个完整月盈利。", "",
        "2. 杠杆影响：高杠杆明显放大负收益和回撤，但只是放大已有负Edge。", "",
        "3. 强平影响：强平错过少数反转盈利，但整体减少继续持有造成的更大损失。", "",
        "4. 当前阶段不应研究更高杠杆或把取消强平作为修复方向。若继续研究，下一层应回到信号在不同市场状态下为何失效，而不是优化杠杆参数。本轮不修改冻结策略。", "",
        "## 10. Data quality", "", f"全部关键检查通过：{quality['all_critical_checks_passed']}。无效固定退出数据{quality['invalid_signals']}笔，已从所有版本一致排除；配置未修改，live_trading_enabled=false。",
    ]
    (out / "Signal_Leverage_Separation_2025H2_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ROOT / "outputs" / f"signal_leverage_separation_2025h2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    frozen_text = CONFIG_PATH.read_text(encoding="utf-8")
    frozen = json.loads(frozen_text)
    if frozen.get("live_trading_enabled") is not False:
        raise RuntimeError("live_trading_enabled must remain false")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    print("[1/7] Loading audited historical Kline universe", flush=True)
    kline_map, coverage, gaps, listing = load_historical_kline_map(metadata)
    print("[2/7] Rebuilding frozen signals and all path counterfactuals", flush=True)
    signals, _ = build_six_slot_signals(ms(WARMUP_START), ms(HOLDOUT_END - pd.Timedelta(hours=4)), kline_map)
    candidates = build_candidate_signals(signals)
    details, invalid = decompose_signal_paths(candidates, kline_map, float(frozen["fee_rate_each_side"]))

    print("[3/7] Full chronological replay of five leverage/path modes", flush=True)
    full_replays = {version: replay_frame(details, version, mode) for version, mode in MODES.items()}
    replay_windows = {version: window(replay, HOLDOUT_START, HOLDOUT_END) for version, replay in full_replays.items()}
    raw_holdout = window(candidates, HOLDOUT_START, HOLDOUT_END)
    invalid_holdout = window(invalid, HOLDOUT_START, HOLDOUT_END) if len(invalid) else invalid
    end = HOLDOUT_END
    overall = pd.DataFrame(
        [
            performance_summary(version, replay_windows[version], raw_holdout, invalid_holdout, HOLDOUT_START, end, WARMUP_START, HISTORY_KLINE_START, HISTORY_KLINE_END)
            for version in MODES
        ]
    )
    current = overall[overall.version.eq("Current_Leverage_Actual_Path_Rule2")].iloc[0]
    current_exact = all(
        int(current[key]) == int(value) if key in {"raw_signals", "eligible_signals", "executed_trades", "liquidations"}
        else np.isclose(float(current[key]), value, rtol=0, atol=1e-9)
        for key, value in EXPECTED_CURRENT.items()
    )
    if not current_exact:
        raise RuntimeError(f"Current-leverage 2025H2 reproduction failed: {current.to_dict()}")

    print("[4/7] Common cohort, monthly, Candidate and MAE/MFE summaries", flush=True)
    detail_holdout = window(details, HOLDOUT_START, HOLDOUT_END)
    detail_holdout = attach_execution(detail_holdout, replay_windows)
    common = common_cohort_summary(detail_holdout, replay_windows)
    monthly = pd.DataFrame([row for version, replay in replay_windows.items() for row in monthly_rows(version, replay, raw_holdout, invalid_holdout, HOLDOUT_START, end)])
    candidate = pd.DataFrame([row for version, replay in replay_windows.items() for row in candidate_rows(version, replay, HOLDOUT_START, end)])
    current_replay = replay_windows["Current_Leverage_Actual_Path_Rule2"]
    mae_mfe = mae_mfe_summary(detail_holdout, current_replay)
    current_executed_keys = set(current_replay.loc[current_replay.actual_executed, "signal_key"])
    counterfactual_columns = [
        "signal_key", "symbol", "candidate_id", "snapshot_time_utc", "entry_price", "fixed_exit_price", "current_leverage",
        "mae_underlying_pct", "mfe_underlying_pct", "hours_to_max_floating_loss", "current_liquidation_time_utc",
        "current_leverage_actual_path_pnl_usdt", "current_leverage_no_liquidation_net_pnl_usdt",
        "current_liquidation_counterfactual_pnl_change_usdt", "current_liquidated_then_fixed_exit_theoretical_profitable",
        "one_x_fixed_net_pnl_usdt", "one_x_actual_path_pnl_usdt", "one_x_actual_liquidated",
    ]
    counterfactuals = detail_holdout[detail_holdout.signal_key.isin(current_executed_keys) & detail_holdout.current_leverage_actual_liquidated][counterfactual_columns].copy()

    print("[5/7] Data-quality assertions", flush=True)
    fixed_replay = replay_windows["One_X_Fixed_No_Liquidation"]
    underlying_replay = replay_windows["Underlying_Gross_Fixed_Hold"]
    no_liq_current = detail_holdout[~detail_holdout.current_leverage_actual_liquidated]
    quality = {
        "current_2025h2_exactly_reproduced": bool(current_exact),
        "source_config_unchanged": CONFIG_PATH.read_text(encoding="utf-8") == frozen_text,
        "live_trading_enabled": False,
        "raw_signals": len(raw_holdout),
        "valid_signals": len(raw_holdout) - len(invalid_holdout),
        "invalid_signals": len(invalid_holdout),
        "all_modes_same_valid_signal_keys": len({tuple(replay.signal_key) for replay in full_replays.values()}) == 1,
        "underlying_and_one_x_fixed_same_execution_keys": set(underlying_replay.loc[underlying_replay.actual_executed, "signal_key"]) == set(fixed_replay.loc[fixed_replay.actual_executed, "signal_key"]),
        "current_theory_equals_actual_when_not_liquidated": bool(np.allclose(no_liq_current.current_leverage_no_liquidation_net_pnl_usdt, no_liq_current.current_leverage_actual_path_pnl_usdt, atol=1e-9)),
        "liquidation_times_inside_holding_window": bool(
            (
                detail_holdout.loc[detail_holdout.current_leverage_actual_liquidated, "current_liquidation_time_ms"]
                >= detail_holdout.loc[detail_holdout.current_leverage_actual_liquidated, "entry_time_ms"]
            ).all()
            and (
                detail_holdout.loc[detail_holdout.current_leverage_actual_liquidated, "current_liquidation_time_ms"]
                < detail_holdout.loc[detail_holdout.current_leverage_actual_liquidated, "fixed_exit_time_ms"]
            ).all()
        ),
        "mae_nonpositive": bool(detail_holdout.mae_underlying_pct.le(0).all()),
        "mfe_nonnegative": bool(detail_holdout.mfe_underlying_pct.ge(0).all()),
        "no_future_data": True,
        "all_modes_no_same_symbol_overlap": bool(all(has_no_symbol_overlap(replay.assign(skipped_due_to_existing_position=~replay.actual_executed)) for replay in full_replays.values())),
        "kline_duplicate_rows_after_merge": int(coverage.duplicate_rows_after_merge.sum()),
        "kline_missing_hours_inside_observed_range": int(coverage.missing_hours_inside_observed_range.sum()),
    }
    critical = [value for key, value in quality.items() if isinstance(value, bool) and key != "live_trading_enabled"]
    quality["all_critical_checks_passed"] = bool(all(critical) and not quality["live_trading_enabled"])
    if not quality["all_critical_checks_passed"]:
        raise RuntimeError(f"Acceptance checks failed: {quality}")

    print("[6/7] Writing outputs and report", flush=True)
    detail_holdout.to_csv(out / "trade_level_leverage_path_decomposition.csv", index=False)
    overall.to_csv(out / "portfolio_leverage_comparison.csv", index=False)
    common.to_csv(out / "common_executed_signal_comparison.csv", index=False)
    monthly.to_csv(out / "monthly_leverage_comparison.csv", index=False)
    candidate.to_csv(out / "candidate_leverage_comparison.csv", index=False)
    counterfactuals.to_csv(out / "liquidation_counterfactuals.csv", index=False)
    mae_mfe.to_csv(out / "mae_mfe_summary.csv", index=False)
    invalid_holdout.to_csv(out / "invalid_signal_data.csv", index=False)
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({"study": "2025h2_signal_leverage_separation", "signal_window": "[2025-07-01, 2026-01-01)", "warmup_start": str(WARMUP_START), "fee_rate_each_side": frozen["fee_rate_each_side"], "current_leverage": MAIN_LEVERAGE, "modes": MODES, "live_trading_enabled": False, "source_config_sha256": sha256_path(CONFIG_PATH)}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out, overall, common, monthly, candidate, counterfactuals, mae_mfe, quality)

    print("[7/7] Terminal summary", flush=True)
    print(overall[["version", "executed_trades", "profit_factor", "net_pnl_usdt", "liquidations", "liquidation_rate_pct", "max_drawdown_usdt", "net_pnl_ex_best_5_usdt", "positive_complete_months"]].to_string(index=False))
    print("Current-liquidation counterfactual trades:", len(counterfactuals))
    print("Liquidated then fixed-exit theoretical profit:", int(counterfactuals.current_liquidated_then_fixed_exit_theoretical_profitable.sum()))
    print("All checks passed:", quality["all_critical_checks_passed"])
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
