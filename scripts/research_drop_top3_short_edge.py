from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CACHE_DIR = ROOT / "data" / "futures_klines_1h"
OUT = ROOT / "outputs" / "drop_top3_short_edge_2026"
START_UTC = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
SNAPSHOT_UTC_HOURS = (0, 16)  # Beijing 08:00 and 00:00.
HOLD_DAYS = tuple(range(1, 8))
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
NOTIONAL_USDT = 100.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0

SUMMARY_COLUMNS = [
    "trades",
    "wins",
    "losses",
    "liquidations",
    "net_pnl_usdt",
    "profit_factor",
    "win_rate_pct",
    "average_return_pct",
    "median_return_pct",
    "max_drawdown_usdt",
    "max_trade_profit_usdt",
    "max_trade_loss_usdt",
    "net_pnl_ex_best_1_usdt",
    "net_pnl_ex_best_3_usdt",
    "net_pnl_ex_best_5_usdt",
]


def ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def utc(time_ms: int) -> pd.Timestamp:
    return pd.to_datetime(time_ms, unit="ms", utc=True)


def drop_bucket(drop: float) -> str:
    if drop <= 0:
        return "no_drop_or_gain"
    if drop < 0.10:
        return "0~10%"
    if drop < 0.20:
        return "10~20%"
    if drop < 0.40:
        return "20~40%"
    if drop < 0.60:
        return "40~60%"
    if drop < 0.80:
        return "60~80%"
    return ">=80%"


DROP_BUCKET_ORDER = ["no_drop_or_gain", "0~10%", "10~20%", "20~40%", "40~60%", "60~80%", ">=80%"]
RETURN_BINS = [-np.inf, -50, -30, -20, -10, 0, 10, 20, 50, np.inf]
RETURN_LABELS = ["<-50%", "-50~-30%", "-30~-20%", "-20~-10%", "-10~0%", "0~10%", "10~20%", "20~50%", ">50%"]


def load_kline_map() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    files = sorted(CACHE_DIR.glob("*_1h.csv"))
    if not files:
        raise RuntimeError(f"No 1H cache files found under {CACHE_DIR}")
    earliest_needed = ms(START_UTC) - 26 * HOUR_MS
    frames: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    for path in files:
        symbol = path.stem.removesuffix("_1h")
        frame = pd.read_csv(
            path,
            usecols=["open_time", "open", "high", "low", "close"],
            dtype={"open_time": "int64", "open": "float64", "high": "float64", "low": "float64", "close": "float64"},
        )
        frame = frame[frame["open_time"] >= earliest_needed].drop_duplicates("open_time", keep="last").sort_values("open_time")
        invalid = (~np.isfinite(frame[["open", "high", "low", "close"]])).any(axis=1) | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        frame = frame[~invalid].set_index("open_time", drop=False)
        if frame.empty:
            continue
        frames[symbol] = frame
        expected = int((int(frame["open_time"].max()) - int(frame["open_time"].min())) / HOUR_MS) + 1
        audit_rows.append(
            {
                "symbol": symbol,
                "rows": len(frame),
                "start_utc": utc(int(frame["open_time"].min())),
                "end_utc": utc(int(frame["open_time"].max())),
                "missing_hour_count": max(0, expected - len(frame)),
                "invalid_rows_removed": int(invalid.sum()),
            }
        )
    return frames, pd.DataFrame(audit_rows)


def snapshot_rankings(snapshot_time: int, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    current_open = snapshot_time - HOUR_MS
    prior_open = current_open - 24 * HOUR_MS
    rows: list[dict[str, Any]] = []
    for symbol, frame in kline_map.items():
        if current_open not in frame.index or prior_open not in frame.index:
            continue
        current_close = float(frame.at[current_open, "close"])
        prior_close = float(frame.at[prior_open, "close"])
        if prior_close <= 0 or not np.isfinite(current_close) or not np.isfinite(prior_close):
            continue
        rows.append(
            {
                "symbol": symbol,
                "current_close": current_close,
                "close_24h_ago": prior_close,
                "change_24h": current_close / prior_close - 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_signals(
    start_time: int,
    end_time: int,
    kline_map: dict[str, pd.DataFrame],
    direction: Literal["short", "long"],
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for day in pd.date_range(utc(start_time).floor("D"), utc(end_time).floor("D"), freq="D", tz="UTC"):
        for hour in SNAPSHOT_UTC_HOURS:
            signal_time = ms(day + pd.Timedelta(hours=hour))
            if signal_time < start_time or signal_time > end_time:
                continue
            ranking = snapshot_rankings(signal_time, kline_map)
            audit.append({"direction": direction, "signal_time_utc": utc(signal_time), "ranked_symbols": len(ranking)})
            if ranking.empty:
                continue
            ascending = direction == "short"
            selected = ranking.sort_values(["change_24h", "symbol"], ascending=[ascending, True]).head(top_n)
            for rank, (_, item) in enumerate(selected.iterrows(), start=1):
                change = float(item["change_24h"])
                rows.append(
                    {
                        "direction": direction,
                        "signal_time_ms": signal_time,
                        "signal_time_utc": utc(signal_time),
                        "signal_time_bj": utc(signal_time).tz_convert("Asia/Shanghai"),
                        "snapshot_hour_bj": utc(signal_time).tz_convert("Asia/Shanghai").strftime("%H:%M"),
                        "symbol": str(item["symbol"]),
                        "rank": rank,
                        "current_close": float(item["current_close"]),
                        "close_24h_ago": float(item["close_24h_ago"]),
                        "change_24h_pct": change * 100,
                        "drop_24h_pct": -change * 100,
                        "drop_bucket": drop_bucket(-change),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(audit)


def trade_return(direction: Literal["short", "long"], entry: float, exit_: float) -> tuple[float, float, float]:
    ratio = exit_ / entry
    gross = 1.0 - ratio if direction == "short" else ratio - 1.0
    fees = FEE_RATE + FEE_RATE * ratio
    net = gross - fees - 2 * SLIPPAGE_RATE
    return gross, fees, net


def path_excursions(direction: Literal["short", "long"], path: pd.DataFrame, entry: float) -> tuple[float, float]:
    if path.empty:
        return np.nan, np.nan
    if direction == "short":
        return (1.0 - float(path["low"].min()) / entry) * 100, (1.0 - float(path["high"].max()) / entry) * 100
    return (float(path["high"].max()) / entry - 1.0) * 100, (float(path["low"].min()) / entry - 1.0) * 100


def simulate_trades(
    signals: pd.DataFrame,
    holding_days: int,
    kline_map: dict[str, pd.DataFrame],
    direction: Literal["short", "long"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    open_until: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time_ms", "rank", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        entry_time = int(signal["signal_time_ms"])
        planned_exit_time = entry_time + holding_days * DAY_MS
        if entry_time < open_until.get(symbol, -1):
            skipped.append({**signal.to_dict(), "holding_days": holding_days, "skip_reason": "symbol_already_open"})
            continue
        frame = kline_map.get(symbol)
        if frame is None or entry_time not in frame.index or planned_exit_time not in frame.index:
            skipped.append({**signal.to_dict(), "holding_days": holding_days, "skip_reason": "missing_entry_or_exit_open"})
            continue
        entry = float(frame.at[entry_time, "open"])
        path = frame[(frame["open_time"] >= entry_time) & (frame["open_time"] < planned_exit_time)]
        liquidation_hits = path[path["high"] >= entry * 2.0] if direction == "short" else pd.DataFrame()
        liquidated = not liquidation_hits.empty
        if liquidated:
            exit_time = int(liquidation_hits.iloc[0]["open_time"])
            exit_ = entry * 2.0
            gross, fees, net = -1.0, np.nan, -1.0
            path = path[path["open_time"] <= exit_time]
            exit_reason = "liquidation_1x_short"
        else:
            exit_time = planned_exit_time
            exit_ = float(frame.at[exit_time, "open"])
            gross, fees, net = trade_return(direction, entry, exit_)
            exit_reason = f"fixed_{holding_days}d"
        mfe, mae = path_excursions(direction, path, entry)
        completed.append(
            {
                **signal.to_dict(),
                "holding_days": holding_days,
                "entry_time_utc": utc(entry_time),
                "exit_time_utc": utc(exit_time),
                "entry_price": entry,
                "exit_price": exit_,
                "gross_return_pct": gross * 100,
                "fee_pct": fees * 100,
                "net_return_pct": net * 100,
                "pnl_usdt": net * NOTIONAL_USDT,
                "is_win": net > 0,
                "liquidated": liquidated,
                "exit_reason": exit_reason,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "month": utc(entry_time).strftime("%Y-%m"),
            }
        )
        open_until[symbol] = exit_time
    return pd.DataFrame(completed), pd.DataFrame(skipped)


def profit_factor(pnl: pd.Series) -> float:
    loss = abs(float(pnl[pnl < 0].sum()))
    return math.inf if loss == 0 and float(pnl[pnl > 0].sum()) > 0 else (float(pnl[pnl > 0].sum()) / loss if loss else np.nan)


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    return float((equity - equity.cummax()).min())


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values(["exit_time_utc", "rank", "symbol"]) if not group.empty else group
    pnl = ordered["pnl_usdt"].astype(float) if not ordered.empty else pd.Series(dtype=float)
    ret = ordered["net_return_pct"].astype(float) if not ordered.empty else pd.Series(dtype=float)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    return {
        "trades": len(group),
        "wins": wins,
        "losses": losses,
        "liquidations": int(group["liquidated"].sum()) if "liquidated" in group else 0,
        "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "win_rate_pct": wins / len(group) * 100 if len(group) else np.nan,
        "average_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "max_drawdown_usdt": max_drawdown(pnl),
        "max_trade_profit_usdt": float(pnl.max()) if len(pnl) else np.nan,
        "max_trade_loss_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(5).sum()) if len(pnl) >= 5 else np.nan,
    }


def grouped_summary(trades: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in trades.groupby(columns, sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(columns, keys)) | summarize(group))
    return pd.DataFrame(rows)


def consecutive_loss_outputs(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hold, group in trades.groupby("holding_days", sort=True):
        ordered = group.sort_values(["exit_time_utc", "rank", "symbol"])
        streaks: list[int] = []
        current = 0
        for loss in ordered["pnl_usdt"].lt(0):
            if loss:
                current += 1
            elif current:
                streaks.append(current)
                current = 0
        if current:
            streaks.append(current)
        rows.append(
            {
                "holding_days": hold,
                "row_type": "summary",
                "streak_length": np.nan,
                "occurrences": len(streaks),
                "max_consecutive_losses": max(streaks, default=0),
                "average_consecutive_losses": float(np.mean(streaks)) if streaks else 0.0,
            }
        )
        for length, count in pd.Series(streaks, dtype="int64").value_counts().sort_index().items():
            rows.append(
                {
                    "holding_days": hold,
                    "row_type": "distribution",
                    "streak_length": int(length),
                    "occurrences": int(count),
                    "max_consecutive_losses": np.nan,
                    "average_consecutive_losses": np.nan,
                }
            )
    return pd.DataFrame(rows)


def return_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hold, group in trades.groupby("holding_days", sort=True):
        bins = pd.cut(group["net_return_pct"], bins=RETURN_BINS, labels=RETURN_LABELS, right=False)
        counts = bins.value_counts(sort=False)
        for label in RETURN_LABELS:
            count = int(counts.get(label, 0))
            rows.append({"holding_days": hold, "return_bin": label, "trades": count, "share_pct": count / len(group) * 100 if len(group) else 0.0})
    return pd.DataFrame(rows)


def mae_mfe_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hold, group in trades.groupby("holding_days", sort=True):
        for metric, col in [("MFE", "mfe_pct"), ("MAE", "mae_pct")]:
            values = group[col].dropna().astype(float)
            rows.append(
                {
                    "horizon_hours": int(hold) * 24,
                    "metric": metric,
                    "trades": len(values),
                    "mean_pct": float(values.mean()),
                    "p25_pct": float(values.quantile(0.25)),
                    "median_pct": float(values.median()),
                    "p75_pct": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def add_month_consistency(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, hold), group in trades.groupby(["direction", "holding_days"], sort=True):
        monthly = grouped_summary(group, ["month"])
        rows.append(
            {
                "direction": direction,
                "holding_days": hold,
                "positive_months": int((monthly["net_pnl_usdt"] > 0).sum()),
                "months_tested": len(monthly),
                "months_pf_gt_1": int((monthly["profit_factor"] > 1).sum()),
            }
        )
    return summary.merge(pd.DataFrame(rows), on=["direction", "holding_days"], how="left")


def fmt(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if np.isinf(value):
        return "inf"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame[columns].copy() if columns else frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(fmt)
    return "\n".join(
        [
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join("---" for _ in view.columns) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy().tolist()],
        ]
    )


def write_report(
    metadata: dict[str, Any],
    overall: pd.DataFrame,
    holding: pd.DataFrame,
    rank: pd.DataFrame,
    bucket: pd.DataFrame,
    monthly: pd.DataFrame,
    streaks: pd.DataFrame,
    mae_mfe: pd.DataFrame,
    short_trades: pd.DataFrame,
) -> None:
    short_overall = overall[overall["direction"].eq("short")].copy()
    strongest = short_overall.sort_values(["profit_factor", "net_pnl_usdt"], ascending=False).iloc[0]
    rank_best = rank.sort_values(["profit_factor", "net_pnl_usdt"], ascending=False).iloc[0]
    rank_best_monthly = short_trades[
        short_trades["holding_days"].eq(rank_best["holding_days"]) & short_trades["rank"].eq(rank_best["rank"])
    ].groupby("month")["pnl_usdt"].sum()
    bucket_eligible = bucket[bucket["trades"] >= 20]
    bucket_best = bucket_eligible.sort_values(["profit_factor", "net_pnl_usdt"], ascending=False).iloc[0] if not bucket_eligible.empty else None
    comparison = overall.pivot(index="holding_days", columns="direction", values=["profit_factor", "net_pnl_usdt", "max_drawdown_usdt", "positive_months", "max_consecutive_losses"]).reset_index()
    comparison.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col) for col in comparison.columns]
    lines = [
        "# Binance Futures 24H 跌幅榜 Top3 做空基础 Edge 研究",
        "",
        f"生成时间：{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 结论",
        "",
        "**当前数据不支持“跌幅榜 Top3 整体存在稳定、可持续的做空 Edge”。** 3D 是唯一同时满足 PF>1、去最佳 5 笔后仍盈利且过半月份盈利的候选周期，但优势较弱且存在明显时段集中，不能据此确认策略有效。",
        "",
        f"按 PF 排序的最强整体持仓是 {int(strongest['holding_days'])}D：PF {fmt(strongest['profit_factor'])}，净收益 {fmt(strongest['net_pnl_usdt'])} USDT，去掉最佳 5 笔后 {fmt(strongest['net_pnl_ex_best_5_usdt'])} USDT，正收益月份 {int(strongest['positive_months'])}/{int(strongest['months_tested'])}。但 2026-04 至 2026-06 连续三个月亏损，2026-07 又只是截至 7 月 11 日的部分月，月度稳定性不足。",
        f"按单个 Rank×持仓组合排序，最强为 Rank{int(rank_best['rank'])} / {int(rank_best['holding_days'])}D：PF {fmt(rank_best['profit_factor'])}，交易 {int(rank_best['trades'])} 笔，净收益 {fmt(rank_best['net_pnl_usdt'])} USDT，去最佳 5 笔后 {fmt(rank_best['net_pnl_ex_best_5_usdt'])} USDT，正收益月份 {int((rank_best_monthly > 0).sum())}/{len(rank_best_monthly)}。这是最值得预注册后做样本外验证的候选，不是已经确认的过滤规则。",
        (f"至少 20 笔样本的跌幅桶中，表面最强为 {bucket_best['drop_bucket']} / {int(bucket_best['holding_days'])}D（PF {fmt(bucket_best['profit_factor'])}，交易 {int(bucket_best['trades'])} 笔）；这是描述性拆解，不是已验证过滤器。" if bucket_best is not None else "没有跌幅桶达到 20 笔，不能据此判断分桶规律。"),
        "",
        "## 研究口径",
        "",
        f"- 信号窗口：{metadata['signal_start_utc']} 至 {metadata['signal_end_utc']}；缓存最新 K 线为 {metadata['cache_end_utc']}。为保证 1D–7D 使用同一完整样本，信号结束时间预留了 7 天退出数据。",
        f"- 宇宙：本地缓存中的 {metadata['symbols']} 个 Binance USD-M USDT 永续合约；每个快照实际可排名合约中位数 {metadata['median_ranked_symbols']}。",
        "- 快照：北京时间 00:00 / 08:00。排行只使用快照前一根已完成 1H K 线的 Close 与 24 小时前 Close；Entry 和固定 Exit 均使用对应 1H Open。",
        f"- 仓位：每笔 {NOTIONAL_USDT:.0f} USDT，1X 隔离保证金；同一持仓版本中，同币已有仓位时跳过新信号。1X 空单若小时 High 触及入场价 2 倍，按保证金全损 -{NOTIONAL_USDT:.0f} USDT 强平。",
        f"- 成本：开仓费 {FEE_RATE:.2%}、平仓费 {FEE_RATE:.2%}（按退出名义价值）、滑点 {SLIPPAGE_RATE:.2%}；未计 Funding、盘口冲击和借贷/清算摩擦。",
        "- 第一阶段没有 Volume、MA/EMA、OI、Funding、链上、波动率、TP、SL、提前退出、动态仓位或杠杆过滤。",
        "- Edge 基础检查事先定义为：PF>1、去最佳 5 笔后仍盈利、且超过半数月份盈利。它是审慎筛查，不是显著性证明。",
        "",
        "## 跌幅榜持仓周期",
        "",
        markdown_table(holding, ["holding_days", *SUMMARY_COLUMNS]),
        "",
        "## Rank 拆解",
        "",
        markdown_table(rank, ["holding_days", "rank", "trades", "net_pnl_usdt", "profit_factor", "win_rate_pct", "median_return_pct", "max_drawdown_usdt"]),
        "",
        "## 跌幅分桶",
        "",
        markdown_table(bucket, ["holding_days", "drop_bucket", "trades", "net_pnl_usdt", "profit_factor", "win_rate_pct", "median_return_pct"]),
        "",
        "注意：`no_drop_or_gain` 与 `0~10%` 被保留，因为第一阶段禁止过滤；市场整体上涨时，跌幅榜末三名也可能仍是正收益币种。",
        "",
        "## 月度稳定性",
        "",
        markdown_table(monthly, ["holding_days", "month", "trades", "net_pnl_usdt", "profit_factor", "win_rate_pct"]),
        "",
        "## MAE / MFE",
        "",
        markdown_table(mae_mfe),
        "",
        "MFE/MAE 为费用前路径价格偏移；做空 MFE 为向下最大有利幅度（正数），MAE 为向上最大不利幅度（负数）。",
        "",
        "## 与涨幅榜 Top3 的同口径比较",
        "",
        "下表比较的是同一信号窗口、无过滤、1X、固定持仓的镜像基准，不是当前已经调优且含退出规则/过滤/动态杠杆的 Rank2/Rank3 主策略。只有这个比较能回答‘排行榜方向本身谁更强’。",
        "",
        markdown_table(comparison),
        "",
        "- PF：做空在 1D–4D 高于做多，做多在 5D–7D 高于做空；各自最强分别是做空 3D PF 1.13 与做多 7D PF 1.11。",
        "- 稳定性：两边都不稳定。做空 3D 只有 4/7 个正收益月；做多所有周期最多只有 3/7 个正收益月。",
        "- 回撤：同为 3D 时做空最大回撤 -1197.11 USDT，低于做多 -1359.79 USDT；但做空延长至 5D–7D 后回撤明显恶化。",
        "- 连续亏损：做空各周期最大连续亏损 6–10 笔，少于做多的 13–19 笔。",
        "- 收益分布：做空 3D 去最佳 5 笔仍有 +403.12 USDT；做多所有周期去最佳 5 笔后均亏损，因此裸做多基准更依赖极少数大涨币。",
        "- 后续优先级：如果继续，只应先对 Rank3 / 6D 做预注册的滚动样本外验证，其次验证 20%–40% 跌幅 / 3D；在验证前不进入影响因子优化。",
        "",
        "完整收益分布、月度一致性与连续亏损数据见八个 CSV。",
        "",
        "## 限制",
        "",
        "- 研究仅覆盖 2026 年至今，只有约六个半月，尚不足以覆盖完整市场周期。",
        "- 本地合约池可能存在当前合约集合带来的幸存者偏差；新上市合约只有在其本地 K 线存在且满 24H 后才进入排行。",
        "- 1H 数据无法给出小时内精确强平时刻；2 倍入场价强平阈值未纳入逐合约维持保证金率和强平手续费。盘口深度、跳空与资金费率也未模拟，极端行情的真实做空成本可能显著更高。",
        "- 分桶、Rank 和持仓优劣均为同一样本内描述，不应直接转化为过滤器；若基础 Edge 被支持，下一步应先做滚动样本外验证。",
    ]
    (OUT / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kline_map, cache_audit = load_kline_map()
    cache_end = min(int(frame["open_time"].max()) for frame in kline_map.values())
    signal_start = ms(START_UTC)
    latest_complete_signal = cache_end - max(HOLD_DAYS) * DAY_MS
    candidates = [t for day in pd.date_range(START_UTC.floor("D"), utc(latest_complete_signal).floor("D"), freq="D", tz="UTC") for t in [ms(day + pd.Timedelta(hours=h)) for h in SNAPSHOT_UTC_HOURS] if t <= latest_complete_signal]
    signal_end = max(candidates)

    short_signals, short_snapshot_audit = build_signals(signal_start, signal_end, kline_map, "short")
    long_signals, long_snapshot_audit = build_signals(signal_start, signal_end, kline_map, "long")
    trade_frames = []
    skipped_frames = []
    for direction, signals in [("short", short_signals), ("long", long_signals)]:
        for hold in HOLD_DAYS:
            completed, skipped = simulate_trades(signals, hold, kline_map, direction)
            trade_frames.append(completed)
            skipped_frames.append(skipped)
    all_trades = pd.concat(trade_frames, ignore_index=True)
    all_skipped = pd.concat(skipped_frames, ignore_index=True)
    short_trades = all_trades[all_trades["direction"].eq("short")].copy()

    holding = grouped_summary(short_trades, ["holding_days"])
    rank = grouped_summary(short_trades, ["holding_days", "rank"])
    bucket = grouped_summary(short_trades, ["holding_days", "drop_bucket"])
    bucket["drop_bucket"] = pd.Categorical(bucket["drop_bucket"], DROP_BUCKET_ORDER, ordered=True)
    bucket = bucket.sort_values(["holding_days", "drop_bucket"])
    monthly = grouped_summary(short_trades, ["holding_days", "month"])
    distribution = return_distribution(short_trades)
    streaks = consecutive_loss_outputs(short_trades)
    mae_mfe = mae_mfe_summary(short_trades)
    overall = grouped_summary(all_trades, ["direction", "holding_days"])
    overall = add_month_consistency(overall, all_trades)
    # Recompute by direction because streak ordering must not mix long and short trades.
    direction_streaks = []
    for direction, group in all_trades.groupby("direction"):
        part = consecutive_loss_outputs(group).query("row_type == 'summary'").copy()
        part["direction"] = direction
        direction_streaks.append(part)
    overall = overall.merge(pd.concat(direction_streaks)[["direction", "holding_days", "max_consecutive_losses", "average_consecutive_losses"]], on=["direction", "holding_days"], how="left")
    overall["candidate_screen"] = np.where(
        (overall["profit_factor"] > 1)
        & (overall["net_pnl_ex_best_5_usdt"] > 0)
        & (overall["positive_months"] > overall["months_tested"] / 2),
        "pass_descriptive_screen",
        "fail_descriptive_screen",
    )

    snapshot_audit = pd.concat([short_snapshot_audit, long_snapshot_audit], ignore_index=True)
    metadata = {
        "signal_start_utc": utc(signal_start),
        "signal_end_utc": utc(signal_end),
        "cache_end_utc": utc(cache_end),
        "symbols": len(kline_map),
        "median_ranked_symbols": int(short_snapshot_audit["ranked_symbols"].median()),
    }
    overall.insert(0, "signal_start_utc", metadata["signal_start_utc"])
    overall.insert(1, "signal_end_utc", metadata["signal_end_utc"])
    overall.insert(2, "cache_end_utc", metadata["cache_end_utc"])
    overall.insert(3, "notional_usdt", NOTIONAL_USDT)
    overall.insert(4, "leverage", "1X")
    overall.insert(5, "entry_fee_rate", FEE_RATE)
    overall.insert(6, "exit_fee_rate", FEE_RATE)
    overall.insert(7, "slippage_rate", SLIPPAGE_RATE)

    outputs = {
        "01_overall_summary.csv": overall,
        "02_holding_period_summary.csv": holding,
        "03_rank_summary.csv": rank,
        "04_drop_bucket_summary.csv": bucket,
        "05_monthly_summary.csv": monthly,
        "06_return_distribution.csv": distribution,
        "07_consecutive_loss_statistics.csv": streaks,
        "08_mae_mfe_statistics.csv": mae_mfe,
        "signal_details.csv": pd.concat([short_signals, long_signals], ignore_index=True),
        "trade_details.csv": all_trades,
        "skipped_signal_details.csv": all_skipped,
        "snapshot_data_quality.csv": snapshot_audit,
        "cache_data_quality.csv": cache_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    write_report(metadata, overall, holding, rank, bucket, monthly, streaks, mae_mfe, short_trades)
    print(holding.to_string(index=False))
    print(f"Wrote research outputs to {OUT}")


if __name__ == "__main__":
    main()
