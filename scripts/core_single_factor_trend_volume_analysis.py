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

from scripts.backtest_futures_top2_fixed_time import calculate_drawdown


OUT = ROOT / "output"
KLINE_1H_DIR = ROOT / "data" / "futures_klines_1h"
INPUT_CANDIDATES = [
    OUT / "futures_top3_7d_rank2_rank3_ex_rave_trades.csv",
    OUT / "futures_top3_7d_rank2_rank3_trades.csv",
    OUT / "futures_top2_fixed_time_trades.csv",
]
STATS_PATH = OUT / "core_single_factor_trend_volume_stats.csv"
TRADES_WITH_FACTORS_PATH = OUT / "trades_with_core_trend_volume_factors.csv"

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
FOUR_HOUR_MS = 4 * HOUR_MS


def load_input_trades() -> tuple[Path, pd.DataFrame]:
    for path in INPUT_CANDIDATES:
        if path.exists():
            trades = pd.read_csv(path)
            if "status" in trades.columns:
                trades = trades[trades["status"].eq("completed")].copy()
            if "holding_days" in trades.columns:
                trades = trades[trades["holding_days"].eq(7)].copy()
            trades = trades[trades["rank"].isin([2, 3])].copy()
            return path, trades.reset_index(drop=True)
    raise FileNotFoundError("No Rank2/Rank3 completed trades CSV found under output/.")


def load_1h(symbol: str) -> pd.DataFrame:
    path = KLINE_1H_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for col in ["open_time", "close_time", "trade_count"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64").astype("int64")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def aggregate_bars(frame: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["bar_open_time"] = (frame["open_time"] // interval_ms) * interval_ms
    grouped = frame.sort_values("open_time").groupby("bar_open_time", sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
    ).reset_index()
    out = out.rename(columns={"bar_open_time": "open_time"})
    out["close_time"] = out["open_time"] + interval_ms
    return out.set_index("open_time", drop=False).sort_index()


def build_kline_maps(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    map_4h: dict[str, pd.DataFrame] = {}
    map_1d: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        one_h = load_1h(symbol)
        if one_h.empty:
            continue
        map_4h[symbol] = aggregate_bars(one_h, FOUR_HOUR_MS)
        map_1d[symbol] = aggregate_bars(one_h, DAY_MS)
    return map_4h, map_1d


def get_bar(indexed: pd.DataFrame, open_time: int) -> pd.Series | None:
    if indexed.empty or open_time not in indexed.index:
        return None
    row = indexed.loc[open_time]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def bucket_ma_structure(close_4h: float, ma7_4h: float, ma21_4h: float) -> str:
    if not all(np.isfinite([close_4h, ma7_4h, ma21_4h])):
        return "missing"
    if close_4h > ma7_4h > ma21_4h:
        return "close > MA7 > MA21"
    if close_4h > ma7_4h and ma7_4h <= ma21_4h:
        return "close > MA7 but MA7 <= MA21"
    if ma7_4h > ma21_4h and close_4h <= ma7_4h:
        return "MA7 > MA21 but close <= MA7"
    if close_4h <= ma7_4h and ma7_4h <= ma21_4h:
        return "close <= MA7 and MA7 <= MA21"
    return "missing"


def bucket_distance_4h_ma7(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    pct = value * 100
    if pct < -3:
        return "<-3%"
    if pct < 0:
        return "-3%~0%"
    if pct < 5:
        return "0%~5%"
    if pct < 15:
        return "5%~15%"
    if pct < 30:
        return "15%~30%"
    return ">30%"


def bucket_distance_1d_ma7(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    pct = value * 100
    if pct < 0:
        return "<0%"
    if pct < 10:
        return "0%~10%"
    if pct < 30:
        return "10%~30%"
    return ">30%"


def bucket_volume_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 1:
        return "<1"
    if value < 1.5:
        return "1~1.5"
    if value < 3:
        return "1.5~3"
    if value < 5:
        return "3~5"
    if value < 10:
        return "5~10"
    return ">10"


def add_factors(trades: pd.DataFrame, map_4h: dict[str, pd.DataFrame], map_1d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"])
        entry_time = int(trade["entry_time_ms"])
        four_h = map_4h.get(symbol, pd.DataFrame())
        one_d = map_1d.get(symbol, pd.DataFrame())
        last_4h_open = entry_time - FOUR_HOUR_MS
        last_1d_open = (entry_time // DAY_MS) * DAY_MS - DAY_MS

        factor: dict[str, Any] = {
            "ma_structure_4h": "missing",
            "distance_to_4h_ma7_pct": np.nan,
            "above_1d_ma7": np.nan,
            "distance_to_1d_ma7_pct": np.nan,
            "volume_4h_ratio_7d": np.nan,
            "volume_24h_ratio_7d": np.nan,
        }

        if not four_h.empty:
            up_to_4h = four_h[four_h["open_time"] <= last_4h_open].copy()
            if len(up_to_4h) >= 21:
                close_4h = float(up_to_4h.iloc[-1]["close"])
                ma7_4h = float(up_to_4h.tail(7)["close"].mean())
                ma21_4h = float(up_to_4h.tail(21)["close"].mean())
                factor["ma_structure_4h"] = bucket_ma_structure(close_4h, ma7_4h, ma21_4h)
                factor["distance_to_4h_ma7_pct"] = (close_4h / ma7_4h - 1.0) * 100 if ma7_4h > 0 else np.nan

            current_4h = get_bar(four_h, last_4h_open)
            previous_42 = four_h[four_h["open_time"] < last_4h_open].tail(42)
            recent_42_including_current = four_h[four_h["open_time"] <= last_4h_open].tail(42)
            recent_6 = four_h[four_h["open_time"] <= last_4h_open].tail(6)
            if current_4h is not None and len(previous_42) == 42:
                avg_4h_volume_7d = float(previous_42["volume"].mean())
                factor["volume_4h_ratio_7d"] = (
                    float(current_4h["volume"]) / avg_4h_volume_7d if avg_4h_volume_7d > 0 else np.nan
                )
            if len(recent_42_including_current) == 42 and len(recent_6) == 6:
                avg_daily_volume_7d = float(recent_42_including_current["volume"].sum()) / 7.0
                volume_24h = float(recent_6["volume"].sum())
                factor["volume_24h_ratio_7d"] = volume_24h / avg_daily_volume_7d if avg_daily_volume_7d > 0 else np.nan

        if not one_d.empty:
            up_to_1d = one_d[one_d["open_time"] <= last_1d_open].copy()
            if len(up_to_1d) >= 7:
                close_1d = float(up_to_1d.iloc[-1]["close"])
                ma7_1d = float(up_to_1d.tail(7)["close"].mean())
                if ma7_1d > 0:
                    factor["above_1d_ma7"] = bool(close_1d > ma7_1d)
                    factor["distance_to_1d_ma7_pct"] = (close_1d / ma7_1d - 1.0) * 100

        rows.append(factor)

    enriched = pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    enriched["distance_to_4h_ma7_bucket"] = enriched["distance_to_4h_ma7_pct"].map(lambda x: bucket_distance_4h_ma7(x / 100 if pd.notna(x) else np.nan))
    enriched["above_1d_ma7_bucket"] = enriched["above_1d_ma7"].map(
        lambda x: "missing" if pd.isna(x) else ("True" if bool(x) else "False")
    )
    enriched["distance_to_1d_ma7_bucket"] = enriched["distance_to_1d_ma7_pct"].map(lambda x: bucket_distance_1d_ma7(x / 100 if pd.notna(x) else np.nan))
    enriched["volume_4h_ratio_7d_bucket"] = enriched["volume_4h_ratio_7d"].map(bucket_volume_ratio)
    enriched["volume_24h_ratio_7d_bucket"] = enriched["volume_24h_ratio_7d"].map(bucket_volume_ratio)
    return enriched


def profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def sample_quality(trade_count: int) -> str:
    if trade_count < 30:
        return "sample_lt_30"
    if trade_count < 50:
        return "sample_30_50_observe"
    return "sample_gt_50_reference"


def summarize_bucket(scope: str, factor_name: str, bucket: str, group: pd.DataFrame) -> dict[str, Any]:
    completed = group.sort_values("entry_time_ms").copy()
    pnl = completed["pnl_u"].astype(float)
    returns = completed["net_return_pct"].astype(float)
    wins = completed[completed["pnl_u"] > 0]
    losses = completed[completed["pnl_u"] < 0]
    sorted_pnl = completed.sort_values("pnl_u", ascending=False)["pnl_u"].reset_index(drop=True)
    trade_count = int(len(completed))
    return {
        "scope": scope,
        "factor_name": factor_name,
        "bucket": bucket,
        "trade_count": trade_count,
        "net_pnl_u": float(pnl.sum()) if trade_count else 0.0,
        "pf": profit_factor(pnl),
        "win_rate": float(len(wins) / trade_count) if trade_count else np.nan,
        "avg_return_pct": float(returns.mean()) if trade_count else np.nan,
        "median_return_pct": float(returns.median()) if trade_count else np.nan,
        "avg_win_pct": float(wins["net_return_pct"].mean()) if len(wins) else np.nan,
        "avg_loss_pct": float(losses["net_return_pct"].mean()) if len(losses) else np.nan,
        "max_win_pct": float(returns.max()) if trade_count else np.nan,
        "max_loss_pct": float(returns.min()) if trade_count else np.nan,
        "max_drawdown_u": calculate_drawdown(pnl),
        "pnl_after_remove_top1_u": float(sorted_pnl.iloc[1:].sum()) if trade_count > 1 else 0.0,
        "pnl_after_remove_top3_u": float(sorted_pnl.iloc[3:].sum()) if trade_count > 3 else 0.0,
        "sample_quality": sample_quality(trade_count),
    }


def factor_stats(enriched: pd.DataFrame) -> pd.DataFrame:
    factors = [
        ("ma_structure_4h", "ma_structure_4h"),
        ("distance_to_4h_ma7_pct", "distance_to_4h_ma7_bucket"),
        ("above_1d_ma7", "above_1d_ma7_bucket"),
        ("distance_to_1d_ma7_pct", "distance_to_1d_ma7_bucket"),
        ("volume_4h_ratio_7d", "volume_4h_ratio_7d_bucket"),
        ("volume_24h_ratio_7d", "volume_24h_ratio_7d_bucket"),
    ]
    scopes = [
        ("rank2_rank3", enriched),
        ("rank2", enriched[enriched["rank"].eq(2)]),
        ("rank3", enriched[enriched["rank"].eq(3)]),
    ]
    rows: list[dict[str, Any]] = []
    for scope_name, scope_frame in scopes:
        for factor_name, bucket_col in factors:
            rows.append(summarize_bucket(scope_name, factor_name, "ALL", scope_frame))
            for bucket, group in scope_frame.groupby(bucket_col, dropna=False, sort=False):
                bucket_name = "missing" if pd.isna(bucket) else str(bucket)
                rows.append(summarize_bucket(scope_name, factor_name, bucket_name, group))
    return pd.DataFrame(rows)


def print_section(title: str, frame: pd.DataFrame) -> None:
    print(f"\n========== {title} ==========")
    if frame.empty:
        print("无")
        return
    cols = ["scope", "factor_name", "bucket", "trade_count", "net_pnl_u", "pf", "win_rate", "median_return_pct", "pnl_after_remove_top1_u", "sample_quality"]
    show = frame[cols].copy()
    for col in ["net_pnl_u", "pf", "win_rate", "median_return_pct", "pnl_after_remove_top1_u"]:
        show[col] = show[col].map(lambda x: "inf" if x == math.inf else ("" if pd.isna(x) else f"{x:.2f}"))
    print(show.to_string(index=False))


def main() -> None:
    input_path, trades = load_input_trades()
    symbols = sorted(trades["symbol"].astype(str).unique())
    map_4h, map_1d = build_kline_maps(symbols)
    enriched = add_factors(trades, map_4h, map_1d)
    stats = factor_stats(enriched)

    keep_cols = [
        "symbol",
        "rank",
        "entry_time_utc",
        "entry_time_bj",
        "snapshot_hour_bj",
        "entry_price",
        "exit_price",
        "gain_24h",
        "pnl_u",
        "net_return_pct",
        "month",
        "ma_structure_4h",
        "distance_to_4h_ma7_pct",
        "above_1d_ma7",
        "distance_to_1d_ma7_pct",
        "volume_4h_ratio_7d",
        "volume_24h_ratio_7d",
        "distance_to_4h_ma7_bucket",
        "above_1d_ma7_bucket",
        "distance_to_1d_ma7_bucket",
        "volume_4h_ratio_7d_bucket",
        "volume_24h_ratio_7d_bucket",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    stats.to_csv(STATS_PATH, index=False, encoding="utf-8-sig")
    enriched[[col for col in keep_cols if col in enriched.columns]].to_csv(
        TRADES_WITH_FACTORS_PATH, index=False, encoding="utf-8-sig"
    )

    all_row = stats[
        (stats["scope"].eq("rank2_rank3"))
        & (stats["factor_name"].eq("ma_structure_4h"))
        & (stats["bucket"].eq("ALL"))
    ].iloc[0]
    all_median = float(all_row["median_return_pct"])

    trend = stats[stats["factor_name"].isin(["ma_structure_4h", "distance_to_4h_ma7_pct", "above_1d_ma7", "distance_to_1d_ma7_pct"])]
    volume = stats[stats["factor_name"].isin(["volume_4h_ratio_7d", "volume_24h_ratio_7d"])]
    worth = stats[
        (stats["trade_count"] >= 50)
        & (stats["pf"] > 1.10)
        & (stats["median_return_pct"] > all_median)
        & (stats["pnl_after_remove_top1_u"] > -500)
        & (~stats["bucket"].eq("ALL"))
    ]
    risky = stats[
        (stats["trade_count"] >= 30)
        & (~stats["bucket"].eq("ALL"))
        & ((stats["pf"] < 0.80) | (stats["median_return_pct"] < all_median - 5))
    ]

    print("========== 第一阶段核心单因子拆解完成 ==========")
    print("研究对象: Rank2 + Rank3")
    print(f"输入文件: {input_path}")
    print(f"完成交易数: {len(enriched)}")
    print(f"时间范围: {enriched['entry_time_utc'].min()} ~ {enriched['entry_time_utc'].max()}")
    print(f"输出: {STATS_PATH}")
    print(f"输出: {TRADES_WITH_FACTORS_PATH}")
    print_section("趋势结构因子", trend[trend["scope"].eq("rank2_rank3")])
    print_section("成交量因子", volume[volume["scope"].eq("rank2_rank3")])
    print_section("值得继续研究的分桶", worth)
    print_section("高风险分桶", risky)

    print("\n========== 最终结论 ==========")
    print("1. 4H MA7/MA21 多头结构需要看 CSV 分桶；不能只因单桶净收益为正就判断有效。")
    print("2. 距离 4H MA7 太远是否变差，以 >30% 与 15%~30% 桶的 PF/中位数/去最大1笔为准。")
    print("3. 1D MA7 上方是否更优，重点比较 above_1d_ma7=True/False 的 PF 和中位数。")
    print("4. 4H 放量健康区间，优先看 trade_count>=50 且 PF>1.10 的分桶。")
    print("5. 24H 放量健康区间同样不能忽略去最大1笔后的净收益。")
    print("6. 第二阶段只应纳入样本充足、PF和中位数同时改善、且长尾依赖不严重的单因子。")
    print("7. 样本 <30 的分桶仅观察，不作为结论。")


if __name__ == "__main__":
    main()
