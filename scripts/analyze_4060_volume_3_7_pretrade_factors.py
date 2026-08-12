from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import (
    CACHE_DIR,
    DAY_MS,
    FOUR_HOUR_MS,
    HOUR_MS,
    OUT,
    aggregate_4h,
    load_kline_map,
    max_drawdown,
    ms_to_utc,
    profit_factor,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt


PREFIX = "recent_half_4060_rank2_volume_3_7_pretrade"
TRADES_PATH = OUT / "recent_half_4060_rank2_volume_3_7_6080_drop_rank2_4060_trades.csv"
SNAPSHOT_HOURS_BJ = {"00:00", "08:00"}


def cached_symbols() -> list[str]:
    return sorted(path.stem.removesuffix("_1h") for path in Path(CACHE_DIR).glob("*_1h.csv"))


def summarize(trades: pd.DataFrame) -> dict[str, float | int]:
    pnl = trades["pnl_u"].astype(float) if not trades.empty else pd.Series(dtype=float)
    returns = trades["net_return_pct"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = trades[pnl > 0]
    losses = trades[pnl < 0]
    return {
        "交易数": int(len(trades)),
        "盈利单": int(len(wins)),
        "亏损单": int(len(losses)),
        "盈利总金额U": float(pnl[pnl > 0].sum()) if len(pnl) else 0.0,
        "亏损总金额U": float(pnl[pnl < 0].sum()) if len(pnl) else 0.0,
        "净收益U": float(pnl.sum()) if len(pnl) else 0.0,
        "PF": profit_factor(pnl),
        "胜率": float(len(wins) / len(trades)) if len(trades) else np.nan,
        "平均收益率": float(returns.mean()) if len(returns) else np.nan,
        "中位数收益率": float(returns.median()) if len(returns) else np.nan,
        "最大回撤U": max_drawdown(pnl),
        "去最大1笔后U": float(pnl.sum() - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "去最大3笔后U": float(pnl.sum() - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
    }


def bucket_past_7d(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 0:
        return "<0%"
    if value < 20:
        return "0%-20%"
    if value < 50:
        return "20%-50%"
    if value < 80:
        return "50%-80%"
    if value < 120:
        return "80%-120%"
    if value < 200:
        return "120%-200%"
    return "200%+"


def bucket_distance(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 15:
        return "<15%"
    if value < 30:
        return "15%-30%"
    return ">30%"


def bucket_volume(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 3:
        return "<3"
    if value < 5:
        return "3-5"
    if value < 7:
        return "5-7"
    return ">=7"


def bucket_top1_gap(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 5:
        return "<5pct"
    if value < 10:
        return "5-10pct"
    if value < 20:
        return "10-20pct"
    return ">=20pct"


def finite_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def finite_median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def add_pretrade_features(trades: pd.DataFrame, kline_map: dict[str, pd.DataFrame], rank_context: pd.DataFrame) -> pd.DataFrame:
    h4_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    context_key = rank_context.set_index(["signal_time", "symbol"], drop=False)

    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"])
        entry_time = int(float(trade["entry_time_ms"]))
        h1 = kline_map.get(symbol, pd.DataFrame())
        h4 = h4_cache.setdefault(symbol, aggregate_4h(h1))
        feature: dict[str, object] = {
            "past_7d_gain_pct": np.nan,
            "distance_to_4h_ma21_pct": np.nan,
            "ma7_slope_3bar_pct": np.nan,
            "ma21_slope_3bar_pct": np.nan,
            "close_gt_ma7": False,
            "ma7_gt_ma21": False,
            "volume_4h_ratio_7d": np.nan,
            "volume_accel_4h_vs_24h": np.nan,
            "top1_gain_24h": np.nan,
            "top1_top2_gap_pct": np.nan,
            "rank_strength_vs_top1": np.nan,
            "fresh_rank2": "unknown",
        }

        if not h1.empty:
            pre_h1 = h1[h1["open_time"] < entry_time].copy()
            seven_days_ago = entry_time - 7 * DAY_MS
            before_7d = h1[h1["open_time"] <= seven_days_ago]
            if not pre_h1.empty and not before_7d.empty:
                last_close = float(pre_h1.iloc[-1]["close"])
                base_open = float(before_7d.iloc[-1]["open"])
                if base_open > 0:
                    feature["past_7d_gain_pct"] = (last_close / base_open - 1.0) * 100

        if not h4.empty:
            last_4h_open = entry_time - FOUR_HOUR_MS
            up_to = h4[h4["open_time"] <= last_4h_open].copy()
            if len(up_to) >= 24:
                close = float(up_to.iloc[-1]["close"])
                ma7_series = up_to["close"].rolling(7).mean()
                ma21_series = up_to["close"].rolling(21).mean()
                ma7 = float(ma7_series.iloc[-1])
                ma21 = float(ma21_series.iloc[-1])
                ma7_prev = float(ma7_series.iloc[-4])
                ma21_prev = float(ma21_series.iloc[-4])
                feature["distance_to_4h_ma21_pct"] = (close / ma21 - 1.0) * 100 if ma21 > 0 else np.nan
                feature["ma7_slope_3bar_pct"] = (ma7 / ma7_prev - 1.0) * 100 if ma7_prev > 0 else np.nan
                feature["ma21_slope_3bar_pct"] = (ma21 / ma21_prev - 1.0) * 100 if ma21_prev > 0 else np.nan
                feature["close_gt_ma7"] = bool(close > ma7)
                feature["ma7_gt_ma21"] = bool(ma7 > ma21)
            recent_42 = up_to.tail(42)
            if len(recent_42) == 42:
                avg_4h_volume = float(recent_42["volume"].mean())
                last_4h_volume = float(recent_42.iloc[-1]["volume"])
                feature["volume_4h_ratio_7d"] = last_4h_volume / avg_4h_volume if avg_4h_volume > 0 else np.nan
                vol24 = float(trade["volume_24h_ratio_7d"])
                feature["volume_accel_4h_vs_24h"] = feature["volume_4h_ratio_7d"] / vol24 if vol24 > 0 else np.nan

        try:
            context_row = context_key.loc[(entry_time, symbol)]
            if isinstance(context_row, pd.DataFrame):
                context_row = context_row.iloc[0]
            feature["top1_gain_24h"] = float(context_row.get("top1_gain_24h", np.nan))
            feature["top1_top2_gap_pct"] = float(context_row.get("top1_top2_gap_pct", np.nan))
            feature["rank_strength_vs_top1"] = float(context_row.get("rank_strength_vs_top1", np.nan))
            feature["fresh_rank2"] = str(context_row.get("fresh_rank2", "unknown"))
        except KeyError:
            pass

        rows.append(feature)

    out = pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["result"] = np.where(out["pnl_u"].astype(float) > 0, "win", "loss")
    out["past_7d_gain_bucket"] = out["past_7d_gain_pct"].apply(bucket_past_7d)
    out["distance_to_4h_ma7_bucket_new"] = out["distance_to_4h_ma7_pct"].apply(bucket_distance)
    out["distance_to_4h_ma21_bucket"] = out["distance_to_4h_ma21_pct"].apply(bucket_distance)
    out["volume_24h_bucket_3_7"] = out["volume_24h_ratio_7d"].apply(bucket_volume)
    out["volume_4h_bucket"] = out["volume_4h_ratio_7d"].apply(
        lambda v: "missing" if pd.isna(v) else ("<1" if v < 1 else ("1-2" if v < 2 else ("2-4" if v < 4 else ">=4")))
    )
    out["top1_gap_bucket"] = out["top1_top2_gap_pct"].apply(bucket_top1_gap)
    return out


def build_rank_context(kline_map: dict[str, pd.DataFrame], start: int, end: int) -> pd.DataFrame:
    all_signals = generate_signals(start, end, kline_map)
    all_signals = all_signals[all_signals["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)].copy()
    top1 = (
        all_signals[all_signals["rank"].eq(1)][["signal_time", "gain_24h"]]
        .drop_duplicates("signal_time")
        .rename(columns={"gain_24h": "top1_gain_24h"})
    )
    out = all_signals.merge(top1, on="signal_time", how="left")
    out["top1_top2_gap_pct"] = (out["top1_gain_24h"] - out["gain_24h"]) * 100
    out["rank_strength_vs_top1"] = out["gain_24h"] / out["top1_gain_24h"]
    out = out.sort_values(["symbol", "signal_time"])
    out["previous_rank"] = out.groupby("symbol")["rank"].shift(1)
    out["previous_signal_time"] = out.groupby("symbol")["signal_time"].shift(1)
    out["fresh_rank2"] = np.where(
        out["rank"].eq(2) & (out["previous_rank"].ne(2) | ((out["signal_time"] - out["previous_signal_time"]) > 12 * HOUR_MS)),
        "fresh",
        "persistent",
    )
    return out


def single_variable_table(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "gain_24h",
        "past_7d_gain_pct",
        "distance_to_4h_ma7_pct",
        "distance_to_4h_ma21_pct",
        "ma7_slope_3bar_pct",
        "ma21_slope_3bar_pct",
        "volume_24h_ratio_7d",
        "volume_4h_ratio_7d",
        "volume_accel_4h_vs_24h",
        "top1_top2_gap_pct",
        "rank_strength_vs_top1",
    ]
    rows = []
    wins = frame[frame["result"].eq("win")]
    losses = frame[frame["result"].eq("loss")]
    for var in variables:
        win_mean = finite_mean(wins[var])
        win_median = finite_median(wins[var])
        loss_mean = finite_mean(losses[var])
        loss_median = finite_median(losses[var])
        diff = win_median - loss_median if np.isfinite(win_median) and np.isfinite(loss_median) else np.nan
        pooled = pd.to_numeric(frame[var], errors="coerce").dropna()
        spread = float(pooled.quantile(0.75) - pooled.quantile(0.25)) if len(pooled) else np.nan
        strength = "NO"
        if np.isfinite(diff) and np.isfinite(spread) and spread > 0:
            ratio = abs(diff) / spread
            strength = "YES" if ratio >= 0.75 else ("WEAK" if ratio >= 0.35 else "NO")
        rows.append(
            {
                "变量": var,
                "win_mean": win_mean,
                "win_median": win_median,
                "loss_mean": loss_mean,
                "loss_median": loss_median,
                "median_diff_win_minus_loss": diff,
                "分离能力": strength,
            }
        )
    return pd.DataFrame(rows)


def categorical_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        for value, group in frame.groupby(col, dropna=False, sort=True):
            stats = summarize(group)
            rows.append({"变量": col, "分组": value, **stats})
    return pd.DataFrame(rows)


def evaluate_rules(frame: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[str, pd.Series]] = []
    candidates.extend(
        [
            ("distance_to_4h_ma7_pct < 30", frame["distance_to_4h_ma7_pct"].lt(30)),
            ("distance_to_4h_ma21_pct < 45", frame["distance_to_4h_ma21_pct"].lt(45)),
            ("past_7d_gain_pct < 200", frame["past_7d_gain_pct"].lt(200)),
            ("volume_24h_ratio_7d < 5", frame["volume_24h_ratio_7d"].lt(5)),
            ("volume_24h_ratio_7d >= 5", frame["volume_24h_ratio_7d"].ge(5)),
            ("volume_4h_ratio_7d < 2", frame["volume_4h_ratio_7d"].lt(2)),
            ("top1_top2_gap_pct < 10", frame["top1_top2_gap_pct"].lt(10)),
            ("fresh_rank2", frame["fresh_rank2"].eq("fresh")),
            ("entry_hour_00", frame["snapshot_hour_bj"].eq("00:00")),
            ("entry_hour_08", frame["snapshot_hour_bj"].eq("08:00")),
            ("ma7_slope_3bar_pct > 0", frame["ma7_slope_3bar_pct"].gt(0)),
            ("ma21_slope_3bar_pct > 0", frame["ma21_slope_3bar_pct"].gt(0)),
        ]
    )
    rows = []
    base = summarize(frame)
    for (name_a, mask_a), (name_b, mask_b) in itertools.combinations(candidates, 2):
        mask = mask_a & mask_b
        subset = frame[mask].copy()
        if len(subset) < 10:
            continue
        stats = summarize(subset)
        rows.append(
            {
                "规则": f"{name_a} AND {name_b}",
                **stats,
                "保留交易占比": len(subset) / len(frame),
                "盈利单捕获率": stats["盈利单"] / base["盈利单"] if base["盈利单"] else np.nan,
                "亏损单保留率": stats["亏损单"] / base["亏损单"] if base["亏损单"] else np.nan,
                "相对基准PF提升": stats["PF"] - base["PF"],
                "相对基准净收益提升U": stats["净收益U"] - base["净收益U"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["PF", "净收益U", "交易数"], ascending=[False, False, False])


def main() -> None:
    trades = pd.read_csv(TRADES_PATH)
    trades = trades[trades["status"].eq("completed")].copy()
    start = int(trades["entry_time_ms"].min()) - 14 * DAY_MS
    end = int(trades["entry_time_ms"].max()) + 8 * DAY_MS
    kline_map = load_kline_map(cached_symbols(), start, end)
    context = build_rank_context(kline_map, start, end)
    enriched = add_pretrade_features(trades, kline_map, context)
    enriched.to_csv(OUT / f"{PREFIX}_trades_enriched.csv", index=False, encoding="utf-8-sig")

    single = single_variable_table(enriched)
    single.to_csv(OUT / f"{PREFIX}_single_variable_win_loss.csv", index=False, encoding="utf-8-sig")

    cats = categorical_table(
        enriched,
        [
            "snapshot_hour_bj",
            "past_7d_gain_bucket",
            "distance_to_4h_ma7_bucket_new",
            "distance_to_4h_ma21_bucket",
            "volume_24h_bucket_3_7",
            "volume_4h_bucket",
            "top1_gap_bucket",
            "fresh_rank2",
            "ma_structure_4h",
            "close_gt_ma7",
            "ma7_gt_ma21",
        ],
    )
    cats.to_csv(OUT / f"{PREFIX}_categorical_stats.csv", index=False, encoding="utf-8-sig")

    rules = evaluate_rules(enriched)
    rules.to_csv(OUT / f"{PREFIX}_candidate_2factor_rules.csv", index=False, encoding="utf-8-sig")

    base = pd.DataFrame([{"规则": "base_59", **summarize(enriched)}])
    base.to_csv(OUT / f"{PREFIX}_base_summary.csv", index=False, encoding="utf-8-sig")

    print("========== Base 59 ==========")
    print(base.to_string(index=False))
    print()
    print("========== Single Variable ==========")
    print(single.to_string(index=False))
    print()
    print("========== Top Candidate Rules ==========")
    print(rules.head(12).to_string(index=False))
    print()
    print(f"files: output/{PREFIX}_*.csv")


if __name__ == "__main__":
    main()
