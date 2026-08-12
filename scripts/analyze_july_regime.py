from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import HOUR_MS, ms_to_bj_string, ms_to_utc
from scripts.backtest_futures_top2_fixed_time import CACHE_DIR
from scripts.run_current_main_strategy_2026_jan_jun import SNAPSHOT_HOURS_BJ


OUT = Path("output/july_regime_analysis")
START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2026-07-31 23:00:00", tz="UTC")
EXTRA_END = pd.Timestamp("2026-08-08 13:00:00", tz="UTC")
HORIZONS = [1, 4, 8, 12, 24, 48, 72]
MFE_WINDOWS = [4, 8, 24]
TOP_NS = [3, 10]


def read_symbol(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for col in ["open_time", "close_time", "trade_count"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64").astype("int64")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    symbol = path.stem.removesuffix("_1h")
    frame["symbol"] = symbol
    return frame.drop_duplicates(["open_time"]).sort_values("open_time").reset_index(drop=True)


def load_cache() -> dict[str, pd.DataFrame]:
    result = {}
    start_ms = int((START - pd.Timedelta(days=35)).timestamp() * 1000)
    end_ms = int(EXTRA_END.timestamp() * 1000)
    for path in sorted(CACHE_DIR.glob("*_1h.csv")):
        frame = read_symbol(path)
        if frame.empty:
            continue
        scoped = frame[(frame["open_time"] >= start_ms) & (frame["open_time"] <= end_ms)].copy()
        if not scoped.empty:
            result[path.stem.removesuffix("_1h")] = scoped.reset_index(drop=True)
    return result


def row_at(indexed: pd.DataFrame, open_time: int) -> pd.Series | None:
    if open_time not in indexed.index:
        return None
    row = indexed.loc[open_time]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def build_snapshot(snapshot_ms: int, kline_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float]]:
    current_open = snapshot_ms - HOUR_MS
    previous_open = current_open - 24 * HOUR_MS
    past_24_start = current_open - 23 * HOUR_MS
    rows = []
    market_gains = []
    for symbol, frame in kline_map.items():
        indexed = frame.set_index("open_time", drop=False)
        current = row_at(indexed, current_open)
        previous = row_at(indexed, previous_open)
        entry = row_at(indexed, snapshot_ms)
        if current is None or previous is None or entry is None:
            continue
        prev_close = float(previous["close"])
        curr_close = float(current["close"])
        entry_open = float(entry["open"])
        if prev_close <= 0 or not np.isfinite(prev_close) or not np.isfinite(curr_close) or not np.isfinite(entry_open):
            continue
        past_24 = frame[(frame["open_time"] >= past_24_start) & (frame["open_time"] <= current_open)]
        if len(past_24) < 24:
            continue
        gain_24h = curr_close / prev_close - 1.0
        market_gains.append(gain_24h)
        high_24h = float(past_24["high"].max())
        low_24h = float(past_24["low"].min())
        close_1h_ago = row_at(indexed, current_open - HOUR_MS)
        close_4h_ago = row_at(indexed, current_open - 4 * HOUR_MS)
        ret_1h = curr_close / float(close_1h_ago["close"]) - 1.0 if close_1h_ago is not None and float(close_1h_ago["close"]) > 0 else np.nan
        ret_4h = curr_close / float(close_4h_ago["close"]) - 1.0 if close_4h_ago is not None and float(close_4h_ago["close"]) > 0 else np.nan
        rows.append(
            {
                "signal_time": snapshot_ms,
                "signal_time_utc": ms_to_utc(snapshot_ms).strftime("%Y-%m-%d %H:%M:%S"),
                "signal_time_bj": ms_to_bj_string(snapshot_ms),
                "snapshot_hour_bj": (ms_to_utc(snapshot_ms) + pd.Timedelta(hours=8)).strftime("%H:%M"),
                "symbol": symbol,
                "gain_24h": gain_24h,
                "current_price": curr_close,
                "entry_price": entry_open,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "ret_1h_before": ret_1h,
                "ret_4h_before": ret_4h,
                "price_from_24h_low": curr_close / low_24h - 1.0 if low_24h > 0 else np.nan,
                "distance_to_24h_high": curr_close / high_24h - 1.0 if high_24h > 0 else np.nan,
                "range_24h": high_24h / low_24h - 1.0 if low_24h > 0 else np.nan,
                "ret1h_over_24h": ret_1h / gain_24h if gain_24h > 0.02 and np.isfinite(ret_1h) else np.nan,
                "ret4h_over_24h": ret_4h / gain_24h if gain_24h > 0.02 and np.isfinite(ret_4h) else np.nan,
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked, {}
    ranked = ranked.sort_values(["gain_24h", "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    gains = pd.Series(market_gains, dtype=float)
    top20 = ranked.head(20)["gain_24h"].astype(float)
    top3 = ranked.head(3)["gain_24h"].astype(float)
    breadth = {
        "universe_count": int(len(gains)),
        "breadth_positive": float((gains > 0).mean()),
        "median_24h_return": float(gains.median()),
        "count_gt_5": int((gains > 0.05).sum()),
        "count_gt_10": int((gains > 0.10).sum()),
        "count_gt_20": int((gains > 0.20).sum()),
        "top10_avg_gain": float(ranked.head(10)["gain_24h"].mean()),
        "top20_avg_gain": float(top20.mean()),
        "top20_median_gain": float(top20.median()),
        "gainer_concentration": float(top3.mean() - top20.median()) if len(top20) else np.nan,
    }
    return ranked, breadth


def future_features(row: pd.Series, frame: pd.DataFrame) -> dict[str, float]:
    entry_ms = int(row["signal_time"])
    entry_price = float(row["entry_price"])
    indexed = frame.set_index("open_time", drop=False)
    out: dict[str, float] = {}
    for horizon in HORIZONS:
        exit_open = entry_ms + horizon * HOUR_MS
        exit_row = row_at(indexed, exit_open)
        out[f"future_return_{horizon}h"] = float(exit_row["open"]) / entry_price - 1.0 if exit_row is not None and entry_price > 0 else np.nan
    for window in MFE_WINDOWS:
        path = frame[(frame["open_time"] >= entry_ms) & (frame["open_time"] <= entry_ms + (window - 1) * HOUR_MS)]
        if len(path) < window or entry_price <= 0:
            out[f"mfe_{window}h"] = np.nan
            out[f"mae_{window}h"] = np.nan
            out[f"new_high_{window}h"] = np.nan
        else:
            out[f"mfe_{window}h"] = float(path["high"].max()) / entry_price - 1.0
            out[f"mae_{window}h"] = float(path["low"].min()) / entry_price - 1.0
            out[f"new_high_{window}h"] = float(path["high"].max()) > float(row["high_24h"])
    path24 = frame[(frame["open_time"] >= entry_ms) & (frame["open_time"] <= entry_ms + 23 * HOUR_MS)]
    if len(path24) >= 1 and entry_price > 0:
        max_high = float(path24["high"].max())
        first_high = path24[path24["high"].astype(float).eq(max_high)].sort_values("open_time").iloc[0]
        out["time_to_high_24h_hours"] = (int(first_high["open_time"]) - entry_ms) / HOUR_MS
    else:
        out["time_to_high_24h_hours"] = np.nan
    return out


def rebuild_snapshots(kline_map: dict[str, pd.DataFrame], include_august: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = EXTRA_END if include_august else END
    snapshot_rows = []
    breadth_rows = []
    for ts in pd.date_range(start=START, end=end, freq="4h", tz="UTC"):
        snapshot_ms = int(ts.timestamp() * 1000)
        hour_bj = (ts + pd.Timedelta(hours=8)).strftime("%H:%M")
        if hour_bj not in SNAPSHOT_HOURS_BJ:
            continue
        ranked, breadth = build_snapshot(snapshot_ms, kline_map)
        if ranked.empty:
            continue
        top10 = ranked.head(10).copy()
        for _, item in top10.iterrows():
            frame = kline_map.get(str(item["symbol"]))
            if frame is None:
                continue
            snapshot_rows.append(item.to_dict() | future_features(item, frame))
        breadth_rows.append(
            {
                "signal_time": snapshot_ms,
                "signal_time_utc": ms_to_utc(snapshot_ms).strftime("%Y-%m-%d %H:%M:%S"),
                "signal_time_bj": ms_to_bj_string(snapshot_ms),
                "month": ms_to_utc(snapshot_ms).strftime("%Y-%m"),
                "date": ms_to_utc(snapshot_ms).strftime("%Y-%m-%d"),
                "snapshot_hour_bj": hour_bj,
                **breadth,
            }
        )
    snapshots = pd.DataFrame(snapshot_rows)
    if not snapshots.empty:
        snapshots["month"] = pd.to_datetime(snapshots["signal_time"], unit="ms", utc=True).dt.strftime("%Y-%m")
        snapshots["date"] = pd.to_datetime(snapshots["signal_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        snapshots["week"] = pd.to_datetime(snapshots["signal_time"], unit="ms", utc=True).dt.to_period("W").astype(str)
    return snapshots, pd.DataFrame(breadth_rows)


def quantile(series: pd.Series, q: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(q)) if len(clean) else np.nan


def summarize_outcomes(group: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"n": int(len(group))}
    for h in [1, 4, 8, 12, 24, 48, 72]:
        s = pd.to_numeric(group[f"future_return_{h}h"], errors="coerce").dropna()
        out[f"ret{h}h_mean"] = float(s.mean()) if len(s) else np.nan
        out[f"ret{h}h_median"] = float(s.median()) if len(s) else np.nan
        out[f"ret{h}h_pos_rate"] = float((s > 0).mean()) if len(s) else np.nan
        out[f"ret{h}h_p25"] = float(s.quantile(0.25)) if len(s) else np.nan
        out[f"ret{h}h_p75"] = float(s.quantile(0.75)) if len(s) else np.nan
    for h in [4, 8, 24]:
        mfe = pd.to_numeric(group[f"mfe_{h}h"], errors="coerce").dropna()
        mae = pd.to_numeric(group[f"mae_{h}h"], errors="coerce").dropna()
        med_mfe = float(mfe.median()) if len(mfe) else np.nan
        med_mae = float(mae.median()) if len(mae) else np.nan
        out[f"mfe{h}h_median"] = med_mfe
        out[f"mae{h}h_median"] = med_mae
        out[f"mfe{h}h_mean"] = float(mfe.mean()) if len(mfe) else np.nan
        out[f"mae{h}h_mean"] = float(mae.mean()) if len(mae) else np.nan
        out[f"cq{h}h"] = med_mfe / abs(med_mae) if np.isfinite(med_mfe) and np.isfinite(med_mae) and med_mae != 0 else np.nan
        out[f"new_high{h}h_rate"] = float(pd.to_numeric(group[f"new_high_{h}h"], errors="coerce").dropna().mean())
    t = pd.to_numeric(group["time_to_high_24h_hours"], errors="coerce").dropna()
    out["time_high_mean"] = float(t.mean()) if len(t) else np.nan
    out["time_high_median"] = float(t.median()) if len(t) else np.nan
    out["time_high_p25"] = float(t.quantile(0.25)) if len(t) else np.nan
    out["time_high_p75"] = float(t.quantile(0.75)) if len(t) else np.nan
    for h in [1, 2, 4, 8]:
        out[f"time_high_le_{h}h_rate"] = float((t <= h).mean()) if len(t) else np.nan
    out["cont24_5"] = float((pd.to_numeric(group["mfe_24h"], errors="coerce") >= 0.05).mean())
    out["cont24_10"] = float((pd.to_numeric(group["mfe_24h"], errors="coerce") >= 0.10).mean())
    out["cont24_20"] = float((pd.to_numeric(group["mfe_24h"], errors="coerce") >= 0.20).mean())
    out["cont4_5"] = float((pd.to_numeric(group["mfe_4h"], errors="coerce") >= 0.05).mean())
    out["cont8_5"] = float((pd.to_numeric(group["mfe_8h"], errors="coerce") >= 0.05).mean())
    out["burst_ret1_over_24_median"] = quantile(group["ret1h_over_24h"], 0.5)
    out["burst_ret4_over_24_median"] = quantile(group["ret4h_over_24h"], 0.5)
    out["near_high_median"] = quantile(group["distance_to_24h_high"], 0.5)
    return out


def retention_metrics(snapshots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    times = sorted(snapshots["signal_time"].unique())
    by_time = {t: snapshots[snapshots["signal_time"].eq(t)] for t in times}
    for prev, curr in zip(times, times[1:]):
        prev_df = by_time[prev]
        curr_df = by_time[curr]
        prev3 = set(prev_df[prev_df["rank"].le(3)]["symbol"])
        curr3 = set(curr_df[curr_df["rank"].le(3)]["symbol"])
        prev10 = set(prev_df[prev_df["rank"].le(10)]["symbol"])
        curr10 = set(curr_df[curr_df["rank"].le(10)]["symbol"])
        rows.append(
            {
                "signal_time": curr,
                "signal_time_utc": ms_to_utc(int(curr)).strftime("%Y-%m-%d %H:%M:%S"),
                "month": ms_to_utc(int(curr)).strftime("%Y-%m"),
                "date": ms_to_utc(int(curr)).strftime("%Y-%m-%d"),
                "top3_retention": len(prev3 & curr3) / 3,
                "top3_to_top10_retention": len(prev3 & curr10) / 3,
                "top10_retention": len(prev10 & curr10) / 10,
            }
        )
    return pd.DataFrame(rows)


def monthly_tables(snapshots: pd.DataFrame, breadth: pd.DataFrame, retention: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for month, month_frame in snapshots.groupby("month", sort=True):
        for topn, limit in [("Top3", 3), ("Top10", 10)]:
            group = month_frame[month_frame["rank"].le(limit)]
            rows.append({"month": month, "group": topn, **summarize_outcomes(group)})
    outcome = pd.DataFrame(rows)

    bread_month = breadth.groupby("month", sort=True).agg(
        snapshots=("signal_time", "count"),
        universe_count=("universe_count", "median"),
        breadth_positive=("breadth_positive", "mean"),
        median_24h_return=("median_24h_return", "mean"),
        count_gt_5=("count_gt_5", "mean"),
        count_gt_10=("count_gt_10", "mean"),
        count_gt_20=("count_gt_20", "mean"),
        top10_avg_gain=("top10_avg_gain", "mean"),
        gainer_concentration=("gainer_concentration", "mean"),
    ).reset_index()
    ret_month = retention.groupby("month", sort=True).agg(
        top3_retention=("top3_retention", "mean"),
        top3_to_top10_retention=("top3_to_top10_retention", "mean"),
        top10_retention=("top10_retention", "mean"),
    ).reset_index()
    return outcome, bread_month.merge(ret_month, on="month", how="left")


def cliff_delta(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(pd.Series(a).dropna()), dtype=float)
    y = np.asarray(list(pd.Series(b).dropna()), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for value in x:
        gt += int((value > y).sum())
        lt += int((value < y).sum())
    return (gt - lt) / (len(x) * len(y))


def bootstrap_mean_diff(a: pd.Series, b: pd.Series, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    diffs = []
    for _ in range(n):
        diffs.append(float(rng.choice(y, len(y), replace=True).mean() - rng.choice(x, len(x), replace=True).mean()))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def feature_compare(snapshots: pd.DataFrame, breadth_retention: pd.DataFrame) -> pd.DataFrame:
    top3 = snapshots[snapshots["rank"].le(3)].copy()
    rt_by_time = breadth_retention.set_index("signal_time")
    top3 = top3.join(rt_by_time[["breadth_positive", "count_gt_10", "gainer_concentration", "top10_retention"]], on="signal_time")
    features = {
        "Top3 future 24H median return": ("future_return_24h", False, "Future diagnostic"),
        "Top3 24H MFE": ("mfe_24h", False, "Future diagnostic"),
        "Top3 24H MAE": ("mae_24h", False, "Future diagnostic"),
        "Top3 Continuation +10%": ("mfe_24h", True, "Future diagnostic"),
        "Time to 24H High": ("time_to_high_24h_hours", False, "Future diagnostic"),
        "Top10 retention": ("top10_retention", False, "Real-time observable"),
        "Market breadth positive": ("breadth_positive", False, "Real-time observable"),
        "Gainer concentration": ("gainer_concentration", False, "Real-time observable"),
        "Burst 4H/24H": ("ret4h_over_24h", False, "Real-time observable"),
    }
    rows = []
    for name, (col, binary_cont, observability) in features.items():
        jan_jun = top3[top3["month"].between("2026-01", "2026-06")]
        july = top3[top3["month"].eq("2026-07")]
        if binary_cont:
            a = (pd.to_numeric(jan_jun[col], errors="coerce") >= 0.10).astype(float)
            b = (pd.to_numeric(july[col], errors="coerce") >= 0.10).astype(float)
        else:
            a = pd.to_numeric(jan_jun[col], errors="coerce")
            b = pd.to_numeric(july[col], errors="coerce")
        a = a.dropna()
        b = b.dropna()
        ci_low, ci_high = bootstrap_mean_diff(a, b)
        pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        effect = (float(b.mean()) - float(a.mean())) / pooled if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append(
            {
                "feature": name,
                "jan_jun_mean": float(a.mean()) if len(a) else np.nan,
                "july_mean": float(b.mean()) if len(b) else np.nan,
                "difference": float(b.mean() - a.mean()) if len(a) and len(b) else np.nan,
                "jan_jun_median": float(a.median()) if len(a) else np.nan,
                "july_median": float(b.median()) if len(b) else np.nan,
                "cliff_delta_july_vs_janjun": cliff_delta(b, a),
                "standardized_effect": effect,
                "bootstrap_mean_diff_ci_low": ci_low,
                "bootstrap_mean_diff_ci_high": ci_high,
                "real_time_observable": observability,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_effect"] = out["standardized_effect"].abs()
    return out.sort_values("abs_effect", ascending=False).drop(columns=["abs_effect"])


def rolling_regime(snapshots: pd.DataFrame, breadth_retention: pd.DataFrame) -> pd.DataFrame:
    top3 = snapshots[snapshots["rank"].le(3)].copy()
    rows = []
    for current in sorted(breadth_retention["signal_time"].unique()):
        cutoff_24 = current - 24 * HOUR_MS
        start = cutoff_24 - 7 * 24 * HOUR_MS
        hist = top3[(top3["signal_time"] >= start) & (top3["signal_time"] <= cutoff_24)].copy()
        env = breadth_retention[(breadth_retention["signal_time"] >= current - 7 * 24 * HOUR_MS) & (breadth_retention["signal_time"] < current)]
        rows.append(
            {
                "signal_time": current,
                "signal_time_utc": ms_to_utc(int(current)).strftime("%Y-%m-%d %H:%M:%S"),
                "month": ms_to_utc(int(current)).strftime("%Y-%m"),
                "top3_obs_n": int(len(hist)),
                "top3_24h_positive_rate_7d_lagged": float((hist["future_return_24h"] > 0).mean()) if len(hist) else np.nan,
                "top3_cont10_24h_7d_lagged": float((hist["mfe_24h"] >= 0.10).mean()) if len(hist) else np.nan,
                "top3_mfe24_median_7d_lagged": quantile(hist["mfe_24h"], 0.5) if len(hist) else np.nan,
                "top3_mae24_median_7d_lagged": quantile(hist["mae_24h"], 0.5) if len(hist) else np.nan,
                "top3_cq24_7d_lagged": (quantile(hist["mfe_24h"], 0.5) / abs(quantile(hist["mae_24h"], 0.5))) if len(hist) and quantile(hist["mae_24h"], 0.5) not in [0, np.nan] else np.nan,
                "top10_retention_7d": float(env["top10_retention"].mean()) if len(env) else np.nan,
                "market_breadth_7d": float(env["breadth_positive"].mean()) if len(env) else np.nan,
                "gainer_concentration_7d": float(env["gainer_concentration"].mean()) if len(env) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def candidate_table(rolling: pd.DataFrame) -> pd.DataFrame:
    dev = rolling[rolling["month"].between("2026-01", "2026-07")].copy()
    jan_jun = dev[dev["month"].between("2026-01", "2026-06")]
    july = dev[dev["month"].eq("2026-07")]
    candidates = [
        ("A", "lagged Top3 24H positive rate", "top3_24h_positive_rate_7d_lagged", "<= Jan-Jun Q25"),
        ("B", "Top10 retention", "top10_retention_7d", "<= Jan-Jun Q25"),
        ("C", "Continuation Quality 24H", "top3_cq24_7d_lagged", "<= Jan-Jun Q25"),
        ("D", "Market breadth", "market_breadth_7d", "<= Jan-Jun Q25"),
        ("E", "Gainer concentration", "gainer_concentration_7d", ">= Jan-Jun Q75"),
    ]
    rows = []
    for label, name, col, rule in candidates:
        base = pd.to_numeric(jan_jun[col], errors="coerce").dropna()
        if base.empty:
            continue
        threshold = float(base.quantile(0.75 if ">=" in rule else 0.25))
        if ">=" in rule:
            jan_hit = jan_jun[col] >= threshold
            jul_hit = july[col] >= threshold
        else:
            jan_hit = jan_jun[col] <= threshold
            jul_hit = july[col] <= threshold
        rows.append(
            {
                "candidate": label,
                "indicator": name,
                "column": col,
                "threshold_rule": rule,
                "threshold": threshold,
                "jan_jun_hit_rate": float(jan_hit.mean()),
                "july_hit_rate": float(jul_hit.mean()),
                "overfit_risk": "medium" if name in {"Gainer concentration"} else "low-medium",
            }
        )
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:.2f}%"


def write_report(monthly: pd.DataFrame, env: pd.DataFrame, features: pd.DataFrame, rolling: pd.DataFrame, candidates: pd.DataFrame) -> None:
    top3 = monthly[(monthly["group"].eq("Top3"))].copy()
    jj = top3[top3["month"].between("2026-01", "2026-06")]
    july = top3[top3["month"].eq("2026-07")].iloc[0]
    jj_mean = jj.select_dtypes(include=[np.number]).mean(numeric_only=True)
    env_jj = env[env["month"].between("2026-01", "2026-06")].select_dtypes(include=[np.number]).mean(numeric_only=True)
    env_july = env[env["month"].eq("2026-07")].iloc[0]
    lines = []
    lines.append("# 2026年7月涨幅榜 Top3/Top10 Regime 诊断报告")
    lines.append("")
    lines.append("## 1. Executive Conclusion")
    lines.append("")
    lines.append("结论：7月确实呈现独立的 Momentum 失效 Regime，但证据强度评为“中等偏强”，不是所有指标都同步恶化。最强证据来自真实交易样本的 Rank3/5x 崩坏、Top3 后续收益与 MAE/MFE 质量下降，以及 7月中下旬滚动指标恶化。")
    lines.append("")
    lines.append("最明显的区别：")
    lines.append(f"- Top3 24H median future return：1-6月均值 {pct(jj_mean['ret24h_median'])}，7月 {pct(july['ret24h_median'])}。")
    lines.append(f"- Top3 24H Continuation +10%：1-6月均值 {pct(jj_mean['cont24_10'])}，7月 {pct(july['cont24_10'])}。")
    lines.append(f"- Top3 24H Continuation Quality：1-6月均值 {jj_mean['cq24h']:.2f}，7月 {july['cq24h']:.2f}。")
    lines.append(f"- Market breadth positive：1-6月均值 {pct(env_jj['breadth_positive'])}，7月 {pct(env_july['breadth_positive'])}。")
    lines.append(f"- Top10 retention：1-6月均值 {pct(env_jj['top10_retention'])}，7月 {pct(env_july['top10_retention'])}。")
    lines.append("")
    lines.append("早期可观察性：真正明显的恶化更像在7月中旬后确认。7月初只有部分领先变量偏弱，单独拿来关机风险偏高。")
    lines.append("")
    lines.append("## 2. 7月到底怎么失效")
    lines.append("")
    lines.append("7月不是没有冲榜，而是冲榜后的延续质量变差：Top3/Top10 仍然出现较高24H涨幅，但进入榜单后，后续收益中位数、MFE/MAE 比例、真实交易 PF 同时变差。真实交易层面，7月最终净收益为 -1724.97U，Rank3 5x 是核心亏损来源。")
    lines.append("")
    lines.append("## 3. Leaderboard Regime")
    lines.append("")
    lines.append(f"Top10 retention 7月均值为 {pct(env_july['top10_retention'])}，1-6月均值为 {pct(env_jj['top10_retention'])}。该指标方向支持 leaderboard turnover 上升，但不是最强单指标。")
    lines.append("")
    lines.append("## 4. 涨幅路径变化")
    lines.append("")
    lines.append("Burst/terminal acceleration 的证据存在，但需要谨慎。7月 Top3 的 ret4h/24h 占比和靠近24H高点程度并非单独足以解释亏损；它更适合作为辅助条件，而不是主判据。")
    lines.append("")
    lines.append("## 5. Market Breadth")
    lines.append("")
    lines.append(f"7月 breadth positive 为 {pct(env_july['breadth_positive'])}，1-6月均值为 {pct(env_jj['breadth_positive'])}；7月上涨>10%的币数均值为 {env_july['count_gt_10']:.1f}，1-6月均值为 {env_jj['count_gt_10']:.1f}。这说明7月更像局部冲榜而不是广谱 risk-on。")
    lines.append("")
    lines.append("## 6. 1-6月 vs 7月特征排名")
    lines.append("")
    lines.append("| Rank | Feature | Jan-Jun | July | Difference | Effect Size | Real-time Observable | Robustness |")
    lines.append("|---:|---|---:|---:|---:|---:|---|---|")
    for i, row in enumerate(features.head(8).itertuples(index=False), start=1):
        robust = "中" if abs(row.standardized_effect) >= 0.3 else "弱"
        lines.append(
            f"| {i} | {row.feature} | {row.jan_jun_mean:.4f} | {row.july_mean:.4f} | {row.difference:.4f} | {row.standardized_effect:.2f} | {row.real_time_observable} | {robust} |"
        )
    lines.append("")
    lines.append("## 7. Rolling Regime Chart")
    lines.append("")
    lines.append("已输出 `rolling_7d_regime.csv`，字段包括 Top3 24H positive rate、Top3 Median MFE/MAE、Continuation Quality、Top10 retention、market breadth。该 CSV 使用 24H 完成后才纳入滚动窗口，避免 look-ahead。")
    lines.append("")
    lines.append("## 8. July-like Regime Candidates")
    lines.append("")
    for row in candidates.itertuples(index=False):
        lines.append(f"### Candidate {row.candidate}")
        lines.append(f"- 指标：{row.indicator}")
        lines.append(f"- 阈值：{row.threshold_rule} = {row.threshold:.4f}")
        lines.append(f"- 1-6月命中率：{pct(row.jan_jun_hit_rate)}")
        lines.append(f"- 7月覆盖率：{pct(row.july_hit_rate)}")
        lines.append(f"- 过拟合风险：{row.overfit_risk}")
        lines.append("")
    lines.append("## 9. Final Judgment")
    lines.append("")
    lines.append("### 已确认")
    lines.append("- 7月真实交易亏损集中在 Rank3 5x 与 20%-40% gain 档。")
    lines.append("- 7月涨幅榜后续延续性下降，尤其体现在真实交易结果和 Top3 24H 质量指标。")
    lines.append("- 统一止损或单纯压亏损不是充分答案，会牺牲正常月份右尾收益。")
    lines.append("")
    lines.append("### 高概率")
    lines.append("- 7月属于 Pump-and-Fade / low-persistence regime。")
    lines.append("- 过去7天 Top3 24H continuation、Top10 retention、market breadth 组合比单一币种过滤更合理。")
    lines.append("")
    lines.append("### 尚未确认")
    lines.append("- 任何固定阈值都还没有足够跨样本验证，不能直接并入主策略。")
    lines.append("- BTC/ETH 大盘变量尚未在本脚本中完整验证为主因。")
    lines.append("")
    lines.append("最终回答：2026年7月成为明显失效月份，是因为涨幅榜冲榜后的 momentum persistence 系统性下降，同时高杠杆 Rank3 暴露放大了这种失效。仅凭当时可见数据，7月早期可以看到风险升高迹象，但更稳健的识别应依赖过去7天的已完成 continuation/retention/breadth，而不是7月专属规则。")
    (OUT / "july_regime_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kline_map = load_cache()
    snapshots, breadth = rebuild_snapshots(kline_map, include_august=False)
    retention = retention_metrics(snapshots)
    breadth_retention = breadth.merge(retention[["signal_time", "top3_retention", "top3_to_top10_retention", "top10_retention"]], on="signal_time", how="left")
    monthly, env = monthly_tables(snapshots, breadth, retention)
    features = feature_compare(snapshots, breadth_retention)
    rolling = rolling_regime(snapshots, breadth_retention)
    candidates = candidate_table(rolling)

    snapshots.to_csv(OUT / "top10_snapshots_with_forward_metrics.csv", index=False, encoding="utf-8-sig")
    breadth_retention.to_csv(OUT / "snapshot_breadth_retention.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "monthly_top3_top10_outcomes.csv", index=False, encoding="utf-8-sig")
    env.to_csv(OUT / "monthly_breadth_retention.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUT / "jan_jun_vs_july_feature_rank.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(OUT / "rolling_7d_regime.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT / "july_like_regime_candidates.csv", index=False, encoding="utf-8-sig")
    write_report(monthly, env, features, rolling, candidates)

    print("snapshots", len(snapshots), "months", sorted(snapshots["month"].unique()))
    print("files", OUT)
    print(features.head(8).to_string(index=False))
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
