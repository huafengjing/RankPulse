from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
SIGNALS_PATH = Path("outputs/signals.csv")
OUT = Path("outputs/compression")
CHARTS = OUT / "charts"

INTERVAL_MS = 5 * 60_000
OBS_HOURS = 240
OBS_MS = OBS_HOURS * 60 * 60_000
FEE = 0.0005
SLIP = 0.0005


def load_klines() -> pd.DataFrame:
    frames = []
    for path in DATA_DIR.glob("*/*.parquet"):
        frame = pd.read_parquet(
            path,
            columns=["symbol", "open_time", "open", "high", "low", "close", "volume", "quote_volume"],
        )
        frame["_mtime"] = path.stat().st_mtime
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("_mtime").drop_duplicates(["symbol", "open_time"], keep="last")
    data = data.drop(columns=["_mtime"]).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    data["open_time_utc"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    return data


def add_base_features(klines: pd.DataFrame) -> pd.DataFrame:
    out = []
    for symbol, g in klines.groupby("symbol", sort=False):
        g = g.sort_values("open_time").copy()
        g["ret_5m"] = g["close"].pct_change()
        for periods, name in [(3, "15m"), (12, "1h"), (48, "4h"), (288, "24h")]:
            g[f"return_{name}"] = g["close"] / g["close"].shift(periods) - 1
        g["ema20_5m"] = g["close"].ewm(span=20, adjust=False).mean()
        g["distance_to_5m_ema20_pct"] = g["close"] / g["ema20_5m"] - 1
        typical = (g["high"] + g["low"] + g["close"]) / 3
        g["vwap_24h"] = (typical * g["volume"]).rolling(288, min_periods=20).sum() / g["volume"].rolling(288, min_periods=20).sum()
        g["distance_to_vwap_pct"] = g["close"] / g["vwap_24h"] - 1
        g["quote_volume_24h"] = g["quote_volume"].rolling(288, min_periods=20).sum()
        g["volume_1h_at_signal"] = g["quote_volume"].rolling(12, min_periods=1).sum()
        g["volume_4h_at_signal"] = g["quote_volume"].rolling(48, min_periods=1).sum()
        g["volume_1h_prev"] = g["quote_volume"].shift(12).rolling(12, min_periods=1).sum()
        g["volume_4h_prev"] = g["quote_volume"].shift(48).rolling(48, min_periods=1).sum()
        g["volume_7d"] = g["quote_volume"].rolling(2016, min_periods=100).sum()
        g["volatility_1h"] = g["ret_5m"].rolling(12, min_periods=3).std()
        g["volatility_4h"] = g["ret_5m"].rolling(48, min_periods=10).std()
        body = (g["close"] - g["open"]).abs()
        candle_range = (g["high"] - g["low"]).replace(0, np.nan)
        g["candle_body_pct_at_signal"] = body / candle_range
        g["upper_wick_pct_at_signal"] = (g["high"] - g[["open", "close"]].max(axis=1)) / candle_range
        g["lower_wick_pct_at_signal"] = (g[["open", "close"]].min(axis=1) - g["low"]) / candle_range
        green = (g["close"] > g["open"]).astype(int)
        g["consecutive_green_5m_count"] = green.groupby((green != green.shift()).cumsum()).cumcount() + 1
        g.loc[green == 0, "consecutive_green_5m_count"] = 0
        g["recent_high_breakout_1h"] = g["high"] >= g["high"].shift(1).rolling(12, min_periods=3).max()
        g["recent_high_breakout_4h"] = g["high"] >= g["high"].shift(1).rolling(48, min_periods=10).max()
        g["market_age_days"] = (g["open_time"] - int(g["open_time"].min())) / 86_400_000
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_resampled_features(klines: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, g in klines.groupby("symbol", sort=False):
        g = g.sort_values("open_time").copy().set_index("open_time_utc")
        close_15 = g["close"].resample("15min", label="right", closed="right").last().dropna()
        open_15 = g["open"].resample("15min", label="right", closed="right").first().dropna()
        ema15 = close_15.ewm(span=20, adjust=False).mean().shift(1).reindex(g.index, method="ffill")
        green15 = (close_15 > open_15).astype(int)
        green15_count = green15.groupby((green15 != green15.shift()).cumsum()).cumcount() + 1
        green15_count.loc[green15 == 0] = 0
        green15_5m = green15_count.shift(1).reindex(g.index, method="ffill").fillna(0)
        close_1h = g["close"].resample("1h", label="right", closed="right").last().dropna()
        ema1h = close_1h.ewm(span=20, adjust=False).mean().shift(1).reindex(g.index, method="ffill")
        g["distance_to_15m_ema20_pct"] = g["close"] / ema15.values - 1
        g["distance_to_1h_ema20_pct"] = g["close"] / ema1h.values - 1
        g["consecutive_green_15m_count"] = green15_5m.values
        g = g.reset_index()
        frames.append(g[["symbol", "open_time", "distance_to_15m_ema20_pct", "distance_to_1h_ema20_pct", "consecutive_green_15m_count"]])
    return klines.merge(pd.concat(frames, ignore_index=True), on=["symbol", "open_time"], how="left")


def build_rankings(klines: pd.DataFrame) -> pd.DataFrame:
    ranked = klines[["symbol", "open_time", "close", "quote_volume"]].copy()
    ranked["prev_24h_close"] = ranked.groupby("symbol")["close"].shift(288)
    ranked = ranked[ranked["prev_24h_close"].notna()].copy()
    ranked["rolling_24h_gain"] = ranked["close"] / ranked["prev_24h_close"] - 1
    ranked["rank"] = ranked.groupby("open_time")["rolling_24h_gain"].rank(method="first", ascending=False)
    ranked["rank"] = ranked["rank"].astype(int)
    return ranked


def btc_features(klines: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    btc = klines[klines["symbol"].eq("BTCUSDT")].sort_values("open_time").copy()
    if btc.empty:
        return pd.DataFrame({"open_time": times.unique(), "btc_regime_label": "unknown"})
    btc["btc_return_15m"] = btc["close"] / btc["close"].shift(3) - 1
    btc["btc_return_1h"] = btc["close"] / btc["close"].shift(12) - 1
    btc["btc_return_4h"] = btc["close"] / btc["close"].shift(48) - 1
    btc["btc_return_24h"] = btc["close"] / btc["close"].shift(288) - 1
    btc["btc_ema20_1h"] = btc["close"].resample("1h", on="open_time_utc", label="right", closed="right").last().ewm(span=20, adjust=False).mean().shift(1).reindex(btc["open_time_utc"], method="ffill").values
    btc["btc_ema20_4h"] = btc["close"].resample("4h", on="open_time_utc", label="right", closed="right").last().ewm(span=20, adjust=False).mean().shift(1).reindex(btc["open_time_utc"], method="ffill").values
    btc["btc_above_1h_ema20"] = btc["close"] > btc["btc_ema20_1h"]
    btc["btc_above_4h_ema20"] = btc["close"] > btc["btc_ema20_4h"]
    btc["btc_regime_label"] = np.where(
        (btc["btc_return_1h"] >= 0) & (btc["btc_return_4h"] >= 0),
        "bullish",
        np.where((btc["btc_return_1h"] < -0.01) | (btc["btc_return_4h"] < -0.02), "bearish", "neutral"),
    )
    cols = ["open_time", "btc_return_15m", "btc_return_1h", "btc_return_4h", "btc_return_24h", "btc_above_1h_ema20", "btc_above_4h_ema20", "btc_regime_label"]
    return btc[cols]


def path_labels(signals: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kmap = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in klines.groupby("symbol")}
    thresholds = [0.10, 0.20, 0.30, 0.50, 1.00]
    for _, sig in signals.iterrows():
        g = kmap.get(sig["symbol"])
        if g is None:
            continue
        future_idx = g.index[g["open_time"] > int(sig["signal_time"])]
        if len(future_idx) == 0:
            continue
        entry_idx = int(future_idx[0])
        entry = g.loc[entry_idx]
        entry_time = int(entry["open_time"])
        entry_price = float(entry["open"])
        path = g[(g["open_time"] >= entry_time) & (g["open_time"] <= entry_time + OBS_MS)]
        max_ret = float(path["high"].max() / entry_price - 1)
        min_ret = float(path["low"].min() / entry_price - 1)
        hit = {f"hit_plus_{int(t*100)}_before_minus_10": False for t in thresholds}
        t_hit = {f"time_to_plus_{int(t*100)}_minutes": np.nan for t in thresholds[:-1]}
        hit_minus_10_first = False
        touched_minus_10_240h = False
        time_to_minus_10 = np.nan
        for _, bar in path.iterrows():
            minutes = (int(bar["open_time"]) - entry_time) / 60_000
            low_ret = float(bar["low"]) / entry_price - 1
            high_ret = float(bar["high"]) / entry_price - 1
            if low_ret <= -0.10:
                touched_minus_10_240h = True
                hit_minus_10_first = not hit["hit_plus_10_before_minus_10"]
                time_to_minus_10 = minutes
                break
            for t in thresholds:
                key = f"hit_plus_{int(t*100)}_before_minus_10"
                if high_ret >= t:
                    hit[key] = True
                    tkey = f"time_to_plus_{int(t*100)}_minutes"
                    if tkey in t_hit and pd.isna(t_hit[tkey]):
                        t_hit[tkey] = minutes
        if hit["hit_plus_50_before_minus_10"]:
            label = "A"
            reason = "hit +50 before -10"
        elif hit["hit_plus_20_before_minus_10"] or hit["hit_plus_30_before_minus_10"]:
            label = "B"
            reason = "hit +20/+30 before -10 but not +50"
        elif hit_minus_10_first:
            label = "C"
            reason = "hit -10 before meaningful continuation"
        else:
            label = "D"
            reason = "no strong continuation and no immediate -10 label"
        rows.append(
            {
                "signal_id": int(sig["signal_id"]),
                **hit,
                "hit_minus_10_first": hit_minus_10_first,
                "touched_minus_10_240h": touched_minus_10_240h,
                "max_forward_return_240h": max_ret,
                "min_forward_return_240h": min_ret,
                **t_hit,
                "time_to_minus_10_minutes": time_to_minus_10,
                "class_label": label,
                "class_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def signal_features(signals: pd.DataFrame, klines: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "symbol", "open_time", "open", "high", "low", "close", "quote_volume", "market_age_days",
        "return_15m", "return_1h", "return_4h", "return_24h", "candle_body_pct_at_signal",
        "upper_wick_pct_at_signal", "lower_wick_pct_at_signal", "consecutive_green_5m_count",
        "consecutive_green_15m_count", "distance_to_5m_ema20_pct", "distance_to_15m_ema20_pct",
        "distance_to_1h_ema20_pct", "distance_to_vwap_pct", "quote_volume_24h", "volume_1h_at_signal",
        "volume_4h_at_signal", "volume_1h_prev", "volume_4h_prev", "volume_7d", "volatility_1h",
        "volatility_4h", "recent_high_breakout_1h", "recent_high_breakout_4h",
    ]
    signals = signals.rename(columns={"close": "close_at_signal", "quote_volume": "quote_volume_at_signal"}).copy()
    snap = klines[feature_cols].drop(columns=["close"]).rename(columns={"open_time": "signal_time", "quote_volume": "volume_5m_at_signal"})
    df = signals.merge(snap, on=["symbol", "signal_time"], how="left")
    df = df.rename(
        columns={
            "rank": "rank_at_signal",
            "rolling_24h_change_pct": "rolling_24h_gain_at_signal",
            "close": "close_at_signal",
        }
    )
    df["rank_bucket"] = pd.cut(df["rank_at_signal"], [0, 3, 5, 10], labels=["1-3", "4-5", "6-10"])
    df["rolling_24h_gain_bucket"] = pd.cut(
        df["rolling_24h_gain_at_signal"],
        [-np.inf, 0.15, 0.30, 0.50, 0.80, 1.20, np.inf],
        labels=["0-15", "15-30", "30-50", "50-80", "80-120", ">120"],
    )
    df["volume_5m_vs_24h_avg"] = df["volume_5m_at_signal"] / (df["quote_volume_24h"] / 288)
    df["volume_1h_vs_24h_avg"] = df["volume_1h_at_signal"] / (df["quote_volume_24h"] / 24)
    df["volume_4h_vs_7d_avg"] = df["volume_4h_at_signal"] / (df["volume_7d"] / 42)
    df["volume_acceleration_1h"] = df["volume_1h_at_signal"] / df["volume_1h_prev"].replace(0, np.nan)
    df["volume_acceleration_4h"] = df["volume_4h_at_signal"] / df["volume_4h_prev"].replace(0, np.nan)
    df["above_5m_ema20"] = df["distance_to_5m_ema20_pct"] > 0
    df["above_15m_ema20"] = df["distance_to_15m_ema20_pct"] > 0
    df["above_1h_ema20"] = df["distance_to_1h_ema20_pct"] > 0
    df = df.merge(btc_features(klines, df["signal_time"]), left_on="signal_time", right_on="open_time", how="left").drop(columns=["open_time"], errors="ignore")

    rank_cols = rankings[["symbol", "open_time", "rank"]].copy()
    for minutes, name in [(60, "1h"), (240, "4h")]:
        past = rank_cols.copy()
        past["signal_time"] = past["open_time"] + minutes * 60_000
        past = past.rename(columns={"rank": f"rank_{name}_ago"})
        df = df.merge(past[["symbol", "signal_time", f"rank_{name}_ago"]], on=["symbol", "signal_time"], how="left")
        df[f"rank_improvement_last_{name}"] = df[f"rank_{name}_ago"] - df["rank_at_signal"]

    top20 = rankings[rankings["rank"] <= 20][["symbol", "open_time"]]
    speeds = []
    for _, row in df.iterrows():
        hist = top20[(top20["symbol"] == row["symbol"]) & (top20["open_time"] <= row["signal_time"])]
        if hist.empty:
            speeds.append(np.nan)
        else:
            speeds.append((row["signal_time"] - int(hist.iloc[-1]["open_time"])) / 60_000)
    df["top10_entry_speed"] = speeds
    return df


def add_posthoc_features(df: pd.DataFrame, rankings: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kmap = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in klines.groupby("symbol")}
    rmap = {s: g.sort_values("open_time") for s, g in rankings.groupby("symbol")}
    for _, row in df.iterrows():
        symbol = row["symbol"]
        t = int(row["signal_time"])
        rg = rmap.get(symbol, pd.DataFrame())
        kg = kmap.get(symbol, pd.DataFrame())
        future_r = rg[rg["open_time"] > t]
        future_k = kg[kg["open_time"] > t].head(48)
        f1 = future_r[future_r["open_time"] <= t + 60 * 60_000]
        f4 = future_r[future_r["open_time"] <= t + 4 * 60 * 60_000]
        f24 = future_r[future_r["open_time"] <= t + 24 * 60 * 60_000]
        in_top10 = future_r[future_r["rank"] <= 10]
        stayed = 0
        if not in_top10.empty:
            for _, rr in future_r.iterrows():
                if int(rr["rank"]) <= 10:
                    stayed += 5
                else:
                    break
        rows.append(
            {
                "signal_id": row["signal_id"],
                "entered_top5_within_1h": bool((f1["rank"] <= 5).any()) if not f1.empty else False,
                "entered_top5_within_4h": bool((f4["rank"] <= 5).any()) if not f4.empty else False,
                "entered_top5_within_24h": bool((f24["rank"] <= 5).any()) if not f24.empty else False,
                "entered_top3_within_24h": bool((f24["rank"] <= 3).any()) if not f24.empty else False,
                "stayed_in_top10_duration_minutes": stayed,
                "dropped_out_top20_within_1h": bool((f1["rank"] > 20).any()) if not f1.empty else False,
                "dropped_out_top20_within_4h": bool((f4["rank"] > 20).any()) if not f4.empty else False,
                "price_held_above_ema20_after_signal_1h": bool((future_k.head(12)["distance_to_5m_ema20_pct"] > 0).all()) if not future_k.empty and "distance_to_5m_ema20_pct" in future_k else False,
                "price_held_above_ema20_after_signal_4h": bool((future_k["distance_to_5m_ema20_pct"] > 0).all()) if not future_k.empty and "distance_to_5m_ema20_pct" in future_k else False,
                "pullback_to_ema20_and_bounced": bool(((future_k["distance_to_5m_ema20_pct"].abs() <= 0.01) & (future_k["close"] > future_k["ema20_5m"])).any()) if not future_k.empty and "ema20_5m" in future_k else False,
                "pullback_to_vwap_and_bounced": bool(((future_k["distance_to_vwap_pct"].abs() <= 0.01) & (future_k["close"] > future_k["vwap_24h"])).any()) if not future_k.empty and "vwap_24h" in future_k else False,
            }
        )
    return df.merge(pd.DataFrame(rows), on="signal_id", how="left")


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df["quote_volume_bucket"] = pd.cut(df["quote_volume_24h"], [-np.inf, 10e6, 20e6, 50e6, 100e6, 300e6, np.inf], labels=["<10M", "10-20M", "20-50M", "50-100M", "100-300M", ">300M"])
    df["distance_to_5m_ema20_bucket"] = pd.cut(df["distance_to_5m_ema20_pct"], [-np.inf, .02, .05, .08, .12, np.inf], labels=["<=2", "2-5", "5-8", "8-12", ">12"])
    df["distance_to_vwap_bucket"] = pd.cut(df["distance_to_vwap_pct"], [-np.inf, .02, .05, .08, .12, np.inf], labels=["<=2", "2-5", "5-8", "8-12", ">12"])
    df["upper_wick_bucket"] = pd.cut(df["upper_wick_pct_at_signal"], [-np.inf, .1, .25, .5, np.inf], labels=["<=10", "10-25", "25-50", ">50"])
    df["volume_acceleration_bucket"] = pd.cut(df["volume_acceleration_1h"], [-np.inf, 1, 1.5, 2, 3, np.inf], labels=["<=1", "1-1.5", "1.5-2", "2-3", ">3"])
    df["stayed_in_top10_duration_bucket"] = pd.cut(df["stayed_in_top10_duration_minutes"], [-np.inf, 30, 120, 360, 1440, np.inf], labels=["<=30m", "30m-2h", "2h-6h", "6h-24h", ">24h"])
    return df


def score(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(0, index=df.index, dtype=float)
    s += np.select([df["rank_at_signal"].between(6, 10), df["rank_at_signal"].between(4, 5), df["rank_at_signal"].between(1, 3)], [20, 15, 5], 0)
    g = df["rolling_24h_gain_at_signal"]
    s += np.select([g.between(.15, .30, inclusive="left"), g.between(.30, .50, inclusive="left"), g.between(.50, .80, inclusive="left"), g.between(.80, 1.20, inclusive="left"), g >= 1.20, g < .15], [20, 18, 12, 5, -10, 5], 0)
    v = df["quote_volume_24h"].fillna(0)
    s += np.select([v > 300e6, v.between(100e6, 300e6), v.between(50e6, 100e6), v.between(20e6, 50e6), v < 20e6], [15, 12, 10, 6, -10], 0)
    d = df["distance_to_5m_ema20_pct"].fillna(9)
    s += np.select([d <= .02, d.between(.02, .05), d.between(.05, .08), d.between(.08, .12), d > .12], [15, 12, 8, 2, -10], 0)
    btc1 = df.get("btc_return_1h", pd.Series(np.nan, index=df.index)).fillna(0)
    btc4 = df.get("btc_return_4h", pd.Series(np.nan, index=df.index)).fillna(0)
    s += np.select([(btc1 >= 0) & (btc4 >= 0), (btc1 >= -.01) & (btc4 >= -.02)], [10, 5], -10)
    s += np.where(df["upper_wick_pct_at_signal"].fillna(0) > .5, -15, 0)
    s += np.where(df["consecutive_green_5m_count"].fillna(0) >= 6, -8, 0)
    s += np.where(g > 1.20, -15, 0)
    s += np.where(df["distance_to_vwap_pct"].fillna(0) > .12, -10, 0)
    s += np.where(v < 10e6, -20, 0)
    return s.clip(lower=0, upper=100)


def class_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "rank_bucket", "rolling_24h_gain_bucket", "quote_volume_bucket", "distance_to_5m_ema20_bucket",
        "distance_to_vwap_bucket", "upper_wick_bucket", "volume_acceleration_bucket", "btc_regime_label",
        "entered_top5_within_4h", "stayed_in_top10_duration_bucket",
    ]
    rows = []
    for field in fields:
        for bucket, g in df.groupby(field, dropna=False):
            total = len(g)
            counts = g["class_label"].value_counts()
            a = int(counts.get("A", 0)); b = int(counts.get("B", 0)); c = int(counts.get("C", 0)); d = int(counts.get("D", 0))
            rows.append(
                {
                    "feature": field, "bucket": str(bucket), "total_count": total,
                    "A_count": a, "B_count": b, "C_count": c, "D_count": d,
                    "A_rate": a / total, "B_rate": b / total, "C_rate": c / total, "D_rate": d / total,
                    "AB_rate": (a + b) / total, "CD_rate": (c + d) / total,
                    "plus50_rate": g["hit_plus_50_before_minus_10"].mean(),
                    "plus100_rate": g["hit_plus_100_before_minus_10"].mean(),
                    "minus10_first_rate": g["hit_minus_10_first"].mean(),
                }
            )
    return pd.DataFrame(rows)


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {}
    masks["rule_rank4_10_gain15_80_vol20m"] = df["rank_at_signal"].between(4, 10) & df["rolling_24h_gain_at_signal"].between(.15, .80) & (df["quote_volume_24h"] >= 20e6)
    masks["rule_gain15_50_rank6_10"] = df["rank_at_signal"].between(6, 10) & df["rolling_24h_gain_at_signal"].between(.15, .50)
    masks["rule_rank4_10_gain15_80_dist8"] = masks["rule_rank4_10_gain15_80_vol20m"] & (df["distance_to_5m_ema20_pct"] <= .08)
    masks["rule_rank4_10_gain15_80_dist8_btc"] = masks["rule_rank4_10_gain15_80_dist8"] & (~df["btc_regime_label"].eq("bearish"))
    masks["rule_rank4_10_gain15_80_dist8_volacc"] = masks["rule_rank4_10_gain15_80_dist8"] & (df["volume_1h_vs_24h_avg"] >= 1.5)
    masks["rule_gain10_30_rank6_10_not_over_vwap12"] = df["rank_at_signal"].between(6, 10) & df["rolling_24h_gain_at_signal"].between(.10, .30) & (df["distance_to_vwap_pct"] <= .12)
    masks["rule_no_upper_wick_gain15_50"] = df["rolling_24h_gain_at_signal"].between(.15, .50) & (df["upper_wick_pct_at_signal"] <= .5) & df["rank_at_signal"].between(4, 10)
    for pct in [10, 15, 20, 25, 30]:
        cutoff = df["signal_quality_score"].quantile(1 - pct / 100)
        masks[f"score_top_{pct}pct"] = df["signal_quality_score"] >= cutoff
    return masks


def compression_summary(df: pd.DataFrame) -> pd.DataFrame:
    base_plus50 = df["hit_plus_50_before_minus_10"].mean()
    base_plus100 = df["hit_plus_100_before_minus_10"].mean()
    total_plus50 = df["hit_plus_50_before_minus_10"].sum()
    total_plus100 = df["hit_plus_100_before_minus_10"].sum()
    rows = []
    for name, mask in rule_masks(df).items():
        g = df[mask.fillna(False)]
        if g.empty:
            continue
        counts = g["class_label"].value_counts()
        rows.append(
            {
                "rule_name": name,
                "selected_count": len(g),
                "selected_pct": len(g) / len(df),
                "A_count": int(counts.get("A", 0)), "B_count": int(counts.get("B", 0)),
                "C_count": int(counts.get("C", 0)), "D_count": int(counts.get("D", 0)),
                "A_rate": (g["class_label"] == "A").mean(),
                "AB_rate": g["class_label"].isin(["A", "B"]).mean(),
                "C_rate": (g["class_label"] == "C").mean(),
                "D_rate": (g["class_label"] == "D").mean(),
                "hit_plus_10_rate": g["hit_plus_10_before_minus_10"].mean(),
                "hit_plus_20_rate": g["hit_plus_20_before_minus_10"].mean(),
                "hit_plus_30_rate": g["hit_plus_30_before_minus_10"].mean(),
                "hit_plus_50_rate": g["hit_plus_50_before_minus_10"].mean(),
                "hit_plus_100_rate": g["hit_plus_100_before_minus_10"].mean(),
                "minus10_first_rate": g["hit_minus_10_first"].mean(),
                "avg_max_forward_return_240h": g["max_forward_return_240h"].mean(),
                "median_max_forward_return_240h": g["max_forward_return_240h"].median(),
                "avg_min_forward_return_240h": g["min_forward_return_240h"].mean(),
                "median_min_forward_return_240h": g["min_forward_return_240h"].median(),
                "plus50_capture_rate": g["hit_plus_50_before_minus_10"].sum() / total_plus50 if total_plus50 else 0,
                "plus100_capture_rate": g["hit_plus_100_before_minus_10"].sum() / total_plus100 if total_plus100 else 0,
                "enrichment_plus50": g["hit_plus_50_before_minus_10"].mean() / base_plus50 if base_plus50 else 0,
                "enrichment_plus100": g["hit_plus_100_before_minus_10"].mean() / base_plus100 if base_plus100 else 0,
                "estimated_trading_frequency_per_day": len(g) / 60,
            }
        )
    return pd.DataFrame(rows).sort_values(["selected_pct", "hit_plus_50_rate"], ascending=[True, False])


def compressed_backtest(df: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    kmap = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in klines.groupby("symbol")}

    def run_one(selected: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, sig in selected.sort_values("signal_time").iterrows():
            g = kmap.get(sig["symbol"])
            if g is None:
                continue
            idx = g.index[g["open_time"] > int(sig["signal_time"])]
            if len(idx) == 0:
                continue
            entry_idx = int(idx[0])
            entry = g.loc[entry_idx]
            entry_time = int(entry["open_time"])
            entry_price = float(entry["open"])
            path = g[(g["open_time"] >= entry_time) & (g["open_time"] <= entry_time + OBS_MS)]
            remaining = 1.0
            net = 0.0
            hits = {10: False, 20: False, 30: False}
            breakeven = False
            exit_reason = "max_holding"
            for _, bar in path.iterrows():
                low_ret = float(bar["low"]) / entry_price - 1
                high_ret = float(bar["high"]) / entry_price - 1
                if low_ret <= (0 if breakeven else -0.10):
                    exit_price = entry_price if breakeven else entry_price * .90
                    net += remaining * (exit_price * (1 - SLIP) / (entry_price * (1 + SLIP)) - 1 - FEE - FEE)
                    remaining = 0
                    exit_reason = "sl_or_be"
                    break
                for pct, frac in [(10, .4), (20, .2), (30, .2)]:
                    if remaining > 0 and not hits[pct] and high_ret >= pct / 100:
                        exit_price = entry_price * (1 + pct / 100)
                        net += frac * (exit_price * (1 - SLIP) / (entry_price * (1 + SLIP)) - 1 - FEE - FEE)
                        remaining -= frac
                        hits[pct] = True
                        if pct == 10:
                            breakeven = True
            if remaining > 0:
                last = path.iloc[-1]
                net += remaining * (float(last["close"]) * (1 - SLIP) / (entry_price * (1 + SLIP)) - 1 - FEE - FEE)
            rows.append({"net_return_pct": net, "mfe_pct": sig["max_forward_return_240h"], "mae_pct": sig["min_forward_return_240h"], **{f"tp{p}_hit": hits[p] for p in [10, 20, 30]}})
        return pd.DataFrame(rows)

    rows = []
    for name, mask in rule_masks(df).items():
        trades = run_one(df[mask.fillna(False)])
        if trades.empty:
            continue
        r = trades["net_return_pct"]
        wins = r[r > 0]
        losses = r[r <= 0]
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        rows.append(
            {
                "rule_name": name,
                "selected_count": int(mask.sum()),
                "total_trades": len(trades),
                "avg_net_return_pct": r.mean(),
                "median_net_return_pct": r.median(),
                "win_rate": (r > 0).mean(),
                "profit_factor": wins.sum() / abs(losses.sum()) if abs(losses.sum()) else np.inf,
                "expectancy_pct": r.mean(),
                "max_drawdown_pct": dd,
                "avg_mfe_pct": trades["mfe_pct"].mean(),
                "avg_mae_pct": trades["mae_pct"].mean(),
                "tp10_hit_rate": trades["tp10_hit"].mean(),
                "tp20_hit_rate": trades["tp20_hit"].mean(),
                "tp30_hit_rate": trades["tp30_hit"].mean(),
                "best_trade_pct": r.max(),
                "worst_trade_pct": r.min(),
                "result_excluding_best_1_trade": r.sort_values(ascending=False).iloc[1:].mean() if len(r) > 1 else np.nan,
                "result_excluding_best_5_trades": r.sort_values(ascending=False).iloc[5:].mean() if len(r) > 5 else np.nan,
                "result_excluding_best_10_trades": r.sort_values(ascending=False).iloc[10:].mean() if len(r) > 10 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("profit_factor", ascending=False)


def charts(df: pd.DataFrame, rules: pd.DataFrame, bt: pd.DataFrame) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    df["class_label"].value_counts().sort_index().plot(kind="bar", title="Class Distribution")
    plt.tight_layout(); plt.savefig(CHARTS / "class_distribution.png"); plt.close()
    pivot = pd.crosstab(df["rolling_24h_gain_bucket"], df["class_label"], normalize="index")
    plt.imshow(pivot.fillna(0), aspect="auto"); plt.xticks(range(len(pivot.columns)), pivot.columns); plt.yticks(range(len(pivot.index)), pivot.index); plt.colorbar(); plt.title("ABCD Feature Heatmap")
    plt.tight_layout(); plt.savefig(CHARTS / "abcd_feature_heatmap.png"); plt.close()
    df["score_decile"] = pd.qcut(df["signal_quality_score"], 10, duplicates="drop")
    df.groupby("score_decile", observed=False)["hit_plus_50_before_minus_10"].mean().plot(kind="bar", title="Plus50 Hit Rate by Score Decile")
    plt.tight_layout(); plt.savefig(CHARTS / "hit_rate_by_score_decile.png"); plt.close()
    rules.set_index("rule_name")["plus50_capture_rate"].plot(kind="bar", title="Plus50 Capture by Rule")
    plt.tight_layout(); plt.savefig(CHARTS / "plus50_capture_by_rule.png"); plt.close()
    rules.set_index("rule_name")[["enrichment_plus50", "enrichment_plus100"]].plot(kind="bar", title="Enrichment by Rule")
    plt.tight_layout(); plt.savefig(CHARTS / "enrichment_by_rule.png"); plt.close()
    rules.plot.scatter("selected_count", "hit_plus_50_rate", title="Selected Count vs Plus50 Rate")
    plt.tight_layout(); plt.savefig(CHARTS / "selected_count_vs_plus50_rate.png"); plt.close()
    bt.set_index("rule_name")["avg_net_return_pct"].plot(kind="bar", title="Compressed Strategy Avg Return")
    plt.tight_layout(); plt.savefig(CHARTS / "compressed_strategy_equity_curves.png"); plt.close()
    pd.Series({"raw_plus50": df["hit_plus_50_before_minus_10"].mean(), "best_rule_plus50": rules["hit_plus_50_rate"].max()}).plot(kind="bar", title="Raw vs Compressed")
    plt.tight_layout(); plt.savefig(CHARTS / "raw_vs_compressed_comparison.png"); plt.close()
    df["signal_quality_score"].hist(bins=30); plt.title("Signal Quality Score Distribution")
    plt.tight_layout(); plt.savefig(CHARTS / "signal_quality_score_distribution.png"); plt.close()


def report(df: pd.DataFrame, rules: pd.DataFrame, bt: pd.DataFrame) -> None:
    class_counts = df["class_label"].value_counts().sort_index()
    target = rules[(rules["selected_pct"].between(.15, .25))].sort_values(["hit_plus_50_rate", "plus50_capture_rate"], ascending=False)
    best = target.iloc[0] if not target.empty else rules.sort_values("hit_plus_50_rate", ascending=False).iloc[0]
    best_bt = bt[bt["rule_name"].eq(best["rule_name"])]
    best_bt_text = best_bt.iloc[0].to_dict() if not best_bt.empty else {}
    text = f"""# Signal Compression Report

## 1. ABCD Class Distribution

{class_counts.to_string()}

Total signals: {len(df)}

## 2. A/B Shared Traits

The most useful immediate features are ranking bucket, rolling 24h gain bucket, distance to 5m EMA20, 24h quote volume, and upper wick. A/B signals should be interpreted as signals that reached meaningful continuation before -10%.

## 3. C/D Shared Traits

C/D labels are dominated by early -10% failures or lack of continuation. High overextension, weak volume context, and large wick/distance penalties are the main interpretable risk flags in this first pass.

## 4. Most Distinguishing Signal-Time Features

See `class_feature_summary.csv`. Use only signal-time fields for entry filters. Top5 conversion and stayed-in-rank fields are post-hoc confirmation fields.

## 5. Can We Compress to 15%-25%?

Best candidate in the 15%-25% band:

{best.to_string()}

## 6. Most Robust Rule

Rule selected by this report: `{best['rule_name']}`.

## 7. Score Top 20%

Review `score_top_20pct` in `compression_rules_summary.csv` and `compressed_backtest_summary.csv`.

## 8. Compressed Backtest

Best candidate backtest snapshot:

{best_bt_text}

## 9. Tail Dependence

Backtest summary includes results excluding best 1/5/10 trades. If those columns deteriorate heavily, returns still depend on rare outliers.

## 10. Next Stage

This is still Research. Do not move to live monitoring or execution yet. The next step is to inspect the candidate rules manually and rerun with stricter universe/history assumptions.
"""
    (OUT / "signal_compression_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading klines/signals...", flush=True)
    signals = pd.read_csv(SIGNALS_PATH)
    klines = add_resampled_features(add_base_features(load_klines()))
    rankings = build_rankings(klines)
    print("Building labels/features...", flush=True)
    labels = path_labels(signals, klines)
    df = signal_features(signals, klines, rankings).merge(labels, on="signal_id", how="inner")
    df = add_posthoc_features(df, rankings, klines)
    df = add_buckets(df)
    df["signal_quality_score"] = score(df)
    front_cols = [
        "signal_id", "symbol", "signal_time_utc", "rank_at_signal", "rolling_24h_gain_at_signal",
        "quote_volume_at_signal", "close_at_signal", "market_age_days", "class_label", "class_reason",
    ]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]
    df.to_csv(OUT / "signal_classification.csv", index=False)
    print("Summarizing classes/rules/backtests...", flush=True)
    csum = class_feature_summary(df)
    csum.to_csv(OUT / "class_feature_summary.csv", index=False)
    rsum = compression_summary(df)
    rsum.to_csv(OUT / "compression_rules_summary.csv", index=False)
    bt = compressed_backtest(df, klines)
    bt.to_csv(OUT / "compressed_backtest_summary.csv", index=False)
    charts(df, rsum, bt)
    report(df, rsum, bt)
    print("Done. Outputs written to outputs/compression", flush=True)
    print(rsum.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
