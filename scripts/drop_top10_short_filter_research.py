from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import INTERVAL_MS, load_cached_klines
from src.data.quality import check_klines
from src.research.ranking import add_rolling_24h_change


OUT = ROOT / "outputs" / "drop_top10_short_filter"
INTERVAL = "5m"
INTERVAL_MINUTES = 5
LOOKBACK_DAYS = 180
COOLDOWN_DAYS = 5
LOOKAHEAD_HOURS = 120
FEE = 0.0005
SLIPPAGE = 0.0005
BASE_DROP_MIN = 0.15
BASE_DROP_MAX = 0.20


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.astype(str).values.tolist()]
    return "\n".join([header, sep, *rows])


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(wins / abs(losses)) if abs(losses) > 0 else np.inf


def pnl_excluding_best(pnl: pd.Series, n: int) -> float:
    if pnl.empty:
        return 0.0
    return float(pnl.sort_values(ascending=False).iloc[n:].sum()) if len(pnl) > n else 0.0


def build_rankings(klines: pd.DataFrame) -> pd.DataFrame:
    frame = add_rolling_24h_change(klines, INTERVAL_MINUTES)
    frame = frame[frame["has_full_24h_history"]].copy()
    frame["drop"] = -frame["rolling_24h_change_pct"]
    frame["drop_rank"] = frame.groupby("open_time")["rolling_24h_change_pct"].rank(method="first", ascending=True).astype(int)
    frame["gain_rank"] = frame.groupby("open_time")["rolling_24h_change_pct"].rank(method="first", ascending=False).astype(int)
    frame["is_drop_top10"] = frame["drop_rank"] <= 10
    frame["is_gain_top10"] = frame["gain_rank"] <= 10
    keep = [
        "open_time",
        "open_time_utc",
        "symbol",
        "drop_rank",
        "gain_rank",
        "rolling_24h_change_pct",
        "drop",
        "close",
        "quote_volume",
        "is_drop_top10",
        "is_gain_top10",
    ]
    return frame[keep].sort_values(["open_time", "drop_rank"]).reset_index(drop=True)


def identify_base_signals(rankings: pd.DataFrame) -> pd.DataFrame:
    cooldown_ms = COOLDOWN_DAYS * 24 * 60 * 60 * 1000
    rows = []
    signal_id = 1
    for symbol, group in rankings.sort_values("open_time").groupby("symbol"):
        hits = group[group["is_drop_top10"]]
        previous_time: int | None = None
        for _, row in hits.iterrows():
            now = int(row["open_time"])
            if previous_time is not None and now - previous_time <= cooldown_ms:
                previous_time = now
                continue
            drop = float(row["drop"])
            if BASE_DROP_MIN <= drop < BASE_DROP_MAX:
                rows.append(
                    {
                        "signal_id": signal_id,
                        "symbol": symbol,
                        "signal_time": now,
                        "signal_time_utc": row["open_time_utc"],
                        "rank_at_signal": int(row["drop_rank"]),
                        "rolling_24h_drop_at_signal": drop,
                        "price_at_signal": float(row["close"]),
                        "quote_volume_at_signal": float(row.get("quote_volume", 0.0)),
                    }
                )
                signal_id += 1
            previous_time = now
    return pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)


def add_kline_features(klines: pd.DataFrame) -> pd.DataFrame:
    out = klines.sort_values(["symbol", "open_time"]).copy()
    grouped = out.groupby("symbol", sort=False)
    out["ema20_5m"] = grouped["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    out["ema20_15m"] = grouped["close"].transform(lambda s: s.ewm(span=60, adjust=False).mean())
    out["ema20_1h"] = grouped["close"].transform(lambda s: s.ewm(span=240, adjust=False).mean())
    out["ma14_4h"] = grouped["close"].transform(lambda s: s.rolling(14 * 48, min_periods=20).mean())
    pv = out["close"] * out["quote_volume"]
    out["_pv"] = pv
    out["vwap_24h"] = grouped["_pv"].transform(lambda s: s.rolling(288, min_periods=20).sum()) / grouped["quote_volume"].transform(
        lambda s: s.rolling(288, min_periods=20).sum()
    )
    for bars, name in [(1, "5m"), (3, "15m"), (12, "1h"), (48, "4h"), (288, "24h")]:
        out[f"quote_volume_{name}_sum"] = grouped["quote_volume"].transform(lambda s, b=bars: s.rolling(b, min_periods=1).sum())
    out["quote_volume_24h_avg_5m"] = grouped["quote_volume"].transform(lambda s: s.rolling(288, min_periods=20).mean())
    return out.drop(columns=["_pv"], errors="ignore")


def asof_row(group: pd.DataFrame, time_ms: int) -> pd.Series | None:
    idx = group.index[group["open_time"] <= time_ms]
    if len(idx) == 0:
        return None
    return group.loc[int(idx[-1])]


def future_entry(group: pd.DataFrame, signal_time: int, delay_minutes: int = 0) -> pd.Series | None:
    target = signal_time + delay_minutes * 60_000
    idx = group.index[group["open_time"] > target]
    if len(idx) == 0:
        return None
    return group.loc[int(idx[0])]


def first_hit_labels(path: pd.DataFrame, entry_time: int, entry_price: float) -> dict[str, object]:
    labels: dict[str, object] = {}
    pairs = [
        ("hit_plus_5_before_minus_10", 0.05, -0.10),
        ("hit_plus_10_before_minus_10", 0.10, -0.10),
        ("hit_plus_10_before_minus_20", 0.10, -0.20),
        ("hit_plus_15_before_minus_20", 0.15, -0.20),
        ("hit_plus_20_before_minus_20", 0.20, -0.20),
        ("hit_minus_5_before_plus_10", -0.05, 0.10),
        ("hit_minus_10_before_plus_10", -0.10, 0.10),
        ("hit_minus_15_before_plus_10", -0.15, 0.10),
        ("hit_minus_20_before_plus_10", -0.20, 0.10),
        ("hit_minus_30_before_plus_10", -0.30, 0.10),
        ("hit_minus_40_before_plus_10", -0.40, 0.10),
        ("hit_minus_50_before_plus_10", -0.50, 0.10),
    ]
    pair_results = {name: None for name, _, _ in pairs}

    time_targets = {0.10: "time_to_plus_10_minutes", -0.10: "time_to_minus_10_minutes", -0.20: "time_to_minus_20_minutes", -0.30: "time_to_minus_30_minutes"}
    for col in time_targets.values():
        labels[col] = pd.NA

    for _, bar in path.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        now = int(bar["open_time"])
        for target, col in time_targets.items():
            if pd.isna(labels[col]):
                if target > 0 and high >= entry_price * (1 + target):
                    labels[col] = int((now - entry_time) / 60_000)
                if target < 0 and low <= entry_price * (1 + target):
                    labels[col] = int((now - entry_time) / 60_000)
        for name, target, barrier in pairs:
            if pair_results[name] is not None:
                continue
            target_hit = high >= entry_price * (1 + target) if target > 0 else low <= entry_price * (1 + target)
            barrier_hit = high >= entry_price * (1 + barrier) if barrier > 0 else low <= entry_price * (1 + barrier)
            if not target_hit and not barrier_hit:
                continue
            if target_hit and barrier_hit:
                # Conservative for shorts: adverse upside wins same-candle ties.
                pair_results[name] = target > 0
            else:
                pair_results[name] = bool(target_hit)
    for name in pair_results:
        labels[name] = bool(pair_results[name]) if pair_results[name] is not None else False
    for horizon, col in [(24, "24h"), (72, "72h"), (120, "120h")]:
        hpath = path[path["open_time"] <= entry_time + horizon * 60 * 60 * 1000]
        labels[f"max_favorable_down_{col}"] = float(1 - hpath["low"].min() / entry_price) if not hpath.empty else 0.0
        labels[f"max_adverse_up_{col}"] = float(hpath["high"].max() / entry_price - 1) if not hpath.empty else 0.0
    return labels


def make_path_labels(signals: pd.DataFrame, kmap: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, sig in signals.iterrows():
        group = kmap[sig["symbol"]]
        entry = future_entry(group, int(sig["signal_time"]))
        if entry is None:
            continue
        entry_time = int(entry["open_time"])
        entry_price = float(entry["open"])
        end = entry_time + LOOKAHEAD_HOURS * 60 * 60 * 1000
        path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end)]
        if path.empty:
            continue
        rows.append(
            {
                "signal_id": int(sig["signal_id"]),
                "symbol": sig["symbol"],
                "signal_time_utc": sig["signal_time_utc"],
                "entry_time": entry_time,
                "entry_time_utc": entry["open_time_utc"],
                "entry_price": entry_price,
                **first_hit_labels(path, entry_time, entry_price),
            }
        )
    return pd.DataFrame(rows)


def consecutive_red(group: pd.DataFrame, pos: int, bars: int) -> int:
    count = 0
    start = max(0, pos - bars + 1)
    for _, row in group.iloc[start : pos + 1].iloc[::-1].iterrows():
        if float(row["close"]) < float(row["open"]):
            count += 1
        else:
            break
    return count


def extract_features(signals: pd.DataFrame, klines: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    kmap = {s: g.reset_index(drop=True) for s, g in klines.groupby("symbol", sort=False)}
    rmap = {int(t): g for t, g in rankings.groupby("open_time", sort=False)}
    btc = kmap.get("BTCUSDT")
    rows = []
    for _, sig in signals.iterrows():
        symbol = sig["symbol"]
        group = kmap[symbol]
        signal_time = int(sig["signal_time"])
        pos_arr = group.index[group["open_time"] == signal_time]
        if len(pos_arr) == 0:
            pos_arr = group.index[group["open_time"] <= signal_time]
        if len(pos_arr) == 0:
            continue
        pos = int(pos_arr[-1])
        row = group.loc[pos]
        close = float(row["close"])
        snap = rmap.get(signal_time, pd.DataFrame())
        top_drop = snap.nsmallest(10, "drop_rank") if not snap.empty else pd.DataFrame()
        top_gain = snap.nsmallest(10, "gain_rank") if not snap.empty else pd.DataFrame()
        old30 = asof_row(rankings[rankings["symbol"] == symbol], signal_time - 30 * 60_000)
        old60 = asof_row(rankings[rankings["symbol"] == symbol], signal_time - 60 * 60_000)
        hist = rankings[(rankings["symbol"] == symbol) & (rankings["open_time"] < signal_time)]
        def was(col: str, days: int) -> bool:
            left = signal_time - days * 24 * 60 * 60 * 1000
            return bool(hist[(hist["open_time"] >= left) & (hist[col])].shape[0])

        def ret_back(bars: int) -> float:
            if pos - bars < 0:
                return np.nan
            return close / float(group.loc[pos - bars, "close"]) - 1.0

        qavg = float(row["quote_volume_24h_avg_5m"]) if pd.notna(row["quote_volume_24h_avg_5m"]) and row["quote_volume_24h_avg_5m"] else np.nan
        vol_1h = float(row["quote_volume_1h_sum"]) / (qavg * 12) if qavg and not pd.isna(qavg) else np.nan
        vol_4h = float(row["quote_volume_4h_sum"]) / (qavg * 48) if qavg and not pd.isna(qavg) else np.nan
        low_15 = float(group.iloc[max(0, pos - 2) : pos + 1]["low"].min())
        low_30 = float(group.iloc[max(0, pos - 5) : pos + 1]["low"].min())
        low_1h = float(group.iloc[max(0, pos - 11) : pos + 1]["low"].min())
        last3 = group.iloc[max(0, pos - 2) : pos + 1]
        lower_sum = ((last3[["open", "close"]].min(axis=1) - last3["low"]).clip(lower=0)).sum()
        range_sum = (last3["high"] - last3["low"]).replace(0, np.nan).sum()
        signal_range = max(float(row["high"]) - float(row["low"]), 1e-12)
        btc_row = asof_row(btc, signal_time) if btc is not None else None
        btc_pos = int(btc.index[btc["open_time"] <= signal_time][-1]) if btc is not None and len(btc.index[btc["open_time"] <= signal_time]) else None
        def btc_ret(bars: int) -> float:
            if btc is None or btc_pos is None or btc_pos - bars < 0:
                return np.nan
            return float(btc.loc[btc_pos, "close"]) / float(btc.loc[btc_pos - bars, "close"]) - 1.0

        features = {
            "signal_id": int(sig["signal_id"]),
            "symbol": symbol,
            "signal_time_utc": sig["signal_time_utc"],
            "rank_at_signal": int(sig["rank_at_signal"]),
            "rolling_24h_drop_at_signal": float(sig["rolling_24h_drop_at_signal"]),
            "rolling_24h_drop_bucket": "15-20%",
            "top5_drop_avg": float(top_drop.nsmallest(5, "drop_rank")["drop"].mean()) if not top_drop.empty else np.nan,
            "top10_drop_avg": float(top_drop["drop"].mean()) if not top_drop.empty else np.nan,
            "top5_max_drop": float(top_drop.nsmallest(5, "drop_rank")["drop"].max()) if not top_drop.empty else np.nan,
            "top10_max_drop": float(top_drop["drop"].max()) if not top_drop.empty else np.nan,
            "drop_rank_improvement_last_30m": (int(old30["drop_rank"]) - int(sig["rank_at_signal"])) if old30 is not None else np.nan,
            "drop_rank_improvement_last_1h": (int(old60["drop_rank"]) - int(sig["rank_at_signal"])) if old60 is not None else np.nan,
            "volume_5m_vs_24h_avg": float(row["quote_volume"]) / qavg if qavg and not pd.isna(qavg) else np.nan,
            "volume_15m_vs_24h_avg": float(row["quote_volume_15m_sum"]) / (qavg * 3) if qavg and not pd.isna(qavg) else np.nan,
            "volume_1h_vs_24h_avg": vol_1h,
            "volume_4h_vs_24h_avg": vol_4h,
            "quote_volume_at_signal": float(sig["quote_volume_at_signal"]),
            "quote_volume_24h": float(row["quote_volume_24h_sum"]),
            "volume_trend_1h_vs_4h": vol_1h / vol_4h if vol_4h and not pd.isna(vol_4h) else np.nan,
            "return_5m_before_signal": ret_back(1),
            "return_15m_before_signal": ret_back(3),
            "return_30m_before_signal": ret_back(6),
            "return_1h_before_signal": ret_back(12),
            "return_4h_before_signal": ret_back(48),
            "consecutive_red_5m_count": consecutive_red(group, pos, 12),
            "consecutive_red_15m_count": consecutive_red(group, pos, 36) // 3,
            "signal_candle_body_pct": abs(float(row["close"]) - float(row["open"])) / signal_range,
            "signal_candle_upper_wick_pct": (float(row["high"]) - max(float(row["open"]), float(row["close"]))) / signal_range,
            "signal_candle_lower_wick_pct": (min(float(row["open"]), float(row["close"])) - float(row["low"])) / signal_range,
            "signal_candle_range_pct": signal_range / close,
            "distance_to_5m_ema20_pct": close / float(row["ema20_5m"]) - 1 if pd.notna(row["ema20_5m"]) else np.nan,
            "distance_to_15m_ema20_pct": close / float(row["ema20_15m"]) - 1 if pd.notna(row["ema20_15m"]) else np.nan,
            "distance_to_1h_ema20_pct": close / float(row["ema20_1h"]) - 1 if pd.notna(row["ema20_1h"]) else np.nan,
            "distance_to_vwap_pct": close / float(row["vwap_24h"]) - 1 if pd.notna(row["vwap_24h"]) else np.nan,
            "distance_to_4h_ma14_pct": close / float(row["ma14_4h"]) - 1 if pd.notna(row["ma14_4h"]) else np.nan,
            "price_below_5m_ema20": close < float(row["ema20_5m"]) if pd.notna(row["ema20_5m"]) else False,
            "price_below_15m_ema20": close < float(row["ema20_15m"]) if pd.notna(row["ema20_15m"]) else False,
            "price_below_vwap": close < float(row["vwap_24h"]) if pd.notna(row["vwap_24h"]) else False,
            "rebound_from_local_low_15m_pct": close / low_15 - 1,
            "rebound_from_local_low_30m_pct": close / low_30 - 1,
            "rebound_from_local_low_1h_pct": close / low_1h - 1,
            "lower_wick_ratio_last_3_5m": float(lower_sum / range_sum) if range_sum and not pd.isna(range_sum) else np.nan,
            "buy_recovery_score": (close / low_30 - 1) + (float(lower_sum / range_sum) if range_sum and not pd.isna(range_sum) else 0.0),
            "btc_return_1h": btc_ret(12),
            "btc_return_4h": btc_ret(48),
            "btc_return_24h": btc_ret(288),
            "market_drop_top10_avg": float(top_drop["drop"].mean()) if not top_drop.empty else np.nan,
            "market_gain_top10_avg": float(top_gain["rolling_24h_change_pct"].mean()) if not top_gain.empty else np.nan,
            "gain_gt_20_count": int((snap["rolling_24h_change_pct"] > 0.20).sum()) if not snap.empty else 0,
            "drop_gt_15_count": int((snap["drop"] > 0.15).sum()) if not snap.empty else 0,
            "board_temperature_label": "panic" if (not top_drop.empty and top_drop["drop"].mean() >= 0.15) else "normal",
            "was_in_drop_top10_last_24h": was("is_drop_top10", 1),
            "was_in_drop_top10_last_3d": was("is_drop_top10", 3),
            "was_in_drop_top10_last_5d": was("is_drop_top10", 5),
            "was_in_gain_top10_last_3d": was("is_gain_top10", 3),
            "was_in_gain_top10_last_5d": was("is_gain_top10", 5),
        }
        rows.append(features)
    return pd.DataFrame(rows)


def short_leg_pnl(entry: float, exit_price: float, weight: float) -> float:
    entry_eff = entry * (1 - SLIPPAGE)
    exit_eff = exit_price * (1 + SLIPPAGE)
    return weight * ((entry_eff - exit_eff) / entry_eff - 2 * FEE)


def simulate_exit(path: pd.DataFrame, entry_price: float, exit_rule: str) -> tuple[float, dict[str, object]]:
    sl = entry_price * 1.10
    tp1 = entry_price * 0.90
    tp2 = entry_price * 0.80
    pnl = 0.0
    tp1_hit = False
    minus20_hit = False
    minus30_hit = bool((path["low"] <= entry_price * 0.70).any())
    first_plus10 = False
    if exit_rule == "Exit C":
        for _, bar in path.iterrows():
            if float(bar["high"]) >= sl:
                return short_leg_pnl(entry_price, sl, 1.0), {"exit_reason": "sl", "tp1_hit": False, "minus20_hit": False, "minus30_hit": minus30_hit, "first_plus10": True}
            if float(bar["low"]) <= tp1:
                return short_leg_pnl(entry_price, tp1, 1.0), {"exit_reason": "tp", "tp1_hit": True, "minus20_hit": False, "minus30_hit": minus30_hit, "first_plus10": False}
        return short_leg_pnl(entry_price, float(path.iloc[-1]["close"]), 1.0), {"exit_reason": "time", "tp1_hit": False, "minus20_hit": False, "minus30_hit": minus30_hit, "first_plus10": False}
    if exit_rule == "Exit A":
        remaining = 1.0
        for _, bar in path.iterrows():
            if float(bar["high"]) >= sl:
                pnl += short_leg_pnl(entry_price, sl, remaining)
                first_plus10 = not tp1_hit
                return pnl, {"exit_reason": "sl", "tp1_hit": tp1_hit, "minus20_hit": minus20_hit, "minus30_hit": minus30_hit, "first_plus10": first_plus10}
            if not tp1_hit and float(bar["low"]) <= tp1:
                pnl += short_leg_pnl(entry_price, tp1, 0.5)
                remaining = 0.5
                tp1_hit = True
            if tp1_hit and float(bar["low"]) <= tp2:
                pnl += short_leg_pnl(entry_price, tp2, remaining)
                minus20_hit = True
                return pnl, {"exit_reason": "tp2", "tp1_hit": tp1_hit, "minus20_hit": minus20_hit, "minus30_hit": minus30_hit, "first_plus10": False}
        pnl += short_leg_pnl(entry_price, float(path.iloc[-1]["close"]), remaining)
        return pnl, {"exit_reason": "time", "tp1_hit": tp1_hit, "minus20_hit": minus20_hit, "minus30_hit": minus30_hit, "first_plus10": False}
    # Exit B
    remaining = 1.0
    trail = None
    low_water = entry_price
    for _, bar in path.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        if high >= sl and not tp1_hit:
            return short_leg_pnl(entry_price, sl, 1.0), {"exit_reason": "sl", "tp1_hit": False, "minus20_hit": False, "minus30_hit": minus30_hit, "first_plus10": True}
        if not tp1_hit and low <= tp1:
            pnl += short_leg_pnl(entry_price, tp1, 0.5)
            remaining = 0.5
            tp1_hit = True
            low_water = min(low_water, low)
            trail = low_water * 1.20
            continue
        if tp1_hit:
            low_water = min(low_water, low)
            trail = low_water * 1.20
            minus20_hit = minus20_hit or low <= tp2
            if high >= trail:
                pnl += short_leg_pnl(entry_price, trail, remaining)
                return pnl, {"exit_reason": "trail", "tp1_hit": tp1_hit, "minus20_hit": minus20_hit, "minus30_hit": minus30_hit, "first_plus10": False}
    pnl += short_leg_pnl(entry_price, float(path.iloc[-1]["close"]), remaining)
    return pnl, {"exit_reason": "time", "tp1_hit": tp1_hit, "minus20_hit": minus20_hit, "minus30_hit": minus30_hit, "first_plus10": False}


def make_backtest_trades(events: pd.DataFrame, kmap: dict[str, pd.DataFrame], rule_name: str, entry_model: str, exit_rule: str) -> pd.DataFrame:
    rows = []
    for _, event in events.iterrows():
        group = kmap[event["symbol"]]
        entry_time = int(event["entry_time"])
        entry_price = float(event["entry_price"])
        path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= entry_time + LOOKAHEAD_HOURS * 60 * 60 * 1000)]
        if path.empty:
            continue
        pnl, meta = simulate_exit(path, entry_price, exit_rule)
        rows.append(
            {
                "rule_name": rule_name,
                "entry_model": entry_model,
                "exit_rule": exit_rule,
                "signal_id": int(event["signal_id"]),
                "symbol": event["symbol"],
                "entry_time_utc": event["entry_time_utc"],
                "net_pnl": pnl,
                **meta,
            }
        )
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame, keys: dict[str, str] | None = None) -> dict[str, object]:
    keys = keys or {}
    if trades.empty:
        return {**keys, "trade_count": 0}
    pnl = trades["net_pnl"]
    return {
        **keys,
        "trade_count": len(trades),
        "win_rate": float((pnl > 0).mean()),
        "first_plus10_rate": float(trades["first_plus10"].mean()),
        "tp1_hit_rate": float(trades["tp1_hit"].mean()),
        "minus20_hit_rate": float(trades["minus20_hit"].mean()),
        "minus30_hit_rate": float(trades["minus30_hit"].mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "max_drawdown": max_drawdown(pnl),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
        "pnl_excluding_best_1": pnl_excluding_best(pnl, 1),
        "pnl_excluding_best_5": pnl_excluding_best(pnl, 5),
        "pnl_excluding_best_10": pnl_excluding_best(pnl, 10),
    }


def monthly_profitable_count(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    tmp = trades.copy()
    tmp["month"] = pd.to_datetime(tmp["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    return int((tmp.groupby("month")["net_pnl"].sum() > 0).sum())


def make_profile(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(labels, on=["signal_id", "symbol", "signal_time_utc"], how="inner")
    bad = df[df["hit_plus_10_before_minus_10"]]
    good = df[df["hit_minus_10_before_plus_10"]]
    rows = []
    skip = {"signal_id"}
    for col in features.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(features[col]):
            continue
        bad_mean = bad[col].mean()
        good_mean = good[col].mean()
        diff = good_mean - bad_mean
        rows.append(
            {
                "feature_name": col,
                "bad_mean": bad_mean,
                "good_mean": good_mean,
                "bad_median": bad[col].median(),
                "good_median": good[col].median(),
                "difference": diff,
                "direction": "higher_in_good" if diff > 0 else "higher_in_bad",
                "comment": "larger among first -10%" if diff > 0 else "larger among first +10%",
            }
        )
    return pd.DataFrame(rows).sort_values("difference", key=lambda s: s.abs(), ascending=False)


def add_simple_pnl(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["simple_pnl"] = np.select(
        [out["hit_plus_10_before_minus_10"], out["hit_minus_10_before_plus_10"]],
        [-0.101, 0.099],
        default=0.0,
    )
    return out


def bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "volume_1h_vs_24h_avg": ([-np.inf, 1.2, 2, 5, 10, np.inf], ["<1.2", "1.2-2", "2-5", "5-10", ">10"]),
        "rank_at_signal": ([0, 3, 5, 10], ["1-3", "4-5", "6-10"]),
        "top10_drop_avg": ([-np.inf, 0.10, 0.15, 0.20, np.inf], ["<10%", "10%-15%", "15%-20%", ">20%"]),
        "return_1h_before_signal": ([-np.inf, -0.20, -0.15, -0.10, -0.05, 0], ["<-20%", "-20%到-15%", "-15%到-10%", "-10%到-5%", "-5%到0%"]),
        "return_4h_before_signal": ([-np.inf, -0.30, -0.20, -0.10, 0], ["<-30%", "-30%到-20%", "-20%到-10%", "-10%到0%"]),
        "distance_to_5m_ema20_pct": ([-np.inf, -0.12, -0.08, -0.05, -0.02, 0], [">12%", "8%-12%", "5%-8%", "2%-5%", "0%-2%"]),
        "distance_to_vwap_pct": ([-np.inf, -0.10, -0.05, -0.02, 0], [">10%", "5%-10%", "2%-5%", "0%-2%"]),
        "signal_candle_lower_wick_pct": ([-np.inf, 0.20, 0.40, 0.60, np.inf], ["<=20%", "20%-40%", "40%-60%", ">60%"]),
        "rebound_from_local_low_30m_pct": ([-np.inf, 0.02, 0.05, 0.10, np.inf], ["<2%", "2%-5%", "5%-10%", ">10%"]),
        "btc_return_1h": ([-np.inf, -0.03, -0.01, 0, np.inf], ["<-3%", "-3%到-1%", "-1%到0%", ">0%"]),
    }
    rows = []
    for feature, (bins, labels) in specs.items():
        tmp = df.copy()
        tmp["_bucket"] = pd.cut(tmp[feature], bins=bins, labels=labels, include_lowest=True)
        for bucket, g in tmp.groupby("_bucket", observed=False):
            if g.empty:
                continue
            pnl = g["simple_pnl"]
            rows.append(
                {
                    "feature_name": feature,
                    "bucket": str(bucket),
                    "signal_count": len(g),
                    "first_plus10_rate": float(g["hit_plus_10_before_minus_10"].mean()),
                    "first_minus10_rate": float(g["hit_minus_10_before_plus_10"].mean()),
                    "first_minus20_rate": float(g["hit_minus_20_before_plus_10"].mean()),
                    "first_minus30_rate": float(g["hit_minus_30_before_plus_10"].mean()),
                    "avg_mfe_down": float(g["max_favorable_down_120h"].mean()),
                    "avg_mae_up": float(g["max_adverse_up_120h"].mean()),
                    "simple_short_profit_factor": profit_factor(pnl),
                    "total_pnl": float(pnl.sum()),
                    "sample_flag": "valid" if len(g) >= 30 else "sample_too_small",
                }
            )
    return pd.DataFrame(rows)


def filter_metrics(name: str, condition: str, selected: pd.DataFrame, original_count: int, trades: pd.DataFrame) -> dict[str, object]:
    pnl = trades["net_pnl"] if not trades.empty else pd.Series(dtype=float)
    return {
        "filter_name": name,
        "condition": condition,
        "original_count": original_count,
        "remaining_count": len(selected),
        "excluded_count": original_count - len(selected),
        "remaining_first_plus10_rate": float(selected["hit_plus_10_before_minus_10"].mean()) if len(selected) else np.nan,
        "remaining_first_minus10_rate": float(selected["hit_minus_10_before_plus_10"].mean()) if len(selected) else np.nan,
        "remaining_first_minus20_rate": float(selected["hit_minus_20_before_plus_10"].mean()) if len(selected) else np.nan,
        "remaining_first_minus30_rate": float(selected["hit_minus_30_before_plus_10"].mean()) if len(selected) else np.nan,
        "remaining_total_pnl": float(pnl.sum()),
        "remaining_profit_factor": profit_factor(pnl),
        "remaining_max_drawdown": max_drawdown(pnl),
        "pnl_excluding_best_1": pnl_excluding_best(pnl, 1),
        "pnl_excluding_best_5": pnl_excluding_best(pnl, 5),
        "monthly_profitable_count": monthly_profitable_count(trades),
    }


def entry_events(model: str, base: pd.DataFrame, kmap: dict[str, pd.DataFrame], rankings: pd.DataFrame) -> pd.DataFrame:
    r_lookup = rankings.set_index(["open_time", "symbol"])
    rows = []
    for _, sig in base.iterrows():
        group = kmap[sig["symbol"]]
        signal_time = int(sig["signal_time"])
        signal_price = float(sig["price_at_signal"])
        entry = None
        if model == "Immediate":
            entry = future_entry(group, signal_time)
        elif model == "Hold Drop Top10 30m":
            check_time = signal_time + 30 * 60_000
            check = asof_row(group, check_time)
            key = (int(check["open_time"]), sig["symbol"]) if check is not None else None
            if check is not None and key in r_lookup.index and bool(r_lookup.loc[key]["is_drop_top10"]) and float(check["high"]) < signal_price * 1.05:
                entry = future_entry(group, signal_time, 30)
        elif model == "Hold Drop Top10 60m":
            check_time = signal_time + 60 * 60_000
            check = asof_row(group, check_time)
            key = (int(check["open_time"]), sig["symbol"]) if check is not None else None
            if check is not None and key in r_lookup.index and bool(r_lookup.loc[key]["is_drop_top10"]) and float(check["high"]) < signal_price * 1.07:
                entry = future_entry(group, signal_time, 60)
        elif model in {"EMA20 Pullback Fail", "VWAP Pullback Fail"}:
            col = "ema20_5m" if model.startswith("EMA20") else "vwap_24h"
            future = group[(group["open_time"] > signal_time) & (group["open_time"] <= signal_time + 4 * 60 * 60 * 1000)]
            touched = False
            for i, bar in future.iterrows():
                ref = float(bar[col]) if pd.notna(bar[col]) else np.nan
                if pd.isna(ref):
                    continue
                if abs(float(bar["high"]) / ref - 1) <= 0.015 or float(bar["high"]) >= ref * 0.985:
                    touched = True
                if touched and float(bar["close"]) < ref:
                    nxt = group.index[group["open_time"] > int(bar["open_time"])]
                    if len(nxt):
                        entry = group.loc[int(nxt[0])]
                    break
        elif model == "Breakdown Continuation":
            sig_bar = asof_row(group, signal_time)
            sig_low = float(sig_bar["low"]) if sig_bar is not None else np.nan
            future = group[(group["open_time"] > signal_time) & (group["open_time"] <= signal_time + 2 * 60 * 60 * 1000)]
            hit = future[future["low"] < sig_low]
            if not hit.empty:
                nxt = group.index[group["open_time"] > int(hit.iloc[0]["open_time"])]
                if len(nxt):
                    entry = group.loc[int(nxt[0])]
        if entry is not None:
            rows.append(
                {
                    **sig.to_dict(),
                    "entry_time": int(entry["open_time"]),
                    "entry_time_utc": entry["open_time_utc"],
                    "entry_price": float(entry["open"]),
                    "entry_delay_minutes": int((int(entry["open_time"]) - signal_time) / 60_000),
                }
            )
    return pd.DataFrame(rows)


def evaluate_entry_models(base: pd.DataFrame, kmap: dict[str, pd.DataFrame], rankings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    models = ["Immediate", "Hold Drop Top10 30m", "Hold Drop Top10 60m", "EMA20 Pullback Fail", "VWAP Pullback Fail", "Breakdown Continuation"]
    summaries = []
    events_by_model = {}
    for model in models:
        events = entry_events(model, base, kmap, rankings)
        events_by_model[model] = events
        label_rows = []
        for _, e in events.iterrows():
            group = kmap[e["symbol"]]
            entry_time = int(e["entry_time"])
            entry_price = float(e["entry_price"])
            path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= entry_time + LOOKAHEAD_HOURS * 60 * 60 * 1000)]
            label_rows.append(first_hit_labels(path, entry_time, entry_price))
        labels = pd.DataFrame(label_rows)
        trades = make_backtest_trades(events, kmap, "base_15_20", model, "Exit C") if not events.empty else pd.DataFrame()
        summaries.append(
            {
                "model_name": model,
                "base_filter": "base_15_20",
                "original_signal_count": len(base),
                "trade_count": len(events),
                "abandoned_count": len(base) - len(events),
                "avg_entry_delay_minutes": float(events["entry_delay_minutes"].mean()) if not events.empty else np.nan,
                "first_plus10_rate": float(labels["hit_plus_10_before_minus_10"].mean()) if not labels.empty else np.nan,
                "first_minus10_rate": float(labels["hit_minus_10_before_plus_10"].mean()) if not labels.empty else np.nan,
                "first_minus20_rate": float(labels["hit_minus_20_before_plus_10"].mean()) if not labels.empty else np.nan,
                "first_minus30_rate": float(labels["hit_minus_30_before_plus_10"].mean()) if not labels.empty else np.nan,
                "total_pnl": float(trades["net_pnl"].sum()) if not trades.empty else 0.0,
                "profit_factor": profit_factor(trades["net_pnl"]) if not trades.empty else np.nan,
                "max_drawdown": max_drawdown(trades["net_pnl"]) if not trades.empty else 0.0,
                "pnl_excluding_best_1": pnl_excluding_best(trades["net_pnl"], 1) if not trades.empty else 0.0,
                "pnl_excluding_best_5": pnl_excluding_best(trades["net_pnl"], 5) if not trades.empty else 0.0,
                "monthly_profitable_count": monthly_profitable_count(trades),
            }
        )
    return pd.DataFrame(summaries), events_by_model


def robustness(summary: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    all_times = pd.to_datetime(trades["entry_time_utc"], utc=True).sort_values()
    split = all_times.iloc[int(len(all_times) * 0.70)]
    grouped = trades.groupby(["rule_name", "entry_model", "exit_rule"], dropna=False)
    for keys, g in grouped:
        t = g[pd.to_datetime(g["entry_time_utc"], utc=True) <= split]
        v = g[pd.to_datetime(g["entry_time_utc"], utc=True) > split]
        train_pf = profit_factor(t["net_pnl"])
        val_pf = profit_factor(v["net_pnl"])
        train_plus = float(t["first_plus10"].mean()) if len(t) else np.nan
        val_plus = float(v["first_plus10"].mean()) if len(v) else np.nan
        overfit = len(v) < 30 or (train_pf >= 1.15 and (val_pf <= 1 or val_plus >= 0.4286))
        rows.append(
            {
                "rule_name": keys[0],
                "entry_model": keys[1],
                "exit_rule": keys[2],
                "train_count": len(t),
                "validation_count": len(v),
                "train_first_plus10_rate": train_plus,
                "validation_first_plus10_rate": val_plus,
                "train_pf": train_pf,
                "validation_pf": val_pf,
                "train_total_pnl": float(t["net_pnl"].sum()),
                "validation_total_pnl": float(v["net_pnl"].sum()),
                "overfit_flag": bool(overfit),
                "conclusion": "valid" if (len(v) >= 30 and val_pf > 1 and val_plus < 0.4286) else "not_valid",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", INTERVAL)
    if klines.empty:
        raise RuntimeError("No cached 5m klines found.")
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    end_ms = int(klines["open_time"].max())
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    klines = klines[(klines["open_time"] >= start_ms) & (klines["open_time"] <= end_ms)].copy()
    klines = klines.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    check_klines(klines, INTERVAL_MS[INTERVAL]).to_csv(OUT / "data_quality.csv", index=False)

    print("Adding kline features and rankings...", flush=True)
    klines = add_kline_features(klines)
    rankings = build_rankings(klines)
    base = identify_base_signals(rankings)
    kmap = {s: g.reset_index(drop=True) for s, g in klines.groupby("symbol", sort=False)}

    entries = []
    for _, sig in base.iterrows():
        entry = future_entry(kmap[sig["symbol"]], int(sig["signal_time"]))
        if entry is not None:
            entries.append({"signal_id": int(sig["signal_id"]), "entry_time_utc": entry["open_time_utc"], "entry_price": float(entry["open"])})
    base_out = base.merge(pd.DataFrame(entries), on="signal_id", how="inner")
    base_out[
        [
            "signal_id",
            "symbol",
            "signal_time_utc",
            "entry_time_utc",
            "entry_price",
            "rank_at_signal",
            "rolling_24h_drop_at_signal",
            "price_at_signal",
            "quote_volume_at_signal",
        ]
    ].to_csv(OUT / "base_signals.csv", index=False)

    print("Labeling paths and extracting signal-time features...", flush=True)
    labels = make_path_labels(base, kmap)
    labels.to_csv(OUT / "path_labels.csv", index=False)
    features = extract_features(base, klines, rankings)
    features.to_csv(OUT / "signal_features.csv", index=False)
    df = add_simple_pnl(features.merge(labels, on=["signal_id", "symbol", "signal_time_utc"], how="inner"))

    make_profile(features, labels).to_csv(OUT / "profile_minus10_vs_plus10.csv", index=False)
    bucket_analysis(df).to_csv(OUT / "bucket_analysis.csv", index=False)

    print("Testing filters and entry models...", flush=True)
    immediate_events = entry_events("Immediate", base, kmap, rankings)
    exit_c_all = make_backtest_trades(immediate_events, kmap, "base_15_20", "Immediate", "Exit C")
    filter_defs = [
        ("Filter A", "1.2 <= volume_1h_vs_24h_avg <= 5", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5)),
        ("Filter B", "volume_1h_vs_24h_avg <= 5", lambda x: x["volume_1h_vs_24h_avg"] <= 5),
        ("Filter C", "rank_at_signal >= 4", lambda x: x["rank_at_signal"] >= 4),
        ("Filter D", "distance_to_5m_ema20_pct <= 8%", lambda x: x["distance_to_5m_ema20_pct"].abs() <= 0.08),
        ("Filter E", "signal_candle_lower_wick_pct <= 40%", lambda x: x["signal_candle_lower_wick_pct"] <= 0.40),
        ("Filter F", "rebound_from_local_low_30m_pct <= 5%", lambda x: x["rebound_from_local_low_30m_pct"] <= 0.05),
        ("Filter G", "btc_return_1h <= 0", lambda x: x["btc_return_1h"] <= 0),
        ("Filter H", "top10_drop_avg >= 10%", lambda x: x["top10_drop_avg"] >= 0.10),
        ("Filter I", "was_in_drop_top10_last_3d = False", lambda x: ~x["was_in_drop_top10_last_3d"].astype(bool)),
        ("Filter J", "was_in_gain_top10_last_5d = False", lambda x: ~x["was_in_gain_top10_last_5d"].astype(bool)),
    ]
    single_rows = []
    for name, cond, fn in filter_defs:
        sel = df[fn(df)].copy()
        ids = set(sel["signal_id"])
        trades = exit_c_all[exit_c_all["signal_id"].isin(ids)]
        single_rows.append(filter_metrics(name, cond, sel, len(df), trades))
    single_summary = pd.DataFrame(single_rows)
    single_summary.to_csv(OUT / "filter_single_summary.csv", index=False)

    combo_defs = [
        ("Combo 1", "vol 1.2-5 + rebound30 <=5", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (x["rebound_from_local_low_30m_pct"] <= 0.05)),
        ("Combo 2", "vol 1.2-5 + dist ema20 <=8", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (x["distance_to_5m_ema20_pct"].abs() <= 0.08)),
        ("Combo 3", "vol 1.2-5 + lower wick <=40", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (x["signal_candle_lower_wick_pct"] <= 0.40)),
        ("Combo 4", "vol 1.2-5 + btc1h <=0", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (x["btc_return_1h"] <= 0)),
        ("Combo 5", "vol 1.2-5 + not drop top10 3d", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (~x["was_in_drop_top10_last_3d"].astype(bool))),
        ("Combo 6", "vol 1.2-5 + rebound30 <=5 + lower wick <=40", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (x["rebound_from_local_low_30m_pct"] <= 0.05) & (x["signal_candle_lower_wick_pct"] <= 0.40)),
        ("Combo 7", "rank>=4 + rebound30 <=5 + lower wick <=40", lambda x: (x["rank_at_signal"] >= 4) & (x["rebound_from_local_low_30m_pct"] <= 0.05) & (x["signal_candle_lower_wick_pct"] <= 0.40)),
        ("Combo 8", "vol 1.2-5 + dist ema20 <=8 + btc1h <=0", lambda x: x["volume_1h_vs_24h_avg"].between(1.2, 5) & (x["distance_to_5m_ema20_pct"].abs() <= 0.08) & (x["btc_return_1h"] <= 0)),
    ]
    combo_rows = []
    combo_id_sets = {"base_15_20": set(df["signal_id"])}
    for name, cond, fn in combo_defs:
        sel = df[fn(df)].copy()
        combo_id_sets[name] = set(sel["signal_id"])
        trades = exit_c_all[exit_c_all["signal_id"].isin(combo_id_sets[name])]
        m = summarize_trades(trades, {"combo_name": name, "conditions": cond})
        m.update(
            {
                "remaining_count": len(sel),
                "remaining_first_plus10_rate": float(sel["hit_plus_10_before_minus_10"].mean()) if len(sel) else np.nan,
                "remaining_first_minus10_rate": float(sel["hit_minus_10_before_plus_10"].mean()) if len(sel) else np.nan,
                "remaining_first_minus20_rate": float(sel["hit_minus_20_before_plus_10"].mean()) if len(sel) else np.nan,
                "remaining_first_minus30_rate": float(sel["hit_minus_30_before_plus_10"].mean()) if len(sel) else np.nan,
                "monthly_profitable_count": monthly_profitable_count(trades),
            }
        )
        combo_rows.append(m)
    pd.DataFrame(combo_rows).to_csv(OUT / "filter_combo_summary.csv", index=False)

    entry_summary, events_by_model = evaluate_entry_models(base, kmap, rankings)
    entry_summary.to_csv(OUT / "entry_model_summary.csv", index=False)

    print("Running backtests...", flush=True)
    trade_frames = []
    rules_to_test = {"base_15_20": set(df["signal_id"])}
    for name, ids in combo_id_sets.items():
        rules_to_test[name] = ids
    for rule_name, ids in rules_to_test.items():
        for model, events in events_by_model.items():
            if events.empty:
                continue
            selected_events = events[events["signal_id"].isin(ids)].copy()
            if selected_events.empty:
                continue
            for exit_rule in ["Exit A", "Exit B", "Exit C"]:
                trade_frames.append(make_backtest_trades(selected_events, kmap, rule_name, model, exit_rule))
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades_all.to_csv(OUT / "backtest_trades.csv", index=False)
    summary_rows = []
    for keys, g in trades_all.groupby(["rule_name", "entry_model", "exit_rule"], dropna=False):
        summary_rows.append(summarize_trades(g, {"rule_name": keys[0], "entry_model": keys[1], "exit_rule": keys[2]}))
    backtest_summary = pd.DataFrame(summary_rows)
    backtest_summary.to_csv(OUT / "backtest_summary.csv", index=False)

    monthly_rows = []
    tmp = trades_all.copy()
    tmp["month"] = pd.to_datetime(tmp["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    for keys, g in tmp.groupby(["rule_name", "entry_model", "exit_rule", "month"], dropna=False):
        monthly_rows.append(
            {
                "rule_name": keys[0],
                "entry_model": keys[1],
                "exit_rule": keys[2],
                "month": keys[3],
                "trade_count": len(g),
                "pnl": float(g["net_pnl"].sum()),
                "PF": profit_factor(g["net_pnl"]),
                "first_plus10_rate": float(g["first_plus10"].mean()),
                "tp1_hit_rate": float(g["tp1_hit"].mean()),
                "minus20_hit_rate": float(g["minus20_hit"].mean()),
                "max_drawdown": max_drawdown(g["net_pnl"]),
            }
        )
    pd.DataFrame(monthly_rows).to_csv(OUT / "summary_by_month.csv", index=False)
    robustness(backtest_summary, trades_all).to_csv(OUT / "robustness_check.csv", index=False)

    best_filters = single_summary.sort_values(["remaining_first_plus10_rate", "remaining_count"], ascending=[True, False]).head(5)
    valid_bt = backtest_summary[(backtest_summary["trade_count"] >= 80) & (backtest_summary["profit_factor"] >= 1.15)].sort_values("profit_factor", ascending=False)
    valid_rob = pd.read_csv(OUT / "robustness_check.csv")
    report = [
        "# Drop Top10 Short Filter Research",
        "",
        f"Data: Binance USDT-M cached 5m klines, UTC {pd.to_datetime(start_ms, unit='ms', utc=True)} to {pd.to_datetime(end_ms, unit='ms', utc=True)}.",
        f"Base sample: first drop Top10 within {COOLDOWN_DAYS}d, trigger rolling 24h drop in [15%, 20%), next 5m open short entry.",
        f"Base signals: {len(df)}.",
        "",
        "## Base Path",
        f"- first +10% before -10%: {df['hit_plus_10_before_minus_10'].mean():.2%}",
        f"- first -10% before +10%: {df['hit_minus_10_before_plus_10'].mean():.2%}",
        f"- first -20% before +10%: {df['hit_minus_20_before_plus_10'].mean():.2%}",
        f"- first -30% before +10%: {df['hit_minus_30_before_plus_10'].mean():.2%}",
        "",
        "## Best Single Filters By Lower First +10%",
        dataframe_to_markdown(best_filters),
        "",
        "## Backtest Candidates PF >= 1.15 And Count >= 80",
        dataframe_to_markdown(valid_bt.head(20)) if not valid_bt.empty else "No candidate met PF >= 1.15 with count >= 80.",
        "",
        "## Validation",
        dataframe_to_markdown(valid_rob.sort_values(["validation_pf", "validation_first_plus10_rate"], ascending=[False, True]).head(20)),
        "",
        "## Answers",
        "1. Features are in profile_minus10_vs_plus10.csv and bucket_analysis.csv; use only buckets with signal_count >= 30.",
        "2. Check filter_single_summary.csv and filter_combo_summary.csv for filters reaching the 35%-38% first +10% target.",
        "3. Hit-rate preservation is reported in every filter and combo summary.",
        "4. 1H volume ratio is tested directly as Filter A/B and in Combo 1-6/8.",
        "5. Lower wick and rebound strength are tested as Filter E/F and several combos.",
        "6. BTC 1H context is tested as Filter G and Combo 4/8.",
        "7. Entry models are compared in entry_model_summary.csv.",
        "8. PF candidates are in backtest_summary.csv.",
        "9. Train/validation is in robustness_check.csv.",
        "10. Do not enter simulation unless validation PF > 1, first +10% below base, and monthly performance is not concentrated.",
        "11. If invalid, main failure is usually rebound/liquidation rate remains high or PF depends on a few large winners.",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
