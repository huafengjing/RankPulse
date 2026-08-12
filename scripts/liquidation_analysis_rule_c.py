from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.downloader import load_cached_klines
from src.research.ranking import build_rankings


SOURCE_OUT = ROOT / "outputs" / "long_consolidation_top10_20_30"
OUT = ROOT / "outputs" / "liquidation_analysis"

EXCLUDE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}
INTERVAL_MINUTES = 5
MAX_HOLD_HOURS = 240
ACCOUNT_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 10.0
FEE = 0.0005
SLIP = 0.0005

LIQ_PCT = -0.10
TP1 = 0.15
TP1_FRAC = 0.50
STAGE2_TRIGGER = 0.50
TRAIL_STAGE1 = 0.20
TRAIL_STAGE2 = 0.30
MA_WINDOW_4H = 14


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) else np.inf


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def exit_pnl(entry_price: float, exit_price_raw: float, fraction: float) -> tuple[float, float, float]:
    notional = MARGIN_PER_TRADE * LEVERAGE * fraction
    entry_eff = entry_price * (1 + SLIP)
    exit_eff = exit_price_raw * (1 - SLIP)
    gross = notional * (exit_eff / entry_eff - 1)
    fees = notional * FEE + notional * (exit_eff / entry_eff) * FEE
    return gross - fees, gross, fees


def add_indicators(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("open_time").copy()
    g["ema20_5m"] = g["close"].ewm(span=20, adjust=False).mean()
    pv = g["close"] * g["volume"]
    vol24 = g["volume"].rolling(288, min_periods=30).sum()
    g["vwap_24h"] = pv.rolling(288, min_periods=30).sum() / vol24.replace(0, np.nan)
    g["vol_avg_24h"] = g["volume"].rolling(288, min_periods=30).mean()
    g["open_time_utc"] = pd.to_datetime(g["open_time"], unit="ms", utc=True)
    indexed = g.set_index("open_time_utc")
    close_4h = indexed["close"].resample("4h", label="right", closed="right").last().dropna()
    ma14 = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
    indexed["ma14_4h_completed"] = ma14.shift(1).reindex(indexed.index, method="ffill").values
    close_15m = indexed["close"].resample("15min", label="right", closed="right").last().dropna()
    ema15 = close_15m.ewm(span=20, adjust=False).mean()
    indexed["ema20_15m_completed"] = ema15.shift(1).reindex(indexed.index, method="ffill").values
    reset = indexed.reset_index()
    reset["is_4h_close_bar"] = ((reset["open_time"] + 5 * 60 * 1000) % (4 * 60 * 60 * 1000)) == 0
    return reset.reset_index(drop=True)


def latest_bar_at_or_before(group: pd.DataFrame, t: int) -> pd.Series | None:
    rows = group[group["open_time"] <= t]
    if rows.empty:
        return None
    return rows.iloc[-1]


def bar_at(group: pd.DataFrame, t: int) -> pd.Series | None:
    rows = group[group["open_time"] == t]
    if rows.empty:
        return latest_bar_at_or_before(group, t)
    return rows.iloc[-1]


def return_before(group: pd.DataFrame, signal_time: int, minutes: int, signal_close: float) -> float:
    past = latest_bar_at_or_before(group, signal_time - minutes * 60 * 1000)
    if past is None or float(past["close"]) <= 0:
        return np.nan
    return signal_close / float(past["close"]) - 1


def count_green_5m(group: pd.DataFrame, signal_time: int) -> int:
    pre = group[group["open_time"] <= signal_time].tail(100)
    count = 0
    for row in reversed(list(pre.itertuples(index=False))):
        if float(row.close) > float(row.open):
            count += 1
        else:
            break
    return count


def count_green_15m(group: pd.DataFrame, signal_time: int) -> int:
    pre = group[group["open_time"] <= signal_time].tail(120).copy()
    if pre.empty:
        return 0
    pre["dt"] = pd.to_datetime(pre["open_time"], unit="ms", utc=True)
    bars = pre.set_index("dt").resample("15min", label="right", closed="right").agg({"open": "first", "close": "last"}).dropna()
    count = 0
    for row in reversed(list(bars.itertuples(index=False))):
        if float(row.close) > float(row.open):
            count += 1
        else:
            break
    return count


def rank_at(ranking_by_symbol: dict[str, pd.DataFrame], symbol: str, t: int) -> float:
    group = ranking_by_symbol.get(symbol)
    if group is None:
        return np.nan
    row = group[group["open_time"] == t]
    if row.empty:
        return np.nan
    return float(row.iloc[-1]["rank"])


def rank_improvement(ranking_by_symbol: dict[str, pd.DataFrame], symbol: str, signal_time: int, rank_now: int, minutes: int) -> float:
    past_rank = rank_at(ranking_by_symbol, symbol, signal_time - minutes * 60 * 1000)
    if pd.isna(past_rank):
        return np.nan
    return past_rank - rank_now


def board_features(rankings: pd.DataFrame, signal_time: int) -> dict:
    snap = rankings[rankings["open_time"] == signal_time].sort_values("rank")
    if snap.empty:
        return {}
    top5 = snap.head(5)["rolling_24h_change_pct"]
    top10 = snap.head(10)["rolling_24h_change_pct"]
    top5_min = float(top5.min())
    if top5_min < 0.15:
        label = "cold"
    elif top5_min < 0.20:
        label = "neutral"
    elif top5_min < 0.30:
        label = "hot"
    else:
        label = "very_hot"
    return {
        "top5_min_gain": top5_min,
        "top5_avg_gain": float(top5.mean()),
        "top10_avg_gain": float(top10.mean()),
        "gain_gt_20_count": int((snap["rolling_24h_change_pct"] >= 0.20).sum()),
        "board_temperature_label": label,
    }


def btc_features(btc: pd.DataFrame, signal_time: int) -> dict:
    bar = latest_bar_at_or_before(btc, signal_time)
    if bar is None:
        return {"btc_return_1h": np.nan, "btc_return_4h": np.nan, "btc_regime_label": "unknown"}
    close = float(bar["close"])
    r1 = return_before(btc, signal_time, 60, close)
    r4 = return_before(btc, signal_time, 240, close)
    if pd.notna(r1) and pd.notna(r4) and r1 >= 0 and r4 >= 0:
        label = "bullish"
    elif pd.notna(r1) and pd.notna(r4) and r1 < 0 and r4 < 0:
        label = "bearish"
    else:
        label = "neutral"
    return {"btc_return_1h": r1, "btc_return_4h": r4, "btc_regime_label": label}


def path_labels(group: pd.DataFrame, signal_time: int) -> dict | None:
    idx = group.index[group["open_time"] > signal_time]
    if len(idx) == 0:
        return None
    entry = group.loc[int(idx[0])]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = entry_time + MAX_HOLD_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    if path.empty:
        return None
    liq = entry_price * 0.90
    thresholds = {15: entry_price * 1.15, 30: entry_price * 1.30, 50: entry_price * 1.50}
    hit = {k: False for k in thresholds}
    times = {k: np.nan for k in thresholds}
    time_minus = np.nan
    minus_seen = False
    for row in path.itertuples(index=False):
        low = float(row.low)
        high = float(row.high)
        t = int(row.open_time)
        if low <= liq:
            minus_seen = True
            time_minus = (t - entry_time) / 60000
            break
        for k, px in thresholds.items():
            if not hit[k] and high >= px:
                hit[k] = True
                times[k] = (t - entry_time) / 60000
    return {
        "entry_time": entry_time,
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "hit_minus10_first": bool(minus_seen and not hit[15]),
        "hit_plus15_before_minus10": hit[15],
        "hit_plus30_before_minus10": hit[30],
        "hit_plus50_before_minus10": hit[50],
        "max_forward_return_240h": float(path["high"].max() / entry_price - 1),
        "min_forward_return_240h": float(path["low"].min() / entry_price - 1),
        "time_to_minus10_minutes": time_minus,
        "time_to_plus15_minutes": times[15],
        "time_to_plus30_minutes": times[30],
        "time_to_plus50_minutes": times[50],
    }


def signal_features(sig: pd.Series, group: pd.DataFrame, rankings: pd.DataFrame, ranking_by_symbol: dict[str, pd.DataFrame], btc: pd.DataFrame) -> dict:
    t = int(sig["signal_time"])
    bar = bar_at(group, t)
    if bar is None:
        return {}
    open_p = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])
    candle_range = max(high - low, 0.0)
    upper = high - max(open_p, close)
    lower = min(open_p, close) - low
    vol_avg_24h = float(bar.get("vol_avg_24h", np.nan))
    volume = float(bar["volume"])
    vol_15m = float(group[(group["open_time"] >= t - 10 * 60 * 1000) & (group["open_time"] <= t)]["volume"].sum())
    vol_1h = float(group[(group["open_time"] >= t - 55 * 60 * 1000) & (group["open_time"] <= t)]["volume"].sum())
    rank_now = int(sig["rank_at_signal"])
    out = {
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "signal_time": t,
        "signal_time_utc": sig["signal_time_utc"],
        "rank_at_signal": rank_now,
        "rolling_24h_gain_at_signal": float(sig["rolling_24h_gain_at_signal"]),
        "quote_volume_at_signal": float(sig["quote_volume_at_signal"]),
        "distance_to_5m_ema20_pct": close / float(bar["ema20_5m"]) - 1 if pd.notna(bar["ema20_5m"]) else np.nan,
        "distance_to_15m_ema20_pct": close / float(bar["ema20_15m_completed"]) - 1 if pd.notna(bar["ema20_15m_completed"]) else np.nan,
        "distance_to_vwap_pct": close / float(bar["vwap_24h"]) - 1 if pd.notna(bar["vwap_24h"]) else np.nan,
        "return_15m_before_signal": return_before(group, t, 15, close),
        "return_1h_before_signal": return_before(group, t, 60, close),
        "return_4h_before_signal": return_before(group, t, 240, close),
        "signal_candle_body_pct": abs(close - open_p) / open_p if open_p else np.nan,
        "signal_candle_upper_wick_pct": upper / candle_range if candle_range else 0.0,
        "signal_candle_lower_wick_pct": lower / candle_range if candle_range else 0.0,
        "signal_candle_range_pct": high / low - 1 if low else np.nan,
        "consecutive_green_5m_count": count_green_5m(group, t),
        "consecutive_green_15m_count": count_green_15m(group, t),
        "volume_5m_vs_24h_avg": volume / vol_avg_24h if vol_avg_24h > 0 else np.nan,
        "volume_15m_vs_24h_avg": vol_15m / (3 * vol_avg_24h) if vol_avg_24h > 0 else np.nan,
        "volume_1h_vs_24h_avg": vol_1h / (12 * vol_avg_24h) if vol_avg_24h > 0 else np.nan,
        "volume_spike_ratio": volume / vol_avg_24h if vol_avg_24h > 0 else np.nan,
        "rank_improvement_last_30m": rank_improvement(ranking_by_symbol, sig["symbol"], t, rank_now, 30),
        "rank_improvement_last_1h": rank_improvement(ranking_by_symbol, sig["symbol"], t, rank_now, 60),
        "rank_improvement_last_4h": rank_improvement(ranking_by_symbol, sig["symbol"], t, rank_now, 240),
    }
    out.update(board_features(rankings, t))
    out.update(btc_features(btc, t))
    return out


def run_trade_from_entry(sig: pd.Series, group: pd.DataFrame, entry_time: int, entry_price: float, model_name: str, delay_minutes: float = 0.0) -> dict | None:
    end_time = entry_time + MAX_HOLD_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    if path.empty:
        return None
    liq_price = entry_price * (1 + LIQ_PCT)
    tp1_price = entry_price * (1 + TP1)
    stage2_price = entry_price * (1 + STAGE2_TRIGGER)
    remaining = 1.0
    net = 0.0
    gross = 0.0
    fees = 0.0
    tp1_hit = False
    stage2_hit = False
    liquidated = False
    runner_exit = False
    highest_after_tp1 = np.nan
    exit_time = int(path.iloc[-1]["open_time"])
    exit_price = float(path.iloc[-1]["close"])
    exit_reason = "max_10d"
    mfe = float(path["high"].max() / entry_price - 1)
    mae = float(path["low"].min() / entry_price - 1)
    for _, bar in path.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        bar_time = int(bar["open_time"])
        if not tp1_hit:
            if low <= liq_price:
                pnl, gp, fee = exit_pnl(entry_price, liq_price, remaining)
                net += pnl
                gross += gp
                fees += fee
                remaining = 0.0
                liquidated = True
                exit_time = bar_time
                exit_price = liq_price
                exit_reason = "liquidation_before_tp1"
                break
            if high >= tp1_price:
                pnl, gp, fee = exit_pnl(entry_price, tp1_price, TP1_FRAC)
                net += pnl
                gross += gp
                fees += fee
                remaining -= TP1_FRAC
                tp1_hit = True
                highest_after_tp1 = max(high, tp1_price)
            else:
                continue
        if tp1_hit and remaining > 0:
            highest_after_tp1 = max(float(highest_after_tp1), high)
            if high >= stage2_price:
                stage2_hit = True
            trail_pct = TRAIL_STAGE2 if stage2_hit else TRAIL_STAGE1
            trail_stop = float(highest_after_tp1) * (1 - trail_pct)
            if low <= trail_stop:
                pnl, gp, fee = exit_pnl(entry_price, trail_stop, remaining)
                net += pnl
                gross += gp
                fees += fee
                remaining = 0.0
                runner_exit = True
                exit_time = bar_time
                exit_price = trail_stop
                exit_reason = f"stage{2 if stage2_hit else 1}_trailing_{int(trail_pct * 100)}pct"
                break
            ma14 = bar.get("ma14_4h_completed")
            if pd.notna(ma14):
                if not stage2_hit and close < float(ma14):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    remaining = 0.0
                    runner_exit = True
                    exit_time = bar_time
                    exit_price = close
                    exit_reason = "stage1_5m_close_below_4h_ma14"
                    break
                if stage2_hit and bool(bar["is_4h_close_bar"]) and close < float(ma14):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    remaining = 0.0
                    runner_exit = True
                    exit_time = bar_time
                    exit_price = close
                    exit_reason = "stage2_4h_close_below_ma14"
                    break
    if remaining > 0:
        last = path.iloc[-1]
        pnl, gp, fee = exit_pnl(entry_price, float(last["close"]), remaining)
        net += pnl
        gross += gp
        fees += fee
        exit_time = int(last["open_time"])
        exit_price = float(last["close"])
    return {
        "model_name": model_name,
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "signal_time_utc": sig["signal_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_delay_minutes": delay_minutes,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp1_hit": tp1_hit,
        "plus30_hit": mfe >= 0.30 and not liquidated,
        "plus50_hit": stage2_hit,
        "minus10_first": liquidated,
        "runner_exit": runner_exit,
        "exit_reason": exit_reason,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "gross_pnl_usd": gross,
        "fees_usd": fees,
        "net_pnl_usd": net,
        "return_on_margin_pct": net / MARGIN_PER_TRADE,
    }


def immediate_entry(sig: pd.Series, group: pd.DataFrame) -> tuple[int, float, float] | None:
    idx = group.index[group["open_time"] > int(sig["signal_time"])]
    if len(idx) == 0:
        return None
    bar = group.loc[int(idx[0])]
    return int(bar["open_time"]), float(bar["open"]), (int(bar["open_time"]) - int(sig["signal_time"])) / 60000


def confirmation_entry(sig: pd.Series, group: pd.DataFrame, rankings: pd.DataFrame, model: str) -> tuple[int, float, float] | None:
    signal_time = int(sig["signal_time"])
    signal_bar = bar_at(group, signal_time)
    if signal_bar is None:
        return None
    signal_low = float(signal_bar["low"])
    signal_high = float(signal_bar["high"])
    if model in {"hold_top10_30m", "hold_top10_60m"}:
        minutes = 30 if model.endswith("30m") else 60
        check_time = signal_time + minutes * 60 * 1000
        observation = group[(group["open_time"] > signal_time) & (group["open_time"] <= check_time)]
        if not observation.empty and float(observation["low"].min()) < signal_low:
            return None
        snap = rankings[(rankings["open_time"] == check_time) & (rankings["symbol"] == sig["symbol"])]
        if snap.empty or int(snap.iloc[-1]["rank"]) > 10:
            return None
        idx = group.index[group["open_time"] > check_time]
        if len(idx) == 0:
            return None
        bar = group.loc[int(idx[0])]
        return int(bar["open_time"]), float(bar["open"]), (int(bar["open_time"]) - signal_time) / 60000
    window = group[(group["open_time"] > signal_time) & (group["open_time"] <= signal_time + 4 * 60 * 60 * 1000)]
    if window.empty:
        return None
    if model == "signal_high_break":
        hits = window[window["high"] > signal_high]
        if hits.empty:
            return None
        hit_time = int(hits.iloc[0]["open_time"])
        idx = group.index[group["open_time"] > hit_time]
    elif model == "ema20_pullback":
        valid = window[(abs(window["close"] / window["ema20_5m"] - 1) <= 0.015) & (window["close"] > window["ema20_5m"])]
        if valid.empty:
            return None
        idx = group.index[group["open_time"] > int(valid.iloc[0]["open_time"])]
    elif model == "vwap_pullback":
        valid = window[(abs(window["close"] / window["vwap_24h"] - 1) <= 0.015) & (window["close"] > window["vwap_24h"])]
        if valid.empty:
            return None
        idx = group.index[group["open_time"] > int(valid.iloc[0]["open_time"])]
    else:
        return None
    if len(idx) == 0:
        return None
    bar = group.loc[int(idx[0])]
    return int(bar["open_time"]), float(bar["open"]), (int(bar["open_time"]) - signal_time) / 60000


def summarize_trades(df: pd.DataFrame, name: str, original_count: int) -> dict:
    if df.empty:
        return {
            "name": name,
            "original_signal_count": original_count,
            "trade_count": 0,
            "abandoned_count": original_count,
            "avg_entry_delay_minutes": np.nan,
            "first_minus10_rate": np.nan,
            "tp15_hit_rate": np.nan,
            "plus30_hit_rate": np.nan,
            "plus50_hit_rate": np.nan,
            "total_pnl": 0.0,
            "profit_factor": np.nan,
            "max_drawdown": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "pnl_excluding_best_5": np.nan,
            "monthly_profitable_count": 0,
        }
    pnl = df["net_pnl_usd"]
    sorted_pnl = pnl.sort_values(ascending=False).reset_index(drop=True)
    monthly = df.assign(month=pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")).groupby("month")["net_pnl_usd"].sum()
    return {
        "name": name,
        "original_signal_count": original_count,
        "trade_count": int(len(df)),
        "abandoned_count": int(original_count - len(df)),
        "avg_entry_delay_minutes": float(df["entry_delay_minutes"].mean()) if "entry_delay_minutes" in df else 5.0,
        "first_minus10_rate": float(df["minus10_first"].mean()),
        "tp15_hit_rate": float(df["tp1_hit"].mean()),
        "plus30_hit_rate": float(df["plus30_hit"].mean()),
        "plus50_hit_rate": float(df["plus50_hit"].mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(df["return_on_margin_pct"]),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
        "monthly_profitable_count": int((monthly > 0).sum()),
    }


def bucketize(value: float, edges: list[tuple[str, float | None, float | None]]) -> str:
    if pd.isna(value):
        return "missing"
    for label, lo, hi in edges:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return label
    return "missing"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading Rule C signals...", flush=True)
    source_features = pd.read_csv(SOURCE_OUT / "signal_features.csv")
    signals = source_features[source_features["rule_c_21d_pass"] == True].copy()
    print(f"Rule C signals={len(signals)}", flush=True)

    print("Loading cached klines and rankings...", flush=True)
    klines_all = load_cached_klines(ROOT / "data", "5m")
    btc = add_indicators(klines_all[klines_all["symbol"] == "BTCUSDT"].copy())
    ranking_klines = klines_all[~klines_all["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    min_t = int(signals["signal_time"].min()) - 24 * 60 * 60 * 1000
    max_t = int(signals["signal_time"].max()) + MAX_HOLD_HOURS * 60 * 60 * 1000
    ranking_klines = ranking_klines[(ranking_klines["open_time"] >= min_t) & (ranking_klines["open_time"] <= max_t)]
    rankings = build_rankings(ranking_klines, INTERVAL_MINUTES)
    ranking_by_symbol = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in rankings.groupby("symbol", sort=False)}
    symbols = set(signals["symbol"])
    kmap = {
        symbol: add_indicators(group)
        for symbol, group in klines_all[klines_all["symbol"].isin(symbols)].groupby("symbol", sort=False)
    }

    print("Generating labels/features...", flush=True)
    label_rows = []
    feature_rows = []
    baseline_rows = []
    for _, sig in signals.iterrows():
        group = kmap.get(sig["symbol"])
        if group is None:
            continue
        labels = path_labels(group, int(sig["signal_time"]))
        feats = signal_features(sig, group, rankings, ranking_by_symbol, btc)
        entry = immediate_entry(sig, group)
        if labels and feats and entry:
            label_rows.append({"signal_id": int(sig["signal_id"]), "symbol": sig["symbol"], "signal_time": int(sig["signal_time"]), **labels})
            feature_rows.append(feats)
            trade = run_trade_from_entry(sig, group, entry[0], entry[1], "immediate_baseline", entry[2])
            if trade:
                baseline_rows.append(trade)

    labels = pd.DataFrame(label_rows)
    features = pd.DataFrame(feature_rows)
    baseline = pd.DataFrame(baseline_rows)
    labels.to_csv(OUT / "signal_liquidation_labels.csv", index=False)
    features.to_csv(OUT / "signal_features.csv", index=False)
    baseline.to_csv(OUT / "baseline_trades.csv", index=False)
    full = features.merge(labels, on=["signal_id", "symbol"], how="inner").merge(
        baseline[["signal_id", "net_pnl_usd", "return_on_margin_pct", "tp1_hit", "plus30_hit", "plus50_hit", "minus10_first", "entry_time_utc"]],
        on="signal_id",
        how="inner",
        suffixes=("", "_trade"),
    )

    numeric_cols = [
        c
        for c in features.columns
        if c
        not in {
            "signal_id",
            "symbol",
            "signal_time",
            "signal_time_utc",
            "board_temperature_label",
            "btc_regime_label",
        }
        and pd.api.types.is_numeric_dtype(features[c])
        and c in full.columns
    ]
    profile_rows = []
    for col in numeric_cols:
        m = full[full["hit_minus10_first"] == True][col]
        n = full[full["hit_minus10_first"] == False][col]
        diff = float(m.mean() - n.mean()) if len(m) and len(n) else np.nan
        profile_rows.append(
            {
                "feature_name": col,
                "minus10_mean": float(m.mean()) if len(m) else np.nan,
                "non_minus10_mean": float(n.mean()) if len(n) else np.nan,
                "minus10_median": float(m.median()) if len(m) else np.nan,
                "non_minus10_median": float(n.median()) if len(n) else np.nan,
                "difference": diff,
                "direction": "higher_in_minus10" if diff > 0 else "lower_in_minus10",
                "comment": "",
            }
        )
    pd.DataFrame(profile_rows).to_csv(OUT / "minus10_profile.csv", index=False)

    bucket_specs = {
        "rank_at_signal": [("1-3", 1, 4), ("4-5", 4, 6), ("6-10", 6, 11)],
        "top5_min_gain": [("<20%", None, 0.20), ("20%-25%", 0.20, 0.25), ("25%-30%", 0.25, 0.30), (">30%", 0.30, None)],
        "distance_to_5m_ema20_pct": [("<=2%", None, 0.02), ("2%-5%", 0.02, 0.05), ("5%-8%", 0.05, 0.08), ("8%-12%", 0.08, 0.12), (">12%", 0.12, None)],
        "distance_to_vwap_pct": [("<=2%", None, 0.02), ("2%-5%", 0.02, 0.05), ("5%-10%", 0.05, 0.10), (">10%", 0.10, None)],
        "signal_candle_upper_wick_pct": [("<=20%", None, 0.20), ("20%-40%", 0.20, 0.40), ("40%-60%", 0.40, 0.60), (">60%", 0.60, None)],
        "return_1h_before_signal": [("<5%", None, 0.05), ("5%-10%", 0.05, 0.10), ("10%-20%", 0.10, 0.20), (">20%", 0.20, None)],
        "return_4h_before_signal": [("<10%", None, 0.10), ("10%-20%", 0.10, 0.20), ("20%-40%", 0.20, 0.40), (">40%", 0.40, None)],
        "volume_1h_vs_24h_avg": [("<1.5", None, 1.5), ("1.5-3", 1.5, 3.0), ("3-5", 3.0, 5.0), (">5", 5.0, None)],
    }
    bucket_rows = []
    for feature, edges in bucket_specs.items():
        tmp = full.copy()
        tmp["bucket"] = tmp[feature].apply(lambda x: bucketize(x, edges))
        for bucket, df in tmp.groupby("bucket"):
            if bucket == "missing":
                continue
            bucket_rows.append(
                {
                    "feature_name": feature,
                    "bucket": bucket,
                    "count": len(df),
                    "minus10_rate": float(df["hit_minus10_first"].mean()),
                    "tp15_hit_rate": float(df["hit_plus15_before_minus10"].mean()),
                    "plus30_hit_rate": float(df["hit_plus30_before_minus10"].mean()),
                    "plus50_hit_rate": float(df["hit_plus50_before_minus10"].mean()),
                    "profit_factor": profit_factor(df["return_on_margin_pct"]),
                    "total_pnl": float(df["net_pnl_usd"].sum()),
                    "avg_trade_pnl": float(df["net_pnl_usd"].mean()),
                    "sample_flag": "ok" if len(df) >= 30 else "sample_too_small",
                }
            )
    pd.DataFrame(bucket_rows).to_csv(OUT / "minus10_bucket_analysis.csv", index=False)

    filters = {
        "Filter A rank>=4": ("rank_at_signal >= 4", lambda d: d["rank_at_signal"] >= 4),
        "Filter B dist_ema5m<=8%": ("distance_to_5m_ema20_pct <= 8%", lambda d: d["distance_to_5m_ema20_pct"] <= 0.08),
        "Filter C dist_vwap<=10%": ("distance_to_vwap_pct <= 10%", lambda d: d["distance_to_vwap_pct"] <= 0.10),
        "Filter D upper_wick<=40%": ("signal_candle_upper_wick_pct <= 40%", lambda d: d["signal_candle_upper_wick_pct"] <= 0.40),
        "Filter E green5m<=5": ("consecutive_green_5m_count <= 5", lambda d: d["consecutive_green_5m_count"] <= 5),
        "Filter F top5_min 20-30%": ("20% <= top5_min_gain < 30%", lambda d: (d["top5_min_gain"] >= 0.20) & (d["top5_min_gain"] < 0.30)),
        "Filter G vol1h 1.5-5": ("1.5 <= volume_1h_vs_24h_avg <= 5", lambda d: (d["volume_1h_vs_24h_avg"] >= 1.5) & (d["volume_1h_vs_24h_avg"] <= 5)),
        "Filter H return1h<=15%": ("return_1h_before_signal <= 15%", lambda d: d["return_1h_before_signal"] <= 0.15),
        "Filter I return4h<=30%": ("return_4h_before_signal <= 30%", lambda d: d["return_4h_before_signal"] <= 0.30),
        "Combo 1 rank>=4 + dist_ema": ("rank_at_signal >= 4 and distance_to_5m_ema20_pct <= 8%", lambda d: (d["rank_at_signal"] >= 4) & (d["distance_to_5m_ema20_pct"] <= 0.08)),
        "Combo 2 dist_ema + wick": ("distance_to_5m_ema20_pct <= 8% and upper_wick <= 40%", lambda d: (d["distance_to_5m_ema20_pct"] <= 0.08) & (d["signal_candle_upper_wick_pct"] <= 0.40)),
        "Combo 3 rank>=4 + top5": ("rank_at_signal >= 4 and 20% <= top5_min_gain < 30%", lambda d: (d["rank_at_signal"] >= 4) & (d["top5_min_gain"] >= 0.20) & (d["top5_min_gain"] < 0.30)),
        "Combo 4 dist_ema + top5": ("distance_to_5m_ema20_pct <= 8% and 20% <= top5_min_gain < 30%", lambda d: (d["distance_to_5m_ema20_pct"] <= 0.08) & (d["top5_min_gain"] >= 0.20) & (d["top5_min_gain"] < 0.30)),
        "Combo 5 rank + dist_ema + wick": ("rank_at_signal >= 4 and distance_to_5m_ema20_pct <= 8% and upper_wick <= 40%", lambda d: (d["rank_at_signal"] >= 4) & (d["distance_to_5m_ema20_pct"] <= 0.08) & (d["signal_candle_upper_wick_pct"] <= 0.40)),
    }
    filter_rows = []
    month_rows = []
    original_count = len(full)
    for name, (cond, fn) in filters.items():
        mask = fn(full).fillna(False)
        rem = full[mask].copy()
        exc = full[~mask].copy()
        sorted_pnl = rem["net_pnl_usd"].sort_values(ascending=False).reset_index(drop=True)
        filter_rows.append(
            {
                "filter_name": name,
                "conditions": cond,
                "original_count": original_count,
                "remaining_count": len(rem),
                "excluded_count": len(exc),
                "excluded_minus10_rate": float(exc["hit_minus10_first"].mean()) if len(exc) else np.nan,
                "remaining_minus10_rate": float(rem["hit_minus10_first"].mean()) if len(rem) else np.nan,
                "remaining_tp15_hit_rate": float(rem["hit_plus15_before_minus10"].mean()) if len(rem) else np.nan,
                "remaining_plus30_hit_rate": float(rem["hit_plus30_before_minus10"].mean()) if len(rem) else np.nan,
                "remaining_plus50_hit_rate": float(rem["hit_plus50_before_minus10"].mean()) if len(rem) else np.nan,
                "remaining_profit_factor": profit_factor(rem["return_on_margin_pct"]) if len(rem) else np.nan,
                "remaining_total_pnl": float(rem["net_pnl_usd"].sum()) if len(rem) else 0.0,
                "remaining_max_drawdown": max_drawdown(rem.sort_values("entry_time_utc")["net_pnl_usd"]) if len(rem) else 0.0,
                "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
            }
        )
        if len(rem):
            rem["month"] = pd.to_datetime(rem["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
            for month, mdf in rem.groupby("month"):
                month_rows.append(month_summary_row(name, month, mdf))
    pd.DataFrame(filter_rows).to_csv(OUT / "filter_candidates.csv", index=False)

    print("Running confirmation entry models...", flush=True)
    models = {
        "Model 0 Immediate baseline": "immediate",
        "Model 1 Hold Top10 30m": "hold_top10_30m",
        "Model 2 Hold Top10 60m": "hold_top10_60m",
        "Model 3 Signal High Break": "signal_high_break",
        "Model 4 EMA20 Pullback": "ema20_pullback",
        "Model 5 VWAP Pullback": "vwap_pullback",
    }
    confirm_summary_rows = []
    confirm_trade_rows = []
    for model_name, key in models.items():
        rows = []
        for _, sig in signals.iterrows():
            group = kmap.get(sig["symbol"])
            if group is None:
                continue
            entry = immediate_entry(sig, group) if key == "immediate" else confirmation_entry(sig, group, rankings, key)
            if entry is None:
                continue
            trade = run_trade_from_entry(sig, group, entry[0], entry[1], model_name, entry[2])
            if trade:
                rows.append(trade)
                confirm_trade_rows.append(trade)
        df = pd.DataFrame(rows)
        s = summarize_trades(df, model_name, len(signals))
        confirm_summary_rows.append(
            {
                "model_name": model_name,
                "original_signal_count": s["original_signal_count"],
                "trade_count": s["trade_count"],
                "abandoned_count": s["abandoned_count"],
                "avg_entry_delay_minutes": s["avg_entry_delay_minutes"],
                "first_minus10_rate": s["first_minus10_rate"],
                "tp15_hit_rate": s["tp15_hit_rate"],
                "plus30_hit_rate": s["plus30_hit_rate"],
                "plus50_hit_rate": s["plus50_hit_rate"],
                "total_pnl": s["total_pnl"],
                "profit_factor": s["profit_factor"],
                "max_drawdown": s["max_drawdown"],
                "avg_trade_pnl": s["avg_trade_pnl"],
                "median_trade_pnl": s["median_trade_pnl"],
                "pnl_excluding_best_5": s["pnl_excluding_best_5"],
                "monthly_profitable_count": s["monthly_profitable_count"],
            }
        )
        if not df.empty:
            df["month"] = pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
            for month, mdf in df.groupby("month"):
                month_rows.append(month_summary_row(model_name, month, mdf))
    pd.DataFrame(confirm_summary_rows).to_csv(OUT / "confirmation_entry_summary.csv", index=False)
    pd.DataFrame(confirm_trade_rows).to_csv(OUT / "confirmation_entry_trades.csv", index=False)
    pd.DataFrame(month_rows).to_csv(OUT / "summary_by_month.csv", index=False)

    report_lines = build_report(OUT)
    (OUT / "liquidation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(pd.DataFrame(confirm_summary_rows).to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


def month_summary_row(name: str, month: str, df: pd.DataFrame) -> dict:
    return {
        "rule_or_model": name,
        "month": month,
        "trade_count": len(df),
        "pnl": float(df["net_pnl_usd"].sum()),
        "PF": profit_factor(df["return_on_margin_pct"]),
        "first_minus10_rate": float(df["minus10_first"].mean() if "minus10_first" in df else df["hit_minus10_first"].mean()),
        "TP15": float(df["tp1_hit"].mean() if "tp1_hit" in df else df["hit_plus15_before_minus10"].mean()),
        "+50%": float(df["plus50_hit"].mean() if "plus50_hit" in df else df["hit_plus50_before_minus10"].mean()),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
    }


def build_report(out: Path) -> list[str]:
    filters = pd.read_csv(out / "filter_candidates.csv")
    confirms = pd.read_csv(out / "confirmation_entry_summary.csv")
    buckets = pd.read_csv(out / "minus10_bucket_analysis.csv")
    best_filters = filters.sort_values(["remaining_minus10_rate", "remaining_profit_factor"], ascending=[True, False]).head(5)
    best_confirms = confirms.sort_values(["first_minus10_rate", "profit_factor"], ascending=[True, False]).head(6)
    risky = buckets[buckets["sample_flag"] == "ok"].sort_values("minus10_rate", ascending=False).head(8)
    return [
        "# Liquidation Analysis",
        "",
        "Scope: Rule C 21d consolidation signals inside first Top10 and 20%-30% rolling gain pool.",
        "",
        "## Highest Minus10 Buckets",
        risky.to_csv(index=False),
        "",
        "## Filter Candidates",
        filters.to_csv(index=False),
        "",
        "## Best Filters By Minus10 Rate",
        best_filters.to_csv(index=False),
        "",
        "## Confirmation Entry Summary",
        confirms.to_csv(index=False),
        "",
        "## Best Confirmation Models By Minus10 Rate",
        best_confirms.to_csv(index=False),
        "",
        "## Conclusion",
        "Use the CSV outputs for audit. A rule is not considered robust unless it reduces first -10%, improves PF, keeps enough trades, and does not depend only on 2026-01/2026-04.",
    ]


if __name__ == "__main__":
    main()
