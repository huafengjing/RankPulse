from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.downloader import load_cached_klines
from src.research.ranking import build_rankings


SIGNALS = ROOT / "outputs/half_year_gain20_30_signals/signals_gain20_30_first_top10.csv"
OUT = ROOT / "outputs/board_sentiment"
CHARTS = OUT / "charts"

EXCLUDE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}
INTERVAL_MINUTES = 5
MAX_HOLD_HOURS = 240
MARGIN_PER_TRADE = 100.0
ACCOUNT_CAPITAL = 1000.0
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


def bucket_top5_min(x: float) -> str:
    if x < 0.15:
        return "<15%"
    if x < 0.20:
        return "15%-20%"
    if x < 0.25:
        return "20%-25%"
    if x < 0.30:
        return "25%-30%"
    return ">30%"


def bucket_top_avg(x: float) -> str:
    if x < 0.15:
        return "<15%"
    if x < 0.20:
        return "15%-20%"
    if x < 0.30:
        return "20%-30%"
    return ">30%"


def bucket_gain_gt20_count(x: int) -> str:
    if x <= 3:
        return "0-3"
    if x <= 7:
        return "4-7"
    if x <= 12:
        return "8-12"
    return ">12"


def board_temperature(top5_min_gain: float) -> str:
    if top5_min_gain < 0.15:
        return "cold"
    if top5_min_gain < 0.20:
        return "neutral"
    if top5_min_gain < 0.30:
        return "hot"
    return "very_hot"


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) else np.inf


def exit_pnl(entry_price: float, exit_price_raw: float, fraction: float) -> tuple[float, float, float]:
    notional = MARGIN_PER_TRADE * LEVERAGE * fraction
    entry_eff = entry_price * (1 + SLIP)
    exit_eff = exit_price_raw * (1 - SLIP)
    gross = notional * (exit_eff / entry_eff - 1)
    fees = notional * FEE + notional * (exit_eff / entry_eff) * FEE
    return gross - fees, gross, fees


def add_ma14(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("open_time").copy()
    g["open_time_utc"] = pd.to_datetime(g["open_time"], unit="ms", utc=True)
    g = g.set_index("open_time_utc")
    close_4h = g["close"].resample("4h", label="right", closed="right").last().dropna()
    ma14 = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
    g["ma14_4h_completed"] = ma14.shift(1).reindex(g.index, method="ffill").values
    reset = g.reset_index()
    reset["is_4h_close_bar"] = ((reset["open_time"] + 5 * 60 * 1000) % (4 * 60 * 60 * 1000)) == 0
    return reset.reset_index(drop=True)


def run_trade(sig: pd.Series, group: pd.DataFrame) -> dict | None:
    idx = group.index[group["open_time"] > int(sig["signal_time"])]
    if len(idx) == 0:
        return None
    entry_idx = int(idx[0])
    entry = group.loc[entry_idx]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
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
    fees = MARGIN_PER_TRADE * LEVERAGE * FEE
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
                liquidated = True
                remaining = 0.0
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
                runner_exit = True
                remaining = 0.0
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
                    runner_exit = True
                    remaining = 0.0
                    exit_time = bar_time
                    exit_price = close
                    exit_reason = "stage1_5m_close_below_4h_ma14"
                    break
                if stage2_hit and bool(bar["is_4h_close_bar"]) and close < float(ma14):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    runner_exit = True
                    remaining = 0.0
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
        exit_reason = "max_10d"

    return {
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp1_hit": tp1_hit,
        "stage2_hit_plus50": stage2_hit,
        "liquidated": liquidated,
        "runner_exit": runner_exit,
        "exit_reason": exit_reason,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "gross_pnl_usd": gross,
        "fees_usd": fees,
        "net_pnl_usd": net,
        "return_on_margin_pct": net / MARGIN_PER_TRADE,
    }


def scan_labels(sig: pd.Series, group: pd.DataFrame) -> dict | None:
    idx = group.index[group["open_time"] > int(sig["signal_time"])]
    if len(idx) == 0:
        return None
    entry = group.loc[int(idx[0])]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = entry_time + MAX_HOLD_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    out = {
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "tp1_15_hit": False,
        "plus30_hit": False,
        "plus50_hit": False,
        "plus100_hit": False,
        "minus10_first": False,
        "avg_mfe": float(path["high"].max() / entry_price - 1) if not path.empty else np.nan,
        "avg_mae": float(path["low"].min() / entry_price - 1) if not path.empty else np.nan,
    }
    thresholds = {0.15: "tp1_15_hit", 0.30: "plus30_hit", 0.50: "plus50_hit", 1.00: "plus100_hit"}
    for _, bar in path.iterrows():
        if float(bar["low"]) <= entry_price * 0.90:
            out["minus10_first"] = not any(out[col] for col in thresholds.values())
            return out
        high = float(bar["high"])
        for threshold, col in thresholds.items():
            if not out[col] and high >= entry_price * (1 + threshold):
                out[col] = True
    return out


def summarize_trade_subset(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trade_count": 0,
            "net_pnl": 0.0,
            "return_on_1000u": 0.0,
            "win_rate": np.nan,
            "tp1_15_hit_rate": np.nan,
            "plus50_hit_rate": np.nan,
            "minus10_first_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
            "pnl_excluding_best_1": np.nan,
            "pnl_excluding_best_5": np.nan,
            "pnl_excluding_best_10": np.nan,
        }
    r = df["return_on_margin_pct"]
    pnl = df["net_pnl_usd"]
    sorted_pnl = pnl.sort_values(ascending=False).reset_index(drop=True)
    return {
        "trade_count": len(df),
        "net_pnl": float(pnl.sum()),
        "return_on_1000u": float(pnl.sum() / ACCOUNT_CAPITAL),
        "win_rate": float((r > 0).mean()),
        "tp1_15_hit_rate": float(df["tp1_hit"].mean()),
        "plus50_hit_rate": float(df["stage2_hit_plus50"].mean()),
        "minus10_first_rate": float(df["liquidated"].mean()),
        "profit_factor": profit_factor(r),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "pnl_excluding_best_1": float(sorted_pnl.iloc[1:].sum()) if len(sorted_pnl) > 1 else 0.0,
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
        "pnl_excluding_best_10": float(sorted_pnl.iloc[10:].sum()) if len(sorted_pnl) > 10 else 0.0,
    }


def make_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(SIGNALS)
    signals["signal_time"] = signals["signal_time"].astype("int64")
    signals["month"] = pd.to_datetime(signals["signal_time_utc"], utc=True).dt.strftime("%Y-%m")
    start_ms = int(signals["signal_time"].min() - 25 * 60 * 60 * 1000)
    end_ms = int(signals["signal_time"].max() + MAX_HOLD_HOURS * 60 * 60 * 1000)

    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines[(klines["open_time"] >= start_ms) & (klines["open_time"] <= end_ms)].copy()
    klines = klines[~klines["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

    print("Building rankings...", flush=True)
    rankings = build_rankings(klines, INTERVAL_MINUTES)
    ranking_by_time = {int(t): g.sort_values("rank") for t, g in rankings.groupby("open_time", sort=False)}
    top10 = rankings[rankings["rank"] <= 10].sort_values(["symbol", "open_time"]).copy()
    top10["previous_top10_time"] = top10.groupby("symbol")["open_time"].shift(1)
    top10_events = top10[(top10["previous_top10_time"].isna()) | (top10["open_time"] - top10["previous_top10_time"] > 5 * 60_000)]
    event_times = np.sort(top10_events["open_time"].astype("int64").values)

    print("Computing board features...", flush=True)
    feature_rows = []
    for _, sig in signals.iterrows():
        t = int(sig["signal_time"])
        snap = ranking_by_time.get(t)
        if snap is None or snap.empty:
            continue
        top1 = snap.nsmallest(1, "rank")
        top3 = snap.nsmallest(3, "rank")
        top5 = snap.nsmallest(5, "rank")
        top10s = snap.nsmallest(10, "rank")
        top20 = snap.nsmallest(20, "rank")
        top5_avg = float(top5["rolling_24h_change_pct"].mean())
        top10_avg = float(top10s["rolling_24h_change_pct"].mean())
        top5_min = float(top5["rolling_24h_change_pct"].min())
        left_1h = t - 60 * 60 * 1000
        left_4h = t - 4 * 60 * 60 * 1000
        left_24h = t - 24 * 60 * 60 * 1000
        row = {
            "signal_id": int(sig["signal_id"]),
            "symbol": sig["symbol"],
            "signal_time_utc": sig["signal_time_utc"],
            "month": sig["month"],
            "rank_at_signal": int(sig["rank"]),
            "rolling_24h_gain_at_signal": float(sig["rolling_24h_change_pct"]),
            "quote_volume_at_signal": float(sig["quote_volume"]),
            "top1_gain": float(top1.iloc[0]["rolling_24h_change_pct"]),
            "top3_avg_gain": float(top3["rolling_24h_change_pct"].mean()),
            "top5_avg_gain": top5_avg,
            "top5_min_gain": top5_min,
            "top10_avg_gain": top10_avg,
            "top10_min_gain": float(top10s["rolling_24h_change_pct"].min()),
            "top20_avg_gain": float(top20["rolling_24h_change_pct"].mean()),
            "top20_min_gain": float(top20["rolling_24h_change_pct"].min()),
            "top1_to_top5_ratio": float(top1.iloc[0]["rolling_24h_change_pct"] / top5_avg) if top5_avg else np.nan,
            "top1_to_top10_ratio": float(top1.iloc[0]["rolling_24h_change_pct"] / top10_avg) if top10_avg else np.nan,
            "top5_to_top10_spread": top5_avg - top10_avg,
            "gain_gt_10_count": int((snap["rolling_24h_change_pct"] > 0.10).sum()),
            "gain_gt_15_count": int((snap["rolling_24h_change_pct"] > 0.15).sum()),
            "gain_gt_20_count": int((snap["rolling_24h_change_pct"] > 0.20).sum()),
            "gain_gt_30_count": int((snap["rolling_24h_change_pct"] > 0.30).sum()),
            "gain_gt_50_count": int((snap["rolling_24h_change_pct"] > 0.50).sum()),
            "new_top10_count_last_1h": int(((event_times > left_1h) & (event_times <= t)).sum()),
            "new_top10_count_last_4h": int(((event_times > left_4h) & (event_times <= t)).sum()),
            "new_top10_count_last_24h": int(((event_times > left_24h) & (event_times <= t)).sum()),
            "board_temperature_label": board_temperature(top5_min),
        }
        feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    features["top5_min_gain_bucket"] = features["top5_min_gain"].map(bucket_top5_min)
    features["top5_avg_gain_bucket"] = features["top5_avg_gain"].map(bucket_top_avg)
    features["top10_avg_gain_bucket"] = features["top10_avg_gain"].map(bucket_top_avg)
    features["gain_gt_20_count_bucket"] = features["gain_gt_20_count"].map(bucket_gain_gt20_count)
    features.to_csv(OUT / "signal_board_features.csv", index=False)

    print("Running fixed strategy and path labels...", flush=True)
    symbols = set(features["symbol"])
    kmap = {}
    for symbol, group in klines[klines["symbol"].isin(symbols)].groupby("symbol", sort=False):
        kmap[symbol] = add_ma14(group)
    trades = []
    labels = []
    signal_lookup = signals.merge(features[["signal_id"]], on="signal_id", how="inner")
    for _, sig in signal_lookup.iterrows():
        group = kmap.get(sig["symbol"])
        if group is None:
            continue
        trade = run_trade(sig, group)
        label = scan_labels(sig, group)
        if trade is not None:
            trades.append(trade)
        if label is not None:
            labels.append(label)
    trades_df = pd.DataFrame(trades)
    labels_df = pd.DataFrame(labels)
    trades_df["entry_time_utc"] = pd.to_datetime(trades_df["entry_time_utc"], utc=True)
    trades_df["month"] = trades_df["entry_time_utc"].dt.strftime("%Y-%m")
    trades_df.to_csv(OUT / "current_strategy_trades.csv", index=False)
    labels_df.to_csv(OUT / "signal_path_labels.csv", index=False)

    enriched = features.merge(labels_df, on=["signal_id", "symbol"], how="left", suffixes=("", "_label"))
    enriched = enriched.merge(trades_df, on=["signal_id", "symbol"], how="left", suffixes=("", "_trade"))

    bucket_rows = []
    bucket_specs = [
        ("top5_min_gain_bucket", ["<15%", "15%-20%", "20%-25%", "25%-30%", ">30%"]),
        ("top5_avg_gain_bucket", ["<15%", "15%-20%", "20%-30%", ">30%"]),
        ("top10_avg_gain_bucket", ["<15%", "15%-20%", "20%-30%", ">30%"]),
        ("gain_gt_20_count_bucket", ["0-3", "4-7", "8-12", ">12"]),
        ("board_temperature_label", ["cold", "neutral", "hot", "very_hot"]),
    ]
    for feature_name, order in bucket_specs:
        for bucket in order:
            subset = enriched[enriched[feature_name] == bucket].copy()
            metric = summarize_trade_subset(subset.dropna(subset=["net_pnl_usd"]))
            bucket_rows.append(
                {
                    "feature_name": feature_name,
                    "bucket": bucket,
                    "signal_count": len(subset),
                    "tp1_15_hit_rate": subset["tp1_15_hit"].mean() if len(subset) else np.nan,
                    "plus30_hit_rate": subset["plus30_hit"].mean() if len(subset) else np.nan,
                    "plus50_hit_rate": subset["plus50_hit"].mean() if len(subset) else np.nan,
                    "plus100_hit_rate": subset["plus100_hit"].mean() if len(subset) else np.nan,
                    "minus10_first_rate": subset["minus10_first"].mean() if len(subset) else np.nan,
                    "avg_mfe": subset["avg_mfe"].mean() if len(subset) else np.nan,
                    "avg_mae": subset["avg_mae"].mean() if len(subset) else np.nan,
                    "profit_factor_using_current_strategy": metric["profit_factor"],
                    "net_pnl_using_current_strategy": metric["net_pnl"],
                    "max_drawdown": metric["max_drawdown"],
                    "average_trade_pnl": metric["avg_trade_pnl"],
                }
            )
    bucket_analysis = pd.DataFrame(bucket_rows)
    bucket_analysis.to_csv(OUT / "board_bucket_analysis.csv", index=False)

    filter_defs = {
        "raw_no_filter": ("no filter", lambda d: pd.Series(True, index=d.index)),
        "Filter A": ("top5_min_gain >= 15%", lambda d: d["top5_min_gain"] >= 0.15),
        "Filter B": ("top5_min_gain >= 20%", lambda d: d["top5_min_gain"] >= 0.20),
        "Filter C": ("top5_min_gain >= 25%", lambda d: d["top5_min_gain"] >= 0.25),
        "Filter D": ("top5_avg_gain >= 20%", lambda d: d["top5_avg_gain"] >= 0.20),
        "Filter E": ("top10_avg_gain >= 20%", lambda d: d["top10_avg_gain"] >= 0.20),
        "Filter F": ("gain_gt_20_count >= 8", lambda d: d["gain_gt_20_count"] >= 8),
        "Filter G": ("top5_min_gain >= 20% and gain_gt_20_count >= 8", lambda d: (d["top5_min_gain"] >= 0.20) & (d["gain_gt_20_count"] >= 8)),
        "Filter H": ("top5_min_gain >= 20% and top10_avg_gain >= 18%", lambda d: (d["top5_min_gain"] >= 0.20) & (d["top10_avg_gain"] >= 0.18)),
        "Filter I": ('board_temperature_label in ["hot", "very_hot"]', lambda d: d["board_temperature_label"].isin(["hot", "very_hot"])),
    }
    filter_summary_rows = []
    filter_month_rows = []
    total_count = len(enriched)
    for name, (condition, fn) in filter_defs.items():
        mask = fn(enriched).fillna(False)
        subset = enriched[mask].dropna(subset=["net_pnl_usd"]).copy()
        metric = summarize_trade_subset(subset)
        filter_summary_rows.append(
            {
                "filter_name": name,
                "conditions": condition,
                "selected_count": len(subset),
                "selected_pct": len(subset) / total_count if total_count else 0,
                "total_net_pnl": metric["net_pnl"],
                "profit_factor": metric["profit_factor"],
                "win_rate": metric["win_rate"],
                "tp1_15_hit_rate": metric["tp1_15_hit_rate"],
                "plus50_hit_rate": metric["plus50_hit_rate"],
                "minus10_first_rate": metric["minus10_first_rate"],
                "max_drawdown": metric["max_drawdown"],
                "avg_trade_pnl": metric["avg_trade_pnl"],
                "median_trade_pnl": metric["median_trade_pnl"],
                "best_trade": metric["best_trade"],
                "worst_trade": metric["worst_trade"],
                "pnl_excluding_best_1": metric["pnl_excluding_best_1"],
                "pnl_excluding_best_5": metric["pnl_excluding_best_5"],
                "pnl_excluding_best_10": metric["pnl_excluding_best_10"],
            }
        )
        for month, mgroup in subset.groupby("month"):
            m = summarize_trade_subset(mgroup)
            filter_month_rows.append(
                {
                    "filter_name": name,
                    "conditions": condition,
                    "month": month,
                    "trade_count": len(mgroup),
                    "net_pnl": m["net_pnl"],
                    "return_on_1000u": m["return_on_1000u"],
                    "win_rate": m["win_rate"],
                    "tp1_15_hit_rate": m["tp1_15_hit_rate"],
                    "plus50_hit_rate": m["plus50_hit_rate"],
                    "minus10_first_rate": m["minus10_first_rate"],
                    "profit_factor": m["profit_factor"],
                    "max_drawdown": m["max_drawdown"],
                }
            )
    filter_summary = pd.DataFrame(filter_summary_rows)
    filter_month = pd.DataFrame(filter_month_rows)
    filter_summary.to_csv(OUT / "filter_summary.csv", index=False)
    filter_month.to_csv(OUT / "filter_summary_by_month.csv", index=False)

    print("Computing BTC comparison if available...", flush=True)
    regime_rows = []
    btc = load_cached_klines(ROOT / "data", "5m")
    btc = btc[btc["symbol"] == "BTCUSDT"].sort_values("open_time").copy()
    if not btc.empty:
        btc["ret_1h"] = btc["close"] / btc["close"].shift(12) - 1
        btc["ret_4h"] = btc["close"] / btc["close"].shift(48) - 1
        btc["ret_1d"] = btc["close"] / btc["close"].shift(288) - 1
        entry_times = enriched[["signal_id", "entry_time_utc"]].dropna().copy()
        entry_times["entry_time"] = pd.to_datetime(entry_times["entry_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))
        btc_features = pd.merge_asof(
            entry_times.sort_values("entry_time"),
            btc[["open_time", "ret_1h", "ret_4h", "ret_1d"]].sort_values("open_time"),
            left_on="entry_time",
            right_on="open_time",
            direction="backward",
        )
        btc_features["btc_regime"] = np.select(
            [
                (btc_features["ret_1h"] >= 0) & (btc_features["ret_4h"] >= 0) & (btc_features["ret_1d"] >= 0),
                (btc_features["ret_4h"] < 0) & (btc_features["ret_1d"] < 0),
            ],
            ["bullish", "bearish"],
            default="neutral",
        )
        enriched = enriched.merge(btc_features[["signal_id", "btc_regime"]], on="signal_id", how="left")
        for label, group in enriched.groupby("btc_regime"):
            metric = summarize_trade_subset(group.dropna(subset=["net_pnl_usd"]))
            regime_rows.append({"regime_type": "btc", "regime_label": label, **metric})
    for label, group in enriched.groupby("board_temperature_label"):
        metric = summarize_trade_subset(group.dropna(subset=["net_pnl_usd"]))
        regime_rows.append({"regime_type": "board", "regime_label": label, **metric})
    regime_comparison = pd.DataFrame(regime_rows)
    regime_comparison.rename(columns={"tp1_15_hit_rate": "tp1_hit_rate", "minus10_first_rate": "minus10_first_rate"}, inplace=True)
    regime_comparison.to_csv(OUT / "regime_comparison.csv", index=False)

    raw_month = filter_month[filter_month["filter_name"] == "raw_no_filter"][["month", "net_pnl"]].rename(columns={"net_pnl": "raw_net_pnl"})
    hot_month = filter_month[filter_month["filter_name"] == "Filter I"][["month", "net_pnl"]].rename(columns={"net_pnl": "board_filter_net_pnl"})
    raw_vs_filter = raw_month.merge(hot_month, on="month", how="outer").fillna(0)
    raw_vs_filter.to_csv(OUT / "monthly_raw_vs_board_filter.csv", index=False)

    temp_pnl = filter_month[filter_month["filter_name"] == "raw_no_filter"].copy()
    temp_bucket = bucket_analysis[bucket_analysis["feature_name"] == "board_temperature_label"]
    make_bar(temp_bucket, "bucket", "net_pnl_using_current_strategy", CHARTS / "pnl_by_board_temperature.png", "PNL by Board Temperature")
    top5_bucket = bucket_analysis[bucket_analysis["feature_name"] == "top5_min_gain_bucket"]
    make_bar(top5_bucket, "bucket", "tp1_15_hit_rate", CHARTS / "tp1_hit_rate_by_top5_min_gain.png", "TP1 Hit Rate by Top5 Min Gain")
    make_bar(top5_bucket, "bucket", "plus50_hit_rate", CHARTS / "plus50_hit_rate_by_top5_min_gain.png", "+50 Hit Rate by Top5 Min Gain")
    make_bar(top5_bucket, "bucket", "minus10_first_rate", CHARTS / "minus10_rate_by_top5_min_gain.png", "-10 First Rate by Top5 Min Gain")
    make_bar(filter_summary, "filter_name", "profit_factor", CHARTS / "filter_comparison_profit_factor.png", "Filter Comparison Profit Factor")
    temp_dist = features.groupby(["month", "board_temperature_label"]).size().reset_index(name="count")
    temp_dist.pivot(index="month", columns="board_temperature_label", values="count").fillna(0).plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("Board Temperature Distribution by Month")
    plt.tight_layout()
    plt.savefig(CHARTS / "board_temperature_distribution_by_month.png")
    plt.close()
    raw_vs_filter.plot(x="month", y=["raw_net_pnl", "board_filter_net_pnl"], kind="bar", figsize=(10, 5))
    plt.title("Monthly PNL Raw vs Board Filter")
    plt.tight_layout()
    plt.savefig(CHARTS / "monthly_pnl_raw_vs_board_filter.png")
    plt.close()
    regime_plot = regime_comparison.pivot(index="regime_label", columns="regime_type", values="profit_factor")
    regime_plot.plot(kind="bar", figsize=(9, 5))
    plt.title("BTC Regime vs Board Regime")
    plt.tight_layout()
    plt.savefig(CHARTS / "btc_regime_vs_board_regime.png")
    plt.close()

    best = filter_summary.sort_values(["profit_factor", "total_net_pnl"], ascending=False).iloc[0]
    report = [
        "# Board Sentiment Research",
        "",
        "## Executive Summary",
        f"- Raw trades: {int(filter_summary.loc[filter_summary['filter_name'] == 'raw_no_filter', 'selected_count'].iloc[0])}",
        f"- Raw PF: {float(filter_summary.loc[filter_summary['filter_name'] == 'raw_no_filter', 'profit_factor'].iloc[0]):.4f}",
        f"- Best PF filter: {best['filter_name']} ({best['conditions']}), PF={best['profit_factor']:.4f}, trades={int(best['selected_count'])}",
        "",
        "## Filter Summary",
        filter_summary.to_csv(index=False),
        "",
        "## Monthly Filter Summary",
        filter_month.to_csv(index=False),
        "",
        "## Bucket Analysis",
        bucket_analysis.to_csv(index=False),
        "",
        "## Regime Comparison",
        regime_comparison.to_csv(index=False),
        "",
        "## Conclusion",
        "Board sentiment filters are evaluated only with signal-time ranking data. See filter_summary_by_month.csv before accepting any total-return improvement.",
    ]
    (OUT / "board_sentiment_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print(filter_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
