from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.binance_client import BinanceFuturesClient
from src.data.cache import DataCache
from src.data.downloader import INTERVAL_MS, download_klines, load_cached_klines
from src.research.features import add_ema20_and_vwap
from src.research.ranking import build_rankings


SIGNALS = ROOT / "outputs/half_year_gain20_30_signals/signals_gain20_30_first_top10.csv"
BASE_TRADES = ROOT / "outputs/half_year_gain20_30_tp15_stage_runner_50_50_ma14/trades.csv"
OUT = ROOT / "outputs/half_year_regime_analysis"

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
CONFIRM_WINDOW_HOURS = 24
LOOKAHEAD_HOURS = 240


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.2f}%"


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    eq = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((eq / eq.cummax() - 1).min())


def ensure_btc_cache(start_ms: int, end_ms: int) -> None:
    cache = DataCache(cache_format="parquet")
    client = BinanceFuturesClient()
    download_klines(client, cache, "BTCUSDT", "5m", start_ms, end_ms, sleep_seconds=0.2)


def add_ma14(grouped: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol, group in grouped.items():
        g = group.sort_values("open_time").copy()
        g["open_time_utc"] = pd.to_datetime(g["open_time"], unit="ms", utc=True)
        g = g.set_index("open_time_utc")
        close_4h = g["close"].resample("4h", label="right", closed="right").last().dropna()
        ma14 = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
        g["ma14_4h_completed"] = ma14.shift(1).reindex(g.index, method="ffill").values
        reset = g.reset_index()
        reset["is_4h_close_bar"] = ((reset["open_time"] + 5 * 60 * 1000) % (4 * 60 * 60 * 1000)) == 0
        out[symbol] = reset.reset_index(drop=True)
    return out


def exit_pnl(entry_price: float, exit_price_raw: float, fraction: float) -> tuple[float, float, float]:
    notional = MARGIN_PER_TRADE * LEVERAGE * fraction
    entry_eff = entry_price * (1 + SLIP)
    exit_eff = exit_price_raw * (1 - SLIP)
    gross = notional * (exit_eff / entry_eff - 1)
    fees = notional * FEE + notional * (exit_eff / entry_eff) * FEE
    return gross - fees, gross, fees


def run_trade(sig: pd.Series, group: pd.DataFrame, trigger_time: int, variant: str) -> dict | None:
    idx = group.index[group["open_time"] > trigger_time]
    if len(idx) == 0:
        return None
    entry_idx = int(idx[0])
    entry = group.loc[entry_idx]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    liq_price = entry_price * (1 + LIQ_PCT)
    tp1_price = entry_price * (1 + TP1)
    stage2_price = entry_price * (1 + STAGE2_TRIGGER)
    path = group[group["open_time"] >= entry_time]

    remaining = 1.0
    net = 0.0
    gross = 0.0
    fees = MARGIN_PER_TRADE * LEVERAGE * FEE
    tp1_hit = False
    stage2_hit = False
    liquidated = False
    runner_exit = False
    exit_reason = "latest_price"
    exit_time = int(path.iloc[-1]["open_time"])
    exit_price = float(path.iloc[-1]["close"])
    highest_after_tp1 = np.nan
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
                exit_time = bar_time
                exit_price = liq_price
                exit_reason = "liquidation_before_tp1"
                liquidated = True
                remaining = 0.0
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
                exit_time = bar_time
                exit_price = trail_stop
                exit_reason = f"stage{2 if stage2_hit else 1}_trailing_{int(trail_pct * 100)}pct"
                runner_exit = True
                remaining = 0.0
                break
            ma14 = bar.get("ma14_4h_completed")
            if pd.notna(ma14):
                if not stage2_hit and close < float(ma14):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    exit_time = bar_time
                    exit_price = close
                    exit_reason = "stage1_5m_close_below_4h_ma14"
                    runner_exit = True
                    remaining = 0.0
                    break
                if stage2_hit and bool(bar["is_4h_close_bar"]) and close < float(ma14):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    exit_time = bar_time
                    exit_price = close
                    exit_reason = "stage2_4h_close_below_ma14"
                    runner_exit = True
                    remaining = 0.0
                    break

    if remaining > 0:
        last = path.iloc[-1]
        pnl, gp, fee = exit_pnl(entry_price, float(last["close"]), remaining)
        net += pnl
        gross += gp
        fees += fee
        exit_time = int(last["open_time"])
        exit_price = float(last["close"])
        exit_reason = "latest_price"

    return {
        "variant": variant,
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "signal_time_utc": sig["signal_time_utc"],
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
        "holding_hours": (exit_time - entry_time) / 3_600_000,
    }


def summarize_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for keys, group in trades.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        r = group["return_on_margin_pct"]
        wins = r[r > 0]
        losses = r[r <= 0]
        row.update(
            {
                "trades": len(group),
                "net_pnl_usd": group["net_pnl_usd"].sum(),
                "return_on_1000u_account_pct": group["net_pnl_usd"].sum() / ACCOUNT_CAPITAL,
                "win_rate": (r > 0).mean(),
                "tp1_rate": group["tp1_hit"].mean(),
                "plus50_rate": group["stage2_hit_plus50"].mean(),
                "minus10_first_rate": group["liquidated"].mean(),
                "profit_factor": wins.sum() / abs(losses.sum()) if abs(losses.sum()) else np.inf,
                "max_drawdown_pct": max_drawdown(group.sort_values("entry_time_utc")["net_pnl_usd"]),
                "avg_mfe_pct": group["mfe_pct"].mean(),
                "avg_mae_pct": group["mae_pct"].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def scan_path_labels(sig: pd.Series, group: pd.DataFrame) -> dict:
    idx = group.index[group["open_time"] > int(sig["signal_time"])]
    if len(idx) == 0:
        return {}
    entry = group.loc[int(idx[0])]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = entry_time + LOOKAHEAD_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    result = {
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "month": sig["signal_month_utc"],
        "hit_plus15_before_minus10": False,
        "hit_plus30_before_minus10": False,
        "hit_plus50_before_minus10": False,
        "minus10_first": False,
        "max_forward_return_240h": float(path["high"].max() / entry_price - 1) if len(path) else np.nan,
        "min_forward_return_240h": float(path["low"].min() / entry_price - 1) if len(path) else np.nan,
    }
    targets = {0.15: "hit_plus15_before_minus10", 0.30: "hit_plus30_before_minus10", 0.50: "hit_plus50_before_minus10"}
    stop = entry_price * 0.90
    for _, bar in path.iterrows():
        if float(bar["low"]) <= stop:
            result["minus10_first"] = not any(result[col] for col in targets.values())
            return result
        high = float(bar["high"])
        for threshold, col in targets.items():
            if not result[col] and high >= entry_price * (1 + threshold):
                result[col] = True
    return result


def build_btc_features(btc: pd.DataFrame, entry_times: pd.Series) -> pd.DataFrame:
    g = btc.sort_values("open_time").copy()
    g["close_1h_ago"] = g["close"].shift(12)
    g["close_4h_ago"] = g["close"].shift(48)
    g["close_24h_ago"] = g["close"].shift(288)
    g["btc_return_1h"] = g["close"] / g["close_1h_ago"] - 1
    g["btc_return_4h"] = g["close"] / g["close_4h_ago"] - 1
    g["btc_return_1d"] = g["close"] / g["close_24h_ago"] - 1
    hourly = g.set_index(pd.to_datetime(g["open_time"], unit="ms", utc=True))["close"].resample("1h").last().dropna()
    fourh = g.set_index(pd.to_datetime(g["open_time"], unit="ms", utc=True))["close"].resample("4h").last().dropna()
    h_ema = hourly.ewm(span=20, adjust=False).mean()
    f_ema = fourh.ewm(span=20, adjust=False).mean()
    g["btc_1h_ema20"] = h_ema.reindex(pd.to_datetime(g["open_time"], unit="ms", utc=True), method="ffill").values
    g["btc_4h_ema20"] = f_ema.reindex(pd.to_datetime(g["open_time"], unit="ms", utc=True), method="ffill").values
    g["btc_above_1h_ema20"] = g["close"] >= g["btc_1h_ema20"]
    g["btc_above_4h_ema20"] = g["close"] >= g["btc_4h_ema20"]
    features = pd.DataFrame({"entry_time": entry_times.astype("int64")}).sort_values("entry_time")
    merged = pd.merge_asof(features, g.sort_values("open_time"), left_on="entry_time", right_on="open_time", direction="backward")
    keep = [
        "entry_time",
        "btc_return_1h",
        "btc_return_4h",
        "btc_return_1d",
        "btc_above_1h_ema20",
        "btc_above_4h_ema20",
    ]
    merged = merged[keep]
    conditions = [
        (merged["btc_return_1h"] >= 0) & (merged["btc_return_4h"] >= 0) & (merged["btc_return_1d"] >= 0),
        (merged["btc_return_4h"] < 0) & (merged["btc_return_1d"] < 0),
    ]
    merged["btc_regime"] = np.select(conditions, ["bullish", "bearish"], default="neutral")
    return merged


def confirmation_trigger(sig: pd.Series, group: pd.DataFrame, ranking_by_symbol: dict[str, pd.DataFrame], variant: str) -> int | None:
    signal_time = int(sig["signal_time"])
    max_time = signal_time + CONFIRM_WINDOW_HOURS * 60 * 60 * 1000
    if variant == "top5_confirm_24h":
        ranks = ranking_by_symbol.get(sig["symbol"])
        if ranks is None:
            return None
        hits = ranks[(ranks["open_time"] > signal_time) & (ranks["open_time"] <= max_time) & (ranks["rank"] <= 5)]
        return None if hits.empty else int(hits.iloc[0]["open_time"])

    path = group[(group["open_time"] > signal_time) & (group["open_time"] <= max_time)]
    if path.empty:
        return None
    signal_bar = group[group["open_time"] == signal_time]
    signal_high = float(signal_bar.iloc[0]["high"]) if not signal_bar.empty else float(path.iloc[0]["high"])

    if variant == "ema20_pullback_hold_24h":
        hits = path[(path["low"] <= path["ema20"]) & (path["close"] >= path["ema20"])]
        return None if hits.empty else int(hits.iloc[0]["open_time"])
    if variant == "vwap_pullback_hold_24h":
        hits = path[(path["low"] <= path["vwap"]) & (path["close"] >= path["vwap"])]
        return None if hits.empty else int(hits.iloc[0]["open_time"])
    if variant == "break_signal_high_24h":
        hits = path[path["high"] > signal_high]
        return None if hits.empty else int(hits.iloc[0]["open_time"])
    raise ValueError(variant)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(SIGNALS)
    signals["signal_time"] = signals["signal_time"].astype("int64")
    signals["signal_month_utc"] = pd.to_datetime(signals["signal_time_utc"], utc=True).dt.strftime("%Y-%m")
    start_ms = int(signals["signal_time"].min() - 10 * 24 * 60 * 60 * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    ensure_btc_cache(start_ms, end_ms)

    print("Loading cached klines and building rankings...", flush=True)
    all_klines = load_cached_klines(ROOT / "data", "5m")
    all_klines = all_klines[(all_klines["open_time"] >= start_ms) & (all_klines["open_time"] <= end_ms)].copy()
    all_klines["open_time_utc"] = pd.to_datetime(all_klines["open_time"], unit="ms", utc=True)
    rankings = build_rankings(all_klines, 5)
    ranking_by_symbol = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in rankings.groupby("symbol", sort=False)}

    selected_symbols = set(signals["symbol"]) | {"BTCUSDT"}
    signal_klines = all_klines[all_klines["symbol"].isin(selected_symbols)].copy()
    signal_klines = add_ema20_and_vwap(signal_klines)
    kmap = add_ma14({s: g.reset_index(drop=True) for s, g in signal_klines.groupby("symbol", sort=False)})

    print("Scanning path labels...", flush=True)
    labels = []
    for _, sig in signals.iterrows():
        group = kmap.get(sig["symbol"])
        if group is not None:
            labels.append(scan_path_labels(sig, group))
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(OUT / "signal_path_labels.csv", index=False)
    monthly_path = labels_df.groupby("month").agg(
        signals=("signal_id", "count"),
        plus15_rate=("hit_plus15_before_minus10", "mean"),
        plus30_rate=("hit_plus30_before_minus10", "mean"),
        plus50_rate=("hit_plus50_before_minus10", "mean"),
        minus10_first_rate=("minus10_first", "mean"),
        avg_max_forward_return_240h=("max_forward_return_240h", "mean"),
        median_max_forward_return_240h=("max_forward_return_240h", "median"),
    ).reset_index()
    monthly_path.to_csv(OUT / "monthly_signal_path_stats.csv", index=False)

    print("Building regime features...", flush=True)
    entry_times = []
    for _, sig in signals.iterrows():
        group = kmap.get(sig["symbol"])
        idx = group.index[group["open_time"] > int(sig["signal_time"])] if group is not None else []
        entry_times.append(int(group.loc[int(idx[0]), "open_time"]) if len(idx) else np.nan)
    signals["entry_time"] = entry_times
    btc_features = build_btc_features(kmap["BTCUSDT"], signals["entry_time"].dropna().astype("int64"))
    signals = signals.merge(btc_features, on="entry_time", how="left")

    market_rows = []
    for _, sig in signals.iterrows():
        snap = rankings[rankings["open_time"] == int(sig["signal_time"])]
        market_rows.append(
            {
                "signal_id": int(sig["signal_id"]),
                "alt_median_24h_gain": snap["rolling_24h_change_pct"].median() if not snap.empty else np.nan,
                "alt_positive_rate": (snap["rolling_24h_change_pct"] > 0).mean() if not snap.empty else np.nan,
                "alt_top50_median_24h_gain": snap.nsmallest(50, "rank")["rolling_24h_change_pct"].median() if not snap.empty else np.nan,
            }
        )
    market_df = pd.DataFrame(market_rows)
    signals = signals.merge(market_df, on="signal_id", how="left")
    signals["alt_regime"] = np.select(
        [
            (signals["alt_median_24h_gain"] > 0) & (signals["alt_positive_rate"] >= 0.55),
            (signals["alt_median_24h_gain"] < 0) & (signals["alt_positive_rate"] < 0.45),
        ],
        ["strong", "weak"],
        default="mixed",
    )
    signals.to_csv(OUT / "signal_regime_features.csv", index=False)

    monthly_regime = signals.groupby("signal_month_utc").agg(
        signals=("signal_id", "count"),
        btc_1h_avg=("btc_return_1h", "mean"),
        btc_4h_avg=("btc_return_4h", "mean"),
        btc_1d_avg=("btc_return_1d", "mean"),
        btc_bullish_rate=("btc_regime", lambda s: (s == "bullish").mean()),
        btc_bearish_rate=("btc_regime", lambda s: (s == "bearish").mean()),
        alt_median_24h_gain_avg=("alt_median_24h_gain", "mean"),
        alt_positive_rate_avg=("alt_positive_rate", "mean"),
        alt_strong_rate=("alt_regime", lambda s: (s == "strong").mean()),
        alt_weak_rate=("alt_regime", lambda s: (s == "weak").mean()),
    ).reset_index()
    monthly_regime.to_csv(OUT / "monthly_btc_alt_regime.csv", index=False)

    print("Testing market filters...", flush=True)
    base_trades = pd.read_csv(BASE_TRADES)
    base_trades["entry_time_utc"] = pd.to_datetime(base_trades["entry_time_utc"], utc=True)
    enriched = base_trades.merge(signals[[
        "signal_id",
        "btc_return_1h",
        "btc_return_4h",
        "btc_return_1d",
        "btc_regime",
        "alt_regime",
        "alt_median_24h_gain",
        "alt_positive_rate",
    ]], on="signal_id", how="left")
    filters = {
        "raw_no_filter": pd.Series(True, index=enriched.index),
        "btc_1h_ge0": enriched["btc_return_1h"] >= 0,
        "btc_4h_ge0": enriched["btc_return_4h"] >= 0,
        "btc_1h_4h_ge0": (enriched["btc_return_1h"] >= 0) & (enriched["btc_return_4h"] >= 0),
        "btc_1d_ge0": enriched["btc_return_1d"] >= 0,
        "btc_bullish": enriched["btc_regime"] == "bullish",
        "btc_not_bearish": enriched["btc_regime"] != "bearish",
        "alt_strong": enriched["alt_regime"] == "strong",
        "btc_not_bearish_and_alt_strong": (enriched["btc_regime"] != "bearish") & (enriched["alt_regime"] == "strong"),
    }
    filter_rows = []
    for name, mask in filters.items():
        subset = enriched[mask.fillna(False)].copy()
        if subset.empty:
            continue
        subset["filter_name"] = name
        subset["month"] = subset["entry_time_utc"].dt.strftime("%Y-%m")
        filter_rows.append(summarize_trades(subset, ["filter_name", "month"]))
        all_row = summarize_trades(subset.assign(month="ALL"), ["filter_name", "month"])
        filter_rows.append(all_row)
    market_filter_summary = pd.concat(filter_rows, ignore_index=True)
    market_filter_summary.to_csv(OUT / "market_filter_summary_by_month.csv", index=False)

    print("Testing confirmation entries...", flush=True)
    variants = ["immediate", "top5_confirm_24h", "ema20_pullback_hold_24h", "vwap_pullback_hold_24h", "break_signal_high_24h"]
    trade_rows = []
    for variant in variants:
        for _, sig in signals.iterrows():
            group = kmap.get(sig["symbol"])
            if group is None:
                continue
            if variant == "immediate":
                trigger = int(sig["signal_time"])
            else:
                trigger = confirmation_trigger(sig, group, ranking_by_symbol, variant)
                if trigger is None:
                    continue
            row = run_trade(sig, group, trigger, variant)
            if row is not None:
                trade_rows.append(row)
    confirmation_trades = pd.DataFrame(trade_rows)
    confirmation_trades["entry_time_utc"] = pd.to_datetime(confirmation_trades["entry_time_utc"], utc=True)
    confirmation_trades["month"] = confirmation_trades["entry_time_utc"].dt.strftime("%Y-%m")
    confirmation_trades.to_csv(OUT / "confirmation_entry_trades.csv", index=False)
    conf_monthly = summarize_trades(confirmation_trades, ["variant", "month"])
    conf_all = summarize_trades(confirmation_trades.assign(month="ALL"), ["variant", "month"])
    confirmation_summary = pd.concat([conf_monthly, conf_all], ignore_index=True)
    confirmation_summary.to_csv(OUT / "confirmation_entry_summary_by_month.csv", index=False)

    report = [
        "# Half-Year Regime And Confirmation Analysis",
        "",
        "## Monthly Path Stats",
        monthly_path.to_csv(index=False),
        "",
        "## Monthly BTC / Alt Regime",
        monthly_regime.to_csv(index=False),
        "",
        "## Market Filter Summary",
        market_filter_summary.to_csv(index=False),
        "",
        "## Confirmation Entry Summary",
        confirmation_summary.to_csv(index=False),
    ]
    (OUT / "regime_analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print(monthly_path.to_string(index=False), flush=True)
    print(monthly_regime.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
