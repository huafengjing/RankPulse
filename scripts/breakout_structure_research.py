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
from src.research.signals import identify_first_top_signals


OUT = ROOT / "outputs/breakout_structure"
CHARTS = OUT / "charts"
EXCLUDE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

INTERVAL_MINUTES = 5
LOOKBACK_DAYS = 180
COOLDOWN_DAYS = 5
PRE_HOURS = 72
TOP20_TO_TOP10_HOURS = 24
MAX_HOLD_HOURS = 240

PRE_RANGE_MAX = 0.25
PRE_RETURN_MAX = 0.30
VOLUME_RATIO_MIN = 1.5

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


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def profit_factor(r: pd.Series) -> float:
    wins = r[r > 0]
    losses = r[r <= 0]
    return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) else np.inf


def gain_bucket(x: float) -> str:
    if x < 0.10:
        return "<10%"
    if x < 0.20:
        return "10%-20%"
    if x < 0.30:
        return "20%-30%"
    if x < 0.50:
        return "30%-50%"
    return ">50%"


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


def exit_pnl(entry_price: float, exit_price_raw: float, fraction: float) -> tuple[float, float, float]:
    notional = MARGIN_PER_TRADE * LEVERAGE * fraction
    entry_eff = entry_price * (1 + SLIP)
    exit_eff = exit_price_raw * (1 - SLIP)
    gross = notional * (exit_eff / entry_eff - 1)
    fees = notional * FEE + notional * (exit_eff / entry_eff) * FEE
    return gross - fees, gross, fees


def run_trade(event: pd.Series, group: pd.DataFrame) -> dict | None:
    idx = group.index[group["open_time"] > int(event["event_time"])]
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
        "event_id": int(event["event_id"]),
        "event_group": event["event_group"],
        "symbol": event["symbol"],
        "event_time_utc": event["event_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp1_hit": tp1_hit,
        "plus30_hit": mfe >= 0.30 and not liquidated,
        "plus50_hit": stage2_hit,
        "plus100_hit": mfe >= 1.00 and not liquidated,
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


def pre_structure_features(event: pd.Series, group: pd.DataFrame) -> dict:
    t = int(event["structure_time"] if "structure_time" in event and pd.notna(event["structure_time"]) else event["event_time"])
    start = t - PRE_HOURS * 60 * 60 * 1000
    pre = group[(group["open_time"] >= start) & (group["open_time"] < t)]
    event_bar = group[group["open_time"] == t]
    if pre.empty or event_bar.empty:
        return {
            "pre_72h_range_pct": np.nan,
            "pre_72h_return": np.nan,
            "volume_ratio_vs_72h_avg": np.nan,
            "is_breakout_like": False,
        }
    event_close = float(event_bar.iloc[0]["close"])
    event_volume = float(event_bar.iloc[0]["volume"])
    first_close = float(pre.iloc[0]["close"])
    avg_vol = float(pre["volume"].mean())
    pre_range = float(pre["high"].max() / pre["low"].min() - 1)
    pre_return = float(event_close / first_close - 1)
    vol_ratio = event_volume / avg_vol if avg_vol > 0 else np.nan
    is_breakout = pre_range <= PRE_RANGE_MAX and pre_return <= PRE_RETURN_MAX and vol_ratio >= VOLUME_RATIO_MIN
    return {
        "pre_72h_range_pct": pre_range,
        "pre_72h_return": pre_return,
        "volume_ratio_vs_72h_avg": vol_ratio,
        "is_breakout_like": bool(is_breakout),
    }


def summarize_trades(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "group": label,
            "signal_count": 0,
            "trade_count": 0,
            "tp15_hit_rate": np.nan,
            "plus30_hit_rate": np.nan,
            "plus50_hit_rate": np.nan,
            "plus100_hit_rate": np.nan,
            "minus10_first_rate": np.nan,
            "avg_mfe": np.nan,
            "avg_mae": np.nan,
            "total_pnl": 0.0,
            "profit_factor": np.nan,
            "max_drawdown": 0.0,
            "monthly_profitable_count": 0,
            "pnl_excluding_best_1": np.nan,
            "pnl_excluding_best_5": np.nan,
        }
    pnl = df["net_pnl_usd"]
    r = df["return_on_margin_pct"]
    by_month = df.assign(month=pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")).groupby("month")["net_pnl_usd"].sum()
    sorted_pnl = pnl.sort_values(ascending=False).reset_index(drop=True)
    return {
        "group": label,
        "signal_count": int(df["event_id"].nunique()),
        "trade_count": len(df),
        "tp15_hit_rate": float(df["tp1_hit"].mean()),
        "plus30_hit_rate": float(df["plus30_hit"].mean()),
        "plus50_hit_rate": float(df["plus50_hit"].mean()),
        "plus100_hit_rate": float(df["plus100_hit"].mean()),
        "minus10_first_rate": float(df["minus10_first"].mean()),
        "avg_mfe": float(df["mfe_pct"].mean()),
        "avg_mae": float(df["mae_pct"].mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(r),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "monthly_profitable_count": int((by_month > 0).sum()),
        "pnl_excluding_best_1": float(sorted_pnl.iloc[1:].sum()) if len(sorted_pnl) > 1 else 0.0,
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
    }


def make_group_events(signals: pd.DataFrame, event_group: str) -> pd.DataFrame:
    out = signals.copy()
    out["event_group"] = event_group
    out["event_time"] = out["signal_time"].astype("int64")
    out["event_time_utc"] = out["signal_time_utc"]
    if "structure_time" not in out.columns:
        out["structure_time"] = out["signal_time"].astype("int64")
    if "structure_time_utc" not in out.columns:
        out["structure_time_utc"] = out["signal_time_utc"]
    out["rolling_24h_gain_at_event"] = out["rolling_24h_change_pct"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    print("Loading cached 180d klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines[~klines["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    end_ms = int(klines["open_time"].max())
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    load_start = start_ms - PRE_HOURS * 60 * 60 * 1000 - 24 * 60 * 60 * 1000
    klines = klines[(klines["open_time"] >= load_start) & (klines["open_time"] <= end_ms)].copy()
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

    print("Building rankings...", flush=True)
    rankings = build_rankings(klines, INTERVAL_MINUTES)
    ranking_window = rankings[(rankings["open_time"] >= start_ms) & (rankings["open_time"] <= end_ms)].copy()
    top20_all = identify_first_top_signals(rankings, top_n=20, cooldown_days=COOLDOWN_DAYS, observation_hours=72)
    top10_all = identify_first_top_signals(rankings, top_n=10, cooldown_days=COOLDOWN_DAYS, observation_hours=72)
    top20 = top20_all[(top20_all["signal_time"] >= start_ms) & (top20_all["signal_time"] <= end_ms)].copy()
    top10 = top10_all[(top10_all["signal_time"] >= start_ms) & (top10_all["signal_time"] <= end_ms)].copy()

    rank_by_symbol = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in rankings.groupby("symbol", sort=False)}
    top20_c = top20.copy()
    confirm_rows = []
    for _, row in top20_c.iterrows():
        group = rank_by_symbol.get(row["symbol"])
        if group is None:
            continue
        start = int(row["signal_time"])
        end = start + TOP20_TO_TOP10_HOURS * 60 * 60 * 1000
        hit = group[(group["open_time"] > start) & (group["open_time"] <= end) & (group["rank"] <= 10)]
        if not hit.empty:
            confirmed = row.copy()
            confirmed["structure_time"] = int(row["signal_time"])
            confirmed["structure_time_utc"] = row["signal_time_utc"]
            confirmed["top20_rank"] = int(row["rank"])
            confirmed["top20_rolling_24h_gain"] = float(row["rolling_24h_change_pct"])
            confirmed["signal_time"] = int(hit.iloc[0]["open_time"])
            confirmed["signal_time_utc"] = hit.iloc[0]["open_time_utc"]
            confirmed["rank"] = int(hit.iloc[0]["rank"])
            confirmed["rolling_24h_change_pct"] = float(hit.iloc[0]["rolling_24h_change_pct"])
            confirmed["close"] = float(hit.iloc[0]["close"])
            confirmed["quote_volume"] = float(hit.iloc[0]["quote_volume"])
            confirm_rows.append(confirmed)
    top20_to_top10 = pd.DataFrame(confirm_rows)

    print("Preparing symbol klines...", flush=True)
    used_symbols = set(top20["symbol"]) | set(top10["symbol"]) | (set(top20_to_top10["symbol"]) if not top20_to_top10.empty else set())
    kmap = {
        symbol: add_ma14(group)
        for symbol, group in klines[klines["symbol"].isin(used_symbols)].groupby("symbol", sort=False)
    }

    event_frames = [
        make_group_events(top20, "all_first_top20"),
        make_group_events(top10, "all_first_top10"),
    ]
    if not top20_to_top10.empty:
        event_frames.append(make_group_events(top20_to_top10, "top20_breakout_then_top10_24h"))
    events = pd.concat(event_frames, ignore_index=True)
    events = events.sort_values(["event_time", "event_group", "symbol"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)

    print("Computing pre-72h structure and running trades...", flush=True)
    feature_rows = []
    trade_rows = []
    for _, event in events.iterrows():
        group = kmap.get(event["symbol"])
        if group is None:
            continue
        features = pre_structure_features(event, group)
        feature_rows.append({**event.to_dict(), **features, "gain_bucket": gain_bucket(float(event["rolling_24h_gain_at_event"]))})
        trade = run_trade(event, group)
        if trade is not None:
            trade_rows.append(trade)
    features_df = pd.DataFrame(feature_rows)
    trades_df = pd.DataFrame(trade_rows)
    events_out = features_df.merge(trades_df, on=["event_id", "event_group", "symbol", "event_time_utc"], how="left", suffixes=("", "_trade"))
    features_df.to_csv(OUT / "event_features.csv", index=False)
    trades_df.to_csv(OUT / "trades.csv", index=False)

    group_defs = []
    for base in ["all_first_top20", "all_first_top10"]:
        subset = events_out[events_out["event_group"] == base]
        group_defs.append((f"全部{base.replace('all_first_', '')}", subset))
        group_defs.append((f"横盘后{base.replace('all_first_', '')}", subset[subset["is_breakout_like"]]))
        group_defs.append((f"非横盘{base.replace('all_first_', '')}", subset[~subset["is_breakout_like"]]))
    if "top20_breakout_then_top10_24h" in set(events_out["event_group"]):
        c = events_out[(events_out["event_group"] == "top20_breakout_then_top10_24h") & (events_out["is_breakout_like"])]
        group_defs.append(("首次Top20横盘过滤后24h进Top10", c))

    group_defs = []
    for base in ["all_first_top20", "all_first_top10"]:
        subset = events_out[events_out["event_group"] == base]
        short = base.replace("all_first_", "")
        group_defs.append((f"all_{short}", subset))
        group_defs.append((f"breakout_like_{short}", subset[subset["is_breakout_like"]]))
        group_defs.append((f"non_breakout_{short}", subset[~subset["is_breakout_like"]]))
    if "top20_breakout_then_top10_24h" in set(events_out["event_group"]):
        c = events_out[(events_out["event_group"] == "top20_breakout_then_top10_24h") & (events_out["is_breakout_like"])]
        group_defs.append(("breakout_top20_then_top10_24h", c))

    summary = pd.DataFrame([summarize_trades(df.dropna(subset=["net_pnl_usd"]).copy(), label) for label, df in group_defs])
    summary.to_csv(OUT / "group_comparison_summary.csv", index=False)

    bucket_rows = []
    for group_name, group_df in group_defs:
        for bucket, bdf in group_df.groupby("gain_bucket"):
            bdf = bdf.dropna(subset=["net_pnl_usd"]).copy()
            s = summarize_trades(bdf, group_name)
            bucket_rows.append(
                {
                    "group": group_name,
                    "gain_bucket": bucket,
                    "signal_count": s["signal_count"],
                    "tp15_hit_rate": s["tp15_hit_rate"],
                    "plus50_hit_rate": s["plus50_hit_rate"],
                    "minus10_first_rate": s["minus10_first_rate"],
                    "profit_factor": s["profit_factor"],
                }
            )
    gain_bucket_df = pd.DataFrame(bucket_rows)
    gain_bucket_df.to_csv(OUT / "gain_bucket_analysis.csv", index=False)

    monthly = []
    for group_name, group_df in group_defs:
        group_df = group_df.dropna(subset=["net_pnl_usd"]).copy()
        if group_df.empty:
            continue
        group_df["month"] = pd.to_datetime(group_df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
        for month, mdf in group_df.groupby("month"):
            s = summarize_trades(mdf, group_name)
            monthly.append({"group": group_name, "month": month, **s})
    monthly_df = pd.DataFrame(monthly)
    monthly_df.to_csv(OUT / "group_comparison_by_month.csv", index=False)

    summary.plot(x="group", y=["profit_factor"], kind="bar", figsize=(11, 5))
    plt.title("Profit Factor by Group")
    plt.tight_layout()
    plt.savefig(CHARTS / "profit_factor_by_group.png")
    plt.close()
    summary.plot(x="group", y=["plus50_hit_rate", "minus10_first_rate"], kind="bar", figsize=(11, 5))
    plt.title("+50 Hit Rate and -10 First Rate by Group")
    plt.tight_layout()
    plt.savefig(CHARTS / "hit_rates_by_group.png")
    plt.close()
    monthly_df.pivot_table(index="month", columns="group", values="total_pnl", aggfunc="sum").fillna(0).plot(kind="bar", figsize=(12, 5))
    plt.title("Monthly PNL by Group")
    plt.tight_layout()
    plt.savefig(CHARTS / "monthly_pnl_by_group.png")
    plt.close()

    report_lines = [
        "# Breakout Structure Research",
        "",
        "## Group Comparison",
        summary.to_csv(index=False),
        "",
        "## Gain Bucket Analysis",
        gain_bucket_df.to_csv(index=False),
        "",
        "## Monthly Comparison",
        monthly_df.to_csv(index=False),
        "",
        "## Notes",
        "- Breakout-like filter uses only pre-event 72h range/return/volume.",
        "- rolling_24h_gain_at_event is not fixed; it is only analyzed by bucket.",
        "- Exit logic is fixed and unchanged from the current TP15/MA14 research model.",
    ]
    (OUT / "breakout_structure_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
