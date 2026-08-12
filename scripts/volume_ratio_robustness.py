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
from src.research.signals import identify_first_top_signals


OUT = ROOT / "outputs" / "volume_ratio_robustness"
EXCLUDE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

INTERVAL_MINUTES = 5
COOLDOWN_DAYS = 5
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

RULES = [
    ("vr_1.2_5", 1.2, 5.0),
    ("vr_1.5_5", 1.5, 5.0),
    ("vr_2.0_5", 2.0, 5.0),
    ("vr_1.5_4", 1.5, 4.0),
    ("vr_1.5_6", 1.5, 6.0),
]


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
    g = group.sort_values("open_time").drop_duplicates("open_time").copy()
    g["vol_avg_24h"] = g["volume"].rolling(288, min_periods=288).mean()
    g["open_time_utc"] = pd.to_datetime(g["open_time"], unit="ms", utc=True)
    indexed = g.set_index("open_time_utc")
    close_4h = indexed["close"].resample("4h", label="right", closed="right").last().dropna()
    ma14 = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
    indexed["ma14_4h_completed"] = ma14.shift(1).reindex(indexed.index, method="ffill").values
    reset = indexed.reset_index()
    reset["is_4h_close_bar"] = ((reset["open_time"] + 5 * 60 * 1000) % (4 * 60 * 60 * 1000)) == 0
    return reset.reset_index(drop=True)


def run_trade(sig: pd.Series, group: pd.DataFrame) -> dict | None:
    idx = group.index[group["open_time"] > int(sig["signal_time"])]
    if len(idx) == 0:
        return None
    entry = group.loc[int(idx[0])]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    end_time = min(entry_time + MAX_HOLD_HOURS * 60 * 60 * 1000, int(group["open_time"].max()))
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
    highest_after_tp1 = np.nan
    runner_exit = False
    exit_time = int(path.iloc[-1]["open_time"])
    exit_price = float(path.iloc[-1]["close"])
    exit_reason = "max_10d_or_latest_close"
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
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "signal_time": int(sig["signal_time"]),
        "signal_time_utc": sig["signal_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "volume_1h_vs_24h_avg": float(sig["volume_1h_vs_24h_avg"]),
        "rolling_24h_gain_at_signal": float(sig["rolling_24h_gain_at_signal"]),
        "rank_at_signal": int(sig["rank_at_signal"]),
        "tp1_hit": tp1_hit,
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


def volume_ratio_at(group: pd.DataFrame, signal_time: int) -> float:
    bar_rows = group[group["open_time"] == signal_time]
    if bar_rows.empty:
        return np.nan
    bar = bar_rows.iloc[-1]
    vol_avg_24h = float(bar.get("vol_avg_24h", np.nan))
    if not np.isfinite(vol_avg_24h) or vol_avg_24h <= 0:
        return np.nan
    vol_1h = float(group[(group["open_time"] >= signal_time - 55 * 60 * 1000) & (group["open_time"] <= signal_time)]["volume"].sum())
    return vol_1h / (12 * vol_avg_24h)


def summarize(df: pd.DataFrame, label: str, lookback_days: int, rule_name: str) -> dict:
    if df.empty:
        return {
            "lookback_days": lookback_days,
            "rule_name": rule_name,
            "segment": label,
            "trade_count": 0,
            "pnl": 0.0,
            "PF": np.nan,
            "TP15": np.nan,
            "+50%": np.nan,
            "first_-10%": np.nan,
            "max_drawdown": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
        }
    pnl = df["net_pnl_usd"]
    return {
        "lookback_days": lookback_days,
        "rule_name": rule_name,
        "segment": label,
        "trade_count": int(len(df)),
        "pnl": float(pnl.sum()),
        "PF": profit_factor(df["return_on_margin_pct"]),
        "TP15": float(df["tp1_hit"].mean()),
        "+50%": float(df["plus50_hit"].mean()),
        "first_-10%": float(df["minus10_first"].mean()),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
    }


def tail_dependency(df: pd.DataFrame, lookback_days: int, rule_name: str) -> dict:
    sorted_pnl = df["net_pnl_usd"].sort_values(ascending=False).reset_index(drop=True) if not df.empty else pd.Series(dtype=float)
    return {
        "lookback_days": lookback_days,
        "rule_name": rule_name,
        "trade_count": int(len(df)),
        "total_pnl": float(sorted_pnl.sum()) if len(sorted_pnl) else 0.0,
        "pnl_excluding_best_1": float(sorted_pnl.iloc[1:].sum()) if len(sorted_pnl) > 1 else 0.0,
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
        "pnl_excluding_best_10": float(sorted_pnl.iloc[10:].sum()) if len(sorted_pnl) > 10 else 0.0,
    }


def train_validation(df: pd.DataFrame, lookback_days: int, rule_name: str) -> list[dict]:
    if df.empty:
        return [summarize(df, "train", lookback_days, rule_name), summarize(df, "validation", lookback_days, rule_name)]
    ordered = df.sort_values("signal_time").copy()
    split_idx = int(len(ordered) * 0.70)
    if split_idx <= 0:
        split_idx = 1
    train = ordered.iloc[:split_idx]
    val = ordered.iloc[split_idx:]
    return [summarize(train, "train", lookback_days, rule_name), summarize(val, "validation", lookback_days, rule_name)]


def run_lookback(klines_all: pd.DataFrame, lookback_days: int, data_start: int, data_end: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_ms = data_end - lookback_days * 24 * 60 * 60 * 1000
    if data_start > start_ms - 24 * 60 * 60 * 1000:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(
            [
                {
                    "lookback_days": lookback_days,
                    "status": "skipped_insufficient_cache",
                    "available_start_utc": pd.to_datetime(data_start, unit="ms", utc=True),
                    "available_end_utc": pd.to_datetime(data_end, unit="ms", utc=True),
                    "available_days": (data_end - data_start) / 86400000,
                    "required_days": lookback_days + 1,
                }
            ]
        )

    load_start = start_ms - max(COOLDOWN_DAYS + 1, 21) * 24 * 60 * 60 * 1000
    scoped = klines_all[(klines_all["open_time"] >= load_start) & (klines_all["open_time"] <= data_end)].copy()
    ranking_klines = scoped[~scoped["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    rankings = build_rankings(ranking_klines, INTERVAL_MINUTES)
    top10 = identify_first_top_signals(rankings, top_n=10, cooldown_days=COOLDOWN_DAYS, observation_hours=72)
    base = top10[
        (top10["signal_time"] >= start_ms)
        & (top10["signal_time"] <= data_end)
        & (top10["rolling_24h_change_pct"] >= 0.20)
        & (top10["rolling_24h_change_pct"] < 0.30)
    ].copy()
    base = base.rename(
        columns={
            "rank": "rank_at_signal",
            "rolling_24h_change_pct": "rolling_24h_gain_at_signal",
            "quote_volume": "quote_volume_at_signal",
        }
    )
    symbols = set(base["symbol"])
    kmap = {
        symbol: add_indicators(group)
        for symbol, group in scoped[scoped["symbol"].isin(symbols)].groupby("symbol", sort=False)
    }
    feature_rows = []
    trade_rows = []
    for _, sig in base.iterrows():
        group = kmap.get(sig["symbol"])
        if group is None:
            continue
        vr = volume_ratio_at(group, int(sig["signal_time"]))
        if not np.isfinite(vr):
            continue
        row = {
            "signal_id": int(sig["signal_id"]),
            "symbol": sig["symbol"],
            "signal_time": int(sig["signal_time"]),
            "signal_time_utc": sig["signal_time_utc"],
            "rank_at_signal": int(sig["rank_at_signal"]),
            "rolling_24h_gain_at_signal": float(sig["rolling_24h_gain_at_signal"]),
            "quote_volume_at_signal": float(sig["quote_volume_at_signal"]),
            "volume_1h_vs_24h_avg": float(vr),
        }
        feature_rows.append(row)
        trade = run_trade(pd.Series(row), group)
        if trade:
            trade_rows.append(trade)
    features = pd.DataFrame(feature_rows)
    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades = trades.sort_values("entry_time_utc").reset_index(drop=True)
    coverage = pd.DataFrame(
        [
            {
                "lookback_days": lookback_days,
                "status": "completed",
                "available_start_utc": pd.to_datetime(data_start, unit="ms", utc=True),
                "available_end_utc": pd.to_datetime(data_end, unit="ms", utc=True),
                "available_days": (data_end - data_start) / 86400000,
                "analysis_start_utc": pd.to_datetime(start_ms, unit="ms", utc=True),
                "base_signal_count": len(features),
            }
        ]
    )
    return features, trades, rankings, coverage


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    data_start = int(klines["open_time"].min())
    data_end = int(klines["open_time"].max())

    coverage_rows = []
    all_monthly = []
    all_summary = []
    all_tv = []
    all_tail = []

    for lookback in [180, 360]:
        print(f"Running lookback={lookback}d", flush=True)
        features, trades, _rankings, coverage = run_lookback(klines, lookback, data_start, data_end)
        coverage_rows.append(coverage)
        if features.empty or trades.empty:
            continue
        features.to_csv(OUT / f"signal_features_{lookback}d.csv", index=False)
        trades.to_csv(OUT / f"base_trades_{lookback}d.csv", index=False)
        for rule_name, lo, hi in RULES:
            rule = trades[(trades["volume_1h_vs_24h_avg"] >= lo) & (trades["volume_1h_vs_24h_avg"] <= hi)].copy()
            rule.to_csv(OUT / f"trades_{lookback}d_{rule_name}.csv", index=False)
            all_summary.append(summarize(rule, "all", lookback, rule_name))
            all_tv.extend(train_validation(rule, lookback, rule_name))
            all_tail.append(tail_dependency(rule, lookback, rule_name))
            if not rule.empty:
                rule["month"] = pd.to_datetime(rule["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
                for month, mdf in rule.groupby("month"):
                    all_monthly.append(summarize(mdf, month, lookback, rule_name))

    coverage_df = pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    summary_df = pd.DataFrame(all_summary)
    tv_df = pd.DataFrame(all_tv)
    monthly_df = pd.DataFrame(all_monthly)
    tail_df = pd.DataFrame(all_tail)
    coverage_df.to_csv(OUT / "data_coverage.csv", index=False)
    summary_df.to_csv(OUT / "robustness_summary.csv", index=False)
    tv_df.to_csv(OUT / "train_validation_summary.csv", index=False)
    monthly_df.to_csv(OUT / "monthly_results.csv", index=False)
    tail_df.to_csv(OUT / "tail_dependency.csv", index=False)

    report = build_report(coverage_df, summary_df, tv_df, monthly_df, tail_df)
    (OUT / "volume_ratio_robustness_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    print(tv_df.to_string(index=False), flush=True)
    print(tail_df.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


def build_report(coverage: pd.DataFrame, summary: pd.DataFrame, tv: pd.DataFrame, monthly: pd.DataFrame, tail: pd.DataFrame) -> list[str]:
    main = summary[(summary["lookback_days"] == 180) & (summary["rule_name"] == "vr_1.5_5")]
    main_tv = tv[(tv["lookback_days"] == 180) & (tv["rule_name"] == "vr_1.5_5")]
    main_month = monthly[(monthly["lookback_days"] == 180) & (monthly["rule_name"] == "vr_1.5_5")]
    return [
        "# Volume Ratio Robustness Report",
        "",
        "Fixed signal: first Top10 within 5 days, rolling 24h gain in [20%, 30%), volume_1h_vs_24h_avg filter.",
        "Fixed exit: -10% initial stop, +15% close 50%, runner managed by 20% trailing/4H MA14 before +50%, 30% trailing/4H MA14 after +50%, max holding 10 days or latest cached close if shorter.",
        "",
        "## Data Coverage",
        coverage.to_csv(index=False),
        "",
        "## Main Rule 1.5-5 Summary",
        main.to_csv(index=False),
        "",
        "## Main Rule Train / Validation",
        main_tv.to_csv(index=False),
        "",
        "## Main Rule Monthly",
        main_month.to_csv(index=False),
        "",
        "## Adjacent Parameter Robustness",
        summary.to_csv(index=False),
        "",
        "## Tail Dependency",
        tail.to_csv(index=False),
        "",
        "## Research Conclusion",
        "Do not treat this as a live-trading edge unless validation PF stays above 1, first -10% remains below the unfiltered baseline, and tail dependency is acceptable.",
    ]


if __name__ == "__main__":
    main()
