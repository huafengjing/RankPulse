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


OUT = ROOT / "outputs/long_consolidation_top10_20_30"
EXCLUDE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

LOOKBACK_DAYS = 180
COOLDOWN_DAYS = 5
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


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def profit_factor(r: pd.Series) -> float:
    wins = r[r > 0]
    losses = r[r <= 0]
    return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) else np.inf


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


def window_features(sig: pd.Series, group: pd.DataFrame, days: int) -> tuple[float, float]:
    t = int(sig["signal_time"])
    start = t - days * 24 * 60 * 60 * 1000
    pre = group[(group["open_time"] >= start) & (group["open_time"] < t)]
    signal_bar = group[group["open_time"] == t]
    if pre.empty or signal_bar.empty:
        return np.nan, np.nan
    signal_close = float(signal_bar.iloc[0]["close"])
    first_close = float(pre.iloc[0]["close"])
    return float(pre["high"].max() / pre["low"].min() - 1), float(signal_close / first_close - 1)


def prior_rank_flag(sig: pd.Series, rankings_by_symbol: dict[str, pd.DataFrame], top_n: int, days: int) -> bool:
    group = rankings_by_symbol.get(sig["symbol"])
    if group is None:
        return False
    t = int(sig["signal_time"])
    start = t - days * 24 * 60 * 60 * 1000
    hits = group[(group["open_time"] >= start) & (group["open_time"] < t) & (group["rank"] <= top_n)]
    return not hits.empty


def summarize(df: pd.DataFrame, rule_name: str, group_name: str, total: int) -> dict:
    if df.empty:
        return {
            "rule_name": rule_name,
            "group": group_name,
            "count": 0,
            "selected_pct": 0.0,
            "tp15_hit_rate": np.nan,
            "plus30_hit_rate": np.nan,
            "plus50_hit_rate": np.nan,
            "plus100_hit_rate": np.nan,
            "minus10_first_rate": np.nan,
            "total_pnl": 0.0,
            "profit_factor": np.nan,
            "max_drawdown": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "pnl_excluding_best_1": np.nan,
            "pnl_excluding_best_5": np.nan,
            "monthly_profitable_count": 0,
        }
    r = df["return_on_margin_pct"]
    pnl = df["net_pnl_usd"]
    sorted_pnl = pnl.sort_values(ascending=False).reset_index(drop=True)
    monthly = df.assign(month=pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")).groupby("month")["net_pnl_usd"].sum()
    return {
        "rule_name": rule_name,
        "group": group_name,
        "count": len(df),
        "selected_pct": len(df) / total if total else 0.0,
        "tp15_hit_rate": float(df["tp1_hit"].mean()),
        "plus30_hit_rate": float(df["plus30_hit"].mean()),
        "plus50_hit_rate": float(df["plus50_hit"].mean()),
        "plus100_hit_rate": float(df["plus100_hit"].mean()),
        "minus10_first_rate": float(df["minus10_first"].mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(r),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
        "pnl_excluding_best_1": float(sorted_pnl.iloc[1:].sum()) if len(sorted_pnl) > 1 else 0.0,
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
        "monthly_profitable_count": int((monthly > 0).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines[~klines["symbol"].isin(["BTCUSDT", "ETHUSDT", "BNBUSDT"])].copy()
    end_ms = int(klines["open_time"].max())
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    load_start = start_ms - 22 * 24 * 60 * 60 * 1000
    klines = klines[(klines["open_time"] >= load_start) & (klines["open_time"] <= end_ms)].copy()
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

    print("Building rankings and Top10 base pool...", flush=True)
    rankings = build_rankings(klines, INTERVAL_MINUTES)
    top10_all = identify_first_top_signals(rankings, top_n=10, cooldown_days=COOLDOWN_DAYS, observation_hours=72)
    base = top10_all[
        (top10_all["signal_time"] >= start_ms)
        & (top10_all["signal_time"] <= end_ms)
        & (top10_all["rolling_24h_change_pct"] >= 0.20)
        & (top10_all["rolling_24h_change_pct"] < 0.30)
    ].copy()
    base = base.sort_values("signal_time").reset_index(drop=True)
    print(f"Base pool signals={len(base)}", flush=True)

    rankings_by_symbol = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in rankings.groupby("symbol", sort=False)}
    symbols = set(base["symbol"])
    kmap = {s: add_ma14(g) for s, g in klines[klines["symbol"].isin(symbols)].groupby("symbol", sort=False)}

    feature_rows = []
    trade_rows = []
    for _, sig in base.iterrows():
        group = kmap.get(sig["symbol"])
        if group is None:
            continue
        r7, ret7 = window_features(sig, group, 7)
        r14, ret14 = window_features(sig, group, 14)
        r21, ret21 = window_features(sig, group, 21)
        row = {
            "signal_id": int(sig["signal_id"]),
            "symbol": sig["symbol"],
            "signal_time": int(sig["signal_time"]),
            "signal_time_utc": sig["signal_time_utc"],
            "rank_at_signal": int(sig["rank"]),
            "rolling_24h_gain_at_signal": float(sig["rolling_24h_change_pct"]),
            "quote_volume_at_signal": float(sig["quote_volume"]),
            "pre_7d_range_pct": r7,
            "pre_14d_range_pct": r14,
            "pre_21d_range_pct": r21,
            "pre_7d_return": ret7,
            "pre_14d_return": ret14,
            "pre_21d_return": ret21,
            "was_in_top20_last_7d": prior_rank_flag(sig, rankings_by_symbol, 20, 7),
            "was_in_top20_last_14d": prior_rank_flag(sig, rankings_by_symbol, 20, 14),
            "was_in_top10_last_14d": prior_rank_flag(sig, rankings_by_symbol, 10, 14),
            "was_in_top50_last_14d": prior_rank_flag(sig, rankings_by_symbol, 50, 14),
        }
        row["rule_a_7d_pass"] = row["pre_7d_range_pct"] <= 0.35 and abs(row["pre_7d_return"]) <= 0.20
        row["rule_b_14d_pass"] = row["pre_14d_range_pct"] <= 0.50 and abs(row["pre_14d_return"]) <= 0.25
        row["rule_c_21d_pass"] = row["pre_21d_range_pct"] <= 0.65 and abs(row["pre_21d_return"]) <= 0.30
        row["rule_d_14d_clean_pass"] = row["rule_b_14d_pass"] and not row["was_in_top20_last_14d"]
        row["rule_e_21d_clean_pass"] = row["rule_c_21d_pass"] and not row["was_in_top20_last_14d"]
        feature_rows.append(row)
        trade = run_trade(sig, group)
        if trade is not None:
            trade_rows.append(trade)

    features = pd.DataFrame(feature_rows)
    trades = pd.DataFrame(trade_rows)
    full = features.merge(trades, on=["signal_id", "symbol"], how="inner")
    features.to_csv(OUT / "signal_features.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)

    rules = {
        "Rule A 7d consolidation": "rule_a_7d_pass",
        "Rule B 14d consolidation": "rule_b_14d_pass",
        "Rule C 21d consolidation": "rule_c_21d_pass",
        "Rule D 14d consolidation + clean Top20": "rule_d_14d_clean_pass",
        "Rule E 21d consolidation + clean Top20": "rule_e_21d_clean_pass",
    }
    summary_rows = []
    month_rows = []
    total = len(full)
    for name, col in rules.items():
        passed = full[full[col]].copy()
        failed = full[~full[col]].copy()
        summary_rows.append(summarize(passed, name, "pass", total))
        summary_rows.append(summarize(failed, name, "fail", total))
        for group_name, df in [("pass", passed), ("fail", failed)]:
            if df.empty:
                continue
            df = df.copy()
            df["month"] = pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
            for month, mdf in df.groupby("month"):
                s = summarize(mdf, name, group_name, total)
                month_rows.append(
                    {
                        "rule_name": name,
                        "group": group_name,
                        "month": month,
                        "trade_count": len(mdf),
                        "pnl": s["total_pnl"],
                        "PF": s["profit_factor"],
                        "TP15": s["tp15_hit_rate"],
                        "+50%": s["plus50_hit_rate"],
                        "-10% first": s["minus10_first_rate"],
                        "max_drawdown": s["max_drawdown"],
                    }
                )

    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(month_rows)
    summary.to_csv(OUT / "rule_summary.csv", index=False)
    monthly.to_csv(OUT / "summary_by_month.csv", index=False)
    report = [
        "# Long Consolidation Top10 20-30 Research",
        "",
        f"Base pool signals: {len(full)}",
        "",
        "## Rule Summary",
        summary.to_csv(index=False),
        "",
        "## Monthly",
        monthly.to_csv(index=False),
    ]
    (OUT / "long_consolidation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
