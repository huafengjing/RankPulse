from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.downloader import load_cached_klines


SOURCE_OUT = ROOT / "outputs" / "long_consolidation_top10_20_30"
OUT = ROOT / "outputs" / "rule_c_tp15_tp30_runner_ma14"

ACCOUNT_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 10.0
FEE = 0.0005
SLIP = 0.0005

LIQ_PCT = -0.10
TP1 = 0.15
TP2 = 0.30
STAGE2_TRIGGER = 0.50
TP1_FRAC = 0.50
TP2_FRAC = 0.25
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

    entry = group.loc[int(idx[0])]
    entry_time = int(entry["open_time"])
    entry_price = float(entry["open"])
    path = group[group["open_time"] >= entry_time]
    if path.empty:
        return None

    liq_price = entry_price * (1 + LIQ_PCT)
    tp1_price = entry_price * (1 + TP1)
    tp2_price = entry_price * (1 + TP2)
    stage2_price = entry_price * (1 + STAGE2_TRIGGER)

    remaining = 1.0
    net = 0.0
    gross = 0.0
    fees = 0.0
    tp1_hit = False
    tp2_hit = False
    stage2_hit = False
    liquidated = False
    highest_after_tp1 = np.nan
    runner_exit = False
    exit_time = int(path.iloc[-1]["open_time"])
    exit_price = float(path.iloc[-1]["close"])
    exit_reason = "latest_close"

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

        if tp1_hit and (not tp2_hit) and remaining > TP2_FRAC:
            if high >= tp2_price:
                pnl, gp, fee = exit_pnl(entry_price, tp2_price, TP2_FRAC)
                net += pnl
                gross += gp
                fees += fee
                remaining -= TP2_FRAC
                tp2_hit = True

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
        exit_reason = "latest_close"

    return {
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "signal_time_utc": sig["signal_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
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


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "month": label,
            "trade_count": 0,
            "net_pnl": 0.0,
            "return_on_1000u": 0.0,
            "win_rate": np.nan,
            "tp1_hit_rate": np.nan,
            "tp2_hit_rate": np.nan,
            "plus50_hit_rate": np.nan,
            "minus10_first_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
        }
    pnl = df["net_pnl_usd"]
    r = df["return_on_margin_pct"]
    return {
        "month": label,
        "trade_count": int(len(df)),
        "net_pnl": float(pnl.sum()),
        "return_on_1000u": float(pnl.sum() / ACCOUNT_CAPITAL),
        "win_rate": float((pnl > 0).mean()),
        "tp1_hit_rate": float(df["tp1_hit"].mean()),
        "tp2_hit_rate": float(df["tp2_hit"].mean()),
        "plus50_hit_rate": float(df["plus50_hit"].mean()),
        "minus10_first_rate": float(df["minus10_first"].mean()),
        "profit_factor": profit_factor(r),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(SOURCE_OUT / "signal_features.csv")
    selected = features[features["rule_c_21d_pass"] == True].copy()
    print(f"Rule C signals={len(selected)}", flush=True)

    klines = load_cached_klines(ROOT / "data", "5m")
    symbols = set(selected["symbol"])
    kmap = {
        symbol: add_ma14(group)
        for symbol, group in klines[klines["symbol"].isin(symbols)].groupby("symbol", sort=False)
    }

    rows = []
    for _, sig in selected.iterrows():
        group = kmap.get(sig["symbol"])
        if group is None:
            continue
        trade = run_trade(sig, group)
        if trade is not None:
            rows.append(trade)

    trades = pd.DataFrame(rows).sort_values("entry_time_utc")
    trades.to_csv(OUT / "trades.csv", index=False)

    summary_rows = [summarize(trades, "ALL")]
    trades["month"] = pd.to_datetime(trades["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    for month, mdf in trades.groupby("month"):
        summary_rows.append(summarize(mdf, month))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary_by_month.csv", index=False)

    exit_counts = trades["exit_reason"].value_counts().reset_index()
    exit_counts.columns = ["exit_reason", "count"]
    exit_counts.to_csv(OUT / "exit_reason_counts.csv", index=False)

    report = [
        "# Rule C TP15 TP30 Runner MA14",
        "",
        "Signal pool: Rule C 21d consolidation from long_consolidation_top10_20_30.",
        "",
        "Exit: -10% initial stop; +15% close 50%; +30% close 25%; remaining 25% uses 20% trailing before +50%, 30% trailing after +50%, plus 4H MA14 management. Open positions exit at latest cached close.",
        "",
        "## Summary",
        summary.to_csv(index=False),
        "",
        "## Exit Reasons",
        exit_counts.to_csv(index=False),
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
