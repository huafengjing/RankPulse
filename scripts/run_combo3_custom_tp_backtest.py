from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import load_cached_klines


OUT = ROOT / "outputs" / "drop_top10_short_filter"
INTERVAL = "5m"
LOOKAHEAD_HOURS = 120
ACCOUNT_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 10.0
NOTIONAL_PER_TRADE = MARGIN_PER_TRADE * LEVERAGE
TAKER_FEE = 0.0005
SLIPPAGE = 0.0005


def short_pnl(entry_price: float, exit_price: float, notional: float) -> tuple[float, float, float]:
    entry_effective = entry_price * (1.0 - SLIPPAGE)
    exit_effective = exit_price * (1.0 + SLIPPAGE)
    gross = notional * ((entry_effective - exit_effective) / entry_effective)
    fees = notional * TAKER_FEE * 2.0
    return gross - fees, gross, fees


def simulate_trade(group: pd.DataFrame, signal: pd.Series) -> dict[str, object] | None:
    entry_time = pd.Timestamp(signal["entry_time_utc"]).timestamp() * 1000
    entry_time = int(entry_time)
    entry_price = float(signal["entry_price"])
    end_time = entry_time + LOOKAHEAD_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    if path.empty:
        return None

    sl_price = entry_price * 1.10
    tp1_price = entry_price * 0.90
    tp2_price = entry_price * 0.85
    remaining_notional = NOTIONAL_PER_TRADE
    realized = 0.0
    gross_total = 0.0
    fees_total = 0.0
    tp1_hit = False
    tp2_hit = False
    exit_reason = "max_holding_time"
    exit_time_utc = path.iloc[-1]["open_time_utc"]
    exit_price = float(path.iloc[-1]["close"])
    tp1_time_utc = pd.NaT
    tp2_time_utc = pd.NaT

    for _, bar in path.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        # Conservative same-candle rule for shorts: adverse +10% stop wins ties.
        if high >= sl_price:
            pnl, gross, fees = short_pnl(entry_price, sl_price, remaining_notional)
            realized += pnl
            gross_total += gross
            fees_total += fees
            exit_reason = "sl_plus10"
            exit_time_utc = bar["open_time_utc"]
            exit_price = sl_price
            remaining_notional = 0.0
            break

        if not tp1_hit and low <= tp1_price:
            pnl, gross, fees = short_pnl(entry_price, tp1_price, NOTIONAL_PER_TRADE * 0.80)
            realized += pnl
            gross_total += gross
            fees_total += fees
            remaining_notional = NOTIONAL_PER_TRADE * 0.20
            tp1_hit = True
            tp1_time_utc = bar["open_time_utc"]

        if tp1_hit and low <= tp2_price:
            pnl, gross, fees = short_pnl(entry_price, tp2_price, remaining_notional)
            realized += pnl
            gross_total += gross
            fees_total += fees
            remaining_notional = 0.0
            tp2_hit = True
            tp2_time_utc = bar["open_time_utc"]
            exit_reason = "tp2_minus15"
            exit_time_utc = bar["open_time_utc"]
            exit_price = tp2_price
            break

    if remaining_notional > 0:
        pnl, gross, fees = short_pnl(entry_price, exit_price, remaining_notional)
        realized += pnl
        gross_total += gross
        fees_total += fees

    return {
        "signal_id": int(signal["signal_id"]),
        "symbol": signal["symbol"],
        "signal_time_utc": signal["signal_time_utc"],
        "entry_time_utc": signal["entry_time_utc"],
        "exit_time_utc": exit_time_utc,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "tp1_time_utc": tp1_time_utc,
        "tp2_time_utc": tp2_time_utc,
        "gross_pnl_usd": gross_total,
        "fees_slippage_model_fee_usd": fees_total,
        "net_pnl_usd": realized,
        "return_on_trade_margin": realized / MARGIN_PER_TRADE,
        "return_on_account": realized / ACCOUNT_CAPITAL,
        "holding_minutes": int((pd.Timestamp(exit_time_utc).timestamp() * 1000 - entry_time) / 60_000),
    }


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(wins / abs(losses)) if abs(losses) else np.inf


def summarize(df: pd.DataFrame) -> dict[str, object]:
    pnl = df["net_pnl_usd"]
    return {
        "trade_count": len(df),
        "win_rate": float((pnl > 0).mean()) if len(df) else 0.0,
        "tp1_hit_rate": float(df["tp1_hit"].mean()) if len(df) else 0.0,
        "tp2_hit_rate": float(df["tp2_hit"].mean()) if len(df) else 0.0,
        "sl_plus10_rate": float(df["exit_reason"].eq("sl_plus10").mean()) if len(df) else 0.0,
        "total_net_pnl_usd": float(pnl.sum()),
        "account_return": float(pnl.sum() / ACCOUNT_CAPITAL),
        "avg_trade_pnl_usd": float(pnl.mean()) if len(df) else 0.0,
        "median_trade_pnl_usd": float(pnl.median()) if len(df) else 0.0,
        "profit_factor": profit_factor(pnl),
        "max_drawdown_pct": max_drawdown(pnl),
        "ending_equity_usd": float(ACCOUNT_CAPITAL + pnl.sum()),
        "pnl_excluding_best_1": float(pnl.sort_values(ascending=False).iloc[1:].sum()) if len(df) > 1 else 0.0,
        "pnl_excluding_best_5": float(pnl.sort_values(ascending=False).iloc[5:].sum()) if len(df) > 5 else 0.0,
    }


def main() -> None:
    features = pd.read_csv(OUT / "signal_features.csv")
    base = pd.read_csv(OUT / "base_signals.csv")
    combo = features[
        features["volume_1h_vs_24h_avg"].between(1.2, 5)
        & (features["signal_candle_lower_wick_pct"] <= 0.40)
    ][["signal_id"]]
    signals = base.merge(combo, on="signal_id", how="inner").sort_values("entry_time_utc")

    klines = load_cached_klines(ROOT / "data", INTERVAL)
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    symbols = set(signals["symbol"])
    klines = klines[klines["symbol"].isin(symbols)].sort_values(["symbol", "open_time"])
    kmap = {symbol: group.reset_index(drop=True) for symbol, group in klines.groupby("symbol", sort=False)}

    rows = []
    for _, signal in signals.iterrows():
        row = simulate_trade(kmap[signal["symbol"]], signal)
        if row is not None:
            rows.append(row)
    trades = pd.DataFrame(rows).sort_values("entry_time_utc")
    trades.to_csv(OUT / "combo3_tp10_15_trades.csv", index=False)

    summary = pd.DataFrame([{**summarize(trades), "rule_name": "Combo3_TP10_80_TP15_20"}])
    summary.to_csv(OUT / "combo3_tp10_15_summary.csv", index=False)

    trades["month"] = pd.to_datetime(trades["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    monthly_rows = []
    for month, group in trades.groupby("month"):
        monthly_rows.append({"month": month, **summarize(group)})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT / "combo3_tp10_15_summary_by_month.csv", index=False)

    print(summary.to_string(index=False))
    print(monthly.to_string(index=False))
    print(OUT / "combo3_tp10_15_trades.csv")
    print(OUT / "combo3_tp10_15_summary_by_month.csv")


if __name__ == "__main__":
    main()
