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
TRADE_MARGIN_USD = 100.0
ACCOUNT_CAPITAL_USD = 1000.0
LEVERAGE = 10.0
NOTIONAL_USD = TRADE_MARGIN_USD * LEVERAGE
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005
TIMEOUT_HOURS = 12
MAX_HOLDING_HOURS = 120


def short_leg_pnl(entry_price: float, exit_price: float, notional_weight: float) -> tuple[float, float, float]:
    notional = NOTIONAL_USD * notional_weight
    entry_effective = entry_price * (1.0 - SLIPPAGE_RATE)
    exit_effective = exit_price * (1.0 + SLIPPAGE_RATE)
    gross = notional * (entry_effective - exit_effective) / entry_effective
    fee = notional * FEE_RATE * 2.0
    return gross - fee, gross, fee


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL_USD + pnl.cumsum()
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(wins / abs(losses)) if abs(losses) > 0 else np.inf


def load_combo3_signals() -> pd.DataFrame:
    features = pd.read_csv(OUT / "signal_features.csv")
    base = pd.read_csv(OUT / "base_signals.csv")
    data = features.merge(base[["signal_id", "entry_time_utc", "entry_price"]], on="signal_id", how="inner")
    return data[
        data["volume_1h_vs_24h_avg"].between(1.2, 5)
        & (data["signal_candle_lower_wick_pct"] <= 0.40)
    ].sort_values("signal_time_utc").reset_index(drop=True)


def simulate_trade(signal: pd.Series, group: pd.DataFrame) -> dict[str, object] | None:
    entry_time = int(pd.Timestamp(signal["entry_time_utc"]).timestamp() * 1000)
    entry_idx = group.index[group["open_time"] == entry_time]
    if len(entry_idx) == 0:
        future_idx = group.index[group["open_time"] >= entry_time]
        if len(future_idx) == 0:
            return None
        entry_idx = [int(future_idx[0])]
        entry_time = int(group.loc[entry_idx[0], "open_time"])
    entry_idx = int(entry_idx[0])
    entry_price = float(group.loc[entry_idx, "open"])
    tp1_price = entry_price * 0.90
    tp2_price = entry_price * 0.85
    liquidation_price = entry_price * 1.10
    timeout_time = entry_time + TIMEOUT_HOURS * 60 * 60 * 1000
    end_time = entry_time + MAX_HOLDING_HOURS * 60 * 60 * 1000
    path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= end_time)]
    if path.empty:
        return None

    pnl = 0.0
    gross = 0.0
    fee = 0.0
    remaining = 1.0
    tp1_hit = False
    tp2_hit = False
    liquidation_hit = False
    timeout_exit = False
    exit_reason = "max_holding"
    exit_time = int(path.iloc[-1]["open_time"])
    exit_price = float(path.iloc[-1]["close"])
    tp1_time = pd.NA
    tp2_time = pd.NA

    for _, bar in path.iterrows():
        now = int(bar["open_time"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # Conservative same-candle priority for shorts.
        if high >= liquidation_price:
            leg_pnl, leg_gross, leg_fee = short_leg_pnl(entry_price, liquidation_price, remaining)
            pnl += leg_pnl
            gross += leg_gross
            fee += leg_fee
            liquidation_hit = True
            exit_reason = "liquidation_plus10"
            exit_time = now
            exit_price = liquidation_price
            remaining = 0.0
            break

        if not tp1_hit and low <= tp1_price:
            leg_pnl, leg_gross, leg_fee = short_leg_pnl(entry_price, tp1_price, 0.80)
            pnl += leg_pnl
            gross += leg_gross
            fee += leg_fee
            remaining = 0.20
            tp1_hit = True
            tp1_time = int((now - entry_time) / 60_000)

        if tp1_hit and low <= tp2_price:
            leg_pnl, leg_gross, leg_fee = short_leg_pnl(entry_price, tp2_price, remaining)
            pnl += leg_pnl
            gross += leg_gross
            fee += leg_fee
            tp2_hit = True
            tp2_time = int((now - entry_time) / 60_000)
            exit_reason = "tp2"
            exit_time = now
            exit_price = tp2_price
            remaining = 0.0
            break

        if not tp1_hit and now >= timeout_time:
            leg_pnl, leg_gross, leg_fee = short_leg_pnl(entry_price, close, remaining)
            pnl += leg_pnl
            gross += leg_gross
            fee += leg_fee
            timeout_exit = True
            exit_reason = "timeout_12h_before_tp1"
            exit_time = now
            exit_price = close
            remaining = 0.0
            break

    if remaining > 0:
        leg_pnl, leg_gross, leg_fee = short_leg_pnl(entry_price, exit_price, remaining)
        pnl += leg_pnl
        gross += leg_gross
        fee += leg_fee

    return {
        "signal_id": int(signal["signal_id"]),
        "symbol": signal["symbol"],
        "signal_time_utc": signal["signal_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "liquidation_hit": liquidation_hit,
        "timeout_exit": timeout_exit,
        "time_to_tp1_minutes": tp1_time,
        "time_to_tp2_minutes": tp2_time,
        "gross_pnl_usd": gross,
        "fee_usd": fee,
        "net_pnl_usd": pnl,
        "return_on_margin": pnl / TRADE_MARGIN_USD,
        "return_on_account": pnl / ACCOUNT_CAPITAL_USD,
        "holding_minutes": int((exit_time - entry_time) / 60_000),
        "volume_1h_vs_24h_avg": float(signal["volume_1h_vs_24h_avg"]),
        "signal_candle_lower_wick_pct": float(signal["signal_candle_lower_wick_pct"]),
    }


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    pnl = trades["net_pnl_usd"]
    return pd.DataFrame(
        [
            {
                "group": "ALL",
                "trade_count": len(trades),
                "win_rate": float((pnl > 0).mean()),
                "tp1_hit_rate": float(trades["tp1_hit"].mean()),
                "tp2_hit_rate": float(trades["tp2_hit"].mean()),
                "liquidation_rate": float(trades["liquidation_hit"].mean()),
                "timeout_rate": float(trades["timeout_exit"].mean()),
                "total_net_pnl_usd": float(pnl.sum()),
                "total_return_on_1000u_account": float(pnl.sum() / ACCOUNT_CAPITAL_USD),
                "avg_trade_pnl_usd": float(pnl.mean()),
                "median_trade_pnl_usd": float(pnl.median()),
                "profit_factor": profit_factor(pnl),
                "max_drawdown_on_1000u_account": max_drawdown(pnl),
                "best_trade_usd": float(pnl.max()),
                "worst_trade_usd": float(pnl.min()),
            }
        ]
    )


def summarize_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["month"] = pd.to_datetime(frame["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    for month, group in frame.groupby("month", sort=True):
        pnl = group["net_pnl_usd"]
        rows.append(
            {
                "month": month,
                "trade_count": len(group),
                "win_rate": float((pnl > 0).mean()),
                "tp1_hit_rate": float(group["tp1_hit"].mean()),
                "tp2_hit_rate": float(group["tp2_hit"].mean()),
                "liquidation_rate": float(group["liquidation_hit"].mean()),
                "timeout_rate": float(group["timeout_exit"].mean()),
                "net_pnl_usd": float(pnl.sum()),
                "return_on_1000u_account": float(pnl.sum() / ACCOUNT_CAPITAL_USD),
                "avg_trade_pnl_usd": float(pnl.mean()),
                "profit_factor": profit_factor(pnl),
                "max_drawdown_on_1000u_account": max_drawdown(pnl),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    signals = load_combo3_signals()
    klines = load_cached_klines(ROOT / "data", "5m")
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    needed = set(signals["symbol"])
    klines = klines[klines["symbol"].isin(needed)].copy()
    kmap = {
        symbol: group.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        for symbol, group in klines.groupby("symbol", sort=False)
    }
    rows = []
    for _, signal in signals.iterrows():
        group = kmap.get(signal["symbol"])
        if group is None:
            continue
        result = simulate_trade(signal, group)
        if result is not None:
            rows.append(result)
    trades = pd.DataFrame(rows)
    trades.to_csv(OUT / "combo3_tp10_tp15_timeout12_trades.csv", index=False)
    summary = summarize(trades)
    monthly = summarize_by_month(trades)
    summary.to_csv(OUT / "combo3_tp10_tp15_timeout12_summary.csv", index=False)
    monthly.to_csv(OUT / "combo3_tp10_tp15_timeout12_summary_by_month.csv", index=False)
    print(summary.to_string(index=False))
    print(monthly.to_string(index=False))
    print(OUT / "combo3_tp10_tp15_timeout12_summary_by_month.csv")


if __name__ == "__main__":
    main()
