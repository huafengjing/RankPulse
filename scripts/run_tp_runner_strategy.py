from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
OUTPUT_DIR = Path("outputs/custom_tp_runner")
SIGNALS_PATH = Path("outputs/signals.csv")

MARGIN_PER_TRADE = 100.0
ACCOUNT_CAPITAL = 1000.0
LEVERAGE = 10.0
TAKER_FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005
TP1 = 0.10
TP2 = 0.30
TP1_FRACTION = 0.40
TP2_FRACTION = 0.20
RUNNER_FRACTION = 0.40
LIQUIDATION_MOVE = -0.10
MA_WINDOW_4H = 14


def load_klines() -> pd.DataFrame:
    frames = []
    for path in DATA_DIR.glob("*/*.parquet"):
        frame = pd.read_parquet(
            path,
            columns=["symbol", "open_time", "open", "high", "low", "close", "quote_volume"],
        )
        frame["_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        raise RuntimeError("No kline parquet files found.")
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["_mtime"]).drop_duplicates(["symbol", "open_time"], keep="last")
    data = data.drop(columns=["_mtime"]).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    data["open_time_utc"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    return data


def add_4h_ma14(klines: pd.DataFrame) -> pd.DataFrame:
    out = []
    for symbol, group in klines.groupby("symbol", sort=False):
        g = group.sort_values("open_time").copy()
        g = g.set_index("open_time_utc")
        four_h = g["close"].resample("4h", label="right", closed="right").last().dropna()
        ma = four_h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
        # Shift one 4h bar so a 5m candle only sees the latest completed 4h MA.
        ma_for_5m = ma.shift(1).reindex(g.index, method="ffill")
        g["ma14_4h_completed"] = ma_for_5m.values
        g = g.reset_index()
        out.append(g)
    return pd.concat(out, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)


def exit_pnl(entry_price: float, exit_price_raw: float, fraction: float, reason: str) -> tuple[float, float, float]:
    notional = MARGIN_PER_TRADE * LEVERAGE * fraction
    exit_effective = exit_price_raw * (1.0 - SLIPPAGE_RATE)
    entry_effective = entry_price * (1.0 + SLIPPAGE_RATE)
    price_return = exit_effective / entry_effective - 1.0
    gross_pnl = notional * price_return
    entry_fee = notional * TAKER_FEE_RATE
    exit_fee = notional * (exit_effective / entry_effective) * TAKER_FEE_RATE
    net_pnl = gross_pnl - entry_fee - exit_fee
    return net_pnl, gross_pnl, entry_fee + exit_fee


def run_strategy(signals: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    kline_map = {symbol: group.sort_values("open_time").reset_index(drop=True) for symbol, group in klines.groupby("symbol")}

    for _, signal in signals.sort_values("signal_time").iterrows():
        symbol = signal["symbol"]
        group = kline_map.get(symbol)
        if group is None:
            continue
        future_idx = group.index[group["open_time"] > int(signal["signal_time"])]
        if len(future_idx) == 0:
            continue
        entry_idx = int(future_idx[0])
        entry = group.loc[entry_idx]
        entry_time = int(entry["open_time"])
        entry_price = float(entry["open"])
        tp1_price = entry_price * (1.0 + TP1)
        tp2_price = entry_price * (1.0 + TP2)
        liq_price = entry_price * (1.0 + LIQUIDATION_MOVE)
        path = group[group["open_time"] >= entry_time]
        if path.empty:
            continue

        remaining = 1.0
        tp1_hit = False
        tp2_hit = False
        runner_exit = False
        liquidated = False
        net_pnl = 0.0
        gross_pnl = 0.0
        fees = MARGIN_PER_TRADE * LEVERAGE * TAKER_FEE_RATE
        exit_time = int(path.iloc[-1]["open_time"])
        exit_reason = "max_observation"
        mfe = float(path["high"].max() / entry_price - 1.0)
        mae = float(path["low"].min() / entry_price - 1.0)

        for _, bar in path.iterrows():
            bar_time = int(bar["open_time"])
            low = float(bar["low"])
            high = float(bar["high"])
            close = float(bar["close"])
            ma14 = bar.get("ma14_4h_completed")

            # Conservative priority: liquidation before profit targets inside the same 5m candle.
            if remaining > 0 and low <= liq_price:
                pnl, gross, fee = exit_pnl(entry_price, liq_price, remaining, "liquidation")
                net_pnl += pnl
                gross_pnl += gross
                fees += fee
                exit_time = bar_time
                exit_reason = "liquidation"
                liquidated = True
                remaining = 0.0
                break

            if not tp1_hit and high >= tp1_price:
                pnl, gross, fee = exit_pnl(entry_price, tp1_price, TP1_FRACTION, "tp1")
                net_pnl += pnl
                gross_pnl += gross
                fees += fee
                remaining -= TP1_FRACTION
                tp1_hit = True
                exit_time = bar_time

            if tp1_hit and not tp2_hit and high >= tp2_price:
                pnl, gross, fee = exit_pnl(entry_price, tp2_price, TP2_FRACTION, "tp2")
                net_pnl += pnl
                gross_pnl += gross
                fees += fee
                remaining -= TP2_FRACTION
                tp2_hit = True
                exit_time = bar_time

            if tp1_hit and remaining > 0 and pd.notna(ma14) and close < float(ma14):
                pnl, gross, fee = exit_pnl(entry_price, close, remaining, "runner_ma14_4h_break")
                net_pnl += pnl
                gross_pnl += gross
                fees += fee
                exit_time = bar_time
                exit_reason = "runner_ma14_4h_break"
                runner_exit = True
                remaining = 0.0
                break

        if remaining > 0:
            last = path.iloc[-1]
            pnl, gross, fee = exit_pnl(entry_price, float(last["close"]), remaining, "data_end")
            net_pnl += pnl
            gross_pnl += gross
            fees += fee
            exit_time = int(last["open_time"])
            exit_reason = "data_end"

        rows.append(
            {
                "signal_id": int(signal["signal_id"]),
                "symbol": symbol,
                "signal_time_utc": signal["signal_time_utc"],
                "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
                "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
                "entry_price": entry_price,
                "rank": int(signal["rank"]),
                "rolling_24h_change_pct": float(signal["rolling_24h_change_pct"]),
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
                "runner_exit": runner_exit,
                "liquidated": liquidated,
                "exit_reason": exit_reason,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "gross_pnl_usd": gross_pnl,
                "fees_usd": fees,
                "net_pnl_usd": net_pnl,
                "return_on_margin_pct": net_pnl / MARGIN_PER_TRADE,
                "return_on_account_pct": net_pnl / ACCOUNT_CAPITAL,
                "holding_hours": (exit_time - entry_time) / 3_600_000,
            }
        )
    return pd.DataFrame(rows)


def max_drawdown_from_pnl(pnl: pd.Series, starting_capital: float) -> float:
    equity = starting_capital + pnl.cumsum()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame, label: str) -> dict:
        returns = group["return_on_margin_pct"]
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        return {
            "period": label,
            "trades": len(group),
            "net_pnl_usd": group["net_pnl_usd"].sum(),
            "return_on_1000u_account_pct": group["net_pnl_usd"].sum() / ACCOUNT_CAPITAL,
            "win_rate": (returns > 0).mean() if len(returns) else 0.0,
            "Tp1": group["tp1_hit"].mean() if len(group) else 0.0,
            "Tp2": group["tp2_hit"].mean() if len(group) else 0.0,
            "liquidation_rate": group["liquidated"].mean() if len(group) else 0.0,
            "profit_factor": wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf,
            "best_trade_return_on_margin_pct": returns.max() if len(returns) else 0.0,
            "worst_trade_return_on_margin_pct": returns.min() if len(returns) else 0.0,
            "max_drawdown_on_1000u_account_pct": max_drawdown_from_pnl(group["net_pnl_usd"], ACCOUNT_CAPITAL),
            "avg_mfe_pct": group["mfe_pct"].mean() if len(group) else 0.0,
            "avg_mae_pct": group["mae_pct"].mean() if len(group) else 0.0,
        }

    trades = trades.sort_values("entry_time_utc").copy()
    rows = [one(trades, "ALL")]
    trades["month"] = pd.to_datetime(trades["entry_time_utc"], utc=True).dt.to_period("M").astype(str)
    for month, group in trades.groupby("month"):
        rows.append(one(group.sort_values("entry_time_utc"), month))
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(SIGNALS_PATH)
    print(f"Loading klines for {len(signals)} signals...", flush=True)
    klines = add_4h_ma14(load_klines())
    print(f"Loaded klines rows={len(klines):,}. Running strategy...", flush=True)
    trades = run_strategy(signals, klines)
    summary = summarize(trades)
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_by_month.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUTPUT_DIR / 'trades.csv'}", flush=True)
    print(f"Wrote {OUTPUT_DIR / 'summary_by_month.csv'}", flush=True)


if __name__ == "__main__":
    main()
