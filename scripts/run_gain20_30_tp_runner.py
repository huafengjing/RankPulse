from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
FEATURES = Path("outputs/robust_filter/signal_features.csv")
OUT = Path("outputs/gain20_30_tp_runner")

MARGIN_PER_TRADE = 100.0
ACCOUNT_CAPITAL = 1000.0
LEVERAGE = 10.0
FEE = 0.0005
SLIP = 0.0005
TP1 = 0.20
TP2 = 0.50
TP1_FRAC = 0.40
TP2_FRAC = 0.30
MA_WINDOW_4H = 14


def load_klines() -> pd.DataFrame:
    frames = []
    for path in DATA_DIR.glob("*/*.parquet"):
        frame = pd.read_parquet(path, columns=["symbol", "open_time", "open", "high", "low", "close"])
        frame["_mtime"] = path.stat().st_mtime
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("_mtime").drop_duplicates(["symbol", "open_time"], keep="last")
    df = df.drop(columns="_mtime").sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["open_time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def add_4h_ma14(klines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, group in klines.groupby("symbol", sort=False):
        g = group.sort_values("open_time").copy().set_index("open_time_utc")
        close_4h = g["close"].resample("4h", label="right", closed="right").last().dropna()
        ma = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
        g["ma14_4h_completed"] = ma.shift(1).reindex(g.index, method="ffill").values
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)


def exit_pnl(entry_price: float, exit_price_raw: float, fraction: float) -> tuple[float, float, float]:
    notional = MARGIN_PER_TRADE * LEVERAGE * fraction
    entry_eff = entry_price * (1 + SLIP)
    exit_eff = exit_price_raw * (1 - SLIP)
    gross = notional * (exit_eff / entry_eff - 1)
    fees = notional * FEE + notional * (exit_eff / entry_eff) * FEE
    return gross - fees, gross, fees


def run(signals: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    kmap = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in klines.groupby("symbol")}
    rows = []
    for _, sig in signals.sort_values("signal_time_utc").iterrows():
        g = kmap.get(sig["symbol"])
        if g is None:
            continue
        idx = g.index[g["open_time"] > int(sig["signal_time"])]
        if len(idx) == 0:
            continue
        entry_idx = int(idx[0])
        entry = g.loc[entry_idx]
        entry_time = int(entry["open_time"])
        if entry_time <= int(sig["signal_time"]):
            raise RuntimeError(
                f"Entry time is not after signal time for signal_id={sig['signal_id']} "
                f"symbol={sig['symbol']}: signal_time={sig['signal_time']} entry_time={entry_time}"
            )
        entry_price = float(entry["open"])
        tp1_price = entry_price * (1 + TP1)
        tp2_price = entry_price * (1 + TP2)
        liq_price = entry_price * 0.90
        path = g[g["open_time"] >= entry_time]

        remaining = 1.0
        tp1_hit = False
        tp2_hit = False
        runner_exit = False
        liquidated = False
        net = 0.0
        gross = 0.0
        fees = MARGIN_PER_TRADE * LEVERAGE * FEE
        exit_time = int(path.iloc[-1]["open_time"])
        exit_price = float(path.iloc[-1]["close"])
        exit_reason = "data_end"
        mfe = float(path["high"].max() / entry_price - 1)
        mae = float(path["low"].min() / entry_price - 1)

        for _, bar in path.iterrows():
            low = float(bar["low"])
            high = float(bar["high"])
            close = float(bar["close"])
            if remaining > 0 and low <= liq_price:
                pnl, gp, fee = exit_pnl(entry_price, liq_price, remaining)
                net += pnl; gross += gp; fees += fee
                exit_time = int(bar["open_time"])
                exit_price = liq_price
                exit_reason = "liquidation"
                liquidated = True
                remaining = 0.0
                break
            if not tp1_hit and high >= tp1_price:
                pnl, gp, fee = exit_pnl(entry_price, tp1_price, TP1_FRAC)
                net += pnl; gross += gp; fees += fee
                remaining -= TP1_FRAC
                tp1_hit = True
                exit_time = int(bar["open_time"])
            if tp1_hit and not tp2_hit and high >= tp2_price:
                pnl, gp, fee = exit_pnl(entry_price, tp2_price, TP2_FRAC)
                net += pnl; gross += gp; fees += fee
                remaining -= TP2_FRAC
                tp2_hit = True
                exit_time = int(bar["open_time"])
            ma14 = bar.get("ma14_4h_completed")
            if tp1_hit and remaining > 0 and pd.notna(ma14) and close < float(ma14):
                pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                net += pnl; gross += gp; fees += fee
                exit_time = int(bar["open_time"])
                exit_price = close
                exit_reason = "runner_ma14_4h_break"
                runner_exit = True
                remaining = 0.0
                break

        if remaining > 0:
            last = path.iloc[-1]
            pnl, gp, fee = exit_pnl(entry_price, float(last["close"]), remaining)
            net += pnl; gross += gp; fees += fee
            exit_time = int(last["open_time"])
            exit_price = float(last["close"])
            exit_reason = "data_end"

        rows.append(
            {
                "signal_id": int(sig["signal_id"]),
                "symbol": sig["symbol"],
                "signal_time_utc": sig["signal_time_utc"],
                "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
                "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "rank_at_signal": sig["rank_at_signal"],
                "rolling_24h_gain_at_signal": sig["rolling_24h_gain_at_signal"],
                "tp1_hit": tp1_hit,
                "tp2_hit": tp2_hit,
                "runner_exit": runner_exit,
                "liquidated": liquidated,
                "exit_reason": exit_reason,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "gross_pnl_usd": gross,
                "fees_usd": fees,
                "net_pnl_usd": net,
                "return_on_margin_pct": net / MARGIN_PER_TRADE,
                "return_on_account_pct": net / ACCOUNT_CAPITAL,
                "holding_hours": (exit_time - entry_time) / 3_600_000,
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(pnl: pd.Series) -> float:
    eq = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((eq / eq.cummax() - 1).min())


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame, period: str) -> dict:
        r = group["return_on_margin_pct"]
        wins = r[r > 0]
        losses = r[r <= 0]
        return {
            "period": period,
            "trades": len(group),
            "net_pnl_usd": group["net_pnl_usd"].sum(),
            "return_on_1000u_account_pct": group["net_pnl_usd"].sum() / ACCOUNT_CAPITAL,
            "win_rate": (r > 0).mean() if len(r) else 0.0,
            "Tp1": group["tp1_hit"].mean() if len(group) else 0.0,
            "Tp2": group["tp2_hit"].mean() if len(group) else 0.0,
            "liquidation_rate": group["liquidated"].mean() if len(group) else 0.0,
            "profit_factor": wins.sum() / abs(losses.sum()) if abs(losses.sum()) else np.inf,
            "max_drawdown_on_1000u_account_pct": max_drawdown(group["net_pnl_usd"]) if len(group) else 0.0,
            "avg_mfe_pct": group["mfe_pct"].mean() if len(group) else 0.0,
            "avg_mae_pct": group["mae_pct"].mean() if len(group) else 0.0,
            "best_trade_return_on_margin_pct": r.max() if len(r) else 0.0,
            "worst_trade_return_on_margin_pct": r.min() if len(r) else 0.0,
        }

    trades = trades.sort_values("entry_time_utc").copy()
    rows = [one(trades, "ALL")]
    trades["month"] = pd.to_datetime(trades["entry_time_utc"], utc=True).dt.to_period("M").astype(str)
    for month, group in trades.groupby("month"):
        rows.append(one(group.sort_values("entry_time_utc"), month))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES)
    features["signal_time"] = pd.to_datetime(features["signal_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    selected = features[(features["rolling_24h_gain_at_signal"] >= 0.20) & (features["rolling_24h_gain_at_signal"] < 0.30)].copy()
    print(f"Selected gain 20-30 signals: {len(selected)}", flush=True)
    klines = add_4h_ma14(load_klines())
    trades = run(selected, klines)
    summary = summarize(trades)
    trades.to_csv(OUT / "trades.csv", index=False)
    summary.to_csv(OUT / "summary_by_month.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT / 'trades.csv'}", flush=True)
    print(f"Wrote {OUT / 'summary_by_month.csv'}", flush=True)


if __name__ == "__main__":
    main()
