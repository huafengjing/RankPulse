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


REFERENCE_FEATURES = ROOT / "outputs/robust_filter/signal_features.csv"
OUT = ROOT / "outputs/top5_gain20_30_same_window"
EXCLUDE_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSDT"}

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
MAX_HOLD_HOURS = 240


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
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
        "signal_time_utc": sig["signal_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "rank_at_signal": int(sig["rank"]),
        "rolling_24h_gain_at_signal": float(sig["rolling_24h_change_pct"]),
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
        "return_on_account_pct": net / ACCOUNT_CAPITAL,
        "holding_hours": (exit_time - entry_time) / 3_600_000,
    }


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame, period: str) -> dict:
        r = group["return_on_margin_pct"]
        return {
            "period": period,
            "trades": len(group),
            "net_pnl_usd": group["net_pnl_usd"].sum(),
            "return_on_1000u_account_pct": group["net_pnl_usd"].sum() / ACCOUNT_CAPITAL,
            "win_rate": (r > 0).mean() if len(r) else 0.0,
            "Tp1": group["tp1_hit"].mean() if len(group) else 0.0,
            "plus50_stage2_rate": group["stage2_hit_plus50"].mean() if len(group) else 0.0,
            "liquidation_rate": group["liquidated"].mean() if len(group) else 0.0,
            "profit_factor": profit_factor(r),
            "max_drawdown_on_1000u_account_pct": max_drawdown(group.sort_values("entry_time_utc")["net_pnl_usd"]) if len(group) else 0.0,
            "avg_mfe_pct": group["mfe_pct"].mean() if len(group) else 0.0,
            "avg_mae_pct": group["mae_pct"].mean() if len(group) else 0.0,
            "best_trade_return_on_margin_pct": r.max() if len(r) else 0.0,
            "worst_trade_return_on_margin_pct": r.min() if len(r) else 0.0,
        }

    trades = trades.sort_values("entry_time_utc").copy()
    rows = [one(trades, "ALL")]
    trades["month"] = pd.to_datetime(trades["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    for month, group in trades.groupby("month"):
        rows.append(one(group, month))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REFERENCE_FEATURES)
    ref_times = pd.to_datetime(ref["signal_time_utc"], utc=True)
    start_ms = int(ref_times.min().timestamp() * 1000)
    end_ms = int(ref_times.max().timestamp() * 1000)
    load_start = start_ms - 25 * 60 * 60 * 1000
    load_end = end_ms + MAX_HOLD_HOURS * 60 * 60 * 1000

    print("Loading cached klines...", flush=True)
    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines[(klines["open_time"] >= load_start) & (klines["open_time"] <= load_end)].copy()
    klines = klines[~klines["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)

    print("Building rankings and Top5 signals...", flush=True)
    rank_input = klines[(klines["open_time"] >= load_start) & (klines["open_time"] <= end_ms)].copy()
    rankings = build_rankings(rank_input, 5)
    all_top5 = identify_first_top_signals(rankings, top_n=5, cooldown_days=3, observation_hours=72)
    all_top5 = all_top5[(all_top5["signal_time"] >= start_ms) & (all_top5["signal_time"] <= end_ms)].copy()
    selected = all_top5[
        (all_top5["rolling_24h_change_pct"] >= 0.20)
        & (all_top5["rolling_24h_change_pct"] < 0.30)
    ].sort_values("signal_time").reset_index(drop=True)
    selected.to_csv(OUT / "signals_top5_gain20_30.csv", index=False)

    selected["month"] = pd.to_datetime(selected["signal_time_utc"], utc=True).dt.strftime("%Y-%m")
    selected.groupby("month").size().reset_index(name="signal_count").to_csv(OUT / "monthly_signal_counts.csv", index=False)

    print(f"Selected Top5 gain20-30 signals={len(selected)}", flush=True)
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
        row = run_trade(sig, group)
        if row is not None:
            rows.append(row)
    trades = pd.DataFrame(rows)
    summary = summarize(trades)
    trades.to_csv(OUT / "trades.csv", index=False)
    summary.to_csv(OUT / "summary_by_month.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
