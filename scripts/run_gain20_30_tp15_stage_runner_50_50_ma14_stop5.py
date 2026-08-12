from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
FEATURES = Path("outputs/robust_filter/signal_features.csv")
OUT = Path("outputs/gain20_30_tp15_stage_runner_50_50_ma14_stop5")

MARGIN_PER_TRADE = 100.0
ACCOUNT_CAPITAL = 1000.0
LEVERAGE = 10.0
FEE = 0.0005
SLIP = 0.0005
LIQ_PCT = -0.05
TP1 = 0.15
TP1_FRAC = 0.50
RUNNER_FRAC = 0.50
STAGE2_TRIGGER = 0.50
TRAIL_STAGE1 = 0.20
TRAIL_STAGE2 = 0.30
MA_WINDOW_4H = 14


def load_klines(symbols: set[str]) -> dict[str, pd.DataFrame]:
    frames = []
    for path in DATA_DIR.glob("*/*.parquet"):
        frame = pd.read_parquet(path, columns=["symbol", "open_time", "open", "high", "low", "close"])
        if frame.empty:
            continue
        frame = frame[frame["symbol"].isin(symbols)]
        if frame.empty:
            continue
        frame["_mtime"] = path.stat().st_mtime
        frames.append(frame)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("_mtime").drop_duplicates(["symbol", "open_time"], keep="last")
    df = df.drop(columns="_mtime").sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["open_time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    rows = []
    for _, group in df.groupby("symbol", sort=False):
        g = group.sort_values("open_time").copy().set_index("open_time_utc")
        close_4h = g["close"].resample("4h", label="right", closed="right").last().dropna()
        ma21 = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
        g["ma21_4h_completed"] = ma21.shift(1).reindex(g.index, method="ffill").values
        reset = g.reset_index()
        # A 5m candle whose close timestamp lands on a 4h boundary is treated as the 4H close.
        reset["is_4h_close_bar"] = ((reset["open_time"] + 5 * 60 * 1000) % (4 * 60 * 60 * 1000)) == 0
        rows.append(reset)
    df = pd.concat(rows, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol", sort=False)}


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
    if entry_time <= int(sig["signal_time"]):
        raise RuntimeError(f"Entry time <= signal time for signal_id={sig['signal_id']}")

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
    runner_exit_price = np.nan
    runner_exit_return = np.nan
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
                runner_exit_price = liq_price
                runner_exit_return = LIQ_PCT
                exit_reason = "stop5_before_tp1"
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
                exit_time = bar_time
                exit_price = tp1_price
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
                runner_exit_price = trail_stop
                runner_exit_return = trail_stop / entry_price - 1
                exit_reason = f"stage{2 if stage2_hit else 1}_trailing_{int(trail_pct * 100)}pct"
                runner_exit = True
                remaining = 0.0
                break

            ma21 = bar.get("ma21_4h_completed")
            if pd.notna(ma21):
                if not stage2_hit and close < float(ma21):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    exit_time = bar_time
                    exit_price = close
                    runner_exit_price = close
                    runner_exit_return = close / entry_price - 1
                    exit_reason = "stage1_5m_close_below_4h_ma21"
                    runner_exit = True
                    remaining = 0.0
                    break
                if stage2_hit and bool(bar["is_4h_close_bar"]) and close < float(ma21):
                    pnl, gp, fee = exit_pnl(entry_price, close, remaining)
                    net += pnl
                    gross += gp
                    fees += fee
                    exit_time = bar_time
                    exit_price = close
                    runner_exit_price = close
                    runner_exit_return = close / entry_price - 1
                    exit_reason = "stage2_4h_close_below_ma21"
                    runner_exit = True
                    remaining = 0.0
                    break

    if remaining > 0:
        last = path.iloc[-1]
        last_close = float(last["close"])
        pnl, gp, fee = exit_pnl(entry_price, last_close, remaining)
        net += pnl
        gross += gp
        fees += fee
        exit_time = int(last["open_time"])
        exit_price = last_close
        runner_exit_price = last_close
        runner_exit_return = last_close / entry_price - 1
        exit_reason = "latest_price"

    return {
        "signal_id": int(sig["signal_id"]),
        "symbol": sig["symbol"],
        "signal_time_utc": sig["signal_time_utc"],
        "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
        "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "runner_exit_price": runner_exit_price,
        "rank_at_signal": sig["rank_at_signal"],
        "rolling_24h_gain_at_signal": sig["rolling_24h_gain_at_signal"],
        "tp1_hit": tp1_hit,
        "stage2_hit_plus50": stage2_hit,
        "runner_exit": runner_exit,
        "liquidated": liquidated,
        "exit_reason": exit_reason,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "runner_exit_return_pct": runner_exit_return,
        "gross_pnl_usd": gross,
        "fees_usd": fees,
        "net_pnl_usd": net,
        "return_on_margin_pct": net / MARGIN_PER_TRADE,
        "return_on_account_pct": net / ACCOUNT_CAPITAL,
        "holding_hours": (exit_time - entry_time) / 3_600_000,
    }


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
            "plus50_stage2_rate": group["stage2_hit_plus50"].mean() if len(group) else 0.0,
            "liquidation_rate": group["liquidated"].mean() if len(group) else 0.0,
            "runner_exit_rate": group["runner_exit"].mean() if len(group) else 0.0,
            "profit_factor": wins.sum() / abs(losses.sum()) if abs(losses.sum()) else np.inf,
            "max_drawdown_on_1000u_account_pct": max_drawdown(group["net_pnl_usd"]) if len(group) else 0.0,
            "avg_mfe_pct": group["mfe_pct"].mean() if len(group) else 0.0,
            "avg_mae_pct": group["mae_pct"].mean() if len(group) else 0.0,
            "avg_runner_exit_return_pct": group["runner_exit_return_pct"].mean() if len(group) else 0.0,
            "best_trade_return_on_margin_pct": r.max() if len(r) else 0.0,
            "worst_trade_return_on_margin_pct": r.min() if len(r) else 0.0,
        }

    trades = trades.sort_values("entry_time_utc").copy()
    rows = [one(trades, "ALL")]
    trades["month"] = pd.to_datetime(trades["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    for month, group in trades.groupby("month"):
        rows.append(one(group.sort_values("entry_time_utc"), month))
    return pd.DataFrame(rows)


def write_chinese_trades(trades: pd.DataFrame) -> None:
    out = pd.DataFrame(
        {
            "信号ID": trades["signal_id"],
            "交易对": trades["symbol"],
            "信号时间UTC": trades["signal_time_utc"],
            "入场时间UTC": trades["entry_time_utc"],
            "出场时间UTC": trades["exit_time_utc"],
            "入场价格": trades["entry_price"],
            "出场价格": trades["exit_price"],
            "Runner出场价格": trades["runner_exit_price"],
            "信号时排名": trades["rank_at_signal"],
            "信号时24H滚动涨幅": trades["rolling_24h_gain_at_signal"].map(lambda x: f"{round(x * 100):.0f}%"),
            "是否触发Tp1": trades["tp1_hit"],
            "是否到达+50%": trades["stage2_hit_plus50"],
            "是否Runner出场": trades["runner_exit"],
            "是否TP1前-5%止损": trades["liquidated"],
            "出场原因": trades["exit_reason"],
            "最大浮盈比例": trades["mfe_pct"].map(lambda x: f"{round(x * 100):.0f}%"),
            "最大浮亏比例": trades["mae_pct"].map(lambda x: f"{round(x * 100):.0f}%"),
            "Runner出场涨幅": trades["runner_exit_return_pct"].map(lambda x: "" if pd.isna(x) else f"{round(x * 100):.0f}%"),
            "毛收益U": trades["gross_pnl_usd"].round(0).astype(int),
            "手续费U": trades["fees_usd"],
            "净收益U": trades["net_pnl_usd"],
            "保证金收益率": trades["return_on_margin_pct"].map(lambda x: f"{round(x * 100):.0f}%"),
            "账户收益率": trades["return_on_account_pct"].map(lambda x: f"{round(x * 100):.0f}%"),
            "持仓小时": trades["holding_hours"].round(0).astype(int),
        }
    )
    out.to_csv(OUT / "trades_中文表头.csv", index=False, encoding="utf-8-sig")


def write_chinese_summary(summary: pd.DataFrame) -> None:
    out = pd.DataFrame(
        {
            "周期": summary["period"],
            "交易数": summary["trades"],
            "净收益U": summary["net_pnl_usd"].round(2),
            "1000U账户收益率": summary["return_on_1000u_account_pct"].map(lambda x: f"{x * 100:.2f}%"),
            "胜率": summary["win_rate"].map(lambda x: f"{x * 100:.2f}%"),
            "Tp1": summary["Tp1"].map(lambda x: f"{x * 100:.2f}%"),
            "到达+50%": summary["plus50_stage2_rate"].map(lambda x: f"{x * 100:.2f}%"),
            "TP1前-5%止损率": summary["liquidation_rate"].map(lambda x: f"{x * 100:.2f}%"),
            "Runner出场率": summary["runner_exit_rate"].map(lambda x: f"{x * 100:.2f}%"),
            "Profit Factor": summary["profit_factor"].round(4),
            "最大回撤": summary["max_drawdown_on_1000u_account_pct"].map(lambda x: f"{x * 100:.2f}%"),
            "平均最大浮盈": summary["avg_mfe_pct"].map(lambda x: f"{x * 100:.2f}%"),
            "平均最大浮亏": summary["avg_mae_pct"].map(lambda x: f"{x * 100:.2f}%"),
            "平均Runner出场涨幅": summary["avg_runner_exit_return_pct"].map(lambda x: f"{x * 100:.2f}%"),
            "最佳单笔保证金收益率": summary["best_trade_return_on_margin_pct"].map(lambda x: f"{x * 100:.2f}%"),
            "最差单笔保证金收益率": summary["worst_trade_return_on_margin_pct"].map(lambda x: f"{x * 100:.2f}%"),
        }
    )
    out.to_csv(OUT / "summary_by_month_中文表头.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES)
    features["signal_time"] = pd.to_datetime(features["signal_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    selected = features[
        (features["rolling_24h_gain_at_signal"] >= 0.20)
        & (features["rolling_24h_gain_at_signal"] < 0.30)
    ].copy()
    klines = load_klines(set(selected["symbol"]))
    rows = []
    for _, sig in selected.sort_values("signal_time_utc").iterrows():
        group = klines.get(sig["symbol"])
        if group is None:
            continue
        row = run_trade(sig, group)
        if row is not None:
            rows.append(row)
    trades = pd.DataFrame(rows)
    summary = summarize(trades)
    trades.to_csv(OUT / "trades.csv", index=False)
    summary.to_csv(OUT / "summary_by_month.csv", index=False)
    write_chinese_trades(trades)
    write_chinese_summary(summary)
    print(summary.to_string(index=False))
    print(f"Wrote {OUT / 'trades.csv'}")
    print(f"Wrote {OUT / 'summary_by_month.csv'}")
    print(f"Wrote {OUT / 'trades_中文表头.csv'}")
    print(f"Wrote {OUT / 'summary_by_month_中文表头.csv'}")


if __name__ == "__main__":
    main()
