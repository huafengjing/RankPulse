from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_futures_top2_fixed_time import BUY_NOTIONAL_U, FEE_RATE, calculate_drawdown, ms_to_bj_string, ms_to_utc


OUT = ROOT / "output"
KLINE_1H_DIR = ROOT / "data" / "futures_klines_1h"
SOURCE_PATH = OUT / "trades_with_core_trend_volume_factors.csv"

DETAIL_PATH = OUT / "exit_strategy_trade_details.csv"
SUMMARY_PATH = OUT / "exit_strategy_summary.csv"
MONTHLY_PATH = OUT / "exit_strategy_monthly.csv"
RANK_PATH = OUT / "exit_strategy_rank_breakdown.csv"
FACTOR_PATH = OUT / "exit_strategy_factor_breakdown.csv"
MFE_MAE_PATH = OUT / "exit_strategy_mfe_mae.csv"
TRAIN_VALIDATION_PATH = OUT / "exit_strategy_train_validation.csv"

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
FOUR_HOUR_MS = 4 * HOUR_MS


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str
    mode: str = "conservative"


STRATEGIES = [
    Strategy("fixed_3d", "fixed"),
    Strategy("fixed_7d", "fixed"),
    Strategy("fixed_14d", "fixed"),
    Strategy("fixed_30d", "fixed"),
    Strategy("strategy_A_72h_activation_14d", "A"),
    Strategy("strategy_B_sl10_72h_activation_14d", "B"),
    Strategy("strategy_C_tp15_runner_ma7", "C"),
    Strategy("strategy_D_tp15_runner_ma21", "D"),
    Strategy("strategy_E_tp15_runner_trailing25", "E"),
    Strategy("strategy_F_sl10_trailing25_weak_ma7", "F"),
    Strategy("strategy_B_sl10_72h_activation_14d_optimistic", "B", "optimistic"),
    Strategy("strategy_C_tp15_runner_ma7_optimistic", "C", "optimistic"),
    Strategy("strategy_D_tp15_runner_ma21_optimistic", "D", "optimistic"),
    Strategy("strategy_E_tp15_runner_trailing25_optimistic", "E", "optimistic"),
    Strategy("strategy_F_sl10_trailing25_weak_ma7_optimistic", "F", "optimistic"),
]


def load_signals() -> pd.DataFrame:
    trades = pd.read_csv(SOURCE_PATH)
    if trades["gain_24h"].abs().median() > 2:
        trades["gain_24h_decimal"] = trades["gain_24h"] / 100.0
    else:
        trades["gain_24h_decimal"] = trades["gain_24h"]
    trades = trades[
        trades["rank"].isin([2, 3])
        & trades["gain_24h_decimal"].le(0.80)
        & trades["entry_price"].notna()
        & trades["entry_time_utc"].notna()
    ].copy()
    if "entry_time_ms" not in trades.columns:
        trades["entry_time_ms"] = pd.to_datetime(trades["entry_time_utc"], utc=True).map(lambda value: int(value.timestamp() * 1000))
    trades["entry_time_ms"] = pd.to_numeric(trades["entry_time_ms"], errors="coerce").astype("int64")
    return trades.sort_values(["entry_time_ms", "rank", "symbol"]).reset_index(drop=True)


def load_1h(symbol: str) -> pd.DataFrame:
    path = KLINE_1H_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for col in ["open_time", "close_time", "trade_count"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64").astype("int64")
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.drop_duplicates("open_time").sort_values("open_time").set_index("open_time", drop=False)


def aggregate_4h(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.reset_index(drop=True).copy()
    work["bar_open_time"] = (work["open_time"] // FOUR_HOUR_MS) * FOUR_HOUR_MS
    out = (
        work.sort_values("open_time")
        .groupby("bar_open_time", sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .reset_index()
        .rename(columns={"bar_open_time": "open_time"})
    )
    out["next_open_time"] = out["open_time"] + FOUR_HOUR_MS
    return out.set_index("open_time", drop=False).sort_index()


def load_maps(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    h1: dict[str, pd.DataFrame] = {}
    h4: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = load_1h(symbol)
        if frame.empty:
            continue
        h1[symbol] = frame
        h4[symbol] = aggregate_4h(frame)
    return h1, h4


def get_open_at_or_latest(frame: pd.DataFrame, open_time: int, entry_time: int) -> tuple[int, float, str]:
    if open_time in frame.index:
        row = frame.loc[open_time]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return int(open_time), float(row["open"]), ""
    available = frame[frame["open_time"] >= entry_time]
    if available.empty:
        return entry_time, np.nan, "data_missing"
    row = available.iloc[-1]
    return int(row["open_time"]), float(row["open"]), "data_latest_available"


def calc_pnl(entry_price: float, exits: list[tuple[float, float]]) -> tuple[float, float]:
    qty_total = BUY_NOTIONAL_U * (1.0 - FEE_RATE) / entry_price
    proceeds = 0.0
    for ratio, exit_price in exits:
        proceeds += qty_total * ratio * exit_price * (1.0 - FEE_RATE)
    pnl = proceeds - BUY_NOTIONAL_U
    return pnl, pnl / BUY_NOTIONAL_U * 100.0


def path_slice(frame: pd.DataFrame, entry_time: int, max_exit_time: int) -> pd.DataFrame:
    return frame[(frame["open_time"] >= entry_time) & (frame["open_time"] <= max_exit_time)].copy()


def mfe_mae(path: pd.DataFrame, entry_price: float) -> tuple[float, float, float, float]:
    if path.empty:
        return np.nan, np.nan, np.nan, np.nan
    max_price = float(path["high"].max())
    min_price = float(path["low"].min())
    return (max_price / entry_price - 1.0) * 100.0, (min_price / entry_price - 1.0) * 100.0, max_price, min_price


def fixed_exit(signal: pd.Series, frame: pd.DataFrame, days: int) -> dict[str, Any]:
    entry_time = int(signal["entry_time_ms"])
    entry_price = float(signal["entry_price"])
    target = entry_time + days * DAY_MS
    exit_time, exit_price, fallback = get_open_at_or_latest(frame, target, entry_time)
    path = path_slice(frame, entry_time, exit_time)
    pnl, net_return = calc_pnl(entry_price, [(1.0, exit_price)])
    mfe, mae, max_price, min_price = mfe_mae(path, entry_price)
    return {
        "exit_time_ms": exit_time,
        "exit_price": exit_price,
        "exit_reason": fallback or f"fixed_{days}d",
        "tp1_pnl_u": 0.0,
        "runner_pnl_u": pnl,
        "pnl_u": pnl,
        "net_return_pct": net_return,
        "gross_return_pct": (exit_price / entry_price - 1.0) * 100.0,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "max_price_during_trade": max_price,
        "min_price_during_trade": min_price,
        "hit_tp1": False,
        "hit_stop_loss": False,
        "runner_exit_reason": "",
    }


def first_hit(
    path: pd.DataFrame,
    entry_price: float,
    stop_pct: float | None,
    tp_pct: float | None,
    mode: str,
) -> tuple[str, int, float] | None:
    stop_price = entry_price * (1.0 + stop_pct) if stop_pct is not None else None
    tp_price = entry_price * (1.0 + tp_pct) if tp_pct is not None else None
    for _, bar in path.iterrows():
        hit_stop = stop_price is not None and float(bar["low"]) <= stop_price
        hit_tp = tp_price is not None and float(bar["high"]) >= tp_price
        if hit_stop and hit_tp:
            if mode == "optimistic":
                return "tp", int(bar["open_time"]), float(tp_price)
            return "stop", int(bar["open_time"]), float(stop_price)
        if hit_stop:
            return "stop", int(bar["open_time"]), float(stop_price)
        if hit_tp:
            return "tp", int(bar["open_time"]), float(tp_price)
    return None


def exit_ma_runner(
    h4: pd.DataFrame,
    h1: pd.DataFrame,
    entry_time: int,
    start_time: int,
    max_exit_time: int,
    ma_len: int,
) -> tuple[int, float, str]:
    bars = h4[(h4["open_time"] >= (start_time // FOUR_HOUR_MS) * FOUR_HOUR_MS) & (h4["open_time"] < max_exit_time)].copy()
    for _, bar in bars.iterrows():
        up_to = h4[h4["open_time"] <= int(bar["open_time"])]
        if len(up_to) < ma_len:
            continue
        ma = float(up_to.tail(ma_len)["close"].mean())
        if float(bar["close"]) < ma:
            next_open_time = int(bar["next_open_time"])
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, next_open_time, entry_time)
            return exit_time, exit_price, fallback or f"tp1_then_ma{ma_len}_exit"
    exit_time, exit_price, fallback = get_open_at_or_latest(h1, max_exit_time, entry_time)
    return exit_time, exit_price, fallback or "max_hold"


def trailing_exit(
    h1: pd.DataFrame,
    entry_time: int,
    start_time: int,
    max_exit_time: int,
    start_high: float,
    trail_pct: float,
) -> tuple[int, float, str]:
    high_water = start_high
    path = h1[(h1["open_time"] >= start_time) & (h1["open_time"] <= max_exit_time)].copy()
    for _, bar in path.iterrows():
        high_water = max(high_water, float(bar["high"]))
        trigger = high_water * (1.0 - trail_pct)
        if float(bar["low"]) <= trigger:
            return int(bar["open_time"]), float(trigger), "tp1_then_trailing_exit"
    exit_time, exit_price, fallback = get_open_at_or_latest(h1, max_exit_time, entry_time)
    return exit_time, exit_price, fallback or "max_hold"


def simulate(signal: pd.Series, strategy: Strategy, h1: pd.DataFrame, h4: pd.DataFrame) -> dict[str, Any]:
    entry_time = int(signal["entry_time_ms"])
    entry_price = float(signal["entry_price"])
    if h1.empty or entry_time not in h1.index:
        return {"exit_time_ms": entry_time, "exit_price": np.nan, "exit_reason": "data_missing", "pnl_u": np.nan}
    if strategy.kind == "fixed":
        days = int(strategy.name.replace("fixed_", "").replace("d", ""))
        return fixed_exit(signal, h1, days)

    max_hold_days = 14 if strategy.kind in ["A", "B"] else 30
    max_exit_time = entry_time + max_hold_days * DAY_MS
    full_path = path_slice(h1, entry_time, max_exit_time)

    if strategy.kind == "A":
        first72 = path_slice(h1, entry_time, entry_time + 72 * HOUR_MS)
        hit = first_hit(first72, entry_price, None, 0.10, strategy.mode)
        exit_target = max_exit_time if hit else entry_time + 72 * HOUR_MS
        exit_time, exit_price, fallback = get_open_at_or_latest(h1, exit_target, entry_time)
        pnl, net_return = calc_pnl(entry_price, [(1.0, exit_price)])
        mfe, mae, max_price, min_price = mfe_mae(path_slice(h1, entry_time, exit_time), entry_price)
        return {
            "exit_time_ms": exit_time,
            "exit_price": exit_price,
            "exit_reason": fallback or ("max_hold" if hit else "weak_exit_72h"),
            "tp1_pnl_u": 0.0,
            "runner_pnl_u": pnl,
            "pnl_u": pnl,
            "net_return_pct": net_return,
            "gross_return_pct": (exit_price / entry_price - 1.0) * 100.0,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "max_price_during_trade": max_price,
            "min_price_during_trade": min_price,
            "hit_tp1": bool(hit),
            "hit_stop_loss": False,
            "runner_exit_reason": "",
        }

    if strategy.kind == "B":
        first72 = path_slice(h1, entry_time, entry_time + 72 * HOUR_MS)
        hit = first_hit(first72, entry_price, -0.10, 0.10, strategy.mode)
        if hit and hit[0] == "stop":
            exit_time, exit_price, reason = hit[1], hit[2], "stop_loss"
        elif not hit or hit[0] != "tp":
            stop_hit = first_hit(first72, entry_price, -0.10, None, strategy.mode)
            if stop_hit:
                exit_time, exit_price, reason = stop_hit[1], stop_hit[2], "stop_loss"
            else:
                exit_time, exit_price, fallback = get_open_at_or_latest(h1, entry_time + 72 * HOUR_MS, entry_time)
                reason = fallback or "weak_exit_72h"
        else:
            after72 = path_slice(h1, hit[1], max_exit_time)
            stop_hit = first_hit(after72, entry_price, -0.10, None, strategy.mode)
            if stop_hit:
                exit_time, exit_price, reason = stop_hit[1], stop_hit[2], "stop_loss"
            else:
                exit_time, exit_price, fallback = get_open_at_or_latest(h1, max_exit_time, entry_time)
                reason = fallback or "max_hold"
        pnl, net_return = calc_pnl(entry_price, [(1.0, exit_price)])
        mfe, mae, max_price, min_price = mfe_mae(path_slice(h1, entry_time, exit_time), entry_price)
        return {
            "exit_time_ms": exit_time,
            "exit_price": exit_price,
            "exit_reason": reason,
            "tp1_pnl_u": 0.0,
            "runner_pnl_u": pnl,
            "pnl_u": pnl,
            "net_return_pct": net_return,
            "gross_return_pct": (exit_price / entry_price - 1.0) * 100.0,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "max_price_during_trade": max_price,
            "min_price_during_trade": min_price,
            "hit_tp1": False,
            "hit_stop_loss": reason == "stop_loss",
            "runner_exit_reason": "",
        }

    if strategy.kind in ["C", "D", "E"]:
        tp_path = path_slice(h1, entry_time, max_exit_time)
        hit = first_hit(tp_path, entry_price, -0.10, 0.15, strategy.mode)
        if hit is None:
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, max_exit_time, entry_time)
            exits = [(1.0, exit_price)]
            reason = fallback or "max_hold"
            hit_tp1 = False
            hit_stop = False
            runner_reason = ""
        elif hit[0] == "stop":
            exit_time, exit_price, reason = hit[1], hit[2], "stop_loss"
            exits = [(1.0, exit_price)]
            hit_tp1 = False
            hit_stop = True
            runner_reason = ""
        else:
            tp_time, tp_price = hit[1], hit[2]
            if strategy.kind == "C":
                runner_exit_time, runner_exit_price, reason = exit_ma_runner(h4, h1, entry_time, tp_time, max_exit_time, 7)
            elif strategy.kind == "D":
                runner_exit_time, runner_exit_price, reason = exit_ma_runner(h4, h1, entry_time, tp_time, max_exit_time, 21)
            else:
                high_at_tp = float(h1.loc[tp_time]["high"]) if tp_time in h1.index else tp_price
                runner_exit_time, runner_exit_price, reason = trailing_exit(h1, entry_time, tp_time, max_exit_time, high_at_tp, 0.25)
            exit_time, exit_price = runner_exit_time, runner_exit_price
            exits = [(0.5, tp_price), (0.5, runner_exit_price)]
            hit_tp1 = True
            hit_stop = False
            runner_reason = reason
        pnl, net_return = calc_pnl(entry_price, exits)
        tp1_pnl, _ = calc_pnl(entry_price, [(0.5, exits[0][1]), (0.5, entry_price)]) if hit_tp1 else (0.0, 0.0)
        runner_pnl = pnl - tp1_pnl
        mfe, mae, max_price, min_price = mfe_mae(path_slice(h1, entry_time, exit_time), entry_price)
        return {
            "exit_time_ms": exit_time,
            "exit_price": exit_price,
            "exit_reason": reason,
            "tp1_pnl_u": tp1_pnl,
            "runner_pnl_u": runner_pnl,
            "pnl_u": pnl,
            "net_return_pct": net_return,
            "gross_return_pct": net_return,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "max_price_during_trade": max_price,
            "min_price_during_trade": min_price,
            "hit_tp1": hit_tp1,
            "hit_stop_loss": hit_stop,
            "runner_exit_reason": runner_reason,
        }

    if strategy.kind == "F":
        high_water = entry_price
        reached_10 = False
        exit_time = None
        exit_price = None
        reason = ""
        for _, bar in full_path.iterrows():
            high_water = max(high_water, float(bar["high"]))
            reached_10 = reached_10 or high_water >= entry_price * 1.10
            stop_hit = float(bar["low"]) <= entry_price * 0.90
            trail_hit = float(bar["low"]) <= high_water * 0.75
            if stop_hit:
                exit_time, exit_price, reason = int(bar["open_time"]), entry_price * 0.90, "stop_loss"
                break
            if trail_hit:
                exit_time, exit_price, reason = int(bar["open_time"]), high_water * 0.75, "trailing_exit"
                break
            if not reached_10 and int(bar["open_time"]) % FOUR_HOUR_MS == 0:
                bar_open = int(bar["open_time"]) - FOUR_HOUR_MS
                if bar_open in h4.index:
                    up_to = h4[h4["open_time"] <= bar_open]
                    if len(up_to) >= 7 and float(up_to.iloc[-1]["close"]) < float(up_to.tail(7)["close"].mean()):
                        exit_time, exit_price, fallback = get_open_at_or_latest(h1, int(bar["open_time"]), entry_time)
                        reason = fallback or "weak_ma7_exit"
                        break
        if exit_time is None:
            exit_time, exit_price, fallback = get_open_at_or_latest(h1, max_exit_time, entry_time)
            reason = fallback or "max_hold"
        pnl, net_return = calc_pnl(entry_price, [(1.0, float(exit_price))])
        mfe, mae, max_price, min_price = mfe_mae(path_slice(h1, entry_time, int(exit_time)), entry_price)
        return {
            "exit_time_ms": int(exit_time),
            "exit_price": float(exit_price),
            "exit_reason": reason,
            "tp1_pnl_u": 0.0,
            "runner_pnl_u": pnl,
            "pnl_u": pnl,
            "net_return_pct": net_return,
            "gross_return_pct": net_return,
            "mfe_pct": mfe,
            "mae_pct": mae,
            "max_price_during_trade": max_price,
            "min_price_during_trade": min_price,
            "hit_tp1": False,
            "hit_stop_loss": reason == "stop_loss",
            "runner_exit_reason": reason if reason != "stop_loss" else "",
        }
    raise ValueError(strategy)


def result_row(signal: pd.Series, strategy: Strategy, result: dict[str, Any]) -> dict[str, Any]:
    entry_time = int(signal["entry_time_ms"])
    exit_time = int(result.get("exit_time_ms", entry_time))
    return {
        "strategy_name": strategy.name,
        "mode": strategy.mode,
        "symbol": signal["symbol"],
        "rank": int(signal["rank"]),
        "entry_time_utc": signal["entry_time_utc"],
        "entry_time_bj": signal["entry_time_bj"],
        "entry_time_ms": entry_time,
        "entry_price": float(signal["entry_price"]),
        "gain_24h": float(signal["gain_24h"]),
        "snapshot_hour_bj": signal["snapshot_hour_bj"],
        "month": signal["month"],
        "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(exit_time),
        "exit_price": result.get("exit_price", np.nan),
        "exit_reason": result.get("exit_reason", "data_missing"),
        "holding_hours": (exit_time - entry_time) / HOUR_MS,
        "holding_days": (exit_time - entry_time) / DAY_MS,
        "gross_return_pct": result.get("gross_return_pct", np.nan),
        "net_return_pct": result.get("net_return_pct", np.nan),
        "pnl_u": result.get("pnl_u", np.nan),
        "mfe_pct": result.get("mfe_pct", np.nan),
        "mae_pct": result.get("mae_pct", np.nan),
        "hit_tp1": result.get("hit_tp1", False),
        "hit_stop_loss": result.get("hit_stop_loss", False),
        "runner_exit_reason": result.get("runner_exit_reason", ""),
        "tp1_pnl_u": result.get("tp1_pnl_u", 0.0),
        "runner_pnl_u": result.get("runner_pnl_u", result.get("pnl_u", np.nan)),
        "is_win": bool(result.get("pnl_u", 0) > 0),
        "max_price_during_trade": result.get("max_price_during_trade", np.nan),
        "min_price_during_trade": result.get("min_price_during_trade", np.nan),
        "volume_24h_ratio_7d_bucket": signal.get("volume_24h_ratio_7d_bucket", ""),
        "ma_structure_4h": signal.get("ma_structure_4h", ""),
        "distance_to_4h_ma7_bucket": signal.get("distance_to_4h_ma7_bucket", ""),
    }


def profit_factor(pnl: pd.Series) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = abs(float(pnl[pnl < 0].sum()))
    return math.inf if gl == 0 and gp > 0 else (gp / gl if gl else 0.0)


def summarize(group: pd.DataFrame, name_cols: dict[str, Any]) -> dict[str, Any]:
    g = group.dropna(subset=["pnl_u"]).copy()
    pnl = g["pnl_u"].astype(float)
    returns = g["net_return_pct"].astype(float)
    wins = g[g["pnl_u"] > 0]
    losses = g[g["pnl_u"] < 0]
    sorted_pnl = g.sort_values("pnl_u", ascending=False)["pnl_u"].reset_index(drop=True)
    best = g.sort_values("pnl_u", ascending=False).head(1)
    worst = g.sort_values("pnl_u", ascending=True).head(1)
    total_deployed = len(g) * BUY_NOTIONAL_U
    dd = calculate_drawdown(g.sort_values("entry_time_ms")["pnl_u"])
    return {
        **name_cols,
        "trade_count": int(len(g)),
        "net_pnl_u": float(pnl.sum()) if len(g) else 0.0,
        "pf": profit_factor(pnl),
        "win_rate": float(len(wins) / len(g)) if len(g) else np.nan,
        "avg_return_pct": float(returns.mean()) if len(g) else np.nan,
        "median_return_pct": float(returns.median()) if len(g) else np.nan,
        "avg_win_pct": float(wins["net_return_pct"].mean()) if len(wins) else np.nan,
        "avg_loss_pct": float(losses["net_return_pct"].mean()) if len(losses) else np.nan,
        "max_win_pct": float(returns.max()) if len(g) else np.nan,
        "max_loss_pct": float(returns.min()) if len(g) else np.nan,
        "max_drawdown_u": dd,
        "max_drawdown_pct_on_total_deployed": float(dd / total_deployed) if total_deployed else np.nan,
        "pnl_after_remove_top1_u": float(sorted_pnl.iloc[1:].sum()) if len(sorted_pnl) > 1 else 0.0,
        "pnl_after_remove_top3_u": float(sorted_pnl.iloc[3:].sum()) if len(sorted_pnl) > 3 else 0.0,
        "pnl_after_remove_top5_u": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
        "best_trade_symbol": "" if best.empty else str(best.iloc[0]["symbol"]),
        "best_trade_pnl_u": np.nan if best.empty else float(best.iloc[0]["pnl_u"]),
        "worst_trade_symbol": "" if worst.empty else str(worst.iloc[0]["symbol"]),
        "worst_trade_pnl_u": np.nan if worst.empty else float(worst.iloc[0]["pnl_u"]),
        "avg_holding_days": float(g["holding_days"].mean()) if len(g) else np.nan,
        "median_holding_days": float(g["holding_days"].median()) if len(g) else np.nan,
    }


def build_stats(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame([summarize(g, {"strategy_name": s}) for s, g in details.groupby("strategy_name", sort=False)])
    monthly = pd.DataFrame(
        [summarize(g, {"strategy_name": s, "month": m}) for (s, m), g in details.groupby(["strategy_name", "month"], sort=False)]
    )
    rank_rows = []
    for strategy, group in details.groupby("strategy_name", sort=False):
        rank_rows.append(summarize(group, {"strategy_name": strategy, "scope": "rank2_rank3"}))
        rank_rows.append(summarize(group[group["rank"].eq(2)], {"strategy_name": strategy, "scope": "rank2"}))
        rank_rows.append(summarize(group[group["rank"].eq(3)], {"strategy_name": strategy, "scope": "rank3"}))
    rank_breakdown = pd.DataFrame(rank_rows)

    factor_rows = []
    for strategy, group in details.groupby("strategy_name", sort=False):
        for factor in ["volume_24h_ratio_7d_bucket", "ma_structure_4h", "distance_to_4h_ma7_bucket"]:
            work = group.copy()
            if factor == "ma_structure_4h":
                work[factor] = work[factor].where(
                    work[factor].isin(["close > MA7 > MA21", "close > MA7 but MA7 <= MA21"]), "other"
                )
            for bucket, bg in work.groupby(factor, dropna=False, sort=False):
                factor_rows.append(summarize(bg, {"strategy_name": strategy, "factor_name": factor, "bucket": "missing" if pd.isna(bucket) else str(bucket)}))
    factor_breakdown = pd.DataFrame(factor_rows)

    ordered = details[details["strategy_name"].eq("fixed_7d")].sort_values("entry_time_ms").reset_index(drop=True)
    split = len(ordered) // 2
    key = ordered[["symbol", "entry_time_ms"]].copy()
    key["period"] = ["train" if i < split else "validation" for i in range(len(key))]
    merged = details.merge(key, on=["symbol", "entry_time_ms"], how="left")
    tv = pd.DataFrame(
        [summarize(g, {"strategy_name": s, "period": p}) for (s, p), g in merged.groupby(["strategy_name", "period"], sort=False)]
    )
    return summary, monthly, rank_breakdown, factor_breakdown, tv


def build_mfe_mae(signals: pd.DataFrame, h1_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    horizons = {"24h": 24 * HOUR_MS, "72h": 72 * HOUR_MS, "7d": 7 * DAY_MS, "14d": 14 * DAY_MS, "30d": 30 * DAY_MS}
    rows = []
    for _, signal in signals.iterrows():
        frame = h1_map.get(str(signal["symbol"]), pd.DataFrame())
        entry_time = int(signal["entry_time_ms"])
        entry_price = float(signal["entry_price"])
        row = {"symbol": signal["symbol"], "rank": int(signal["rank"]), "entry_time_utc": signal["entry_time_utc"], "entry_time_ms": entry_time}
        for name, horizon in horizons.items():
            path = path_slice(frame, entry_time, entry_time + horizon)
            mfe, mae, _, _ = mfe_mae(path, entry_price)
            row[f"mfe_{name}"] = mfe
            row[f"mae_{name}"] = mae
        rows.append(row)
    detail = pd.DataFrame(rows)
    dist_rows = []
    for horizon in horizons:
        for threshold in [5, 10, 15, 30, 50, 100]:
            dist_rows.append({"metric": f"mfe_{horizon}_gte_{threshold}", "count": int((detail[f"mfe_{horizon}"] >= threshold).sum()), "ratio": float((detail[f"mfe_{horizon}"] >= threshold).mean())})
        for threshold in [-5, -10, -15, -20, -30]:
            dist_rows.append({"metric": f"mae_{horizon}_lte_{threshold}", "count": int((detail[f"mae_{horizon}"] <= threshold).sum()), "ratio": float((detail[f"mae_{horizon}"] <= threshold).mean())})
    no_10_24 = detail["mfe_24h"] < 10
    no_10_72 = detail["mfe_72h"] < 10
    yes_10_72 = detail["mfe_72h"] >= 10
    big_50 = detail["mfe_30d"] >= 50
    dist_rows.extend(
        [
            {"metric": "no_10pct_24h_then_50pct_30d", "count": int((no_10_24 & big_50).sum()), "ratio": float((no_10_24 & big_50).sum() / max(no_10_24.sum(), 1))},
            {"metric": "no_10pct_72h_then_50pct_30d", "count": int((no_10_72 & big_50).sum()), "ratio": float((no_10_72 & big_50).sum() / max(no_10_72.sum(), 1))},
            {"metric": "yes_10pct_72h_then_50pct_30d", "count": int((yes_10_72 & big_50).sum()), "ratio": float((yes_10_72 & big_50).sum() / max(yes_10_72.sum(), 1))},
            {"metric": "mae_10pct_then_50pct_30d", "count": int(((detail["mae_30d"] <= -10) & big_50).sum()), "ratio": float(((detail["mae_30d"] <= -10) & big_50).mean())},
        ]
    )
    dist = pd.DataFrame(dist_rows)
    return detail.merge(dist, how="cross")


def main() -> None:
    signals = load_signals()
    symbols = sorted(signals["symbol"].astype(str).unique())
    h1_map, h4_map = load_maps(symbols)
    rows = []
    for _, signal in signals.iterrows():
        h1 = h1_map.get(str(signal["symbol"]), pd.DataFrame())
        h4 = h4_map.get(str(signal["symbol"]), pd.DataFrame())
        for strategy in STRATEGIES:
            result = simulate(signal, strategy, h1, h4)
            rows.append(result_row(signal, strategy, result))
    details = pd.DataFrame(rows)
    summary, monthly, rank_breakdown, factor_breakdown, tv = build_stats(details)
    mfe_mae = build_mfe_mae(signals, h1_map)

    OUT.mkdir(parents=True, exist_ok=True)
    details.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    monthly.to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    rank_breakdown.to_csv(RANK_PATH, index=False, encoding="utf-8-sig")
    factor_breakdown.to_csv(FACTOR_PATH, index=False, encoding="utf-8-sig")
    mfe_mae.to_csv(MFE_MAE_PATH, index=False, encoding="utf-8-sig")
    tv.to_csv(TRAIN_VALIDATION_PATH, index=False, encoding="utf-8-sig")

    main_summary = summary[~summary["strategy_name"].str.contains("optimistic")].copy()
    print("========== 出场策略研究完成 ==========")
    print("研究对象: Rank2 / Rank3, EX-RAVE, 删除 24H >80%")
    print(f"信号数: {len(signals)}")
    print(f"时间范围: {signals['entry_time_utc'].min()} ~ {signals['entry_time_utc'].max()}")
    print("\n========== 固定持有基准 ==========")
    print(main_summary[main_summary["strategy_name"].str.startswith("fixed_")][["strategy_name", "trade_count", "net_pnl_u", "pf", "win_rate", "median_return_pct", "max_drawdown_u", "pnl_after_remove_top1_u", "avg_holding_days"]].to_string(index=False))
    print("\n========== 趋势型出场策略结果 ==========")
    print(main_summary[~main_summary["strategy_name"].str.startswith("fixed_")][["strategy_name", "trade_count", "net_pnl_u", "pf", "win_rate", "median_return_pct", "max_drawdown_u", "pnl_after_remove_top1_u", "max_win_pct", "avg_holding_days"]].to_string(index=False))
    dist = mfe_mae[["metric", "count", "ratio"]].drop_duplicates()
    print("\n========== MFE / MAE 观察 ==========")
    print(dist[dist["metric"].isin(["no_10pct_24h_then_50pct_30d", "no_10pct_72h_then_50pct_30d", "yes_10pct_72h_then_50pct_30d", "mae_10pct_then_50pct_30d"])].to_string(index=False))
    print("\n输出文件:")
    for path in [DETAIL_PATH, SUMMARY_PATH, MONTHLY_PATH, RANK_PATH, FACTOR_PATH, MFE_MAE_PATH, TRAIN_VALIDATION_PATH]:
        print(path)


if __name__ == "__main__":
    main()
