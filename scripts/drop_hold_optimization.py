from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import load_cached_klines


SOURCE = ROOT / "outputs" / "drop_top10_short_filter"
OUT = ROOT / "outputs" / "drop_hold_optimization"
LOOKAHEAD_MS = 120 * 60 * 60 * 1000
FEE = 0.0005
SLIPPAGE = 0.0005


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(wins / abs(losses)) if abs(losses) else np.inf


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def pnl_excluding_best(pnl: pd.Series, n: int) -> float:
    if len(pnl) <= n:
        return 0.0
    return float(pnl.sort_values(ascending=False).iloc[n:].sum())


def short_leg_pnl(entry: float, exit_price: float, weight: float) -> float:
    entry_eff = entry * (1.0 - SLIPPAGE)
    exit_eff = exit_price * (1.0 + SLIPPAGE)
    return weight * ((entry_eff - exit_eff) / entry_eff - 2.0 * FEE)


def simulate_exit_a(path: pd.DataFrame, entry_price: float) -> tuple[float, dict[str, object]]:
    sl = entry_price * 1.10
    tp1 = entry_price * 0.90
    tp2 = entry_price * 0.80
    pnl = 0.0
    remaining = 1.0
    tp1_hit = False
    minus20_hit = False
    minus30_hit = bool((path["low"] <= entry_price * 0.70).any())

    for _, bar in path.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])
        if high >= sl:
            pnl += short_leg_pnl(entry_price, sl, remaining)
            return pnl, {
                "first_plus10": not tp1_hit,
                "first_minus10": tp1_hit,
                "first_minus20": minus20_hit,
                "first_minus30": minus30_hit,
                "tp1_hit": tp1_hit,
                "exit_reason": "sl_plus10",
            }
        if not tp1_hit and low <= tp1:
            pnl += short_leg_pnl(entry_price, tp1, 0.5)
            remaining = 0.5
            tp1_hit = True
        if tp1_hit and low <= tp2:
            pnl += short_leg_pnl(entry_price, tp2, remaining)
            minus20_hit = True
            return pnl, {
                "first_plus10": False,
                "first_minus10": True,
                "first_minus20": True,
                "first_minus30": minus30_hit,
                "tp1_hit": True,
                "exit_reason": "tp2_minus20",
            }
    pnl += short_leg_pnl(entry_price, float(path.iloc[-1]["close"]), remaining)
    return pnl, {
        "first_plus10": False,
        "first_minus10": tp1_hit,
        "first_minus20": minus20_hit,
        "first_minus30": minus30_hit,
        "tp1_hit": tp1_hit,
        "exit_reason": "max_holding_time",
    }


def summarize(trades: pd.DataFrame, model_name: str) -> dict[str, object]:
    if trades.empty:
        return {
            "model_name": model_name,
            "original_signal_count": 0,
            "trade_count": 0,
        }
    pnl = trades["net_pnl"]
    return {
        "model_name": model_name,
        "original_signal_count": int(trades["original_signal_count"].iloc[0]),
        "trade_count": len(trades),
        "abandoned_count": int(trades["original_signal_count"].iloc[0]) - len(trades),
        "avg_entry_delay_minutes": float(trades["entry_delay_minutes"].mean()),
        "first_plus10_rate": float(trades["first_plus10"].mean()),
        "first_minus10_rate": float(trades["first_minus10"].mean()),
        "first_minus20_rate": float(trades["first_minus20"].mean()),
        "first_minus30_rate": float(trades["first_minus30"].mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "max_drawdown": max_drawdown(pnl),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
        "pnl_excluding_best_1": pnl_excluding_best(pnl, 1),
        "pnl_excluding_best_5": pnl_excluding_best(pnl, 5),
        "monthly_profitable_count": monthly_profitable_count(trades),
    }


def summarize_combo(trades: pd.DataFrame, model_name: str, conditions: str) -> dict[str, object]:
    out = summarize(trades, model_name)
    out.pop("original_signal_count", None)
    out.pop("abandoned_count", None)
    out.pop("avg_entry_delay_minutes", None)
    out["combo_name"] = model_name
    out["conditions"] = conditions
    return out


def monthly_profitable_count(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    tmp = trades.copy()
    tmp["month"] = pd.to_datetime(tmp["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    return int((tmp.groupby("month")["net_pnl"].sum() > 0).sum())


def asof_row(group: pd.DataFrame, time_ms: int) -> pd.Series | None:
    hits = group[group["open_time"] <= time_ms]
    if hits.empty:
        return None
    return hits.iloc[-1]


def next_row(group: pd.DataFrame, time_ms: int) -> pd.Series | None:
    hits = group[group["open_time"] > time_ms]
    if hits.empty:
        return None
    return hits.iloc[0]


def build_events(
    base: pd.DataFrame,
    kmap: dict[str, pd.DataFrame],
    rlookup: pd.DataFrame,
    original_count: int,
    model_name: str,
    hold_minutes: int,
    rank_n: int,
    rebound_limit: float | None = None,
) -> pd.DataFrame:
    rows = []
    for _, signal in base.iterrows():
        symbol = signal["symbol"]
        group = kmap[symbol]
        signal_time = int(signal["signal_time"])
        confirm_time = signal_time + hold_minutes * 60_000
        confirm_bar = asof_row(group, confirm_time)
        if confirm_bar is None:
            continue
        key = (int(confirm_bar["open_time"]), symbol)
        if key not in rlookup.index:
            continue
        rank_row = rlookup.loc[key]
        if isinstance(rank_row, pd.DataFrame):
            rank_row = rank_row.iloc[0]
        if int(rank_row["drop_rank"]) > rank_n:
            continue

        candidate = next_row(group, signal_time)
        entry = next_row(group, confirm_time)
        if candidate is None or entry is None:
            continue
        candidate_price = float(candidate["open"])
        window = group[(group["open_time"] >= int(candidate["open_time"])) & (group["open_time"] <= confirm_time)]
        max_rebound = float(window["high"].max() / candidate_price - 1.0) if not window.empty else 0.0
        if rebound_limit is not None and max_rebound > rebound_limit:
            continue

        rows.append(
            {
                **signal.to_dict(),
                "model_name": model_name,
                "entry_time": int(entry["open_time"]),
                "entry_time_utc": entry["open_time_utc"],
                "entry_price": float(entry["open"]),
                "entry_delay_minutes": int((int(entry["open_time"]) - signal_time) / 60_000),
                "confirm_rank": int(rank_row["drop_rank"]),
                "max_rebound_before_entry": max_rebound,
                "original_signal_count": original_count,
            }
        )
    return pd.DataFrame(rows)


def backtest_events(events: pd.DataFrame, kmap: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, event in events.iterrows():
        group = kmap[event["symbol"]]
        entry_time = int(event["entry_time"])
        entry_price = float(event["entry_price"])
        path = group[(group["open_time"] >= entry_time) & (group["open_time"] <= entry_time + LOOKAHEAD_MS)]
        if path.empty:
            continue
        pnl, meta = simulate_exit_a(path, entry_price)
        rows.append(
            {
                "model_name": event["model_name"],
                "signal_id": int(event["signal_id"]),
                "symbol": event["symbol"],
                "signal_time_utc": event["signal_time_utc"],
                "entry_time_utc": event["entry_time_utc"],
                "entry_price": entry_price,
                "entry_delay_minutes": int(event["entry_delay_minutes"]),
                "confirm_rank": int(event["confirm_rank"]),
                "max_rebound_before_entry": float(event["max_rebound_before_entry"]),
                "original_signal_count": int(event["original_signal_count"]),
                "net_pnl": pnl,
                **meta,
            }
        )
    return pd.DataFrame(rows)


def robustness(all_trades: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    split = pd.to_datetime(original["signal_time_utc"], utc=True).sort_values().iloc[int(len(original) * 0.70)]
    baseline = all_trades[all_trades["model_name"].eq("Hold Top10 60m")]
    base_val = baseline[pd.to_datetime(baseline["signal_time_utc"], utc=True) > split]
    base_val_plus = float(base_val["first_plus10"].mean()) if not base_val.empty else np.nan
    base_val_minus20 = float(base_val["first_minus20"].mean()) if not base_val.empty else np.nan
    rows = []
    for model, group in all_trades.groupby("model_name"):
        train = group[pd.to_datetime(group["signal_time_utc"], utc=True) <= split]
        validation = group[pd.to_datetime(group["signal_time_utc"], utc=True) > split]
        val_plus = float(validation["first_plus10"].mean()) if len(validation) else np.nan
        val_minus20 = float(validation["first_minus20"].mean()) if len(validation) else np.nan
        val_pf = profit_factor(validation["net_pnl"])
        overfit = (
            len(validation) < 30
            or val_pf <= 1
            or (pd.notna(base_val_plus) and val_plus > base_val_plus + 0.03)
            or (pd.notna(base_val_minus20) and val_minus20 < base_val_minus20 - 0.05)
        )
        rows.append(
            {
                "model_name": model,
                "train_count": len(train),
                "validation_count": len(validation),
                "train_first_plus10_rate": float(train["first_plus10"].mean()) if len(train) else np.nan,
                "validation_first_plus10_rate": val_plus,
                "train_first_minus20_rate": float(train["first_minus20"].mean()) if len(train) else np.nan,
                "validation_first_minus20_rate": val_minus20,
                "train_pf": profit_factor(train["net_pnl"]),
                "validation_pf": val_pf,
                "train_total_pnl": float(train["net_pnl"].sum()),
                "validation_total_pnl": float(validation["net_pnl"].sum()),
                "overfit_flag": bool(overfit),
                "conclusion": "valid" if not overfit else "not_valid",
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    return "\n".join(
        [
            "| " + " | ".join(text.columns) + " |",
            "| " + " | ".join("---" for _ in text.columns) + " |",
            *["| " + " | ".join(row) + " |" for row in text.astype(str).values.tolist()],
        ]
    )


def write_report(
    hold: pd.DataFrame,
    rank: pd.DataFrame,
    rebound: pd.DataFrame,
    combo: pd.DataFrame,
    robust: pd.DataFrame,
) -> None:
    lines = [
        "# Drop Hold Optimization",
        "",
        "Scope: 5d first drop Top10, trigger 24h drop in [15%, 20%), Exit A fixed.",
        "",
        "## Hold Time",
        dataframe_to_markdown(hold),
        "",
        "## Rank Hold",
        dataframe_to_markdown(rank),
        "",
        "## Rebound Filter",
        dataframe_to_markdown(rebound),
        "",
        "## Final Combos",
        dataframe_to_markdown(combo),
        "",
        "## Robustness",
        dataframe_to_markdown(robust),
        "",
        "## Recommendation",
        "Keep the original Hold Top10 60m + Exit A unless an enhancement improves validation PF, first +10%, and -20% without a large sample loss.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(SOURCE / "base_signals.csv")
    base["signal_time"] = pd.to_datetime(base["signal_time_utc"], utc=True).map(lambda x: int(x.timestamp() * 1000))
    base["entry_candidate_price"] = base["entry_price"]
    base[
        [
            "signal_id",
            "symbol",
            "signal_time_utc",
            "rank_at_signal",
            "rolling_24h_drop_at_signal",
            "price_at_signal",
            "entry_candidate_price",
            "quote_volume_at_signal",
        ]
    ].to_csv(OUT / "base_signals.csv", index=False)

    symbols = set(base["symbol"])
    klines = load_cached_klines(ROOT / "data", "5m")
    klines = klines[klines["symbol"].isin(symbols)].sort_values(["symbol", "open_time"])
    klines["open_time_utc"] = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    kmap = {symbol: group.reset_index(drop=True) for symbol, group in klines.groupby("symbol", sort=False)}

    rankings = pd.read_csv(SOURCE / "signal_features.csv")[["signal_id"]]  # Placeholder to keep path explicit.
    # Reuse full ranking snapshots from source are not persisted, so rebuild a lightweight drop ranking for these symbols.
    all_klines = load_cached_klines(ROOT / "data", "5m")
    all_klines["open_time_utc"] = pd.to_datetime(all_klines["open_time"], unit="ms", utc=True)
    all_klines = all_klines.sort_values(["symbol", "open_time"])
    prev = all_klines.groupby("symbol")["close"].shift(288)
    rank_frame = all_klines[prev.notna()].copy()
    rank_frame["rolling_24h_change_pct"] = rank_frame["close"] / prev[prev.notna()] - 1.0
    rank_frame["drop_rank"] = rank_frame.groupby("open_time")["rolling_24h_change_pct"].rank(method="first", ascending=True).astype(int)
    rlookup = rank_frame[["open_time", "symbol", "drop_rank"]].set_index(["open_time", "symbol"])

    model_specs = {
        "Hold Top10 30m": (30, 10, None),
        "Hold Top10 60m": (60, 10, None),
        "Hold Top10 90m": (90, 10, None),
        "Hold Top7 60m": (60, 7, None),
        "Hold Top5 60m": (60, 5, None),
        "Hold Top10 60m rebound<=5": (60, 10, 0.05),
        "Hold Top10 60m rebound<=7": (60, 10, 0.07),
        "Hold Top10 60m rebound<=10": (60, 10, 0.10),
        "Combo 1": (60, 10, 0.07),
        "Combo 2": (60, 7, None),
        "Combo 3": (60, 7, 0.07),
    }

    all_trade_frames = []
    for model_name, (hold_minutes, rank_n, rebound_limit) in model_specs.items():
        events = build_events(base, kmap, rlookup, len(base), model_name, hold_minutes, rank_n, rebound_limit)
        trades = backtest_events(events, kmap) if not events.empty else pd.DataFrame()
        all_trade_frames.append(trades)
    all_trades = pd.concat(all_trade_frames, ignore_index=True) if all_trade_frames else pd.DataFrame()
    all_trades.to_csv(OUT / "trades_all_models.csv", index=False)

    hold = pd.DataFrame([summarize(all_trades[all_trades["model_name"].eq(name)], name) for name in ["Hold Top10 30m", "Hold Top10 60m", "Hold Top10 90m"]])
    rank = pd.DataFrame([summarize(all_trades[all_trades["model_name"].eq(name)], name) for name in ["Hold Top10 60m", "Hold Top7 60m", "Hold Top5 60m"]])
    rebound = pd.DataFrame(
        [
            summarize(all_trades[all_trades["model_name"].eq(name)], name)
            for name in ["Hold Top10 60m", "Hold Top10 60m rebound<=5", "Hold Top10 60m rebound<=7", "Hold Top10 60m rebound<=10"]
        ]
    )
    combo = pd.DataFrame(
        [
            summarize_combo(all_trades[all_trades["model_name"].eq("Combo 1")], "Combo 1", "Hold Top10 60m + rebound<=7"),
            summarize_combo(all_trades[all_trades["model_name"].eq("Combo 2")], "Combo 2", "Hold Top7 60m"),
            summarize_combo(all_trades[all_trades["model_name"].eq("Combo 3")], "Combo 3", "Hold Top7 60m + rebound<=7"),
        ]
    )
    robust = robustness(all_trades, base)

    hold.to_csv(OUT / "hold_time_summary.csv", index=False)
    rank.to_csv(OUT / "rank_hold_summary.csv", index=False)
    rebound.to_csv(OUT / "rebound_filter_summary.csv", index=False)
    combo.to_csv(OUT / "final_combo_summary.csv", index=False)
    robust.to_csv(OUT / "robustness_check.csv", index=False)

    monthly_rows = []
    all_trades["month"] = pd.to_datetime(all_trades["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
    for (model, month), group in all_trades.groupby(["model_name", "month"]):
        monthly_rows.append(
            {
                "model_name": model,
                "month": month,
                "trade_count": len(group),
                "pnl": float(group["net_pnl"].sum()),
                "PF": profit_factor(group["net_pnl"]),
                "first_plus10_rate": float(group["first_plus10"].mean()),
                "first_minus10_rate": float(group["first_minus10"].mean()),
                "first_minus20_rate": float(group["first_minus20"].mean()),
                "max_drawdown": max_drawdown(group["net_pnl"]),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT / "summary_by_month.csv", index=False)
    write_report(hold, rank, rebound, combo, robust)

    print("Hold time")
    print(hold.to_string(index=False))
    print("Rank hold")
    print(rank.to_string(index=False))
    print("Rebound")
    print(rebound.to_string(index=False))
    print("Combos")
    print(combo.to_string(index=False))
    print("Robustness")
    print(robust.to_string(index=False))


if __name__ == "__main__":
    main()
