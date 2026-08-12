from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw/klines/5m")
FEATURES = Path("outputs/robust_filter/signal_features.csv")
OUT = Path("outputs/tp1_runner_ma21_analysis")

TP1 = 0.15
LIQ = -0.10
MAX_HOLD_HOURS = 240
MA_WINDOW_4H = 21
TRAILING_STOPS = [0.10, 0.15, 0.20]
LEVERAGE = 10.0
RUNNER_MARGIN = 50.0


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

    with_ma = []
    for _, group in df.groupby("symbol", sort=False):
        g = group.sort_values("open_time").copy().set_index("open_time_utc")
        close_4h = g["close"].resample("4h", label="right", closed="right").last().dropna()
        ma21 = close_4h.rolling(MA_WINDOW_4H, min_periods=MA_WINDOW_4H).mean()
        g["ma21_4h_completed"] = ma21.shift(1).reindex(g.index, method="ffill").values
        with_ma.append(g.reset_index())
    df = pd.concat(with_ma, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol", sort=False)}


def find_entry(group: pd.DataFrame, signal_time: int) -> int | None:
    idx = group.index[group["open_time"] > signal_time]
    if len(idx) == 0:
        return None
    return int(idx[0])


def find_tp1(group: pd.DataFrame, entry_idx: int, end_time: int, entry_price: float) -> int | None:
    tp1_price = entry_price * (1 + TP1)
    liq_price = entry_price * (1 + LIQ)
    path = group[(group["open_time"] >= int(group.loc[entry_idx, "open_time"])) & (group["open_time"] <= end_time)]
    for idx, bar in path.iterrows():
        if float(bar["low"]) <= liq_price:
            return None
        if float(bar["high"]) >= tp1_price:
            return int(idx)
    return None


def pure_ma21_exit(group: pd.DataFrame, tp1_idx: int, end_time: int) -> dict:
    path = group[(group["open_time"] >= int(group.loc[tp1_idx, "open_time"])) & (group["open_time"] <= end_time)]
    peak_price = float(path["high"].max())
    peak_idx = int(path["high"].idxmax())
    for idx, bar in path.iterrows():
        ma21 = bar.get("ma21_4h_completed")
        if pd.notna(ma21) and float(bar["close"]) < float(ma21):
            return {
                "exit_idx": int(idx),
                "exit_time": int(bar["open_time"]),
                "exit_price": float(bar["close"]),
                "exit_reason": "ma21_4h_break",
                "peak_price": peak_price,
                "peak_time": int(group.loc[peak_idx, "open_time"]),
            }
    last = path.iloc[-1]
    return {
        "exit_idx": int(last.name),
        "exit_time": int(last["open_time"]),
        "exit_price": float(last["close"]),
        "exit_reason": "max_10d",
        "peak_price": peak_price,
        "peak_time": int(group.loc[peak_idx, "open_time"]),
    }


def trailing_exit(group: pd.DataFrame, tp1_idx: int, end_time: int, entry_price: float, stop_pct: float) -> dict:
    path = group[(group["open_time"] >= int(group.loc[tp1_idx, "open_time"])) & (group["open_time"] <= end_time)]
    highest = max(float(group.loc[tp1_idx, "high"]), entry_price * (1 + TP1))
    for idx, bar in path.iterrows():
        ma21 = bar.get("ma21_4h_completed")
        highest = max(highest, float(bar["high"]))
        stop_price = highest * (1 - stop_pct)
        if float(bar["low"]) <= stop_price:
            return {
                "exit_idx": int(idx),
                "exit_time": int(bar["open_time"]),
                "exit_price": stop_price,
                "exit_reason": f"trailing_{int(stop_pct * 100)}pct",
            }
        if pd.notna(ma21) and float(bar["close"]) < float(ma21):
            return {
                "exit_idx": int(idx),
                "exit_time": int(bar["open_time"]),
                "exit_price": float(bar["close"]),
                "exit_reason": "ma21_4h_break",
            }
    last = path.iloc[-1]
    return {
        "exit_idx": int(last.name),
        "exit_time": int(last["open_time"]),
        "exit_price": float(last["close"]),
        "exit_reason": "max_10d",
    }


def bucket_return(ret: float, final: bool) -> str:
    if final and ret < 0.15:
        return "<+15%"
    if ret < 0.20:
        return "+15%~+20%"
    if ret < 0.30:
        return "+20%~+30%"
    if ret < 0.50:
        return "+30%~+50%"
    if ret < 1.00:
        return "+50%~+100%"
    if ret < 2.00:
        return "+100%~+200%"
    return ">=+200%"


def summarize_bucket(rows: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    out = rows.groupby(col).size().reindex(order, fill_value=0).reset_index()
    out.columns = ["区间", "数量"]
    out["比例"] = out["数量"] / len(rows)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
        entry_idx = find_entry(group, int(sig["signal_time"]))
        if entry_idx is None:
            continue
        entry = group.loc[entry_idx]
        entry_time = int(entry["open_time"])
        entry_price = float(entry["open"])
        end_time = entry_time + MAX_HOLD_HOURS * 60 * 60 * 1000
        tp1_idx = find_tp1(group, entry_idx, end_time, entry_price)
        if tp1_idx is None:
            continue

        tp1_bar = group.loc[tp1_idx]
        pure = pure_ma21_exit(group, tp1_idx, end_time)
        final_return = pure["exit_price"] / entry_price - 1
        peak_return = pure["peak_price"] / entry_price - 1
        peak_to_exit_drawdown = pure["exit_price"] / pure["peak_price"] - 1
        post_tp1_path = group[
            (group["open_time"] >= pure["exit_time"])
            & (group["open_time"] <= end_time)
        ]

        row = {
            "signal_id": int(sig["signal_id"]),
            "symbol": sig["symbol"],
            "signal_time_utc": sig["signal_time_utc"],
            "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
            "entry_price": entry_price,
            "tp1_time_utc": pd.to_datetime(int(tp1_bar["open_time"]), unit="ms", utc=True),
            "tp1_price": entry_price * (1 + TP1),
            "pure_exit_time_utc": pd.to_datetime(pure["exit_time"], unit="ms", utc=True),
            "pure_exit_price": pure["exit_price"],
            "pure_exit_reason": pure["exit_reason"],
            "final_return_pct": final_return,
            "peak_return_pct": peak_return,
            "peak_to_exit_drawdown_pct": peak_to_exit_drawdown,
            "final_bucket": bucket_return(final_return, final=True),
            "peak_bucket": bucket_return(peak_return, final=False),
            "group": "A_peak_lt_30" if peak_return < 0.30 else "B_peak_ge_30",
        }
        for stop in TRAILING_STOPS:
            trail = trailing_exit(group, tp1_idx, end_time, entry_price, stop)
            trail_ret = trail["exit_price"] / entry_price - 1
            future_after_trail = group[
                (group["open_time"] >= trail["exit_time"])
                & (group["open_time"] <= pure["exit_time"])
            ]
            future_max_ret = (float(future_after_trail["high"].max()) / entry_price - 1) if not future_after_trail.empty else trail_ret
            row[f"trail_{int(stop * 100)}_exit_return_pct"] = trail_ret
            row[f"trail_{int(stop * 100)}_exit_reason"] = trail["exit_reason"]
            row[f"trail_{int(stop * 100)}_early_exit"] = trail["exit_time"] < pure["exit_time"]
            row[f"trail_{int(stop * 100)}_cut_monster"] = trail["exit_time"] < pure["exit_time"] and future_max_ret >= 0.50
        rows.append(row)

    runners = pd.DataFrame(rows)
    runners.to_csv(OUT / "tp1_runner_paths.csv", index=False)

    final_order = ["<+15%", "+15%~+20%", "+20%~+30%", "+30%~+50%", "+50%~+100%", "+100%~+200%", ">=+200%"]
    peak_order = ["+15%~+20%", "+20%~+30%", "+30%~+50%", "+50%~+100%", "+100%~+200%", ">=+200%"]
    final_dist = summarize_bucket(runners, "final_bucket", final_order)
    peak_dist = summarize_bucket(runners, "peak_bucket", peak_order)
    final_dist.to_csv(OUT / "runner_final_distribution.csv", index=False)
    peak_dist.to_csv(OUT / "runner_peak_distribution.csv", index=False)

    group_rows = []
    for name, group in runners.groupby("group"):
        group_rows.append(
            {
                "group": name,
                "count": len(group),
                "avg_peak_return": group["peak_return_pct"].mean(),
                "median_peak_return": group["peak_return_pct"].median(),
                "avg_final_return": group["final_return_pct"].mean(),
                "median_final_return": group["final_return_pct"].median(),
                "avg_peak_to_exit_drawdown": group["peak_to_exit_drawdown_pct"].mean(),
                "median_peak_to_exit_drawdown": group["peak_to_exit_drawdown_pct"].median(),
                "worst_peak_to_exit_drawdown": group["peak_to_exit_drawdown_pct"].min(),
            }
        )
    drawdown_groups = pd.DataFrame(group_rows)
    drawdown_groups.to_csv(OUT / "drawdown_groups.csv", index=False)

    comparison = [
        {
            "exit_model": "pure_ma21",
            "avg_exit_return": runners["final_return_pct"].mean(),
            "median_exit_return": runners["final_return_pct"].median(),
            "early_exit_count": 0,
            "cut_monster_count": 0,
            "runner_avg_pnl_50u_margin_10x": runners["final_return_pct"].mean() * RUNNER_MARGIN * LEVERAGE,
            "worst_exit_return": runners["final_return_pct"].min(),
        }
    ]
    for stop in TRAILING_STOPS:
        key = int(stop * 100)
        ret_col = f"trail_{key}_exit_return_pct"
        comparison.append(
            {
                "exit_model": f"trailing_{key}pct_plus_ma21",
                "avg_exit_return": runners[ret_col].mean(),
                "median_exit_return": runners[ret_col].median(),
                "early_exit_count": int(runners[f"trail_{key}_early_exit"].sum()),
                "cut_monster_count": int(runners[f"trail_{key}_cut_monster"].sum()),
                "runner_avg_pnl_50u_margin_10x": runners[ret_col].mean() * RUNNER_MARGIN * LEVERAGE,
                "worst_exit_return": runners[ret_col].min(),
            }
        )
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT / "trailing_stop_comparison.csv", index=False)

    report = [
        "# TP1 Runner MA21 Analysis",
        "",
        f"- selected_signals: {len(selected)}",
        f"- tp1_plus15_hits_before_minus10: {len(runners)}",
        "- pure runner exit: 5m close breaks completed 4H MA21, or max 10 days from entry",
        "",
        "## Runner Final Distribution",
        markdown_table(final_dist),
        "",
        "## Runner Peak Distribution",
        markdown_table(peak_dist),
        "",
        "## Drawdown Groups",
        markdown_table(drawdown_groups),
        "",
        "## Trailing Stop Comparison",
        markdown_table(comp),
    ]
    (OUT / "tp1_runner_ma21_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"TP1 runners={len(runners)}")
    print(final_dist.to_string(index=False))
    print(peak_dist.to_string(index=False))
    print(drawdown_groups.to_string(index=False))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
