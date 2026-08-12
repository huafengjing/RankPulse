from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.downloader import load_cached_klines


EVENTS = ROOT / "outputs/breakout_structure/event_features.csv"
TRADES = ROOT / "outputs/breakout_structure/trades.csv"
OUT = ROOT / "outputs/strict_compression_retest"

PRE_72H_MS = 72 * 60 * 60 * 1000
PRE_5D_MS = 5 * 24 * 60 * 60 * 1000
PRE_7D_MS = 7 * 24 * 60 * 60 * 1000
ACCOUNT_CAPITAL = 1000.0


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def profit_factor(r: pd.Series) -> float:
    wins = r[r > 0]
    losses = r[r <= 0]
    return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) else np.inf


def summarize(df: pd.DataFrame, event_type: str, rule_name: str, group: str, total_events: int) -> dict:
    if df.empty:
        return {
            "event_type": event_type,
            "rule_name": rule_name,
            "group": group,
            "signal_count": 0,
            "pass_rate": 0.0,
            "tp15_hit_rate": np.nan,
            "plus30_hit_rate": np.nan,
            "plus50_hit_rate": np.nan,
            "plus100_hit_rate": np.nan,
            "minus10_first_rate": np.nan,
            "total_pnl": 0.0,
            "profit_factor": np.nan,
            "max_drawdown": 0.0,
            "monthly_profitable_count": 0,
            "pnl_excluding_best_5": np.nan,
        }
    r = df["return_on_margin_pct"]
    pnl = df["net_pnl_usd"]
    sorted_pnl = pnl.sort_values(ascending=False).reset_index(drop=True)
    monthly = df.assign(month=pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")).groupby("month")["net_pnl_usd"].sum()
    return {
        "event_type": event_type,
        "rule_name": rule_name,
        "group": group,
        "signal_count": int(df["event_id"].nunique()),
        "pass_rate": int(df["event_id"].nunique()) / total_events if total_events else 0.0,
        "tp15_hit_rate": float(df["tp1_hit"].mean()),
        "plus30_hit_rate": float(df["plus30_hit"].mean()),
        "plus50_hit_rate": float(df["plus50_hit"].mean()),
        "plus100_hit_rate": float(df["plus100_hit"].mean()),
        "minus10_first_rate": float(df["minus10_first"].mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(r),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "monthly_profitable_count": int((monthly > 0).sum()),
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
    }


def pre_window_features(event: pd.Series, group: pd.DataFrame, lookback_ms: int) -> tuple[float, float]:
    t = int(event["structure_time"] if "structure_time" in event and pd.notna(event["structure_time"]) else event["event_time"])
    start = t - lookback_ms
    pre = group[(group["open_time"] >= start) & (group["open_time"] < t)]
    event_bar = group[group["open_time"] == t]
    if pre.empty or event_bar.empty:
        return np.nan, np.nan
    event_close = float(event_bar.iloc[0]["close"])
    first_close = float(pre.iloc[0]["close"])
    return float(pre["high"].max() / pre["low"].min() - 1), float(event_close / first_close - 1)


def was_in_top50_last_7d(event: pd.Series, rankings_by_symbol: dict[str, pd.DataFrame]) -> bool:
    symbol = event["symbol"]
    group = rankings_by_symbol.get(symbol)
    if group is None:
        return False
    t = int(event["structure_time"] if "structure_time" in event and pd.notna(event["structure_time"]) else event["event_time"])
    start = t - PRE_7D_MS
    hits = group[(group["open_time"] >= start) & (group["open_time"] < t) & (group["rank"] <= 50)]
    return not hits.empty


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(EVENTS)
    trades = pd.read_csv(TRADES)
    merged = events.merge(trades, on=["event_id", "event_group", "symbol", "event_time_utc"], how="inner", suffixes=("", "_trade"))
    relevant_types = ["all_first_top20", "all_first_top10", "top20_breakout_then_top10_24h"]
    merged = merged[merged["event_group"].isin(relevant_types)].copy()

    print("Loading klines for strict pre-window features...", flush=True)
    klines = load_cached_klines(ROOT / "data", "5m")
    symbols = set(merged["symbol"])
    kmap = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in klines[klines["symbol"].isin(symbols)].groupby("symbol", sort=False)}

    print("Loading rankings for top50 history...", flush=True)
    rankings_path = ROOT / "outputs/board_sentiment/signal_board_features.csv"
    # Use event_features' rank only for events; rebuild small symbol ranking history from breakout output is not enough.
    # Instead, load full ranking-derived event_features already contains event ranks, while top50 history needs local ranking.
    from src.research.ranking import build_rankings

    min_t = int(merged["structure_time"].fillna(merged["event_time"]).min()) - PRE_7D_MS - 24 * 60 * 60 * 1000
    max_t = int(merged["event_time"].max())
    rank_input = klines[(klines["open_time"] >= min_t) & (klines["open_time"] <= max_t)].copy()
    rank_input["open_time_utc"] = pd.to_datetime(rank_input["open_time"], unit="ms", utc=True)
    rankings = build_rankings(rank_input[~rank_input["symbol"].isin(["BTCUSDT", "ETHUSDT", "BNBUSDT"])], 5)
    rankings_by_symbol = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in rankings.groupby("symbol", sort=False)}

    rows = []
    for _, event in merged.iterrows():
        group = kmap.get(event["symbol"])
        if group is None:
            continue
        pre5_range, pre5_return = pre_window_features(event, group, PRE_5D_MS)
        top50 = was_in_top50_last_7d(event, rankings_by_symbol)
        row = event.to_dict()
        row["pre_5d_range_pct"] = pre5_range
        row["pre_5d_return"] = pre5_return
        row["was_in_top50_last_7d"] = top50
        row["rule_a_pass"] = row["pre_72h_range_pct"] <= 0.15 and abs(row["pre_72h_return"]) <= 0.10
        row["rule_b_pass"] = pre5_range <= 0.25 and abs(pre5_return) <= 0.15
        row["rule_c_pass"] = pre5_range <= 0.30 and abs(pre5_return) <= 0.20 and not top50
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "strict_event_features_with_trades.csv", index=False)

    rule_map = {
        "Rule A 72h strict compression": "rule_a_pass",
        "Rule B 5d compression": "rule_b_pass",
        "Rule C clean board + compression": "rule_c_pass",
    }
    summary_rows = []
    monthly_rows = []
    for event_type in relevant_types:
        et = out[out["event_group"] == event_type].copy()
        total = int(et["event_id"].nunique())
        for rule_name, col in rule_map.items():
            passed = et[et[col]].copy()
            failed = et[~et[col]].copy()
            summary_rows.append(summarize(passed, event_type, rule_name, "pass", total))
            summary_rows.append(summarize(failed, event_type, rule_name, "fail", total))
            for group_name, df in [("pass", passed), ("fail", failed)]:
                if df.empty:
                    continue
                df = df.copy()
                df["month"] = pd.to_datetime(df["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
                for month, mdf in df.groupby("month"):
                    s = summarize(mdf, event_type, rule_name, group_name, total)
                    monthly_rows.append({"month": month, **s})

    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    summary.to_csv(OUT / "strict_rule_summary.csv", index=False)
    monthly.to_csv(OUT / "strict_rule_summary_by_month.csv", index=False)

    report = [
        "# Strict Compression Retest",
        "",
        "## Summary",
        summary.to_csv(index=False),
        "",
        "## Monthly",
        monthly.to_csv(index=False),
        "",
        "## Notes",
        "- Rule A: pre_72h_range_pct <= 15% and abs(pre_72h_return) <= 10%.",
        "- Rule B: pre_5d_range_pct <= 25% and abs(pre_5d_return) <= 15%.",
        "- Rule C: pre_5d_range_pct <= 30%, abs(pre_5d_return) <= 20%, and was_in_top50_last_7d = False.",
    ]
    (OUT / "strict_compression_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
