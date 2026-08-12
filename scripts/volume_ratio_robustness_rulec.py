from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "liquidation_analysis"
OUT = ROOT / "outputs" / "volume_ratio_robustness_rulec"

ACCOUNT_CAPITAL = 1000.0

RULES = [
    ("vr_1.2_5", 1.2, 5.0),
    ("vr_1.5_5", 1.5, 5.0),
    ("vr_2.0_5", 2.0, 5.0),
    ("vr_1.5_4", 1.5, 4.0),
    ("vr_1.5_6", 1.5, 6.0),
]


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    return float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) else np.inf


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = ACCOUNT_CAPITAL + pnl.cumsum()
    return float((equity / equity.cummax() - 1).min())


def summarize(df: pd.DataFrame, segment: str, rule_name: str) -> dict:
    if df.empty:
        return {
            "rule_name": rule_name,
            "segment": segment,
            "trade_count": 0,
            "pnl": 0.0,
            "PF": np.nan,
            "TP15": np.nan,
            "+50%": np.nan,
            "first_-10%": np.nan,
            "max_drawdown": 0.0,
            "avg_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
        }
    pnl = df["net_pnl_usd"]
    return {
        "rule_name": rule_name,
        "segment": segment,
        "trade_count": int(len(df)),
        "pnl": float(pnl.sum()),
        "PF": profit_factor(df["return_on_margin_pct"]),
        "TP15": float(df["tp1_hit"].mean()),
        "+50%": float(df["plus50_hit"].mean()),
        "first_-10%": float(df["minus10_first"].mean()),
        "max_drawdown": max_drawdown(df.sort_values("entry_time_utc")["net_pnl_usd"]),
        "avg_trade_pnl": float(pnl.mean()),
        "median_trade_pnl": float(pnl.median()),
    }


def train_validation(df: pd.DataFrame, rule_name: str) -> list[dict]:
    ordered = df.sort_values("signal_time").copy()
    split_idx = int(len(ordered) * 0.70)
    train = ordered.iloc[:split_idx]
    validation = ordered.iloc[split_idx:]
    return [summarize(train, "train", rule_name), summarize(validation, "validation", rule_name)]


def tail_dependency(df: pd.DataFrame, rule_name: str) -> dict:
    sorted_pnl = df["net_pnl_usd"].sort_values(ascending=False).reset_index(drop=True) if not df.empty else pd.Series(dtype=float)
    return {
        "rule_name": rule_name,
        "trade_count": int(len(df)),
        "total_pnl": float(sorted_pnl.sum()) if len(sorted_pnl) else 0.0,
        "pnl_excluding_best_1": float(sorted_pnl.iloc[1:].sum()) if len(sorted_pnl) > 1 else 0.0,
        "pnl_excluding_best_5": float(sorted_pnl.iloc[5:].sum()) if len(sorted_pnl) > 5 else 0.0,
        "pnl_excluding_best_10": float(sorted_pnl.iloc[10:].sum()) if len(sorted_pnl) > 10 else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(SOURCE / "signal_features.csv")
    trades = pd.read_csv(SOURCE / "baseline_trades.csv")
    cols = ["signal_id", "signal_time", "volume_1h_vs_24h_avg"]
    merged = trades.merge(features[cols], on="signal_id", how="left")
    merged["signal_time"] = merged["signal_time"].astype("int64")
    merged.to_csv(OUT / "rulec_base_trades_with_volume.csv", index=False)

    summary_rows = []
    tv_rows = []
    monthly_rows = []
    tail_rows = []
    for rule_name, lo, hi in RULES:
        rule = merged[(merged["volume_1h_vs_24h_avg"] >= lo) & (merged["volume_1h_vs_24h_avg"] <= hi)].copy()
        rule = rule.sort_values("entry_time_utc").reset_index(drop=True)
        rule.to_csv(OUT / f"trades_rulec_{rule_name}.csv", index=False)
        summary_rows.append(summarize(rule, "all", rule_name))
        tv_rows.extend(train_validation(rule, rule_name))
        tail_rows.append(tail_dependency(rule, rule_name))
        if not rule.empty:
            rule["month"] = pd.to_datetime(rule["entry_time_utc"], utc=True).dt.strftime("%Y-%m")
            for month, mdf in rule.groupby("month"):
                monthly_rows.append(summarize(mdf, month, rule_name))

    summary = pd.DataFrame(summary_rows)
    tv = pd.DataFrame(tv_rows)
    monthly = pd.DataFrame(monthly_rows)
    tail = pd.DataFrame(tail_rows)
    summary.to_csv(OUT / "robustness_summary.csv", index=False)
    tv.to_csv(OUT / "train_validation_summary.csv", index=False)
    monthly.to_csv(OUT / "monthly_results.csv", index=False)
    tail.to_csv(OUT / "tail_dependency.csv", index=False)

    report = [
        "# Rule C + Volume Ratio Robustness",
        "",
        "Scope: Rule C 21d consolidation signals from liquidation_analysis baseline, then volume_1h_vs_24h_avg filters.",
        "",
        "## Robustness Summary",
        summary.to_csv(index=False),
        "",
        "## Train / Validation",
        tv.to_csv(index=False),
        "",
        "## Monthly",
        monthly.to_csv(index=False),
        "",
        "## Tail Dependency",
        tail.to_csv(index=False),
    ]
    (OUT / "volume_ratio_robustness_rulec_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(tv.to_string(index=False))
    print(tail.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
