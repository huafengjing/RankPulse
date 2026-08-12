from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("outputs/drop_top10_short_all")


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    return float(wins / abs(losses)) if abs(losses) else np.inf


def main() -> None:
    features = pd.read_csv(OUT / "signal_features.csv")
    labels = pd.read_csv(OUT / "path_labels.csv")
    df = features.merge(labels, on=["signal_id", "symbol", "signal_time_utc"])
    df["signal_time"] = pd.to_datetime(df["signal_time_utc"], utc=True)
    df = df.sort_values("signal_time").reset_index(drop=True)
    df["simple_pnl"] = np.select(
        [df["hit_plus_10_before_minus_10"].astype(bool), df["hit_minus_10_before_plus_10"].astype(bool)],
        [-0.101, 0.099],
        default=0.0,
    )
    split = df["signal_time"].iloc[int(len(df) * 0.70)]
    base_rate = float(df["hit_plus_10_before_minus_10"].mean())
    rule_defs = [
        ("base_all", lambda x: pd.Series(True, index=x.index)),
        ("Combo 1", lambda x: x["rolling_24h_drop_at_signal"].between(0.10, 0.20) & x["volume_1h_vs_24h_avg"].between(1.2, 5)),
        (
            "Combo 5",
            lambda x: x["rolling_24h_drop_at_signal"].between(0.10, 0.20)
            & x["volume_1h_vs_24h_avg"].between(1.2, 5)
            & (x["signal_candle_lower_wick_pct"] <= 0.40),
        ),
        (
            "Combo 9",
            lambda x: x["rolling_24h_drop_at_signal"].between(0.10, 0.25)
            & (x["rank_at_signal"] >= 4)
            & (x["signal_candle_lower_wick_pct"] <= 0.40),
        ),
    ]
    rows = []
    for name, fn in rule_defs:
        selected = df[fn(df)]
        train = selected[selected["signal_time"] <= split]
        validation = selected[selected["signal_time"] > split]
        validation_rate = float(validation["hit_plus_10_before_minus_10"].mean()) if len(validation) else np.nan
        validation_pf = profit_factor(validation["simple_pnl"])
        rows.append(
            {
                "rule_name": name,
                "train_count": len(train),
                "validation_count": len(validation),
                "train_first_plus10_rate": float(train["hit_plus_10_before_minus_10"].mean()) if len(train) else np.nan,
                "validation_first_plus10_rate": validation_rate,
                "train_pf": profit_factor(train["simple_pnl"]),
                "validation_pf": validation_pf,
                "train_total_pnl": float(train["simple_pnl"].sum()),
                "validation_total_pnl": float(validation["simple_pnl"].sum()),
                "conclusion": "valid" if len(validation) >= 30 and validation_pf > 1 and validation_rate < base_rate else "not_valid",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "quick_robustness_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
