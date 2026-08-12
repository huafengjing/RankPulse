from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_basic_charts(trades: pd.DataFrame, summary: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if trades.empty:
        return

    equity = (1.0 + trades.sort_values("entry_time_utc")["net_return_pct"]).cumprod()
    plt.figure()
    equity.plot(title="Equity Curve")
    plt.xlabel("Trade")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out / "equity_curve.png")
    plt.close()

    for col, name in [
        ("net_return_pct", "return_distribution.png"),
        ("mfe_pct", "mfe_distribution.png"),
        ("mae_pct", "mae_distribution.png"),
    ]:
        plt.figure()
        trades[col].hist(bins=30)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out / name)
        plt.close()

    plt.figure()
    trades.groupby("rank_at_signal")["tp1_hit"].mean().plot(kind="bar", title="TP1 Hit Rate by Rank")
    plt.xlabel("Rank at Signal")
    plt.ylabel("TP1 Hit Rate")
    plt.tight_layout()
    plt.savefig(out / "tp1_hit_rate_by_rank.png")
    plt.close()

    if not summary.empty:
        plt.figure()
        summary.groupby("sl_pct")["expectancy_pct"].mean().plot(kind="bar", title="Parameter Sensitivity: SL")
        plt.xlabel("SL")
        plt.ylabel("Average Expectancy")
        plt.tight_layout()
        plt.savefig(out / "parameter_sensitivity_sl.png")
        plt.close()
