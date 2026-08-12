from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_report(summary: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        result = "No trades were generated; no edge can be inferred."
    else:
        median_expectancy = summary["expectancy_pct"].median()
        result = (
            "Based on the tested period and assumptions, the strategy shows positive median expectancy."
            if median_expectancy > 0
            else "Based on the tested period and assumptions, the strategy does not show positive median expectancy."
        )
    text = f"""# Research Report

## 1. Strategy Hypothesis

First entry into Binance USDT-M rolling 24H Top10 may capture attention-driven continuation into Top5.

## 2. Data Source

Binance USDT-M Futures public REST API and local cached klines.

## 3. Universe Selection

PERPETUAL USDT contracts with configurable quote volume threshold.

## 4. Signal Definition

First Top10 entry after a cooldown window, with entry at next candle open.

## 5. Backtest Assumptions

Long only, TP/SL checked by candle high/low, SL first if both trigger in one candle, taker fees and slippage included.

## 6. Parameter Grid

See `config/parameter_grid.yaml`.

## 7. Main Results

{result}

## 8. Best / Median / Worst Parameter Sets

Review `outputs/summary.csv`.

## 9. Top10 to Top5 Conversion Analysis

Review `top10_to_top5_conversion_rate` in `outputs/summary.csv`.

## 10. TP1 Hit Rate Analysis

Review `tp1_hit_rate` in `outputs/summary.csv`.

## 11. MFE / MAE Analysis

Review `avg_mfe_pct`, `avg_mae_pct`, and distributions in charts.

## 12. BTC Regime Analysis

BTC regime columns are placeholders in the minimal version and should be expanded before relying on this section.

## 13. Failure Cases

Review losing trades in `outputs/trades.csv`.

## 14. Limitations and Biases

Initial universe uses currently available exchange info and ticker volume, which can introduce survivorship or future universe bias unless historical listings and volume are reconstructed.

## 15. Conclusion

This project is research only. Backtest results do not represent future returns and are not investment advice.
"""
    path.write_text(text, encoding="utf-8")
