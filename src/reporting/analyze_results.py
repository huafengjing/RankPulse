from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _parameter_line(row: pd.Series) -> str:
    return (
        f"{row['parameter_set_id']} | variant={row['strategy_variant']} | "
        f"volume={row['min_quote_volume']} | SL={_fmt_pct(row['sl_pct'])} | "
        f"TP={_fmt_pct(row['tp1_pct'])} | hold={int(row['max_holding_hours'])}h"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _recompute_expectancy_without_best_trade(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter_set_id, group in trades.groupby("parameter_set_id"):
        trimmed = group.sort_values("net_return_pct", ascending=False).iloc[1:]
        if trimmed.empty:
            continue
        returns = trimmed["net_return_pct"]
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        win_rate = float((returns > 0).mean())
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        rows.append(
            {
                "parameter_set_id": parameter_set_id,
                "trades_after_removing_best": len(trimmed),
                "expectancy_without_best_trade": win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss),
                "avg_return_without_best_trade": float(returns.mean()),
            }
        )
    return pd.DataFrame(rows)


def analyze_outputs(output_dir: str | Path = "outputs") -> str:
    output_path = Path(output_dir)
    summary_path = output_path / "summary.csv"
    trades_path = output_path / "trades.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing trades file: {trades_path}")

    summary = pd.read_csv(summary_path)
    trades = pd.read_csv(trades_path)
    if summary.empty:
        text = "No summary rows found. No edge can be inferred."
        _write_analysis(output_path, text)
        return text

    sort_cols = ["expectancy_pct", "profit_factor", "avg_return_pct"]
    ordered = summary.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)
    best = ordered.iloc[0]
    worst = ordered.iloc[-1]
    median_idx = len(ordered) // 2
    median = ordered.iloc[median_idx]

    positive_expectancy = summary["expectancy_pct"] > 0
    profit_factor_gt_1 = summary["profit_factor"] > 1
    positive_count = int(positive_expectancy.sum())
    pf_count = int(profit_factor_gt_1.sum())
    stable_count = int((positive_expectancy & profit_factor_gt_1).sum())
    total_sets = len(summary)
    median_expectancy = float(summary["expectancy_pct"].median())
    median_profit_factor = float(summary["profit_factor"].median())
    median_drawdown = float(summary["max_drawdown_pct"].median())
    total_trades = int(summary["total_trades"].sum())
    unique_signals = int(summary["total_signals"].max()) if "total_signals" in summary else 0

    robustness = _recompute_expectancy_without_best_trade(trades)
    merged = summary.merge(robustness, on="parameter_set_id", how="left")
    still_positive_without_best = int((merged["expectancy_without_best_trade"] > 0).sum())
    best_robust = merged.loc[merged["parameter_set_id"] == best["parameter_set_id"]].iloc[0]

    by_sl = summary.groupby("sl_pct")["expectancy_pct"].agg(["count", "mean", "median"]).reset_index()
    by_hold = summary.groupby("max_holding_hours")["expectancy_pct"].agg(["count", "mean", "median"]).reset_index()
    by_volume = summary.groupby("min_quote_volume")["expectancy_pct"].agg(["count", "mean", "median"]).reset_index()

    drawdown_acceptable = median_drawdown > -0.5 and float(summary["max_drawdown_pct"].min()) > -0.7
    edge_reliable = (
        total_sets > 0
        and stable_count / total_sets >= 0.6
        and median_expectancy > 0
        and median_profit_factor > 1
        and still_positive_without_best / total_sets >= 0.6
        and drawdown_acceptable
    )
    if edge_reliable:
        conclusion = (
            "Based on this run, the tested parameter grid shows broad positive expectancy after fees/slippage. "
            "This is not enough for live trading; next checks should focus on data bias, 1m execution order, BTC regime, and out-of-sample stability."
        )
    elif median_expectancy > 0 and median_profit_factor > 1 and not drawdown_acceptable:
        conclusion = (
            "Based on this run, returns are broadly positive across the tested grid, but the drawdown is too severe to treat the edge as reliable. "
            "The strategy needs position sizing, portfolio-level exposure controls, stricter execution checks, and bias review before any later-stage work."
        )
    elif positive_count > 0:
        conclusion = (
            "Based on this run, some parameter sets are positive, but the edge should be treated as not yet reliable. "
            "Check whether performance survives parameter changes, removal of outlier trades, and stricter data assumptions."
        )
    else:
        conclusion = "Based on this run, the tested strategy does not show positive expectancy."

    lines = [
        "# Result Analysis",
        "",
        "## Data Scope",
        "",
        f"- parameter_sets: {total_sets}",
        f"- unique_signals_per_set: {unique_signals}",
        f"- total_trade_rows_across_parameter_sets: {total_trades}",
        "",
        "## Stability Snapshot",
        "",
        f"- positive_expectancy_sets: {positive_count}/{total_sets}",
        f"- profit_factor_gt_1_sets: {pf_count}/{total_sets}",
        f"- both_positive_expectancy_and_pf_gt_1: {stable_count}/{total_sets}",
        f"- median_expectancy: {_fmt_pct(median_expectancy)}",
        f"- median_profit_factor: {_fmt_num(median_profit_factor)}",
        f"- median_max_drawdown: {_fmt_pct(median_drawdown)}",
        f"- positive_after_removing_each_set_best_trade: {still_positive_without_best}/{total_sets}",
        "",
        "## Best Parameter Set",
        "",
        f"- {_parameter_line(best)}",
        f"- expectancy: {_fmt_pct(best['expectancy_pct'])}",
        f"- avg_return: {_fmt_pct(best['avg_return_pct'])}",
        f"- profit_factor: {_fmt_num(best['profit_factor'])}",
        f"- win_rate: {_fmt_pct(best['win_rate'])}",
        f"- tp1_hit_rate: {_fmt_pct(best['tp1_hit_rate'])}",
        f"- max_drawdown: {_fmt_pct(best['max_drawdown_pct'])}",
        f"- expectancy_without_best_trade: {_fmt_pct(best_robust['expectancy_without_best_trade'])}",
        "",
        "## Median Parameter Set",
        "",
        f"- {_parameter_line(median)}",
        f"- expectancy: {_fmt_pct(median['expectancy_pct'])}",
        f"- avg_return: {_fmt_pct(median['avg_return_pct'])}",
        f"- profit_factor: {_fmt_num(median['profit_factor'])}",
        f"- max_drawdown: {_fmt_pct(median['max_drawdown_pct'])}",
        "",
        "## Worst Parameter Set",
        "",
        f"- {_parameter_line(worst)}",
        f"- expectancy: {_fmt_pct(worst['expectancy_pct'])}",
        f"- avg_return: {_fmt_pct(worst['avg_return_pct'])}",
        f"- profit_factor: {_fmt_num(worst['profit_factor'])}",
        f"- max_drawdown: {_fmt_pct(worst['max_drawdown_pct'])}",
        "",
        "## Sensitivity",
        "",
        "### By SL",
        _markdown_table(by_sl),
        "",
        "### By Max Holding Hours",
        _markdown_table(by_hold),
        "",
        "### By Min Quote Volume",
        _markdown_table(by_volume),
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Limitations",
        "",
        "- This analysis is based on current output files only.",
        "- The current minimal backtest uses the implemented top10_immediate variant.",
        "- BTC regime fields are placeholders in the current minimal version.",
        "- Historical universe/listing and liquidity bias still need stricter handling before any trading-system stage.",
        "- Backtest results are not investment advice and do not imply future returns.",
    ]
    text = "\n".join(lines)
    _write_analysis(output_path, text)
    return text


def _write_analysis(output_path: Path, text: str) -> None:
    report_dir = output_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "result_analysis.md").write_text(text, encoding="utf-8")
