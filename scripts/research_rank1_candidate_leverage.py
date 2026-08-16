from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, add_entry_factors, load_kline_map, max_drawdown, ms_to_utc, profit_factor
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS
from scripts.regime_adaptive_leverage_walkforward import LIQUIDATION_THRESHOLDS_PCT, simulate_trade_with_leverage
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "rank1_candidate_leverage_research"
LEVERAGES = [1, 2, 3, 5]
GAIN_BUCKETS = [("20-40%", 0.20, 0.40), ("40-60%", 0.40, 0.60)]
VR_BUCKETS = [("2-3", 2.0, 3.0), ("5-6", 5.0, 6.0)]


def bucket_label(value: float, buckets: list[tuple[str, float, float]]) -> str | None:
    for label, lower, upper in buckets:
        if lower <= value < upper:
            return label
    return None


def evaluated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return pd.DataFrame()
    return frame[frame["status"].isin(["completed", "open_mark_to_market"])].copy()


def summarize(frame: pd.DataFrame, raw_signals: int) -> dict[str, Any]:
    done = evaluated(frame).sort_values("entry_time_ms")
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    liqs = done[done["liquidated"].astype(bool)] if len(done) and "liquidated" in done else pd.DataFrame()
    non_liq = done[~done["liquidated"].astype(bool)] if len(done) and "liquidated" in done else done
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    net = float(pnl.sum()) if len(pnl) else 0.0
    liq_loss = float(pd.to_numeric(liqs.get("pnl_u", pd.Series(dtype=float)), errors="coerce").sum()) if len(liqs) else 0.0
    non_liq_pnl = pd.to_numeric(non_liq.get("pnl_u", pd.Series(dtype=float)), errors="coerce") if len(non_liq) else pd.Series(dtype=float)
    return {
        "raw_signals": int(raw_signals),
        "trades": int(len(done)),
        "conflict_skips": int(frame["status"].eq("skipped").sum()) if "status" in frame else 0,
        "open_mtm": int(frame["status"].eq("open_mark_to_market").sum()) if "status" in frame else 0,
        "net_pnl_u": net,
        "pf": profit_factor(pnl),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "avg_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "ex_top1_pnl_u": float(net - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "pnl_per_trade_u": float(net / len(pnl)) if len(pnl) else np.nan,
        "max_dd_u": max_drawdown(pnl),
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
        "liquidations": int(len(liqs)),
        "liq_rate": float(len(liqs) / len(done)) if len(done) else np.nan,
        "total_liq_loss_u": liq_loss,
        "net_ex_liq_trades_u": float(non_liq_pnl.sum()) if len(non_liq_pnl) else 0.0,
    }


def simulate_pool(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int, leverage: int, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        common = {
            "pool": label,
            "research_leverage": leverage,
            "gain_bucket": signal["gain_bucket"],
            "vr_bucket": signal["vr_bucket"],
        }
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            rows.append(
                {
                    "symbol": symbol,
                    "rank": 1,
                    "entry_time_ms": signal_time,
                    "entry_time_utc": ms_to_utc(signal_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_time_bj": signal.get("snapshot_time_bj", ""),
                    "month": ms_to_utc(signal_time).strftime("%Y-%m"),
                    "status": "skipped",
                    "skip_reason": "symbol_already_open",
                    **common,
                }
            )
            continue
        trade = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, leverage)
        rows.append(trade | common)
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            extra = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + extra
    return pd.DataFrame(rows)


def add_rows(rows: list[dict[str, Any]], kind: str, key: dict[str, Any], signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int) -> list[pd.DataFrame]:
    frames = []
    for leverage in LEVERAGES:
        label_parts = [kind] + [f"{k}={v}" for k, v in key.items()]
        label = "|".join(label_parts)
        trades = simulate_pool(signals, kline_map, cutoff_ms, leverage, label)
        frames.append(trades)
        rows.append({"kind": kind, **key, "leverage": leverage, **summarize(trades, len(signals))})
    return frames


def add_efficiency(summary: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = summary.copy()
    out["leverage_efficiency"] = np.nan
    for _, group in out.groupby(keys, dropna=False):
        base = group[group["leverage"].eq(1)]
        if base.empty:
            continue
        base_net = float(base.iloc[0]["net_pnl_u"])
        if base_net <= 0:
            continue
        for idx, row in group.iterrows():
            lev = int(row["leverage"])
            out.loc[idx, "leverage_efficiency"] = float(row["net_pnl_u"]) / (lev * base_net)
    return out


def incremental(summary: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key_values, group in summary.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key = dict(zip(keys, key_values))
        by_lev = group.set_index("leverage")
        for a, b in [(1, 2), (2, 3), (3, 5)]:
            if a not in by_lev.index or b not in by_lev.index:
                continue
            rows.append(
                key
                | {
                    "from_to": f"{a}x->{b}x",
                    "incremental_pnl_u": float(by_lev.loc[b, "net_pnl_u"] - by_lev.loc[a, "net_pnl_u"]),
                    "incremental_liquidations": int(by_lev.loc[b, "liquidations"] - by_lev.loc[a, "liquidations"]),
                    "incremental_dd_u": float(by_lev.loc[b, "max_dd_u"] - by_lev.loc[a, "max_dd_u"]),
                }
            )
    return pd.DataFrame(rows)


def tolerance(row: pd.Series) -> str:
    if row["trades"] < 8:
        return "Unclear / insufficient sample"
    if row["liq_rate"] >= 0.25 or row["median_return_pct"] <= -80:
        return "High leverage fragile"
    if row["liq_rate"] > 0:
        return "Moderate leverage tolerant"
    return "Low leverage tolerant"


def markdown_table(frame: pd.DataFrame) -> str:
    work = frame.copy()
    for column in work.columns:
        if pd.api.types.is_float_dtype(work[column]):
            work[column] = work[column].map(lambda x: "" if pd.isna(x) else ("inf" if np.isinf(x) else f"{x:.3f}"))
    rows = ["|" + "|".join(map(str, work.columns)) + "|", "|" + "|".join(["---"] * len(work.columns)) + "|"]
    rows.extend("|" + "|".join(map(str, row)) + "|" for row in work.itertuples(index=False, name=None))
    return "\n".join(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [symbol for symbol in cached_symbols() if symbol not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)

    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    signals = raw[
        raw["snapshot_hour_bj"].isin(SNAPSHOT_HOURS_BJ)
        & raw["rank"].eq(1)
        & raw["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    signals = add_entry_factors(signals, kline_map)
    signals["gain_bucket"] = signals["gain_24h"].astype(float).map(lambda x: bucket_label(x, GAIN_BUCKETS))
    signals["vr_bucket"] = signals["volume_24h_ratio_7d"].astype(float).map(lambda x: bucket_label(x, VR_BUCKETS))
    scoped = signals[signals["gain_bucket"].notna() & signals["vr_bucket"].notna()].copy()
    scoped = scoped.sort_values(["signal_time", "symbol"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []

    all_frames.extend(add_rows(rows, "overall", {}, scoped, kline_map, common_end))
    for gain in [b[0] for b in GAIN_BUCKETS]:
        all_frames.extend(add_rows(rows, "gain", {"gain_bucket": gain}, scoped[scoped["gain_bucket"].eq(gain)].copy(), kline_map, common_end))
    for vr in [b[0] for b in VR_BUCKETS]:
        all_frames.extend(add_rows(rows, "volume", {"vr_bucket": vr}, scoped[scoped["vr_bucket"].eq(vr)].copy(), kline_map, common_end))
    for gain in [b[0] for b in GAIN_BUCKETS]:
        for vr in [b[0] for b in VR_BUCKETS]:
            cell_signals = scoped[scoped["gain_bucket"].eq(gain) & scoped["vr_bucket"].eq(vr)].copy()
            all_frames.extend(add_rows(rows, "cell", {"gain_bucket": gain, "vr_bucket": vr, "cell": f"{gain}|{vr}"}, cell_signals, kline_map, common_end))

    summary = pd.DataFrame(rows)
    summary["tolerance"] = summary.apply(tolerance, axis=1)
    overall = add_efficiency(summary[summary["kind"].eq("overall")].copy(), ["kind"])
    cell = add_efficiency(summary[summary["kind"].eq("cell")].copy(), ["kind", "cell"])
    gain = add_efficiency(summary[summary["kind"].eq("gain")].copy(), ["kind", "gain_bucket"])
    volume = add_efficiency(summary[summary["kind"].eq("volume")].copy(), ["kind", "vr_bucket"])

    all_trades = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    done = evaluated(all_trades)
    for (kind, pool, lev, month), group in done.groupby(["pool", "pool", "research_leverage", "month"], sort=True):
        pnl = pd.to_numeric(group["pnl_u"], errors="coerce")
        monthly_rows.append(
            {
                "pool": pool,
                "leverage": lev,
                "month": month,
                "trades": int(len(group)),
                "net_pnl_u": float(pnl.sum()),
                "liquidations": int(group["liquidated"].astype(bool).sum()),
                "is_incomplete_month": month == ms_to_utc(common_end).strftime("%Y-%m"),
            }
        )
    monthly = pd.DataFrame(monthly_rows)

    liq_detail = done[done["liquidated"].astype(bool)].copy()
    winners = []
    for (pool, lev), group in done.groupby(["pool", "research_leverage"], sort=True):
        for rank, (_, row) in enumerate(group.sort_values("pnl_u", ascending=False).head(3).iterrows(), start=1):
            winners.append(
                {
                    "pool": pool,
                    "leverage": lev,
                    "winner_rank": rank,
                    "symbol": row["symbol"],
                    "entry_time_utc": row["entry_time_utc"],
                    "entry_time_bj": row["entry_time_bj"],
                    "gain_bucket": row["gain_bucket"],
                    "vr_bucket": row["vr_bucket"],
                    "pnl_u": row["pnl_u"],
                    "net_return_pct": row["net_return_pct"],
                    "liquidated": row["liquidated"],
                }
            )

    risk_rows = []
    for _, row in cell.iterrows():
        risk_rows.append(
            {
                "region": row["cell"],
                "leverage": f"{int(row['leverage'])}x",
                "display": f"{row['net_pnl_u']:.1f} / {row['pf']:.2f} / {row['liq_rate']:.1%}",
            }
        )
    risk = pd.DataFrame(risk_rows).pivot(index="region", columns="leverage", values="display").reset_index()

    overall.to_csv(OUT_DIR / "overall_leverage_summary.csv", index=False, encoding="utf-8-sig")
    cell.to_csv(OUT_DIR / "cell_leverage_summary.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(OUT_DIR / "gain_leverage_summary.csv", index=False, encoding="utf-8-sig")
    volume.to_csv(OUT_DIR / "volume_leverage_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_leverage_summary.csv", index=False, encoding="utf-8-sig")
    liq_detail.to_csv(OUT_DIR / "liquidation_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(winners).to_csv(OUT_DIR / "top_winner_by_leverage.csv", index=False, encoding="utf-8-sig")
    pd.concat([overall, cell, gain, volume], ignore_index=True).to_csv(OUT_DIR / "leverage_efficiency.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(OUT_DIR / "risk_reward_matrix.csv", index=False, encoding="utf-8-sig")
    incremental(overall, ["kind"]).to_csv(OUT_DIR / "overall_incremental_leverage.csv", index=False, encoding="utf-8-sig")
    incremental(cell, ["kind", "cell"]).to_csv(OUT_DIR / "cell_incremental_leverage.csv", index=False, encoding="utf-8-sig")
    scoped.to_csv(OUT_DIR / "raw_candidate_signals.csv", index=False, encoding="utf-8-sig")
    all_trades.to_csv(OUT_DIR / "candidate_leverage_trade_details.csv", index=False, encoding="utf-8-sig")

    liq_model = pd.DataFrame(
        [{"leverage": f"{lev}x", "liquidation_threshold_underlying_mae": LIQUIDATION_THRESHOLDS_PCT[lev]} for lev in LEVERAGES]
    )
    liq_model.to_csv(OUT_DIR / "liquidation_model.csv", index=False, encoding="utf-8-sig")

    report_cols = ["leverage", "trades", "net_pnl_u", "pf", "win_rate", "median_return_pct", "ex_top1_pnl_u", "ex_top3_pnl_u", "liquidations", "liq_rate", "pnl_per_trade_u", "max_dd_u", "tolerance"]
    lines = [
        "# Rank1 Candidate Leverage Response",
        "",
        "## Scope",
        "",
        f"- Cutoff: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        f"- Raw candidate signals: {len(scoped)}.",
        "- Candidate region: Rank1, Gain 20-40 or 40-60, V/R 2-3 or 5-6, BJ 00:00 and 08:00.",
        "- Exit: current 4H extreme weak, 12H weak, otherwise 6D. Fee 0.1% per side, slippage 0.",
        "",
        "## Liquidation Model",
        "",
        markdown_table(liq_model),
        "",
        "## Overall Candidate Pool",
        "",
        markdown_table(overall[report_cols].round(3)),
        "",
        "## Four Cells",
        "",
        markdown_table(cell[["cell"] + report_cols].round(3)),
        "",
        "## Gain Aggregation",
        "",
        markdown_table(gain[["gain_bucket"] + report_cols].round(3)),
        "",
        "## V/R Aggregation",
        "",
        markdown_table(volume[["vr_bucket"] + report_cols].round(3)),
        "",
        "## Incremental Leverage",
        "",
        markdown_table(incremental(overall, ["kind"]).round(3)),
        "",
        "## Final Judgment",
        "",
        "- This is a leverage response study only; no Rank1 rule is added to the main strategy.",
        "- 2x/3x/5x are path-resimulated, not linear multiples of 1x.",
        "- Use `cell_incremental_leverage.csv` and `monthly_leverage_summary.csv` for detailed audit.",
    ]
    (OUT_DIR / "final_judgment.md").write_text("\n".join(lines), encoding="utf-8")

    print("output", OUT_DIR)
    print("cutoff_utc", ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"))
    print("\nOVERALL")
    print(overall[report_cols].round(3).to_string(index=False))
    print("\nVOLUME")
    print(volume[["vr_bucket"] + report_cols].round(3).to_string(index=False))
    print("\nCELL")
    print(cell[["cell"] + report_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
