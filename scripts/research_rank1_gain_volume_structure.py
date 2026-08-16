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

from scripts.backfill_old_half_and_run_main_strategy import (
    DAY_MS,
    HOUR_MS,
    OUT,
    add_entry_factors,
    load_kline_map,
    max_drawdown,
    ms_to_utc,
    profit_factor,
)
from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS
from scripts.regime_adaptive_leverage_walkforward import simulate_trade_with_leverage
import scripts.regime_adaptive_leverage_walkforward as leverage_engine
from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, SNAPSHOT_HOURS_BJ, cache_common_end_ms, cached_symbols


OUT_DIR = OUT / "rank1_gain_volume_structure"
GAIN_BUCKETS = [
    ("20-40%", 0.20, 0.40),
    ("40-60%", 0.40, 0.60),
]
VR_BUCKETS = [
    ("2-3", 2.0, 3.0),
    ("3-4", 3.0, 4.0),
    ("4-5", 4.0, 5.0),
    ("5-6", 5.0, 6.0),
]
HOURS_BJ = ["00:00", "08:00"]


def bucket_label(value: float, buckets: list[tuple[str, float, float]]) -> str | None:
    for label, lower, upper in buckets:
        if lower <= value < upper:
            return label
    return None


def sample_size_label(n: int) -> str:
    if n < 8:
        return "Very Small"
    if n < 15:
        return "Small"
    if n < 30:
        return "Medium"
    return "Better"


def evaluated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame:
        return pd.DataFrame()
    return frame[frame["status"].isin(["completed", "open_mark_to_market"])].copy()


def safe_pf(pnl: pd.Series) -> float:
    return profit_factor(pd.to_numeric(pnl, errors="coerce"))


def top_share(pnl: pd.Series, n: int) -> float:
    positives = pd.to_numeric(pnl, errors="coerce")
    positives = positives[positives > 0]
    gross_profit = float(positives.sum())
    if gross_profit <= 0:
        return np.nan
    return float(positives.nlargest(n).sum() / gross_profit)


def summarize(frame: pd.DataFrame, raw_signals: int) -> dict[str, Any]:
    done = evaluated(frame).sort_values("entry_time_ms")
    pnl = pd.to_numeric(done["pnl_u"], errors="coerce") if len(done) else pd.Series(dtype=float)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    net = float(pnl.sum()) if len(pnl) else 0.0
    return {
        "signals": int(raw_signals),
        "trades": int(len(done)),
        "conflict_skips": int(frame["status"].eq("skipped").sum()) if "status" in frame else 0,
        "open_mtm": int(frame["status"].eq("open_mark_to_market").sum()) if "status" in frame else 0,
        "net_pnl_u": net,
        "gross_profit_u": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss_u": float(losses.sum()) if len(losses) else 0.0,
        "pf": safe_pf(pnl),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else np.nan,
        "median_return_pct": float(ret.median()) if len(ret) else np.nan,
        "avg_return_pct": float(ret.mean()) if len(ret) else np.nan,
        "max_dd_u": max_drawdown(pnl),
        "ex_top1_pnl_u": float(net - pnl.nlargest(1).sum()) if len(pnl) >= 1 else np.nan,
        "ex_top3_pnl_u": float(net - pnl.nlargest(3).sum()) if len(pnl) >= 3 else np.nan,
        "top1_contribution_pct": float(pnl.nlargest(1).sum() / net * 100.0) if net > 0 and len(pnl) >= 1 else np.nan,
        "top3_contribution_pct": float(pnl.nlargest(3).sum() / net * 100.0) if net > 0 and len(pnl) >= 3 else np.nan,
        "top1_share_gross_profit": top_share(pnl, 1),
        "top3_share_gross_profit": top_share(pnl, 3),
        "best_trade_u": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade_u": float(pnl.min()) if len(pnl) else np.nan,
        "sample_size": sample_size_label(len(done)),
    }


def return_distribution(frame: pd.DataFrame, raw_signals: int) -> dict[str, Any]:
    stats = summarize(frame, raw_signals)
    done = evaluated(frame)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    for q, name in [(0.10, "p10"), (0.25, "p25"), (0.75, "p75"), (0.90, "p90")]:
        stats[name] = float(ret.quantile(q)) if len(ret) else np.nan
    stats["mean"] = stats["avg_return_pct"]
    stats["median"] = stats["median_return_pct"]
    return stats


def loser_density(frame: pd.DataFrame) -> dict[str, float]:
    done = evaluated(frame)
    ret = pd.to_numeric(done["net_return_pct"], errors="coerce") if len(done) else pd.Series(dtype=float)
    if len(ret) == 0:
        return {"ret_le_-10_pct": np.nan, "ret_le_-20_pct": np.nan, "ret_le_-30_pct": np.nan, "ret_le_-50_pct": np.nan}
    return {
        "ret_le_-10_pct": float((ret <= -10).mean()),
        "ret_le_-20_pct": float((ret <= -20).mean()),
        "ret_le_-30_pct": float((ret <= -30).mean()),
        "ret_le_-50_pct": float((ret <= -50).mean()),
    }


def current_price_at(frame: pd.DataFrame, open_time: int) -> float:
    rows = frame[frame["open_time"].eq(open_time)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[-1]["open"])


def path_stats(signal: pd.Series, kline_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    entry_time = int(signal["signal_time"])
    h1 = kline_map.get(symbol, pd.DataFrame())
    entry = current_price_at(h1, entry_time)
    out: dict[str, Any] = {}
    if not np.isfinite(entry) or h1.empty:
        return out
    for hours in [24, 48, 72]:
        t = entry_time + hours * HOUR_MS
        px = current_price_at(h1, t)
        out[f"return_{hours}h_pct"] = (px / entry - 1.0) * 100.0 if np.isfinite(px) else np.nan
        path = h1[(h1["open_time"] >= entry_time) & (h1["open_time"] <= t - HOUR_MS)].copy()
        if path.empty:
            out[f"mfe_{hours}h_pct"] = np.nan
            out[f"mae_{hours}h_pct"] = np.nan
        else:
            out[f"mfe_{hours}h_pct"] = (float(path["high"].max()) / entry - 1.0) * 100.0
            out[f"mae_{hours}h_pct"] = (float(path["low"].min()) / entry - 1.0) * 100.0
    t6 = entry_time + 6 * DAY_MS
    px6 = current_price_at(h1, t6)
    out["return_6d_raw_pct"] = (px6 / entry - 1.0) * 100.0 if np.isfinite(px6) else np.nan
    return out


def simulate_cell(signals: pd.DataFrame, kline_map: dict[str, pd.DataFrame], cutoff_ms: int, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    for _, signal in signals.sort_values(["signal_time", "symbol"]).iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        common = {
            "cell": label,
            "gain_bucket": signal["gain_bucket"],
            "vr_bucket": signal["vr_bucket"],
            "time_bj": signal["snapshot_hour_bj"],
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
        trade = simulate_trade_with_leverage(signal, kline_map, cutoff_ms, 1)
        rows.append(trade | common | path_stats(signal, kline_map))
        if trade.get("status") in {"completed", "open_mark_to_market"}:
            extra = 1 if trade.get("status") == "open_mark_to_market" else 0
            open_until_by_symbol[symbol] = int(float(trade["exit_time_ms"])) + extra
    return pd.DataFrame(rows)


def classify(row: pd.Series) -> str:
    n = int(row["trades"])
    if n < 8:
        return "Insufficient"
    if row["net_pnl_u"] > 0 and row["ex_top1_pnl_u"] > 0 and row["pf"] > 1:
        return "Positive Observation"
    if row["net_pnl_u"] < 0 and row["pf"] < 0.9:
        return "Negative Observation"
    return "Neutral / Mixed"


def heatmap(summary: pd.DataFrame, value: str, filename: str, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    pivot = summary.pivot(index="gain_bucket", columns="vr_bucket", values=value).reindex(
        [b[0] for b in GAIN_BUCKETS], columns=[b[0] for b in VR_BUCKETS]
    )
    data = pivot.astype(float).to_numpy()
    finite = data[np.isfinite(data)]
    vmin = float(np.nanmin(finite)) if len(finite) else 0.0
    vmax = float(np.nanmax(finite)) if len(finite) else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0

    cell_w, cell_h = 150, 78
    left, top = 118, 78
    width = left + cell_w * len(pivot.columns) + 24
    height = top + cell_h * len(pivot.index) + 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def color_for(val: float) -> tuple[int, int, int]:
        if not np.isfinite(val):
            return (230, 230, 230)
        x = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
        if x < 0.5:
            t = x / 0.5
            return (int(215 + (255 - 215) * t), int(48 + (255 - 48) * t), int(39 + (191 - 39) * t))
        t = (x - 0.5) / 0.5
        return (int(255 + (26 - 255) * t), int(255 + (152 - 255) * t), int(191 + (80 - 191) * t))

    draw.text((left, 20), title, fill=(0, 0, 0), font=font)
    for j, col in enumerate(pivot.columns):
        draw.text((left + j * cell_w + 46, top - 28), str(col), fill=(0, 0, 0), font=font)
    for i, idx in enumerate(pivot.index):
        draw.text((18, top + i * cell_h + 30), str(idx), fill=(0, 0, 0), font=font)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + cell_h - 2], fill=color_for(val), outline=(255, 255, 255))
            text = "NA" if not np.isfinite(val) else (f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}")
            draw.text((x0 + 46, y0 + 32), text, fill=(0, 0, 0), font=font)
    image.save(OUT_DIR / filename)


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
    leverage_engine.LIQUIDATION_THRESHOLDS_PCT[1] = -100.0

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

    trade_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []

    for gain_label, _, _ in GAIN_BUCKETS:
        for vr_label, _, _ in VR_BUCKETS:
            cell = f"{gain_label}|{vr_label}"
            cell_signals = scoped[scoped["gain_bucket"].eq(gain_label) & scoped["vr_bucket"].eq(vr_label)].copy()
            trades = simulate_cell(cell_signals, kline_map, common_end, cell)
            trade_frames.append(trades)
            row = {"gain_bucket": gain_label, "vr_bucket": vr_label, "cell": cell}
            row |= summarize(trades, len(cell_signals))
            row |= loser_density(trades)
            row["classification"] = classify(pd.Series(row))
            cell_rows.append(row)

            dist_rows.append({"gain_bucket": gain_label, "vr_bucket": vr_label, "cell": cell} | return_distribution(trades, len(cell_signals)))

            done = evaluated(trades)
            frow: dict[str, Any] = {"gain_bucket": gain_label, "vr_bucket": vr_label, "cell": cell, "trades": int(len(done))}
            for col in ["return_24h_pct", "return_48h_pct", "return_72h_pct", "return_6d_raw_pct", "mfe_24h_pct", "mfe_48h_pct", "mfe_72h_pct", "mae_24h_pct", "mae_48h_pct", "mae_72h_pct"]:
                frow[f"avg_{col}"] = float(pd.to_numeric(done.get(col, pd.Series(dtype=float)), errors="coerce").mean()) if len(done) else np.nan
                frow[f"median_{col}"] = float(pd.to_numeric(done.get(col, pd.Series(dtype=float)), errors="coerce").median()) if len(done) else np.nan
            forward_rows.append(frow)

            for month in pd.period_range("2026-01", ms_to_utc(common_end).strftime("%Y-%m"), freq="M").astype(str):
                m = done[done["month"].eq(month)] if len(done) else pd.DataFrame()
                pnl = pd.to_numeric(m.get("pnl_u", pd.Series(dtype=float)), errors="coerce")
                monthly_rows.append(
                    {
                        "gain_bucket": gain_label,
                        "vr_bucket": vr_label,
                        "cell": cell,
                        "month": month,
                        "is_incomplete_month": month == ms_to_utc(common_end).strftime("%Y-%m"),
                        "count": int(len(m)),
                        "pnl_u": float(pnl.sum()) if len(pnl) else 0.0,
                    }
                )

            for hour in HOURS_BJ:
                hsig = cell_signals[cell_signals["snapshot_hour_bj"].eq(hour)].copy()
                htrades = simulate_cell(hsig, kline_map, common_end, f"{cell}|{hour}")
                hrow = {"gain_bucket": gain_label, "vr_bucket": vr_label, "time_bj": hour}
                hrow |= summarize(htrades, len(hsig))
                time_rows.append(hrow)

    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    cell_summary = pd.DataFrame(cell_rows)
    dist = pd.DataFrame(dist_rows)
    forward = pd.DataFrame(forward_rows)
    monthly = pd.DataFrame(monthly_rows)
    time_split = pd.DataFrame(time_rows)

    gain_rows = []
    for gain_label, _, _ in GAIN_BUCKETS:
        gsignals = scoped[scoped["gain_bucket"].eq(gain_label)].copy()
        gtrades = simulate_cell(gsignals, kline_map, common_end, f"gain={gain_label}")
        gain_rows.append({"gain_bucket": gain_label} | summarize(gtrades, len(gsignals)))
    volume_rows = []
    for vr_label, _, _ in VR_BUCKETS:
        vsignals = scoped[scoped["vr_bucket"].eq(vr_label)].copy()
        vtrades = simulate_cell(vsignals, kline_map, common_end, f"vr={vr_label}")
        volume_rows.append({"vr_bucket": vr_label} | summarize(vtrades, len(vsignals)))

    breadth_rows = []
    for row in cell_summary.itertuples(index=False):
        sub = monthly[monthly["cell"].eq(row.cell)]
        complete = sub[~sub["is_incomplete_month"]]
        nonzero = complete[complete["count"] > 0]
        breadth_rows.append(
            {
                "gain_bucket": row.gain_bucket,
                "vr_bucket": row.vr_bucket,
                "cell": row.cell,
                "positive_months": int((nonzero["pnl_u"] > 0).sum()),
                "negative_months": int((nonzero["pnl_u"] < 0).sum()),
                "zero_or_no_signal_months": int((complete["count"] == 0).sum() + ((nonzero["pnl_u"] == 0).sum())),
                "best_month": str(sub.sort_values("pnl_u", ascending=False).iloc[0]["month"]) if len(sub) else "",
                "best_month_pnl_u": float(sub["pnl_u"].max()) if len(sub) else np.nan,
                "worst_month": str(sub.sort_values("pnl_u", ascending=True).iloc[0]["month"]) if len(sub) else "",
                "worst_month_pnl_u": float(sub["pnl_u"].min()) if len(sub) else np.nan,
            }
        )
    breadth = pd.DataFrame(breadth_rows)

    winners = []
    for (gain, vr, cell), group in evaluated(all_trades).groupby(["gain_bucket", "vr_bucket", "cell"], sort=True):
        pnl = pd.to_numeric(group["pnl_u"], errors="coerce")
        net = float(pnl.sum())
        for rank, (_, trade) in enumerate(group.assign(_pnl=pnl).sort_values("_pnl", ascending=False).head(3).iterrows(), start=1):
            winners.append(
                {
                    "gain_bucket": gain,
                    "vr_bucket": vr,
                    "cell": cell,
                    "winner_rank": rank,
                    "symbol": trade["symbol"],
                    "entry_time_utc": trade["entry_time_utc"],
                    "entry_time_bj": trade["entry_time_bj"],
                    "gain_24h": trade["gain_24h"],
                    "volume_24h_ratio_7d": trade["volume_24h_ratio_7d"],
                    "return_pct": trade["net_return_pct"],
                    "pnl_u": trade["pnl_u"],
                    "contribution_to_net_pct": float(trade["pnl_u"] / net * 100.0) if net > 0 else np.nan,
                }
            )

    neighbor_rows = []
    summary_by_cell = cell_summary.set_index(["gain_bucket", "vr_bucket"]).to_dict("index")
    gain_order = [b[0] for b in GAIN_BUCKETS]
    vr_order = [b[0] for b in VR_BUCKETS]
    for gain in gain_order:
        for vr in vr_order:
            for ngain, nvr, relation in [
                (gain_order[gain_order.index(gain) - 1], vr, "prev_gain") if gain_order.index(gain) > 0 else (None, None, ""),
                (gain_order[gain_order.index(gain) + 1], vr, "next_gain") if gain_order.index(gain) < len(gain_order) - 1 else (None, None, ""),
                (gain, vr_order[vr_order.index(vr) - 1], "prev_vr") if vr_order.index(vr) > 0 else (None, None, ""),
                (gain, vr_order[vr_order.index(vr) + 1], "next_vr") if vr_order.index(vr) < len(vr_order) - 1 else (None, None, ""),
            ]:
                if not ngain:
                    continue
                n = summary_by_cell.get((ngain, nvr), {})
                neighbor_rows.append(
                    {
                        "cell": f"{gain}|{vr}",
                        "neighbor_cell": f"{ngain}|{nvr}",
                        "relation": relation,
                        "neighbor_trades": n.get("trades", 0),
                        "neighbor_net_pnl_u": n.get("net_pnl_u", np.nan),
                        "neighbor_pf": n.get("pf", np.nan),
                        "neighbor_ex_top1_pnl_u": n.get("ex_top1_pnl_u", np.nan),
                        "neighbor_ex_top3_pnl_u": n.get("ex_top3_pnl_u", np.nan),
                        "neighbor_classification": n.get("classification", ""),
                    }
                )

    cell_summary.to_csv(OUT_DIR / "cell_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gain_rows).to_csv(OUT_DIR / "gain_bucket_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(volume_rows).to_csv(OUT_DIR / "volume_bucket_summary.csv", index=False, encoding="utf-8-sig")
    time_split.to_csv(OUT_DIR / "time_split_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT_DIR / "monthly_cell_summary.csv", index=False, encoding="utf-8-sig")
    all_trades.to_csv(OUT_DIR / "cell_trade_details.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(winners).to_csv(OUT_DIR / "cell_top_winners.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(OUT_DIR / "cell_return_distribution.csv", index=False, encoding="utf-8-sig")
    forward.to_csv(OUT_DIR / "cell_forward_path.csv", index=False, encoding="utf-8-sig")
    breadth.to_csv(OUT_DIR / "monthly_breadth.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(neighbor_rows).to_csv(OUT_DIR / "neighbor_stability.csv", index=False, encoding="utf-8-sig")
    scoped.to_csv(OUT_DIR / "raw_rank1_signals_scoped.csv", index=False, encoding="utf-8-sig")

    heatmap(cell_summary, "trades", "heatmap_trade_count.png", "Rank1 Trade Count")
    heatmap(cell_summary, "net_pnl_u", "heatmap_net_pnl.png", "Rank1 Net PnL 1x")
    heatmap(cell_summary, "pf", "heatmap_pf.png", "Rank1 PF 1x")
    heatmap(cell_summary, "median_return_pct", "heatmap_median_return.png", "Rank1 Median Return 1x")
    heatmap(cell_summary, "ex_top1_pnl_u", "heatmap_ex_top1.png", "Rank1 Ex Top1 PnL 1x")
    heatmap(cell_summary, "ex_top3_pnl_u", "heatmap_ex_top3.png", "Rank1 Ex Top3 PnL 1x")

    gain_summary = pd.DataFrame(gain_rows)
    volume_summary = pd.DataFrame(volume_rows)
    positive = cell_summary[cell_summary["classification"].eq("Positive Observation")]
    negative = cell_summary[cell_summary["classification"].eq("Negative Observation")]
    mixed = cell_summary[cell_summary["classification"].eq("Neutral / Mixed")]
    insufficient = cell_summary[cell_summary["classification"].eq("Insufficient")]
    ex_top1_pos = cell_summary[cell_summary["ex_top1_pnl_u"].gt(0)]
    ex_top3_pos = cell_summary[cell_summary["ex_top3_pnl_u"].gt(0)]

    lines = [
        "# Rank1 Gain x V/R Structure Judgment",
        "",
        "## 1. Scope",
        "",
        f"- Cutoff: {ms_to_utc(common_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        f"- Signal end: {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC.",
        "- Universe: Binance USD-M Futures local cache; Rank1 only; RAVEUSDT filtered; BTWUSDT excluded if cache gap remains; VELVETUSDT retained.",
        "- Time: Beijing 00:00 and 08:00 both retained.",
        "- Cells: Gain 20-40 / 40-60 x V/R 2-3 / 3-4 / 4-5 / 5-6.",
        "- V/R 1-2 is removed from this research scope.",
        "- Leverage: 1x only. Fee 0.1% per side, slippage 0, 6D default exit, current 4H/12H early exits, same-symbol lock.",
        "",
        "## 2. 8-Cell Matrix",
        "",
        markdown_table(cell_summary[
            [
                "gain_bucket",
                "vr_bucket",
                "signals",
                "trades",
                "net_pnl_u",
                "pf",
                "win_rate",
                "median_return_pct",
                "avg_return_pct",
                "max_dd_u",
                "ex_top1_pnl_u",
                "ex_top3_pnl_u",
                "classification",
                "sample_size",
            ]
        ].round(3)),
        "",
        "## 3. Gain Structure",
        "",
        markdown_table(gain_summary[["gain_bucket", "trades", "net_pnl_u", "pf", "win_rate", "median_return_pct", "ex_top1_pnl_u", "ex_top3_pnl_u"]].round(3)),
        "",
        "## 4. Volume Structure",
        "",
        markdown_table(volume_summary[["vr_bucket", "trades", "net_pnl_u", "pf", "win_rate", "median_return_pct", "ex_top1_pnl_u", "ex_top3_pnl_u"]].round(3)),
        "",
        "## 5. 2D Structure",
        "",
        f"- Positive Observation cells: {', '.join(positive['cell'].tolist()) if len(positive) else 'none'}.",
        f"- Neutral / Mixed cells: {', '.join(mixed['cell'].tolist()) if len(mixed) else 'none'}.",
        f"- Negative Observation cells: {', '.join(negative['cell'].tolist()) if len(negative) else 'none'}.",
        f"- Insufficient cells: {', '.join(insufficient['cell'].tolist()) if len(insufficient) else 'none'}.",
        "- The goal is structure observation only; no Rank1 cell is added to strategy.",
        "",
        "## 6. Tail Dependence",
        "",
        f"- exTop1 positive cells: {', '.join(ex_top1_pos['cell'].tolist()) if len(ex_top1_pos) else 'none'}.",
        f"- exTop3 positive cells: {', '.join(ex_top3_pos['cell'].tolist()) if len(ex_top3_pos) else 'none'}.",
        "- Top winner details are saved in `cell_top_winners.csv`.",
        "",
        "## 7. Time Split",
        "",
        "Time split is saved in `time_split_summary.csv`. This run only observes BJ00/BJ08 differences and does not create a time filter.",
        "",
        "## 8. Monthly Stability",
        "",
        "Monthly cell PnL is saved in `monthly_cell_summary.csv`; breadth is saved in `monthly_breadth.csv`.",
        "",
        "## 9. Forward Duration",
        "",
        "Forward 24H/48H/72H/6D path metrics are saved in `cell_forward_path.csv`.",
        "",
        "## 10. Final Judgment",
        "",
        "- Gain 60-80 is removed from this research scope because the prior structure pass marked it invalid.",
        "- Current Rank1 evidence is still dominated by sparse right-tail winners rather than a broad continuous positive region.",
        "- Continue Rank1 research only if the next step focuses on structure validation, not immediate production integration.",
    ]
    (OUT_DIR / "final_structure_judgment.md").write_text("\n".join(lines), encoding="utf-8")

    print("output", OUT_DIR)
    print("cutoff_utc", ms_to_utc(common_end).strftime("%Y-%m-%d %H:%M:%S"))
    print(cell_summary[["gain_bucket", "vr_bucket", "signals", "trades", "net_pnl_u", "pf", "win_rate", "median_return_pct", "ex_top1_pnl_u", "ex_top3_pnl_u", "classification", "sample_size"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
