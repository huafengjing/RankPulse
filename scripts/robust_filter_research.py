from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SRC = Path("outputs/compression/signal_classification.csv")
KLINE_DIR = Path("data/raw/klines/5m")
OUT = Path("outputs/robust_filter")
CHARTS = OUT / "charts"
OBS_MS = 240 * 60 * 60_000
RAW_PLUS50 = 133 / 1071
RAW_PLUS100 = 57 / 1071
VALIDATION_TOL = 0.10


def load_base() -> pd.DataFrame:
    if not SRC.exists():
        raise RuntimeError("Run scripts/signal_compression_research.py first.")
    df = pd.read_csv(SRC)
    df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True)
    return df.sort_values("signal_time_utc").reset_index(drop=True)


def load_klines() -> pd.DataFrame:
    frames = []
    for p in KLINE_DIR.glob("*/*.parquet"):
        f = pd.read_parquet(p, columns=["symbol", "open_time", "open", "high", "low", "close"])
        f["_mtime"] = p.stat().st_mtime
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("_mtime").drop_duplicates(["symbol", "open_time"], keep="last").drop(columns="_mtime").sort_values(["symbol", "open_time"])


def regenerate_labels(base: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    kmap = {s: g.sort_values("open_time").reset_index(drop=True) for s, g in klines.groupby("symbol")}
    thresholds = [10, 20, 30, 40, 50, 100, 200]
    rows = []
    for _, s in base.iterrows():
        g = kmap.get(s["symbol"])
        idx = g.index[g["open_time"] > int(s["signal_time"])]
        if len(idx) == 0:
            continue
        entry = g.loc[int(idx[0])]
        entry_time = int(entry["open_time"])
        entry_price = float(entry["open"])
        path = g[(g["open_time"] >= entry_time) & (g["open_time"] <= entry_time + OBS_MS)]
        hit = {f"hit_plus_{t}_before_minus_10": False for t in thresholds}
        times = {f"time_to_plus_{t}_minutes": np.nan for t in [10, 20, 30, 50, 100]}
        hit_minus_first = False
        t_minus = np.nan
        max_ret = float(path["high"].max() / entry_price - 1)
        min_ret = float(path["low"].min() / entry_price - 1)
        for _, bar in path.iterrows():
            mins = (int(bar["open_time"]) - entry_time) / 60_000
            if float(bar["low"]) / entry_price - 1 <= -0.10:
                hit_minus_first = not hit["hit_plus_10_before_minus_10"]
                t_minus = mins
                break
            high_ret = float(bar["high"]) / entry_price - 1
            for t in thresholds:
                if high_ret >= t / 100:
                    hit[f"hit_plus_{t}_before_minus_10"] = True
                    key = f"time_to_plus_{t}_minutes"
                    if key in times and pd.isna(times[key]):
                        times[key] = mins
        rows.append(
            {
                "signal_id": int(s["signal_id"]),
                "symbol": s["symbol"],
                "signal_time_utc": s["signal_time_utc"],
                "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
                "entry_price": entry_price,
                **hit,
                "hit_minus_10_first": hit_minus_first,
                "max_forward_return_240h": max_ret,
                "min_forward_return_240h": min_ret,
                **times,
                "time_to_minus_10_minutes": t_minus,
            }
        )
    return pd.DataFrame(rows)


def split_train_val(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    start = df["signal_time_utc"].min()
    split = start + pd.Timedelta(days=30)
    return df["signal_time_utc"] < split, df["signal_time_utc"] >= split, split


def feature_table(base: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "return_15m": "return_15m_before_signal",
        "return_1h": "return_1h_before_signal",
        "return_4h": "return_4h_before_signal",
        "return_24h": "return_24h_before_signal",
        "top10_entry_speed": "top20_to_top10_time_minutes",
    }
    allowed = [
        "signal_id", "symbol", "signal_time_utc", "rank_at_signal", "rolling_24h_gain_at_signal",
        "quote_volume_at_signal", "market_age_days", "rank_bucket", "rolling_24h_gain_bucket",
        "rank_improvement_last_1h", "rank_improvement_last_4h", "top10_entry_speed",
        "volume_5m_at_signal", "volume_1h_at_signal", "volume_4h_at_signal", "volume_5m_vs_24h_avg",
        "volume_1h_vs_24h_avg", "volume_4h_vs_7d_avg", "volume_acceleration_1h", "volume_acceleration_4h",
        "quote_volume_bucket", "return_15m", "return_1h", "return_4h", "return_24h",
        "distance_to_5m_ema20_pct", "distance_to_15m_ema20_pct", "distance_to_1h_ema20_pct",
        "distance_to_vwap_pct", "candle_body_pct_at_signal", "upper_wick_pct_at_signal",
        "lower_wick_pct_at_signal", "consecutive_green_5m_count", "consecutive_green_15m_count",
        "volatility_1h", "volatility_4h", "btc_return_15m", "btc_return_1h", "btc_return_4h",
        "btc_return_24h", "btc_above_1h_ema20", "btc_above_4h_ema20", "btc_regime_label",
    ]
    f = base[[c for c in allowed if c in base.columns]].rename(columns=rename).copy()
    f = f.merge(labels[["signal_id", "entry_time_utc"]], on="signal_id", how="left")
    f["symbol_age_bucket"] = pd.cut(f["market_age_days"], [-1, 7, 30, 90, np.inf], labels=["<7d", "7-30d", "30-90d", ">90d"])
    f["gain_bucket"] = pd.cut(f["rolling_24h_gain_at_signal"], [0, .1, .15, .2, .3, .5, .8, np.inf], labels=["0-10", "10-15", "15-20", "20-30", "30-50", "50-80", ">80"], include_lowest=True)
    f["distance_to_5m_ema20_bucket"] = pd.cut(f["distance_to_5m_ema20_pct"], [-np.inf, .02, .05, .08, .12, np.inf], labels=["<=2", "2-5", "5-8", "8-12", ">12"])
    f["distance_to_vwap_bucket"] = pd.cut(f["distance_to_vwap_pct"], [-np.inf, .02, .05, .08, .12, np.inf], labels=["<=2", "2-5", "5-8", "8-12", ">12"])
    f["return_1h_bucket"] = pd.cut(f["return_1h_before_signal"], [-np.inf, 0, .03, .06, .10, np.inf], labels=["<0", "0-3", "3-6", "6-10", ">10"])
    f["return_4h_bucket"] = pd.cut(f["return_4h_before_signal"], [-np.inf, 0, .05, .10, .20, np.inf], labels=["<0", "0-5", "5-10", "10-20", ">20"])
    f["upper_wick_bucket"] = pd.cut(f["upper_wick_pct_at_signal"], [-np.inf, .1, .25, .5, np.inf], labels=["<=10", "10-25", "25-50", ">50"])
    f["volume_acceleration_1h_bucket"] = pd.cut(f["volume_acceleration_1h"], [-np.inf, 1, 1.5, 2, 3, np.inf], labels=["<=1", "1-1.5", "1.5-2", "2-3", ">3"])
    f["volume_acceleration_4h_bucket"] = pd.cut(f["volume_acceleration_4h"], [-np.inf, 1, 1.5, 2, 3, np.inf], labels=["<=1", "1-1.5", "1.5-2", "2-3", ">3"])
    return f


def metric_row(name: str, conditions: str, df: pd.DataFrame, mask: pd.Series, train: pd.Series, val: pd.Series, dims: int) -> dict:
    g = df[mask.fillna(False)]
    gt = df[mask.fillna(False) & train]
    gv = df[mask.fillna(False) & val]

    def rate(x, col):
        return float(x[col].mean()) if len(x) else np.nan

    train50 = rate(gt, "hit_plus_50_before_minus_10")
    val50 = rate(gv, "hit_plus_50_before_minus_10")
    sample = len(g) >= 50 and len(gt) >= 30 and len(gv) >= 30
    stable = sample and abs(val50 - train50) <= VALIDATION_TOL
    outperf = sample and train50 >= RAW_PLUS50 * 1.5 and val50 >= RAW_PLUS50 * 1.3
    compression = 50 <= len(g) <= 200
    return {
        "rule_name": name,
        "conditions": conditions,
        "condition_count": dims,
        "selected_count_total": len(g),
        "selected_count_train": len(gt),
        "selected_count_validation": len(gv),
        "selected_pct_total": len(g) / len(df),
        "train_plus10_rate": rate(gt, "hit_plus_10_before_minus_10"),
        "val_plus10_rate": rate(gv, "hit_plus_10_before_minus_10"),
        "train_plus30_rate": rate(gt, "hit_plus_30_before_minus_10"),
        "val_plus30_rate": rate(gv, "hit_plus_30_before_minus_10"),
        "train_plus50_rate": train50,
        "val_plus50_rate": val50,
        "train_plus100_rate": rate(gt, "hit_plus_100_before_minus_10"),
        "val_plus100_rate": rate(gv, "hit_plus_100_before_minus_10"),
        "train_minus10_first_rate": rate(gt, "hit_minus_10_first"),
        "val_minus10_first_rate": rate(gv, "hit_minus_10_first"),
        "train_plus50_enrichment": train50 / RAW_PLUS50 if pd.notna(train50) else np.nan,
        "val_plus50_enrichment": val50 / RAW_PLUS50 if pd.notna(val50) else np.nan,
        "train_plus100_enrichment": rate(gt, "hit_plus_100_before_minus_10") / RAW_PLUS100 if len(gt) else np.nan,
        "val_plus100_enrichment": rate(gv, "hit_plus_100_before_minus_10") / RAW_PLUS100 if len(gv) else np.nan,
        "sample_size_pass": sample,
        "stability_pass": stable,
        "compression_pass": compression,
        "baseline_outperformance_pass": outperf,
        "overall_pass": sample and stable and compression and outperf,
    }


def single_dimension(df: pd.DataFrame, train: pd.Series, val: pd.Series) -> pd.DataFrame:
    features = [
        "rank_bucket", "gain_bucket", "quote_volume_bucket", "distance_to_5m_ema20_bucket",
        "distance_to_vwap_bucket", "return_1h_bucket", "return_4h_bucket", "upper_wick_bucket",
        "volume_acceleration_1h_bucket", "volume_acceleration_4h_bucket", "btc_regime_label", "symbol_age_bucket",
    ]
    rows = []
    for feat in features:
        for bucket in df[feat].dropna().astype(str).unique():
            mask = df[feat].astype(str).eq(bucket)
            row = metric_row(f"{feat}={bucket}", f"{feat} == {bucket}", df, mask, train, val, 1)
            row["feature_name"] = feat
            row["bucket"] = bucket
            row["total_count"] = row["selected_count_total"]
            row["train_count"] = row["selected_count_train"]
            row["validation_count"] = row["selected_count_validation"]
            row["conclusion"] = "valid" if row["sample_size_pass"] and row["stability_pass"] and row["baseline_outperformance_pass"] else "invalid"
            rows.append(row)
    return pd.DataFrame(rows)


def candidate_atoms(single: pd.DataFrame, max_atoms: int = 10) -> list[tuple[str, str]]:
    s = single[
        (single["selected_count_total"] >= 80)
        & (single["selected_count_total"] <= 800)
        & (single["selected_count_train"] >= 30)
        & (single["selected_count_validation"] >= 30)
    ].copy()
    s["rank_score"] = s["train_plus50_enrichment"].fillna(0) + s["val_plus50_enrichment"].fillna(0) - (s["train_minus10_first_rate"].fillna(1) + s["val_minus10_first_rate"].fillna(1)) * .25
    s = s.sort_values("rank_score", ascending=False)
    atoms = []
    used_features = set()
    for _, r in s.iterrows():
        if r["feature_name"] in used_features and len(atoms) >= 6:
            continue
        atoms.append((r["feature_name"], r["bucket"]))
        used_features.add(r["feature_name"])
        if len(atoms) >= max_atoms:
            break
    return atoms


def combo_rules(df: pd.DataFrame, train: pd.Series, val: pd.Series, atoms: list[tuple[str, str]], dims: int) -> pd.DataFrame:
    rows = []
    for combo in combinations(atoms, dims):
        features = [x[0] for x in combo]
        if len(set(features)) < len(features):
            continue
        mask = pd.Series(True, index=df.index)
        parts = []
        for feat, bucket in combo:
            mask &= df[feat].astype(str).eq(str(bucket))
            parts.append(f"{feat} == {bucket}")
        rows.append(metric_row(" AND ".join([f"{f}:{b}" for f, b in combo]), " AND ".join(parts), df, mask, train, val, dims))
    return pd.DataFrame(rows).sort_values(["overall_pass", "val_plus50_enrichment", "selected_count_total"], ascending=[False, False, False]) if rows else pd.DataFrame()


def simple_score(df: pd.DataFrame, train: pd.Series, val: pd.Series) -> pd.DataFrame:
    score = pd.Series(0, index=df.index, dtype=float)
    score += np.where(df["rank_at_signal"].between(6, 10), 25, np.where(df["rank_at_signal"].between(4, 5), 15, 5))
    g = df["rolling_24h_gain_at_signal"]
    score += np.select([g.between(.15, .30), g.between(.30, .50), g.between(.50, .80), g < .15], [25, 20, 10, 5], 0)
    score += np.select([df["quote_volume_at_signal"] > 300e6, df["quote_volume_at_signal"] > 100e6, df["quote_volume_at_signal"] > 20e6], [20, 15, 8], 0)
    d = df["distance_to_5m_ema20_pct"].fillna(9)
    score += np.select([d <= .02, d <= .05, d <= .08, d > .12], [20, 15, 10, -10], 0)
    tmp = df.copy()
    tmp["simple_score"] = score
    rows = []
    for pct in [10, 15, 20, 25]:
        cutoff = tmp["simple_score"].quantile(1 - pct / 100)
        rows.append(metric_row(f"simple_score_top_{pct}pct", f"simple_score >= {cutoff:.2f}", tmp, tmp["simple_score"] >= cutoff, train, val, 1))
    return pd.DataFrame(rows)


def backtest_rule(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in selected.iterrows():
        net = 0.0
        remaining = 1.0
        breakeven = False
        for pct, frac in [(10, .4), (20, .2), (30, .2)]:
            if bool(r[f"hit_plus_{pct}_before_minus_10"]):
                exit_price = 1 + pct / 100
                net += frac * ((exit_price * (1 - .0005)) / (1 * (1 + .0005)) - 1 - .001)
                remaining -= frac
                if pct == 10:
                    breakeven = True
        if bool(r["hit_minus_10_first"]):
            exit_ret = 0 if breakeven else -.10
            net += remaining * (((1 + exit_ret) * (1 - .0005)) / (1 * (1 + .0005)) - 1 - .001)
        else:
            exit_ret = min(r["max_forward_return_240h"], .30) if breakeven else r["min_forward_return_240h"]
            net += remaining * (((1 + exit_ret) * (1 - .0005)) / (1 * (1 + .0005)) - 1 - .001)
        rows.append({"signal_id": r["signal_id"], "net_return_pct": net, "split": r["split"]})
    return pd.DataFrame(rows)


def backtest_summaries(df: pd.DataFrame, rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trades = []
    rows = []
    for _, rule in rules.iterrows():
        mask = eval_condition(df, rule["conditions"])
        sel = df[mask].copy()
        tr = backtest_rule(df, sel)
        tr["rule_name"] = rule["rule_name"]
        all_trades.append(tr)

        def stat(x):
            ret = x["net_return_pct"]
            wins = ret[ret > 0]; losses = ret[ret <= 0]
            eq = (1 + ret).cumprod()
            return {
                "avg": ret.mean(), "median": ret.median(), "win": (ret > 0).mean(),
                "pf": wins.sum() / abs(losses.sum()) if abs(losses.sum()) else np.inf,
                "dd": (eq / eq.cummax() - 1).min(),
            }
        total = stat(tr)
        train = stat(tr[tr["split"].eq("train")])
        val = stat(tr[tr["split"].eq("validation")])
        sorted_ret = tr["net_return_pct"].sort_values(ascending=False)
        rows.append({
            "rule_name": rule["rule_name"],
            "selected_count_total": len(sel),
            "selected_count_train": int(sel["split"].eq("train").sum()),
            "selected_count_validation": int(sel["split"].eq("validation").sum()),
            "avg_net_return_total": total["avg"], "avg_net_return_train": train["avg"], "avg_net_return_validation": val["avg"],
            "median_net_return_total": total["median"], "median_net_return_train": train["median"], "median_net_return_validation": val["median"],
            "win_rate_total": total["win"], "win_rate_train": train["win"], "win_rate_validation": val["win"],
            "profit_factor_total": total["pf"], "profit_factor_train": train["pf"], "profit_factor_validation": val["pf"],
            "expectancy_total": total["avg"], "expectancy_train": train["avg"], "expectancy_validation": val["avg"],
            "max_drawdown_total": total["dd"], "max_drawdown_train": train["dd"], "max_drawdown_validation": val["dd"],
            "result_excluding_best_1_trade": sorted_ret.iloc[1:].mean() if len(sorted_ret) > 1 else np.nan,
            "result_excluding_best_5_trades": sorted_ret.iloc[5:].mean() if len(sorted_ret) > 5 else np.nan,
            "result_excluding_best_10_trades": sorted_ret.iloc[10:].mean() if len(sorted_ret) > 10 else np.nan,
            "is_tail_dependent": total["avg"] > 0 and sorted_ret.iloc[5:].mean() < 0,
        })
    return pd.concat(all_trades, ignore_index=True), pd.DataFrame(rows)


def eval_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    if condition.startswith("simple_score"):
        raise ValueError("simple_score condition not supported in final backtest eval")
    mask = pd.Series(True, index=df.index)
    for part in condition.split(" AND "):
        feat, val = [x.strip() for x in part.split("==")]
        mask &= df[feat].astype(str).eq(val)
    return mask


def charts(single: pd.DataFrame, rules: pd.DataFrame, final_bt: pd.DataFrame) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    pd.Series({"raw +50": RAW_PLUS50, "best val +50": rules["val_plus50_rate"].max()}).plot(kind="bar", title="Raw vs Filtered +50")
    plt.tight_layout(); plt.savefig(CHARTS / "raw_vs_filtered_hit_rates.png"); plt.close()
    rules.head(20).plot(x="rule_name", y=["train_plus50_rate", "val_plus50_rate"], kind="bar", title="Train vs Validation +50")
    plt.tight_layout(); plt.savefig(CHARTS / "train_vs_validation_plus50_rate.png"); plt.close()
    rules.plot.scatter("selected_count_total", "val_plus50_rate", title="Selected Count vs Val +50")
    plt.tight_layout(); plt.savefig(CHARTS / "rule_selected_count_vs_plus50_rate.png"); plt.close()
    pivot = single.pivot_table(index="feature_name", values="val_plus50_enrichment", aggfunc="max")
    plt.imshow(pivot.fillna(0), aspect="auto"); plt.yticks(range(len(pivot.index)), pivot.index); plt.colorbar(); plt.title("Single Dimension Enrichment")
    plt.tight_layout(); plt.savefig(CHARTS / "single_dimension_enrichment_heatmap.png"); plt.close()
    rules["stability_gap"] = (rules["train_plus50_rate"] - rules["val_plus50_rate"]).abs()
    rules.head(30).plot(x="rule_name", y="stability_gap", kind="bar", title="Candidate Stability Gap")
    plt.tight_layout(); plt.savefig(CHARTS / "candidate_rules_validation_stability.png"); plt.close()
    if not final_bt.empty:
        tr = pd.read_csv(OUT / "final_rule_backtest.csv")
        first = tr["rule_name"].iloc[0]
        eq = (1 + tr[tr["rule_name"].eq(first)]["net_return_pct"]).cumprod()
        eq.plot(title="Final Rule Equity Curve"); plt.tight_layout(); plt.savefig(CHARTS / "final_rule_equity_curve.png"); plt.close()
        tr[tr["rule_name"].eq(first)]["net_return_pct"].hist(bins=30); plt.title("Final Rule Return Distribution"); plt.tight_layout(); plt.savefig(CHARTS / "final_rule_return_distribution.png"); plt.close()
        final_bt.set_index("rule_name")[["avg_net_return_train", "avg_net_return_validation"]].plot(kind="bar", title="Train vs Validation Backtest")
        plt.tight_layout(); plt.savefig(CHARTS / "final_rule_train_vs_validation.png"); plt.close()


def write_report(single, two, multi, score, candidates, bt, split_time):
    passes = candidates[candidates["overall_pass"]]
    final = passes.iloc[0] if not passes.empty else None
    closest = candidates.head(3)
    if final is None:
        rec = "No robust rule found"
        conds = "No rule met all hard constraints."
    else:
        rec = final["rule_name"]
        conds = final["conditions"]
    text = f"""# Robust Filter Report

## Executive Summary

Final recommendation: {rec}

Conditions: {conds}

If no robust rule is found, do not force a filter. The closest candidates are listed below.

## Raw Baseline

- raw_total_signals: 1071
- raw_plus10_rate: 50.14%
- raw_plus20_rate: 32.31%
- raw_plus30_rate: 22.22%
- raw_plus50_rate: 12.42%
- raw_plus100_rate: 5.32%

## Anti-Overfitting Rules

Minimum selected total 50, train 30, validation 30. Validation +50 must be within train +/- 10 percentage points and must beat raw +50 by 1.3x.

## Train / Validation Split

Time split: signal_time < {split_time} is train; later signals are validation.

## Single Dimension Findings

See `single_dimension_analysis.csv`. Valid rows are those with `conclusion=valid`.

## Invalidated Conditions

Any condition with sample size failure, validation instability, or insufficient validation enrichment is invalidated.

## Two-Dimension Rule Results

See `two_dimension_rules.csv`.

## Multi-Dimension Rule Results

See `multi_dimension_rules.csv`.

## Final Recommended Rule

{rec}

Recommended filter conditions:

{conds}

## Final Rule Validation Table

Closest candidates:

{closest[['rule_name','conditions','selected_count_total','selected_count_train','selected_count_validation','train_plus50_rate','val_plus50_rate','train_plus50_enrichment','val_plus50_enrichment','overall_pass']].to_string(index=False)}

## Backtest Result of Final Rule

See `final_rule_backtest_summary.csv`.

{bt.to_string(index=False) if not bt.empty else 'No backtest candidates.'}

## Tail Dependency Analysis

Check `result_excluding_best_1_trade`, `result_excluding_best_5_trades`, and `is_tail_dependent` in the backtest summary.

## Robustness Conclusion

If `overall_pass` is false for all candidates, no stable immediately tradable compression rule was found.

## Next Step Recommendation

If no robust rule is found, shift from signal-time filtering to post-signal confirmation such as Top5 continuation, pullback-hold behavior, and volume persistence. Do not develop live trading.
"""
    (OUT / "robust_filter_report.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading base/features...", flush=True)
    base = load_base()
    klines = load_klines()
    labels = regenerate_labels(base, klines)
    labels.to_csv(OUT / "signal_labels.csv", index=False)
    features = feature_table(base, labels)
    features.to_csv(OUT / "signal_features.csv", index=False)
    df = features.merge(labels, on=["signal_id", "symbol", "signal_time_utc", "entry_time_utc"], how="inner")
    train, val, split_time = split_train_val(df)
    df["split"] = np.where(train, "train", "validation")
    print("Analyzing rules...", flush=True)
    single = single_dimension(df, train, val)
    single.to_csv(OUT / "single_dimension_analysis.csv", index=False)
    atoms = candidate_atoms(single, 10)
    two = combo_rules(df, train, val, atoms, 2)
    two.to_csv(OUT / "two_dimension_rules.csv", index=False)
    three = combo_rules(df, train, val, atoms, 3)
    four = combo_rules(df, train, val, atoms, 4)
    multi = pd.concat([three, four], ignore_index=True) if not three.empty or not four.empty else pd.DataFrame()
    multi.to_csv(OUT / "multi_dimension_rules.csv", index=False)
    score = simple_score(df, train, val)
    score.to_csv(OUT / "simple_score_rules.csv", index=False)
    candidates = pd.concat([two, multi, score], ignore_index=True)
    candidates = candidates.sort_values(["overall_pass", "val_plus50_enrichment", "selected_count_total"], ascending=[False, False, False])
    # Final backtest supports condition strings, so skip simple_score rows for detailed trade export.
    bt_candidates = candidates[~candidates["conditions"].str.startswith("simple_score")].head(4)
    trades, bt = backtest_summaries(df, bt_candidates)
    trades.to_csv(OUT / "final_rule_backtest.csv", index=False)
    bt.to_csv(OUT / "final_rule_backtest_summary.csv", index=False)
    charts(single, candidates, bt)
    write_report(single, two, multi, score, candidates, bt, split_time)
    print("Done. Outputs written to outputs/robust_filter", flush=True)
    print(candidates.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
