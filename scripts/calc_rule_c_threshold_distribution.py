from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "long_consolidation_top10_20_30"
KLINE_ROOT = ROOT / "data" / "raw" / "klines" / "5m"
MAX_HOLD_HOURS = 240
THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00]


def load_symbol_klines(symbol: str) -> pd.DataFrame:
    files = sorted((KLINE_ROOT / symbol).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return df


def hit_before_minus_10(path: pd.DataFrame, entry_price: float, threshold: float) -> bool:
    liq_price = entry_price * 0.90
    target_price = entry_price * (1.0 + threshold)
    for row in path.itertuples(index=False):
        low = float(row.low)
        high = float(row.high)
        if low <= liq_price:
            return False
        if high >= target_price:
            return True
    return False


def minus_10_first(path: pd.DataFrame, entry_price: float) -> bool:
    liq_price = entry_price * 0.90
    for row in path.itertuples(index=False):
        if float(row.low) <= liq_price:
            return True
    return False


def main() -> None:
    features = pd.read_csv(OUT / "signal_features.csv")
    trades = pd.read_csv(OUT / "trades.csv")
    rule_c_ids = set(features.loc[features["rule_c_21d_pass"] == True, "signal_id"].astype(int))
    selected = trades[trades["signal_id"].astype(int).isin(rule_c_ids)].copy()

    symbol_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for trade in selected.itertuples(index=False):
        symbol = str(trade.symbol)
        if symbol not in symbol_cache:
            symbol_cache[symbol] = load_symbol_klines(symbol)
        klines = symbol_cache[symbol]
        entry_time = int(pd.Timestamp(trade.entry_time_utc).timestamp() * 1000)
        entry_price = float(trade.entry_price)
        end_time = entry_time + MAX_HOLD_HOURS * 60 * 60 * 1000
        path = klines[(klines["open_time"] >= entry_time) & (klines["open_time"] <= end_time)]
        if path.empty:
            continue

        row = {
            "signal_id": int(trade.signal_id),
            "symbol": symbol,
            "entry_time_utc": trade.entry_time_utc,
            "entry_price": entry_price,
            "minus10_first": minus_10_first(path, entry_price),
        }
        for threshold in THRESHOLDS:
            row[f"hit_plus_{int(threshold * 100)}_before_minus10"] = hit_before_minus_10(
                path, entry_price, threshold
            )
        rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "rule_c_threshold_detail.csv", index=False)

    total = len(detail)
    not_minus10 = int((~detail["minus10_first"]).sum())
    summary_rows = []
    for threshold in THRESHOLDS:
        col = f"hit_plus_{int(threshold * 100)}_before_minus10"
        count = int(detail[col].sum())
        summary_rows.append(
            {
                "threshold": f"+{int(threshold * 100)}%",
                "hit_count": count,
                "pct_of_185": count / total if total else 0.0,
                "pct_of_not_minus10_first": count / not_minus10 if not_minus10 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "rule_c_threshold_distribution.csv", index=False)

    print(f"total={total} not_minus10_first={not_minus10} minus10_first={total - not_minus10}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
