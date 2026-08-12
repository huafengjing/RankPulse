from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = ROOT / "output" / "futures_top2_fixed_time_trades.csv"
KLINE_DIR = ROOT / "data" / "futures_klines_1h"
OUT_PATH = ROOT / "output" / "futures_top2_fixed_time_top10_profit_5x_liquidation.csv"


def to_ms(value: object) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def load_symbol_klines(symbol: str) -> pd.DataFrame:
    path = KLINE_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce").astype("Int64").astype("int64")
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("open_time").drop_duplicates("open_time")


def main() -> None:
    trades = pd.read_csv(TRADES_PATH)
    completed = trades[trades["status"].eq("completed")].copy()
    completed["pnl_u"] = pd.to_numeric(completed["pnl_u"], errors="coerce")
    completed["entry_price"] = pd.to_numeric(completed["entry_price"], errors="coerce")
    completed["exit_price"] = pd.to_numeric(completed["exit_price"], errors="coerce")
    completed["net_return_pct"] = pd.to_numeric(completed["net_return_pct"], errors="coerce")
    completed["gain_24h"] = pd.to_numeric(completed["gain_24h"], errors="coerce")
    top10 = completed.sort_values("pnl_u", ascending=False).head(10).reset_index(drop=True)

    rows = []
    for idx, trade in top10.iterrows():
        symbol = str(trade["symbol"])
        entry_ms = int(trade["entry_time_ms"]) if "entry_time_ms" in trade and pd.notna(trade["entry_time_ms"]) else to_ms(trade["entry_time_utc"])
        exit_ms = int(trade["exit_time_ms"]) if "exit_time_ms" in trade and pd.notna(trade["exit_time_ms"]) else to_ms(trade["exit_time_utc"])
        klines = load_symbol_klines(symbol)
        scoped = klines[(klines["open_time"] >= entry_ms) & (klines["open_time"] <= exit_ms)]
        entry_price = float(trade["entry_price"])
        liquidation_price = entry_price * 0.80
        min_low = float(scoped["low"].min()) if not scoped.empty else np.nan
        mae = min_low / entry_price - 1.0 if np.isfinite(min_low) and entry_price > 0 else np.nan
        rows.append(
            {
                "rank_by_pnl": idx + 1,
                "symbol": symbol,
                "entry_time_bj": trade["entry_time_bj"],
                "entry_price": entry_price,
                "exit_time_bj": trade["exit_time_bj"],
                "exit_price": float(trade["exit_price"]),
                "pnl_u": float(trade["pnl_u"]),
                "net_return_pct": float(trade["net_return_pct"]),
                "gain_24h": float(trade["gain_24h"]),
                "liquidation_price_5x": liquidation_price,
                "min_low_7d": min_low,
                "max_adverse_excursion_pct": mae,
                "would_liquidate_5x": bool(np.isfinite(min_low) and min_low <= liquidation_price),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    liq_count = int(result["would_liquidate_5x"].sum()) if not result.empty else 0
    no_liq_count = int(len(result) - liq_count)
    deepest = result.sort_values("max_adverse_excursion_pct", ascending=True).iloc[0] if not result.empty else None
    result["distance_to_liq_pct"] = result["min_low_7d"] / result["liquidation_price_5x"] - 1.0
    closest = result.sort_values("distance_to_liq_pct", ascending=True).iloc[0] if not result.empty else None

    print("========== Top10 Profit 5x Liquidation Check ==========")
    print(result.to_string(index=False))
    print("\n========== Summary ==========")
    print(f"Top10盈利单数量: {len(result)}")
    print(f"5x会爆仓数量: {liq_count}")
    print(f"5x不会爆仓数量: {no_liq_count}")
    if deepest is not None:
        print(
            "最大盘中回撤最深的一笔: "
            f"rank={int(deepest['rank_by_pnl'])}, symbol={deepest['symbol']}, "
            f"MAE={float(deepest['max_adverse_excursion_pct']) * 100:.2f}%"
        )
    if closest is not None:
        print(
            "最接近爆仓的一笔: "
            f"rank={int(closest['rank_by_pnl'])}, symbol={closest['symbol']}, "
            f"min_low={float(closest['min_low_7d']):.8g}, liq={float(closest['liquidation_price_5x']):.8g}, "
            f"distance={float(closest['distance_to_liq_pct']) * 100:.2f}%"
        )
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
