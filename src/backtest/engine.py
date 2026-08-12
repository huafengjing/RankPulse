from __future__ import annotations

import pandas as pd

from src.backtest.execution import apply_long_entry_slippage, apply_long_exit_slippage, net_return_pct


def run_top10_immediate_backtest(
    signals: pd.DataFrame,
    klines: pd.DataFrame,
    rankings: pd.DataFrame,
    parameter_set_id: str,
    strategy_variant: str = "top10_immediate",
    interval_minutes: int = 5,
    tp1_pct: float = 0.10,
    sl_pct: float = -0.05,
    max_holding_hours: int = 24,
    taker_fee_rate: float = 0.0005,
    slippage_rate: float = 0.0005,
) -> pd.DataFrame:
    klines = klines.drop_duplicates(["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    rankings = rankings.drop_duplicates(["open_time", "symbol"]).sort_values(["open_time", "symbol"])
    kline_map = {symbol: group.sort_values("open_time").reset_index(drop=True) for symbol, group in klines.groupby("symbol")}
    ranking_lookup = rankings.set_index(["open_time", "symbol"]).sort_index()
    rows = []
    trade_id = 1
    max_bars = int(max_holding_hours * 60 / interval_minutes)

    for _, signal in signals[signals["eligible_for_entry"]].sort_values("signal_time").iterrows():
        symbol = signal["symbol"]
        if symbol not in kline_map:
            continue
        group = kline_map[symbol]
        future_idx = group.index[group["open_time"] > int(signal["signal_time"])]
        if len(future_idx) == 0:
            continue
        entry_idx = int(future_idx[0])
        entry = group.loc[entry_idx]
        entry_price_raw = float(entry["open"])
        entry_price_effective = apply_long_entry_slippage(entry_price_raw, slippage_rate)
        tp_raw = entry_price_raw * (1.0 + tp1_pct)
        sl_raw = entry_price_raw * (1.0 + sl_pct)
        path = group.iloc[entry_idx : entry_idx + max_bars + 1]
        if path.empty:
            continue

        exit_row = path.iloc[-1]
        exit_price_raw = float(exit_row["close"])
        exit_reason = "max_holding_time"
        tp1_hit = False
        sl_hit = False
        for _, bar in path.iterrows():
            # Conservative same-candle rule: stop loss wins if TP and SL both print in one candle.
            if float(bar["low"]) <= sl_raw:
                exit_row = bar
                exit_price_raw = sl_raw
                exit_reason = "sl"
                sl_hit = True
                break
            if float(bar["high"]) >= tp_raw:
                exit_row = bar
                exit_price_raw = tp_raw
                exit_reason = "tp1"
                tp1_hit = True
                break

        exit_price_effective = apply_long_exit_slippage(exit_price_raw, slippage_rate)
        net_ret, fee_paid = net_return_pct(entry_price_effective, exit_price_effective, taker_fee_rate)
        gross_ret = exit_price_raw / entry_price_raw - 1.0
        mfe = float(path["high"].max() / entry_price_raw - 1.0)
        mae = float(path["low"].min() / entry_price_raw - 1.0)
        entry_rank = ranking_lookup.loc[(int(entry["open_time"]), symbol)] if (int(entry["open_time"]), symbol) in ranking_lookup.index else None
        if isinstance(entry_rank, pd.DataFrame):
            entry_rank = entry_rank.iloc[0]
        rows.append(
            {
                "trade_id": trade_id,
                "strategy_variant": strategy_variant,
                "symbol": symbol,
                "signal_time_utc": signal["signal_time_utc"],
                "entry_time_utc": entry["open_time_utc"],
                "exit_time_utc": exit_row["open_time_utc"],
                "entry_price_raw": entry_price_raw,
                "entry_price_effective": entry_price_effective,
                "exit_price_raw": exit_price_raw,
                "exit_price_effective": exit_price_effective,
                "rank_at_signal": int(signal["rank"]),
                "rank_at_entry": int(entry_rank["rank"]) if entry_rank is not None else None,
                "rolling_24h_change_pct_at_signal": float(signal["rolling_24h_change_pct"]),
                "rolling_24h_change_pct_at_entry": float(entry_rank["rolling_24h_change_pct"]) if entry_rank is not None else None,
                "quote_volume_at_signal": float(signal.get("quote_volume", 0)),
                "top10_first_entry": bool(signal["is_first_top10"]),
                "reached_top5_after_top10": bool(signal["entered_top5_later"]),
                "time_to_top5_minutes": signal["time_to_top5_minutes"],
                "tp1_hit": tp1_hit,
                "sl_hit": sl_hit,
                "exit_reason": exit_reason,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "gross_return_pct": gross_ret,
                "net_return_pct": net_ret,
                "fee_paid_pct": fee_paid,
                "slippage_paid_pct": slippage_rate * 2,
                "holding_minutes": int((int(exit_row["open_time"]) - int(entry["open_time"])) / 60_000),
                "btc_1h_return_at_entry": 0.0,
                "btc_4h_return_at_entry": 0.0,
                "ema20_at_entry": entry.get("ema20"),
                "vwap_at_entry": entry.get("vwap"),
                "distance_to_ema20_pct": entry.get("distance_to_ema20_pct"),
                "distance_to_vwap_pct": entry.get("distance_to_vwap_pct"),
                "is_local_top_entry": bool(mae < sl_pct / 2),
                "parameter_set_id": parameter_set_id,
            }
        )
        trade_id += 1
    return pd.DataFrame(rows)
