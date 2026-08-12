from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def summarize_trades(trades: pd.DataFrame, parameter_set_id: str, params: dict) -> dict:
    returns = trades["net_return_pct"] if not trades.empty else pd.Series(dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else (float("inf") if wins.sum() > 0 else 0.0)
    return {
        "parameter_set_id": parameter_set_id,
        "strategy_variant": params.get("strategy_variant", "top10_immediate"),
        "lookback_days": params.get("lookback_days"),
        "interval": params.get("interval"),
        "min_quote_volume": params.get("min_quote_volume"),
        "sl_pct": params.get("sl_pct"),
        "tp1_pct": params.get("tp1_pct"),
        "max_holding_hours": params.get("max_holding_hours"),
        "total_signals": params.get("total_signals", 0),
        "total_trades": int(len(trades)),
        "win_rate": win_rate,
        "tp1_hit_rate": float(trades["tp1_hit"].mean()) if len(trades) else 0.0,
        "avg_return_pct": float(returns.mean()) if len(returns) else 0.0,
        "median_return_pct": float(returns.median()) if len(returns) else 0.0,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
        "expectancy_pct": win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss),
        "max_drawdown_pct": max_drawdown(returns),
        "sharpe_like": float(returns.mean() / returns.std(ddof=0) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0,
        "avg_mfe_pct": float(trades["mfe_pct"].mean()) if len(trades) else 0.0,
        "avg_mae_pct": float(trades["mae_pct"].mean()) if len(trades) else 0.0,
        "median_mfe_pct": float(trades["mfe_pct"].median()) if len(trades) else 0.0,
        "median_mae_pct": float(trades["mae_pct"].median()) if len(trades) else 0.0,
        "top10_to_top5_conversion_rate": float(trades["reached_top5_after_top10"].mean()) if len(trades) else 0.0,
        "avg_time_to_top5_minutes": float(trades["time_to_top5_minutes"].dropna().mean()) if len(trades) else 0.0,
        "local_top_entry_rate": float(trades["is_local_top_entry"].mean()) if len(trades) else 0.0,
        "btc_up_regime_return": float(trades.loc[trades["btc_1h_return_at_entry"] >= 0, "net_return_pct"].mean()) if len(trades) else 0.0,
        "btc_down_regime_return": float(trades.loc[trades["btc_1h_return_at_entry"] < 0, "net_return_pct"].mean()) if len(trades) else 0.0,
        "best_trade_pct": float(returns.max()) if len(returns) else 0.0,
        "worst_trade_pct": float(returns.min()) if len(returns) else 0.0,
    }
