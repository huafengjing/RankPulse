from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_drop_top3_short_edge import DAY_MS, HOUR_MS, max_drawdown, profit_factor
from scripts.research_losers_rank10_extension import (
    apply_position_conflict,
    block_bootstrap,
    common_entry_events,
    complete_months,
    precompute_outcomes,
    summarize_trades,
)


def synthetic_frame(start: int, multiplier: float = 1.0) -> pd.DataFrame:
    rows = []
    for hour in range(25 * 8):
        price = 100 * multiplier * (1 - hour * 0.0005)
        rows.append({"open_time": start + hour * HOUR_MS, "open": price, "high": price * 1.01, "low": price * 0.99, "close": price})
    return pd.DataFrame(rows).set_index("open_time", drop=False)


def outcome_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_time_ms": 0,
                "snapshot_time_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                "snapshot_hour_bj": "08:00",
                "symbol": "AUSDT",
                "rank": 1,
                "entry_time_ms": 0,
                "entry_time_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                "exit_time_ms": DAY_MS,
                "holding_days": 1,
                "net_return_pct": 10.0,
                "pnl_usdt_at_100": 10.0,
                "fees_usdt_at_100": 0.2,
                "liquidated": False,
                "mfe_pct": 12.0,
                "mae_pct": -2.0,
            },
            {
                "snapshot_time_ms": HOUR_MS,
                "snapshot_time_utc": pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                "snapshot_hour_bj": "09:00",
                "symbol": "AUSDT",
                "rank": 1,
                "entry_time_ms": HOUR_MS,
                "entry_time_utc": pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                "exit_time_ms": DAY_MS + HOUR_MS,
                "holding_days": 1,
                "net_return_pct": 5.0,
                "pnl_usdt_at_100": 5.0,
                "fees_usdt_at_100": 0.2,
                "liquidated": False,
                "mfe_pct": 6.0,
                "mae_pct": -1.0,
            },
        ]
    )


def test_position_conflict_and_fixed_capital_allocation() -> None:
    trades = apply_position_conflict(outcome_rows(), {1}, 1, {1: 300.0}, "Top1", "fixed_snapshot_capital")
    assert trades.notional_usdt.tolist() == [300.0, 300.0]
    assert trades.skipped_due_to_existing_position.tolist() == [False, True]
    assert trades.iloc[0].pnl_usdt == pytest.approx(30.0)


def test_common_event_requires_leave_before_reentry() -> None:
    signals = pd.DataFrame(
        [
            {"snapshot_time_ms": 0, "symbol": "A", "rank": 3},
            {"snapshot_time_ms": 1, "symbol": "A", "rank": 3},
            {"snapshot_time_ms": 2, "symbol": "A", "rank": 4},
            {"snapshot_time_ms": 3, "symbol": "A", "rank": 3},
        ]
    )
    events = common_entry_events(signals, 3, 3, 0)
    assert events.snapshot_time_ms.tolist() == [0, 3]


def test_precomputed_entry_and_fixed_exit_use_hourly_open() -> None:
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    frame = synthetic_frame(start)
    signals = pd.DataFrame(
        [
            {
                "snapshot_time_ms": start,
                "snapshot_time_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                "snapshot_hour_bj": "08:00",
                "symbol": "AUSDT",
                "rank": 1,
                "current_close": 90.0,
                "close_24h_ago": 100.0,
                "return_24h_pct": -10.0,
                "drop_24h_pct": 10.0,
                "drop_bucket": "10~20%",
                "entry_time_ms": start,
                "entry_time_utc": pd.Timestamp("2026-01-01", tz="UTC"),
            }
        ]
    )
    cfg = {"holding_days": [1], "liquidation_price_multiple": 2.0, "per_symbol_notional_usdt": 100.0, "fee_rate": 0.001, "slippage_rate": 0.0}
    outcomes = precompute_outcomes(signals, {"AUSDT": frame}, cfg)
    assert outcomes.iloc[0].entry_price == pytest.approx(frame.at[start, "open"])
    assert outcomes.iloc[0].exit_price == pytest.approx(frame.at[start + DAY_MS, "open"])


def test_metrics_pf_drawdown_and_tail_removal() -> None:
    trades = apply_position_conflict(outcome_rows().iloc[[0]], {1}, 1, {1: 100.0}, "Rank1", "fixed_per_symbol")
    extra = trades.copy()
    extra["symbol"] = "B"
    extra["pnl_usdt"] = -5.0
    extra["net_return_pct"] = -5.0
    combined = pd.concat([trades, extra], ignore_index=True)
    stats = summarize_trades(combined, {"2026-01"})
    assert stats["profit_factor"] == pytest.approx(2.0)
    assert stats["net_pnl_ex_best_1_usdt"] == pytest.approx(-5.0)
    assert profit_factor(pd.Series([10.0, -5.0])) == pytest.approx(2.0)
    assert max_drawdown(pd.Series([10.0, -5.0])) == pytest.approx(-5.0)


def test_complete_months_excludes_partial_last_month() -> None:
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2026-03-15", tz="UTC").timestamp() * 1000)
    assert complete_months(start, end) == {"2026-01", "2026-02"}


def test_block_bootstrap_is_seed_reproducible_and_preserves_snapshot_blocks() -> None:
    trades = apply_position_conflict(outcome_rows(), {1}, 1, {1: 100.0}, "Rank1", "fixed_per_symbol")
    first = block_bootstrap(trades, "candidate", "snapshot", 20, 123)
    second = block_bootstrap(trades, "candidate", "snapshot", 20, 123)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first.total_net_pnl_usdt).all()
