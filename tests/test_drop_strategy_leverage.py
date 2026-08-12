import pandas as pd

from scripts.research_drop_strategy_leverage import leveraged_outcome, replay


def test_leverage_notional_fees_and_exit_hour_excluded() -> None:
    path = pd.DataFrame({"open_time": [0, 1], "high": [110.0, 149.0]})
    result = leveraged_outcome(100.0, 90.0, path, leverage=2, fee_rate=0.001)
    assert not result["liquidated"]
    assert round(result["gross_pnl_usdt"], 8) == 20.0
    assert round(result["fees_usdt"], 8) == 0.38
    assert round(result["net_pnl_usdt"], 8) == 19.62


def test_liquidation_is_full_margin_loss() -> None:
    path = pd.DataFrame({"open_time": [0, 1], "high": [120.0, 151.0]})
    result = leveraged_outcome(100.0, 90.0, path, leverage=2, fee_rate=0.001)
    assert result["liquidated"]
    assert result["first_liquidation_time_ms"] == 1
    assert result["net_pnl_usdt"] == -100.0
    assert result["fees_usdt"] == 0.0


def test_earlier_liquidation_releases_lock_for_later_signal() -> None:
    rows = []
    for leverage, exit_time, liquidated in [(1, 24, False), (2, 6, True)]:
        rows.extend([
            {"candidate_id": "A", "leverage": leverage, "entry_time_ms": 0, "exit_time_ms": exit_time, "rank": 1, "symbol": "X", "net_pnl_usdt": -100.0 if liquidated else 1.0, "return_on_margin_pct": -100.0 if liquidated else 1.0, "entry_notional_usdt": 100 * leverage, "liquidated": liquidated},
            {"candidate_id": "A", "leverage": leverage, "entry_time_ms": 8, "exit_time_ms": 32, "rank": 1, "symbol": "X", "net_pnl_usdt": 2.0, "return_on_margin_pct": 2.0, "entry_notional_usdt": 100 * leverage, "liquidated": False},
        ])
    outcomes = pd.DataFrame(rows)
    one_x = replay(outcomes, {"A": 1}, True, "1x")
    two_x = replay(outcomes, {"A": 2}, True, "2x")
    assert one_x.execution_status.tolist() == ["executed", "skipped_existing_position"]
    assert two_x.execution_status.tolist() == ["executed", "executed"]
