import pandas as pd

from scripts.research_combined_recommended_drop_strategy import apply_global_position_lock


def test_global_lock_skips_cross_candidate_without_resetting_exit() -> None:
    rows = [
        {"entry_time_ms": 0, "exit_time_ms": 24, "rank": 1, "symbol": "X", "candidate_id": "A", "pnl_usdt_at_100": 1.0, "fees_usdt_at_100": 0.2},
        {"entry_time_ms": 8, "exit_time_ms": 56, "rank": 1, "symbol": "X", "candidate_id": "B", "pnl_usdt_at_100": 2.0, "fees_usdt_at_100": 0.2},
        {"entry_time_ms": 20, "exit_time_ms": 92, "rank": 3, "symbol": "Y", "candidate_id": "C", "pnl_usdt_at_100": 3.0, "fees_usdt_at_100": 0.2},
        {"entry_time_ms": 24, "exit_time_ms": 48, "rank": 1, "symbol": "X", "candidate_id": "A", "pnl_usdt_at_100": 4.0, "fees_usdt_at_100": 0.2},
    ]
    result = apply_global_position_lock(pd.DataFrame(rows))
    assert result.execution_status.tolist() == ["executed", "skipped_global_existing_position", "executed", "executed"]
    assert result.iloc[1].blocked_by_candidate_id == "A"
    assert result.iloc[3].pnl_usdt == 4.0
