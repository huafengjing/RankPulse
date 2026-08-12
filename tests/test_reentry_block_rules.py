import pandas as pd

from scripts.research_reentry_block_rules import (
    DAY_MS,
    RULE_1_REASON,
    blocks_post_liquidation,
    blocks_profit_reentry,
    replay_with_block_rules,
)


def profitable_exit(exit_time: int = 0) -> dict:
    return {"liquidated": False, "net_pnl_usdt": 1.0, "exit_reason": "fixed_exit", "exit_time_ms": exit_time}


def liquidation(exit_time: int = 0) -> dict:
    return {"liquidated": True, "net_pnl_usdt": -100.0, "exit_reason": "liquidation", "exit_time_ms": exit_time}


def test_rule_boundaries() -> None:
    assert not blocks_profit_reentry(profitable_exit(DAY_MS), DAY_MS)
    assert blocks_profit_reentry(profitable_exit(), DAY_MS)
    assert not blocks_profit_reentry(profitable_exit(), DAY_MS + 1)
    assert not blocks_post_liquidation(liquidation(), 5 * DAY_MS)
    assert blocks_post_liquidation(liquidation(), 5 * DAY_MS + 1)
    assert blocks_post_liquidation(liquidation(), 30 * DAY_MS)
    assert not blocks_post_liquidation(liquidation(), 30 * DAY_MS + 1)


def test_blocked_signal_does_not_reset_profit_exit_window() -> None:
    rows = []
    for entry_time, exit_time, pnl in [(0, DAY_MS, 10.0), (DAY_MS + 12 * 3_600_000, 3 * DAY_MS, 5.0), (DAY_MS + 25 * 3_600_000, 4 * DAY_MS, 7.0)]:
        rows.append(
            {
                "candidate_id": "A", "snapshot_time_ms": entry_time, "snapshot_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
                "symbol": "X", "entry_time_ms": entry_time, "entry_time_utc": pd.to_datetime(entry_time, unit="ms", utc=True),
                "exit_time_ms": exit_time, "exit_time_utc": pd.to_datetime(exit_time, unit="ms", utc=True), "rank": 1,
                "net_pnl_usdt": pnl, "return_on_margin_pct": pnl, "liquidated": False, "exit_reason": "fixed_exit",
            }
        )
    replay = replay_with_block_rules(pd.DataFrame(rows), "test", use_rule_1=True, use_rule_2=False)

    assert replay.actual_executed.tolist() == [True, False, True]
    assert replay.block_reason.tolist() == ["", RULE_1_REASON, ""]
