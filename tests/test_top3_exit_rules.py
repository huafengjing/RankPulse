from __future__ import annotations

from src.research.top3_strategy_rules import (
    DAY_MS,
    HOUR_MS,
    early_exit_time_ms,
    planned_exit_time_ms,
    should_exit_early_12h,
)


def test_holding_full_6d_exits_at_entry_plus_6d() -> None:
    entry_time_ms = 1_700_000_000_000

    assert planned_exit_time_ms(entry_time_ms) == entry_time_ms + 6 * DAY_MS


def test_12h_weak_exit_condition_requires_all_three_parts() -> None:
    assert should_exit_early_12h(mfe_12h=0.049, close_return_12h=-0.001, mae_12h=-0.051) is True
    assert should_exit_early_12h(mfe_12h=0.050, close_return_12h=-0.001, mae_12h=-0.051) is False
    assert should_exit_early_12h(mfe_12h=0.049, close_return_12h=0.0, mae_12h=-0.051) is False
    assert should_exit_early_12h(mfe_12h=0.049, close_return_12h=-0.001, mae_12h=-0.050) is False


def test_12h_early_exit_time_is_entry_plus_12h() -> None:
    entry_time_ms = 1_700_000_000_000

    assert early_exit_time_ms(entry_time_ms) == entry_time_ms + 12 * HOUR_MS
