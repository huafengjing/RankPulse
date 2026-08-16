from __future__ import annotations

from src.research.rankpulse_strategy_rules import DAY_MS, is_duplicate_position


def test_same_symbol_open_position_is_not_reopened() -> None:
    entry_time_ms = 1_700_000_000_000
    open_until_by_symbol = {"AAAUSDT": entry_time_ms + 6 * DAY_MS}

    assert is_duplicate_position("AAAUSDT", entry_time_ms + DAY_MS, open_until_by_symbol) is True


def test_same_symbol_can_reopen_after_previous_position_exits() -> None:
    entry_time_ms = 1_700_000_000_000
    open_until_by_symbol = {"AAAUSDT": entry_time_ms + 6 * DAY_MS}

    assert is_duplicate_position("AAAUSDT", entry_time_ms + 6 * DAY_MS, open_until_by_symbol) is False


def test_different_symbol_is_not_blocked_by_existing_position() -> None:
    entry_time_ms = 1_700_000_000_000
    open_until_by_symbol = {"AAAUSDT": entry_time_ms + 6 * DAY_MS}

    assert is_duplicate_position("BBBUSDT", entry_time_ms + DAY_MS, open_until_by_symbol) is False
