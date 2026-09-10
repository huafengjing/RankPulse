from __future__ import annotations

from src.research.rankpulse_strategy_rules import DAY_MS, is_duplicate_position, same_symbol_reentry_block_reason


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


def test_losing_same_symbol_reentry_is_blocked_from_14_to_27_days_after_entry() -> None:
    entry_time_ms = 1_700_000_000_000
    last_entry_by_symbol = {"AAAUSDT": entry_time_ms}
    last_pnl_by_symbol = {"AAAUSDT": -12.5}

    assert same_symbol_reentry_block_reason("AAAUSDT", entry_time_ms + 13 * DAY_MS, last_entry_by_symbol, last_pnl_by_symbol) is None
    assert same_symbol_reentry_block_reason("AAAUSDT", entry_time_ms + 14 * DAY_MS, last_entry_by_symbol, last_pnl_by_symbol) is not None
    assert same_symbol_reentry_block_reason("AAAUSDT", entry_time_ms + 26 * DAY_MS, last_entry_by_symbol, last_pnl_by_symbol) is not None
    assert same_symbol_reentry_block_reason("AAAUSDT", entry_time_ms + 27 * DAY_MS, last_entry_by_symbol, last_pnl_by_symbol) is None
    assert same_symbol_reentry_block_reason("BBBUSDT", entry_time_ms + 20 * DAY_MS, last_entry_by_symbol, last_pnl_by_symbol) is None


def test_winning_same_symbol_reentry_is_blocked_before_30_days() -> None:
    entry_time_ms = 1_700_000_000_000
    last_entry_by_symbol = {"AAAUSDT": entry_time_ms}

    assert (
        same_symbol_reentry_block_reason(
            "AAAUSDT",
            entry_time_ms + 13 * DAY_MS,
            last_entry_by_symbol,
            {"AAAUSDT": 12.5},
        )
        is not None
    )
    assert (
        same_symbol_reentry_block_reason(
            "AAAUSDT",
            entry_time_ms + 29 * DAY_MS,
            last_entry_by_symbol,
            {"AAAUSDT": 12.5},
        )
        is not None
    )
    assert (
        same_symbol_reentry_block_reason(
            "AAAUSDT",
            entry_time_ms + 30 * DAY_MS,
            last_entry_by_symbol,
            {"AAAUSDT": 12.5},
        )
        is None
    )
    assert (
        same_symbol_reentry_block_reason(
            "AAAUSDT",
            entry_time_ms + 13 * DAY_MS,
            last_entry_by_symbol,
            {"AAAUSDT": -12.5},
        )
        is None
    )
