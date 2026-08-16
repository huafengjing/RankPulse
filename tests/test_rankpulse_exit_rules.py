from __future__ import annotations

from src.research.rankpulse_strategy_rules import (
    ENABLE_12H_WEAK_EXIT,
    ENABLE_4H_EXTREME_WEAK_EXIT,
    DAY_MS,
    HOUR_MS,
    early_exit_time_ms,
    extreme_weak_exit_time_ms,
    planned_exit_time_ms,
    Top3Signal,
    should_exit_extreme_weak_4h,
    should_exit_early_12h,
)


def test_holding_full_6d_exits_at_entry_plus_6d() -> None:
    entry_time_ms = 1_700_000_000_000

    assert planned_exit_time_ms(entry_time_ms) == entry_time_ms + 6 * DAY_MS


def test_rank1_tuned_signal_exits_at_entry_plus_5d() -> None:
    entry_time_ms = 1_700_000_000_000
    signal = Top3Signal(symbol="AAAUSDT", rank=1, gain_24h=0.25, volume_24h_ratio_7d=2.5)

    assert planned_exit_time_ms(entry_time_ms, signal) == entry_time_ms + 5 * DAY_MS


def test_rank1_5x_tuned_signal_exits_at_entry_plus_2d() -> None:
    entry_time_ms = 1_700_000_000_000
    signal = Top3Signal(symbol="AAAUSDT", rank=1, gain_24h=0.45, volume_24h_ratio_7d=2.5)

    assert planned_exit_time_ms(entry_time_ms, signal) == entry_time_ms + 2 * DAY_MS


def test_12h_weak_exit_condition_requires_mfe_and_negative_close_return_only() -> None:
    assert should_exit_early_12h(mfe_12h=0.049, close_return_12h=-0.001, mae_12h=-0.001) is True
    assert should_exit_early_12h(mfe_12h=0.050, close_return_12h=-0.001, mae_12h=-0.051) is False
    assert should_exit_early_12h(mfe_12h=0.049, close_return_12h=0.0, mae_12h=-0.051) is False
    assert should_exit_early_12h(mfe_12h=0.049, close_return_12h=-0.001, mae_12h=0.0) is True


def test_12h_weak_exit_is_enabled_by_default() -> None:
    assert ENABLE_12H_WEAK_EXIT is True


def test_12h_weak_exit_can_be_disabled_explicitly() -> None:
    assert should_exit_early_12h(
        mfe_12h=0.049,
        close_return_12h=-0.001,
        mae_12h=-0.051,
        enabled=False,
    ) is False


def test_4h_extreme_weak_exit_condition_requires_low_mfe_and_deep_mae() -> None:
    assert should_exit_extreme_weak_4h(mfe_4h=0.019, mae_4h=-0.081) is True
    assert should_exit_extreme_weak_4h(mfe_4h=0.020, mae_4h=-0.081) is False
    assert should_exit_extreme_weak_4h(mfe_4h=0.019, mae_4h=-0.080) is False


def test_4h_extreme_weak_exit_is_enabled_by_default() -> None:
    assert ENABLE_4H_EXTREME_WEAK_EXIT is True


def test_4h_extreme_weak_exit_can_be_disabled_explicitly() -> None:
    assert should_exit_extreme_weak_4h(
        mfe_4h=0.019,
        mae_4h=-0.081,
        enabled=False,
    ) is False


def test_12h_early_exit_time_is_entry_plus_12h() -> None:
    entry_time_ms = 1_700_000_000_000

    assert early_exit_time_ms(entry_time_ms) == entry_time_ms + 12 * HOUR_MS


def test_4h_extreme_weak_exit_time_is_entry_plus_4h() -> None:
    entry_time_ms = 1_700_000_000_000

    assert extreme_weak_exit_time_ms(entry_time_ms) == entry_time_ms + 4 * HOUR_MS
