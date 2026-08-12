from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config.modes import SignalMode, TradingMode
from src.config.settings import (
    LIVE_TRADING_CONFIRMATION_PHRASE,
    AppSettings,
    LiveTradingDisabledError,
)
from src.config.schedule import (
    exit_after_ms,
    hourly_window_time_ms,
    information_window_time_ms,
    is_entry_signal_time,
    market_preflight_window_time_ms,
)
from src.config.state_paths import state_paths_for


BEIJING = timezone(timedelta(hours=8))


def bj_ms(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 6, 17, hour, minute, tzinfo=BEIJING).timestamp() * 1000)


def test_settings_defaults_are_safe() -> None:
    settings = AppSettings.from_env({})

    assert settings.trading_mode == TradingMode.PAPER
    assert settings.signal_mode == SignalMode.PRODUCTION
    assert settings.enforce_safety_lock is True
    assert settings.max_open_positions == 10
    assert settings.enable_12h_weak_exit is True
    assert settings.enable_4h_extreme_weak_exit is True
    assert settings.top3_regime_enabled is False
    assert settings.top3_regime_context_auto_generate is True


def test_test_fast_settings_from_env() -> None:
    settings = AppSettings.from_env(
        {
            "TRADING_MODE": "testnet",
            "SIGNAL_MODE": "test_fast",
            "POSITION_MARGIN_USDT": "10",
            "MAX_OPEN_POSITIONS": "2",
            "SIGNAL_TEST_INTERVAL_MINUTES": "5",
            "TEST_EXTREME_WEAK_EXIT_AFTER_MINUTES": "5",
            "TEST_WEAK_EXIT_AFTER_MINUTES": "15",
            "TEST_PLANNED_EXIT_AFTER_MINUTES": "60",
            "TELEGRAM_BOT_TOKEN": "123:token",
            "TELEGRAM_CHAT_ID": "456",
            "TOP3_REGIME_ENABLED": "true",
            "TOP3_REGIME_CONTEXT_AUTO_GENERATE": "false",
        }
    )

    assert settings.trading_mode == TradingMode.TESTNET
    assert settings.signal_mode == SignalMode.TEST_FAST
    assert settings.position_margin_usdt == 10
    assert settings.max_open_positions == 2
    assert settings.telegram_bot_token == "123:token"
    assert settings.telegram_chat_id == "456"
    assert settings.top3_regime_enabled is True
    assert settings.top3_regime_context_auto_generate is False


def test_enforce_safety_lock_blocks_run_when_true() -> None:
    settings = AppSettings(enforce_safety_lock=True)

    with pytest.raises(LiveTradingDisabledError, match="ENFORCE_SAFETY_LOCK"):
        settings.assert_can_run()


def test_enforce_safety_lock_allows_run_when_false() -> None:
    settings = AppSettings(enforce_safety_lock=False)

    assert settings.assert_can_run() is None


def test_live_mode_is_rejected_without_live_guard_even_if_safety_lock_is_off() -> None:
    settings = AppSettings.from_env({"TRADING_MODE": "live", "ENFORCE_SAFETY_LOCK": "false"})

    with pytest.raises(LiveTradingDisabledError, match="ALLOW_LIVE_TRADING"):
        settings.assert_can_run()


def test_live_mode_requires_exact_confirmation_phrase() -> None:
    settings = AppSettings.from_env(
        {
            "TRADING_MODE": "live",
            "ENFORCE_SAFETY_LOCK": "false",
            "ALLOW_LIVE_TRADING": "true",
            "LIVE_ORDER_CONFIRMATION": "wrong",
        }
    )

    with pytest.raises(LiveTradingDisabledError, match="LIVE_ORDER_CONFIRMATION"):
        settings.assert_can_run()


def test_live_mode_can_run_only_with_all_guards_disabled_explicitly() -> None:
    settings = AppSettings.from_env(
        {
            "TRADING_MODE": "live",
            "ENFORCE_SAFETY_LOCK": "false",
            "ALLOW_LIVE_TRADING": "true",
            "LIVE_ORDER_CONFIRMATION": LIVE_TRADING_CONFIRMATION_PHRASE,
        }
    )

    assert settings.assert_can_run() is None


def test_production_schedule_uses_beijing_00_and_08() -> None:
    settings = AppSettings(signal_mode=SignalMode.PRODUCTION)

    assert is_entry_signal_time(bj_ms(0), settings) is True
    assert is_entry_signal_time(bj_ms(8), settings) is True
    assert is_entry_signal_time(bj_ms(8, 5), settings) is False


def test_test_fast_schedule_uses_configured_interval() -> None:
    settings = AppSettings(signal_mode=SignalMode.TEST_FAST, signal_test_interval_minutes=5)

    assert is_entry_signal_time(bj_ms(8, 0), settings) is True
    assert is_entry_signal_time(bj_ms(8, 5), settings) is True
    assert is_entry_signal_time(bj_ms(8, 3), settings) is False


def test_exit_timing_is_compressed_only_in_test_fast() -> None:
    production = AppSettings(signal_mode=SignalMode.PRODUCTION)
    test_fast = AppSettings(signal_mode=SignalMode.TEST_FAST)

    assert exit_after_ms("weak", production) == 12 * 60 * 60 * 1000
    assert exit_after_ms("extreme_weak", production) == 4 * 60 * 60 * 1000
    assert exit_after_ms("planned", production) == 6 * 24 * 60 * 60 * 1000
    assert exit_after_ms("weak", test_fast) == 15 * 60 * 1000
    assert exit_after_ms("extreme_weak", test_fast) == 5 * 60 * 1000
    assert exit_after_ms("planned", test_fast) == 60 * 60 * 1000


def test_hourly_exit_check_only_runs_at_beijing_top_of_hour() -> None:
    assert hourly_window_time_ms(bj_ms(8, 0)) == bj_ms(8, 0)
    assert hourly_window_time_ms(bj_ms(8, 0) + 45_000) == bj_ms(8, 0)
    assert hourly_window_time_ms(bj_ms(8, 5)) is None


def test_production_information_window_is_beijing_23_only() -> None:
    production = AppSettings(signal_mode=SignalMode.PRODUCTION)

    assert information_window_time_ms(bj_ms(23, 0), production) == bj_ms(23, 0)
    assert information_window_time_ms(bj_ms(23, 0) + 45_000, production) == bj_ms(23, 0)
    assert information_window_time_ms(bj_ms(0, 0), production) is None
    assert information_window_time_ms(bj_ms(23, 5), production) is None


def test_production_market_preflight_runs_30_minutes_before_observation_and_entry() -> None:
    production = AppSettings(signal_mode=SignalMode.PRODUCTION)

    assert market_preflight_window_time_ms(bj_ms(22, 30), production) == bj_ms(22, 30)
    assert market_preflight_window_time_ms(bj_ms(23, 30), production) == bj_ms(23, 30)
    assert market_preflight_window_time_ms(bj_ms(7, 30), production) == bj_ms(7, 30)
    assert market_preflight_window_time_ms(bj_ms(23, 0), production) is None
    assert market_preflight_window_time_ms(bj_ms(8, 0), production) is None
    assert market_preflight_window_time_ms(bj_ms(7, 31), production) is None


def test_state_paths_are_isolated_by_trading_and_signal_mode() -> None:
    paper = state_paths_for(TradingMode.PAPER, SignalMode.PRODUCTION)
    testnet = state_paths_for(TradingMode.TESTNET, SignalMode.TEST_FAST)
    live = state_paths_for(TradingMode.LIVE, SignalMode.PRODUCTION)

    assert str(paper.state_path).endswith("data\\paper\\production\\state.json") or str(paper.state_path).endswith("data/paper/production/state.json")
    assert str(testnet.state_path).endswith("data\\testnet\\test_fast\\state.json") or str(testnet.state_path).endswith("data/testnet/test_fast/state.json")
    assert str(live.state_path).endswith("data\\live\\production\\state.json") or str(live.state_path).endswith("data/live/production/state.json")
    assert paper.state_path != testnet.state_path
    assert live.state_path != testnet.state_path
