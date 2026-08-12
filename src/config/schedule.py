from __future__ import annotations

from datetime import datetime, timezone, timedelta

from src.config.modes import SignalMode
from src.config.settings import AppSettings


BEIJING_TZ = timezone(timedelta(hours=8))


def is_entry_signal_time(timestamp_ms: int, settings: AppSettings) -> bool:
    return signal_window_time_ms(timestamp_ms, settings) is not None


def signal_window_time_ms(timestamp_ms: int, settings: AppSettings) -> int | None:
    bj_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
    if settings.signal_mode == SignalMode.PRODUCTION:
        if bj_time.minute != 0 or bj_time.hour not in {0, 8}:
            return None
        window = bj_time.replace(minute=0, second=0, microsecond=0)
        return int(window.timestamp() * 1000)
    if settings.signal_mode == SignalMode.TEST_FAST:
        if bj_time.minute % settings.signal_test_interval_minutes != 0:
            return None
        window_minute = bj_time.minute - (bj_time.minute % settings.signal_test_interval_minutes)
        window = bj_time.replace(minute=window_minute, second=0, microsecond=0)
        return int(window.timestamp() * 1000)
    return None


def hourly_window_time_ms(timestamp_ms: int) -> int | None:
    bj_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
    if bj_time.minute != 0:
        return None
    window = bj_time.replace(minute=0, second=0, microsecond=0)
    return int(window.timestamp() * 1000)


def information_window_time_ms(
    timestamp_ms: int,
    settings: AppSettings,
) -> int | None:
    if settings.signal_mode != SignalMode.PRODUCTION:
        return None
    bj_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
    if bj_time.hour != 23 or bj_time.minute != 0:
        return None
    window = bj_time.replace(minute=0, second=0, microsecond=0)
    return int(window.timestamp() * 1000)


def market_preflight_window_time_ms(
    timestamp_ms: int,
    settings: AppSettings,
) -> int | None:
    if settings.signal_mode != SignalMode.PRODUCTION:
        return None
    bj_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
    if bj_time.minute != 30 or bj_time.hour not in {7, 22, 23}:
        return None
    window = bj_time.replace(minute=30, second=0, microsecond=0)
    return int(window.timestamp() * 1000)


def exit_after_ms(kind: str, settings: AppSettings) -> int:
    if kind == "extreme_weak":
        if settings.signal_mode == SignalMode.TEST_FAST:
            return settings.test_extreme_weak_exit_after_minutes * 60 * 1000
        return 4 * 60 * 60 * 1000
    if kind == "weak":
        if settings.signal_mode == SignalMode.TEST_FAST:
            return settings.test_weak_exit_after_minutes * 60 * 1000
        return 12 * 60 * 60 * 1000
    if kind == "planned":
        if settings.signal_mode == SignalMode.TEST_FAST:
            return settings.test_planned_exit_after_minutes * 60 * 1000
        return 6 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unknown exit kind: {kind}")
