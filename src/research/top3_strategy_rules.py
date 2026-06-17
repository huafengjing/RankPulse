from __future__ import annotations

from dataclasses import dataclass


DAY_MS = 24 * 60 * 60 * 1000
HOUR_MS = 60 * 60 * 1000

EXCLUDED_SYMBOLS = {"RAVEUSDT"}
SIGNAL_HOURS_BJ = {"00:00", "08:00"}


@dataclass(frozen=True)
class Top3Signal:
    symbol: str
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float | None
    snapshot_hour_bj: str = "00:00"


def is_trade_signal(signal: Top3Signal) -> bool:
    """Return whether a signal passes the current Top3 main strategy entry rules."""
    if signal.symbol in EXCLUDED_SYMBOLS:
        return False
    if signal.snapshot_hour_bj not in SIGNAL_HOURS_BJ:
        return False
    if signal.rank not in {2, 3}:
        return False
    if signal.gain_24h < 0.10:
        return False
    if signal.gain_24h >= 0.80:
        return False

    if 0.10 <= signal.gain_24h < 0.20:
        return True

    if 0.20 <= signal.gain_24h < 0.40:
        return _volume_between(signal.volume_24h_ratio_7d, 1.5, 5.0)

    if 0.40 <= signal.gain_24h < 0.60:
        return signal.rank == 2 and _volume_between(signal.volume_24h_ratio_7d, 3.0, 6.0)

    if 0.60 <= signal.gain_24h < 0.80:
        return False

    return False


def leverage_for_signal(signal: Top3Signal) -> int | None:
    """Return configured leverage for a passing signal, or None when it should not trade."""
    if not is_trade_signal(signal):
        return None
    if 0.10 <= signal.gain_24h < 0.20:
        return 3
    if 0.20 <= signal.gain_24h < 0.40:
        return 3 if signal.rank == 2 else 5
    if 0.40 <= signal.gain_24h < 0.60:
        return 2
    return None


def liquidation_threshold_for_leverage(leverage: int) -> float | None:
    thresholds = {2: -0.50, 3: -0.33, 5: -0.20}
    return thresholds.get(leverage)


def should_exit_early_12h(mfe_12h: float, close_return_12h: float, mae_12h: float) -> bool:
    return mfe_12h < 0.05 and close_return_12h < 0 and mae_12h < -0.05


def planned_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 6 * DAY_MS


def early_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 12 * HOUR_MS


def is_duplicate_position(symbol: str, signal_time_ms: int, open_until_by_symbol: dict[str, int]) -> bool:
    open_until = open_until_by_symbol.get(symbol)
    return open_until is not None and signal_time_ms < open_until


def _volume_between(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value < upper
