from __future__ import annotations

from dataclasses import dataclass


DAY_MS = 24 * 60 * 60 * 1000
HOUR_MS = 60 * 60 * 1000

EXCLUDED_SYMBOLS = {"RAVEUSDT"}
SIGNAL_HOURS_BJ = {"00:00", "08:00"}
ENABLE_12H_WEAK_EXIT = True
ENABLE_4H_EXTREME_WEAK_EXIT = True
RANK1_3X_HOLD_DAYS = 5
RANK1_5X_HOLD_DAYS = 2
RANK23_HOLD_DAYS = 6


@dataclass(frozen=True)
class Top3Signal:
    symbol: str
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float | None
    snapshot_hour_bj: str = "00:00"


@dataclass(frozen=True)
class Top3RegimeContext:
    """As-of Bucket B Rank3 FR3/YR1 regime state for one signal observation."""

    state: str = "GREEN"
    recovery_signal: bool = False
    recovery_streak: int = 0
    model: str = "FR_avg_return24_l3_gt_0_fr3_yr1"


def is_trade_signal(signal: Top3Signal) -> bool:
    """Return whether a signal passes the current Top3 main strategy entry rules."""
    if signal.symbol in EXCLUDED_SYMBOLS:
        return False
    if signal.snapshot_hour_bj not in SIGNAL_HOURS_BJ:
        return False
    if signal.rank not in {1, 2, 3}:
        return False
    if signal.rank == 1:
        return rank1_leverage_for_signal(signal) is not None
    if signal.gain_24h < 0.10:
        return False
    if signal.gain_24h >= 0.80:
        return False

    if 0.10 <= signal.gain_24h < 0.20:
        return True

    if 0.20 <= signal.gain_24h < 0.40:
        if signal.rank == 2:
            return _volume_between(signal.volume_24h_ratio_7d, 1.5, 5.0)
        if signal.rank == 3:
            return _volume_between(signal.volume_24h_ratio_7d, 1.2, 5.0)
        return False

    if 0.40 <= signal.gain_24h < 0.60:
        return signal.rank == 2 and _volume_between(signal.volume_24h_ratio_7d, 3.0, 5.5)

    if 0.60 <= signal.gain_24h < 0.80:
        return False

    return False


def signal_rejection_reason(signal: Top3Signal) -> str | None:
    """Return a human-readable reason if the signal should NOT trade, or None if it passes."""
    if signal.symbol in EXCLUDED_SYMBOLS:
        return f"Excluded symbol: {signal.symbol}"
    if signal.snapshot_hour_bj not in SIGNAL_HOURS_BJ:
        return f"Non-signal time window: {signal.snapshot_hour_bj} (requires BJ {SIGNAL_HOURS_BJ})"
    if signal.rank not in {1, 2, 3}:
        return f"Rank {signal.rank} is outside tradable ranks (Rank 1/2/3 only)"
    if signal.rank == 1:
        if rank1_leverage_for_signal(signal) is None:
            return _rank1_rejection_reason(signal)
        return None
    if signal.gain_24h < 0.10:
        return f"Gain {signal.gain_24h:.1%} is below minimum 10%"
    if signal.gain_24h >= 0.80:
        return f"Gain {signal.gain_24h:.1%} is at/above 80%; no trade"

    if 0.10 <= signal.gain_24h < 0.20:
        return None

    if 0.20 <= signal.gain_24h < 0.40:
        vr = signal.volume_24h_ratio_7d
        if vr is None:
            return "Volume ratio unavailable (None)"
        lower = 1.5 if signal.rank == 2 else 1.2
        if not lower <= vr < 5.0:
            return f"Volume ratio {vr:.2f} is outside Rank{signal.rank} 20-40% range [{lower}, 5.0)"
        return None

    if 0.40 <= signal.gain_24h < 0.60:
        if signal.rank != 2:
            return f"40-60% bucket only allows Rank2 for Rank2/3 strategy (current Rank {signal.rank})"
        vr = signal.volume_24h_ratio_7d
        if vr is None:
            return "Volume ratio unavailable (None)"
        if not 3.0 <= vr < 5.5:
            return f"Volume ratio {vr:.2f} is outside 40-60% Rank2 range [3.0, 5.5)"
        return None

    if 0.60 <= signal.gain_24h < 0.80:
        return f"Gain {signal.gain_24h:.1%} is in disabled 60-80% range"

    return f"Gain {signal.gain_24h:.1%} does not match any rule"


def leverage_for_signal(signal: Top3Signal, regime_context: Top3RegimeContext | None = None) -> int | None:
    """Return configured leverage for a passing signal, or None when it should not trade."""
    if not is_trade_signal(signal):
        return None
    rank1_leverage = rank1_leverage_for_signal(signal)
    if rank1_leverage is not None:
        return rank1_leverage
    if 0.10 <= signal.gain_24h < 0.20:
        return 3
    if 0.20 <= signal.gain_24h < 0.40:
        adaptive = bucket_b_fr3_yr1_leverage(signal, regime_context)
        if adaptive is not None:
            return adaptive
        return 3 if signal.rank == 2 else 5
    if 0.40 <= signal.gain_24h < 0.60:
        return 2
    return None


def bucket_b_fr3_yr1_leverage(signal: Top3Signal, regime_context: Top3RegimeContext | None) -> int | None:
    """Return FR3/YR1 adaptive leverage for Bucket B, or None to use base leverage."""
    if regime_context is None:
        return None
    if signal.rank not in {2, 3}:
        return None
    if not (0.20 <= signal.gain_24h < 0.40):
        return None

    state = regime_context.state.upper()
    if state == "GREEN":
        return 3 if signal.rank == 2 else 5

    if state == "YELLOW":
        if signal.rank == 2:
            return 3
        if signal.rank == 3:
            return 5 if regime_context.recovery_signal else 3
        return None

    if state == "RED":
        if signal.rank == 2:
            return 3 if regime_context.recovery_signal else 2
        if signal.rank == 3:
            if not regime_context.recovery_signal:
                return 1
            return 5 if regime_context.recovery_streak >= 2 else 3
        return None

    return None


def rank1_leverage_for_signal(signal: Top3Signal) -> int | None:
    """Return fixed Rank1 tuned leverage. Rank1 is not adjusted by FR3/YR1."""
    if signal.rank != 1:
        return None
    volume = signal.volume_24h_ratio_7d
    if 0.20 <= signal.gain_24h < 0.40:
        if _volume_between(volume, 2.0, 3.0):
            return 3
        if _volume_between(volume, 5.0, 6.0):
            return 5
    if 0.40 <= signal.gain_24h < 0.60 and _volume_between(volume, 2.0, 3.0):
        return 5
    return None


def planned_hold_days_for_signal(signal: Top3Signal | None = None) -> int:
    if signal is not None:
        rank1_leverage = rank1_leverage_for_signal(signal)
        if rank1_leverage == 3:
            return RANK1_3X_HOLD_DAYS
        if rank1_leverage == 5:
            return RANK1_5X_HOLD_DAYS
    return RANK23_HOLD_DAYS


def liquidation_threshold_for_leverage(leverage: int) -> float | None:
    thresholds = {1: -1.00, 2: -0.50, 3: -0.33, 5: -0.20}
    return thresholds.get(leverage)


def volume_24h_ratio_7d(completed_4h_volumes: list[float]) -> float | None:
    if len(completed_4h_volumes) < 42:
        return None

    latest_42 = completed_4h_volumes[-42:]
    recent_24h_volume = sum(latest_42[-6:])
    seven_day_avg_daily_volume = sum(latest_42) / 7
    if seven_day_avg_daily_volume == 0:
        return None
    return recent_24h_volume / seven_day_avg_daily_volume


def should_exit_early_12h(
    mfe_12h: float,
    close_return_12h: float,
    mae_12h: float,
    enabled: bool = ENABLE_12H_WEAK_EXIT,
) -> bool:
    if not enabled:
        return False
    return mfe_12h < 0.05 and close_return_12h < 0


def should_exit_extreme_weak_4h(
    mfe_4h: float,
    mae_4h: float,
    enabled: bool = ENABLE_4H_EXTREME_WEAK_EXIT,
) -> bool:
    if not enabled:
        return False
    return mfe_4h < 0.02 and mae_4h < -0.08


def planned_exit_time_ms(entry_time_ms: int, signal: Top3Signal | None = None) -> int:
    return entry_time_ms + planned_hold_days_for_signal(signal) * DAY_MS


def early_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 12 * HOUR_MS


def extreme_weak_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 4 * HOUR_MS


def is_duplicate_position(symbol: str, signal_time_ms: int, open_until_by_symbol: dict[str, int]) -> bool:
    open_until = open_until_by_symbol.get(symbol)
    return open_until is not None and signal_time_ms < open_until


def _volume_between(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value < upper


def _rank1_rejection_reason(signal: Top3Signal) -> str:
    vr = signal.volume_24h_ratio_7d
    if vr is None:
        return "Rank1 volume ratio unavailable (None)"
    return (
        f"Rank1 outside tuned candidate cells: gain={signal.gain_24h:.1%}, "
        f"volume_24h_ratio_7d={vr:.2f}; allowed cells are 20-40/V2-3, 20-40/V5-6, 40-60/V2-3"
    )
