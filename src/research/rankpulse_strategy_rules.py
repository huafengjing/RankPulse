from __future__ import annotations

from dataclasses import dataclass


DAY_MS = 24 * 60 * 60 * 1000
HOUR_MS = 60 * 60 * 1000

EXCLUDED_SYMBOLS = {"RAVEUSDT"}
SIGNAL_HOURS_BJ = {"00:00", "08:00"}
ENABLE_12H_WEAK_EXIT = True
ENABLE_4H_EXTREME_WEAK_EXIT = True
ENABLE_RANK1_24H_WEAK_EXIT = True
ENABLE_MFE_AGING_PARTIAL_TP = True
MFE_AGING_PARTIAL_TP_ACTIVATION_U = 400.0
MFE_AGING_PARTIAL_TP_STALE_HOURS = 24
MFE_AGING_PARTIAL_TP_RATIO = 0.40
RANK1_3X_HOLD_DAYS = 5
RANK1_5X_HOLD_DAYS = 2
RANK23_HOLD_DAYS = 6
PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS = 14
PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS = 27
PREV_WIN_SAME_SYMBOL_REENTRY_BLOCK_DAYS = 30


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
        return f"交易对在黑名单中：{signal.symbol}"
    if signal.snapshot_hour_bj not in SIGNAL_HOURS_BJ:
        return f"非交易信号时间：{signal.snapshot_hour_bj}，仅允许北京时间 00:00 / 08:00"
    if signal.rank not in {1, 2, 3}:
        return f"排名 Rank{signal.rank} 不在可交易范围内，仅允许 Rank1/Rank2/Rank3"
    if signal.rank == 1:
        if rank1_leverage_for_signal(signal) is None:
            return _rank1_rejection_reason(signal)
        return None
    if signal.gain_24h < 0.10:
        return f"涨幅 {signal.gain_24h:.1%} 低于最低要求 10%"
    if signal.gain_24h >= 0.80:
        return f"涨幅 {signal.gain_24h:.1%} 达到或超过 80%，不交易"

    if 0.10 <= signal.gain_24h < 0.20:
        return None

    if 0.20 <= signal.gain_24h < 0.40:
        vr = signal.volume_24h_ratio_7d
        if vr is None:
            return "量比不可用"
        lower = 1.5 if signal.rank == 2 else 1.2
        if not lower <= vr < 5.0:
            return f"量比 {vr:.2f} 不在 Rank{signal.rank} 的 20%-40% 区间要求 [{lower}, 5.0) 内"
        return None

    if 0.40 <= signal.gain_24h < 0.60:
        if signal.rank != 2:
            return f"40%-60% 区间仅允许 Rank2，当前为 Rank{signal.rank}"
        vr = signal.volume_24h_ratio_7d
        if vr is None:
            return "量比不可用"
        if not 3.0 <= vr < 5.5:
            return f"量比 {vr:.2f} 不在 40%-60% Rank2 区间要求 [3.0, 5.5) 内"
        return None

    if 0.60 <= signal.gain_24h < 0.80:
        return f"涨幅 {signal.gain_24h:.1%} 位于 60%-80% 禁用区间，不交易"

    return f"涨幅 {signal.gain_24h:.1%} 不匹配任何交易规则"


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


def should_exit_rank1_weak_24h(
    rank: int | None,
    mfe_24h: float,
    close_return_24h: float,
    enabled: bool = ENABLE_RANK1_24H_WEAK_EXIT,
) -> bool:
    if not enabled:
        return False
    if rank not in {1, 2}:
        return False
    return mfe_24h < 0.08 and close_return_24h < 0


def should_trigger_mfe_aging_partial_tp(
    running_mfe_u: float,
    last_mfe_high_time_ms: int | None,
    now_ms: int,
    enabled: bool = ENABLE_MFE_AGING_PARTIAL_TP,
    activation_mfe_u: float = MFE_AGING_PARTIAL_TP_ACTIVATION_U,
    stale_hours: int = MFE_AGING_PARTIAL_TP_STALE_HOURS,
) -> bool:
    if not enabled:
        return False
    if last_mfe_high_time_ms is None:
        return False
    return running_mfe_u >= activation_mfe_u and now_ms - last_mfe_high_time_ms >= stale_hours * HOUR_MS


def planned_exit_time_ms(entry_time_ms: int, signal: Top3Signal | None = None) -> int:
    return entry_time_ms + planned_hold_days_for_signal(signal) * DAY_MS


def early_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 12 * HOUR_MS


def extreme_weak_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 4 * HOUR_MS


def rank1_weak_exit_time_ms(entry_time_ms: int) -> int:
    return entry_time_ms + 24 * HOUR_MS


def is_duplicate_position(symbol: str, signal_time_ms: int, open_until_by_symbol: dict[str, int]) -> bool:
    open_until = open_until_by_symbol.get(symbol)
    return open_until is not None and signal_time_ms < open_until


def same_symbol_reentry_block_reason(
    symbol: str,
    signal_time_ms: int,
    last_entry_time_by_symbol: dict[str, int],
    last_pnl_by_symbol: dict[str, float] | None = None,
) -> str | None:
    last_entry_time_ms = last_entry_time_by_symbol.get(symbol)
    if last_entry_time_ms is None:
        return None
    days_since_entry = (signal_time_ms - last_entry_time_ms) / DAY_MS
    last_pnl = None if last_pnl_by_symbol is None else last_pnl_by_symbol.get(symbol)
    if (
        last_pnl is not None
        and last_pnl > 0
        and 0 <= days_since_entry < PREV_WIN_SAME_SYMBOL_REENTRY_BLOCK_DAYS
    ):
        return (
            f"同交易对盈利后冷却：{symbol} 上次开仓距今 "
            f"{days_since_entry:.1f} 天，PnL={last_pnl:.4f}；"
            f"盈利单 {PREV_WIN_SAME_SYMBOL_REENTRY_BLOCK_DAYS} 天内不重复开仓"
        )
    if (
        last_pnl is not None
        and last_pnl <= 0
        and PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS <= days_since_entry < PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS
    ):
        return (
            f"同交易对亏损后冷却：{symbol} 上次开仓距今 "
            f"{days_since_entry:.1f} 天，PnL={last_pnl:.4f}；冷却区间为 "
            f"[{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MIN_DAYS}, "
            f"{PREV_LOSS_SAME_SYMBOL_REENTRY_BLOCK_MAX_DAYS}) 天"
        )
    return None


def _volume_between(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and lower <= value < upper


def _rank1_rejection_reason(signal: Top3Signal) -> str:
    vr = signal.volume_24h_ratio_7d
    if vr is None:
        return "Rank1 量比不可用"
    return (
        f"Rank1 不在允许交易区间：涨幅={signal.gain_24h:.1%}，"
        f"量比={vr:.2f}；允许区间：20%-40%/V2-3、20%-40%/V5-6、40%-60%/V2-3"
    )

