from __future__ import annotations

import pytest

from src.paper.trading import (
    LIVE_ORDER_CONFIRMATION_PHRASE,
    LiveTradingDisabledError,
    PaperExtremeWeakExitCheck,
    PaperSignal,
    PaperTradingConfig,
    PaperTradingEngine,
    PaperWeakExitCheck,
)
from src.research.rankpulse_strategy_rules import DAY_MS, HOUR_MS
from src.research.rankpulse_strategy_rules import Top3RegimeContext


def test_paper_trading_opens_long_position_with_simulated_fill_price() -> None:
    engine = PaperTradingEngine()
    signal = PaperSignal(
        symbol="AAAUSDT",
        rank=3,
        gain_24h=0.35,
        volume_24h_ratio_7d=4.2,
        snapshot_hour_bj="08:00",
        signal_time_ms=1_700_000_000_000,
        fill_price=2.5,
    )

    position = engine.on_signal(signal)

    assert position is not None
    assert position.symbol == "AAAUSDT"
    assert position.side == "LONG"
    assert position.entry_price == 2.5
    assert position.leverage == 5
    assert position.margin_usdt == 100.0
    assert position.planned_exit_time_ms == signal.signal_time_ms + 6 * DAY_MS
    assert position.extreme_weak_exit_check_time_ms == signal.signal_time_ms + 4 * HOUR_MS


def test_paper_trading_uses_regime_context_leverage() -> None:
    engine = PaperTradingEngine()
    signal = PaperSignal(
        symbol="AAAUSDT",
        rank=3,
        gain_24h=0.35,
        volume_24h_ratio_7d=4.2,
        snapshot_hour_bj="08:00",
        signal_time_ms=1_700_000_000_000,
        fill_price=2.5,
        regime_context=Top3RegimeContext(state="RED"),
    )

    position = engine.on_signal(signal)

    assert position is not None
    assert position.leverage == 1


def test_paper_trading_opens_rank1_with_tuned_leverage_and_5d_exit() -> None:
    engine = PaperTradingEngine()
    signal = PaperSignal(
        symbol="RANK1USDT",
        rank=1,
        gain_24h=0.45,
        volume_24h_ratio_7d=2.5,
        snapshot_hour_bj="08:00",
        signal_time_ms=1_700_000_000_000,
        fill_price=2.5,
        regime_context=Top3RegimeContext(state="RED"),
    )

    position = engine.on_signal(signal)

    assert position is not None
    assert position.leverage == 5
    assert position.planned_exit_time_ms == signal.signal_time_ms + 2 * DAY_MS


def test_paper_trading_blocks_same_symbol_until_actual_exit() -> None:
    engine = PaperTradingEngine()
    first_signal_time_ms = 1_700_000_000_000
    signal = PaperSignal(
        symbol="AAAUSDT",
        rank=2,
        gain_24h=0.25,
        volume_24h_ratio_7d=2.5,
        snapshot_hour_bj="00:00",
        signal_time_ms=first_signal_time_ms,
        fill_price=10.0,
    )

    assert engine.on_signal(signal) is not None
    duplicate = PaperSignal(
        symbol="AAAUSDT",
        rank=2,
        gain_24h=0.25,
        volume_24h_ratio_7d=2.5,
        snapshot_hour_bj="08:00",
        signal_time_ms=first_signal_time_ms + HOUR_MS,
        fill_price=10.5,
    )

    assert engine.on_signal(duplicate) is None


def test_paper_trading_12h_weak_exit_is_enabled_by_default() -> None:
    engine = PaperTradingEngine()
    entry_time_ms = 1_700_000_000_000
    engine.on_signal(
        PaperSignal(
            symbol="AAAUSDT",
            rank=2,
            gain_24h=0.25,
            volume_24h_ratio_7d=2.5,
            snapshot_hour_bj="00:00",
            signal_time_ms=entry_time_ms,
            fill_price=10.0,
        )
    )

    trade_exit = engine.on_weak_exit_check(
        PaperWeakExitCheck(
            symbol="AAAUSDT",
            check_time_ms=entry_time_ms + 12 * HOUR_MS,
            fill_price=9.7,
            mfe_12h=0.049,
            close_return_12h=-0.001,
            mae_12h=-0.001,
        )
    )

    assert trade_exit is not None
    assert trade_exit.symbol == "AAAUSDT"
    assert trade_exit.exit_reason == "weak_12h"
    assert trade_exit.exit_price == 9.7
    assert engine.open_position("AAAUSDT") is None


def test_paper_trading_can_disable_12h_weak_exit_explicitly() -> None:
    engine = PaperTradingEngine(PaperTradingConfig(enable_12h_weak_exit=False))
    entry_time_ms = 1_700_000_000_000
    engine.on_signal(
        PaperSignal(
            symbol="AAAUSDT",
            rank=2,
            gain_24h=0.25,
            volume_24h_ratio_7d=2.5,
            snapshot_hour_bj="00:00",
            signal_time_ms=entry_time_ms,
            fill_price=10.0,
        )
    )

    trade_exit = engine.on_weak_exit_check(
        PaperWeakExitCheck(
            symbol="AAAUSDT",
            check_time_ms=entry_time_ms + 12 * HOUR_MS,
            fill_price=9.7,
            mfe_12h=0.049,
            close_return_12h=-0.001,
            mae_12h=-0.051,
        )
    )

    assert trade_exit is None
    assert engine.open_position("AAAUSDT") is not None


def test_paper_trading_4h_extreme_weak_exit_is_enabled_by_default() -> None:
    engine = PaperTradingEngine()
    entry_time_ms = 1_700_000_000_000
    engine.on_signal(
        PaperSignal(
            symbol="AAAUSDT",
            rank=2,
            gain_24h=0.25,
            volume_24h_ratio_7d=2.5,
            snapshot_hour_bj="00:00",
            signal_time_ms=entry_time_ms,
            fill_price=10.0,
        )
    )

    trade_exit = engine.on_extreme_weak_exit_check(
        PaperExtremeWeakExitCheck(
            symbol="AAAUSDT",
            check_time_ms=entry_time_ms + 4 * HOUR_MS,
            fill_price=9.1,
            mfe_4h=0.019,
            mae_4h=-0.081,
        )
    )

    assert trade_exit is not None
    assert trade_exit.exit_reason == "extreme_weak_4h"
    assert trade_exit.exit_price == 9.1
    assert engine.open_position("AAAUSDT") is None


def test_live_order_guard_is_disabled_by_default_and_requires_second_confirmation() -> None:
    default_engine = PaperTradingEngine()

    with pytest.raises(LiveTradingDisabledError):
        default_engine.assert_live_orders_allowed()

    unconfirmed_engine = PaperTradingEngine(PaperTradingConfig(live_trading_enabled=True))
    with pytest.raises(LiveTradingDisabledError):
        unconfirmed_engine.assert_live_orders_allowed()

    confirmed_engine = PaperTradingEngine(
        PaperTradingConfig(
            live_trading_enabled=True,
            live_order_confirmation=LIVE_ORDER_CONFIRMATION_PHRASE,
        )
    )
    assert confirmed_engine.assert_live_orders_allowed() is True
