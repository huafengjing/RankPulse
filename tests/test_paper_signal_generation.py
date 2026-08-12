from __future__ import annotations

from datetime import datetime, timezone, timedelta

from src.market.binance_futures import Kline, Ticker24hrStat
from src.paper.signals import (
    bj_snapshot_hour,
    generate_binance_ticker_rank_signals,
    is_signal_time_bj,
)


HOUR_MS = 60 * 60 * 1000
FOUR_HOUR_MS = 4 * HOUR_MS
BEIJING = timezone(timedelta(hours=8))


def bj_time_ms(hour: int) -> int:
    return int(datetime(2026, 6, 17, hour, 0, tzinfo=BEIJING).timestamp() * 1000)


def make_4h_kline(open_time_ms: int, volume: float) -> Kline:
    return Kline(
        open_time_ms=open_time_ms,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=volume,
        close_time_ms=open_time_ms + FOUR_HOUR_MS - 1,
    )


def test_signal_time_uses_beijing_00_and_08() -> None:
    assert is_signal_time_bj(bj_time_ms(0)) is True
    assert bj_snapshot_hour(bj_time_ms(0)) == "00:00"
    assert is_signal_time_bj(bj_time_ms(8)) is True
    assert bj_snapshot_hour(bj_time_ms(8)) == "08:00"
    assert is_signal_time_bj(bj_time_ms(9)) is False


def test_generate_binance_ticker_rank_signals_uses_rank2_and_rank3_with_latest_fill_price() -> None:
    signal_time_ms = bj_time_ms(0)
    start = signal_time_ms - 42 * FOUR_HOUR_MS
    klines = [make_4h_kline(start + index * FOUR_HOUR_MS, volume) for index, volume in enumerate([1.0] * 36 + [2.0] * 6)]

    signals = generate_binance_ticker_rank_signals(
        signal_time_ms=signal_time_ms,
        ticker_stats_by_symbol={
            "RANK1USDT": Ticker24hrStat(symbol="RANK1USDT", price_change_percent=30.0),
            "RANK2USDT": Ticker24hrStat(symbol="RANK2USDT", price_change_percent=25.0),
            "RANK3USDT": Ticker24hrStat(symbol="RANK3USDT", price_change_percent=20.0),
        },
        four_hour_klines_by_symbol={
            "RANK1USDT": klines,
            "RANK2USDT": klines,
            "RANK3USDT": klines,
        },
        latest_prices_by_symbol={
            "RANK1USDT": 11.0,
            "RANK2USDT": 22.0,
            "RANK3USDT": 33.0,
        },
    )

    assert [(signal.rank, signal.symbol, signal.gain_24h, signal.fill_price) for signal in signals] == [
        (2, "RANK2USDT", 0.25, 22.0),
        (3, "RANK3USDT", 0.20, 33.0),
    ]
    assert signals[0].snapshot_hour_bj == "00:00"
    assert signals[0].volume_24h_ratio_7d == 12.0 / (48.0 / 7.0)


def test_generate_binance_ticker_rank_signals_returns_empty_outside_signal_hours() -> None:
    signals = generate_binance_ticker_rank_signals(
        signal_time_ms=bj_time_ms(9),
        ticker_stats_by_symbol={"AAAUSDT": Ticker24hrStat(symbol="AAAUSDT", price_change_percent=30.0)},
        four_hour_klines_by_symbol={},
        latest_prices_by_symbol={"AAAUSDT": 1.0},
    )

    assert signals == []
