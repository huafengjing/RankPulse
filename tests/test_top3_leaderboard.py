from __future__ import annotations

from src.market.binance_futures import Kline
from src.market.top3_leaderboard import build_top3_leaderboard, gain_24h_from_1h_klines


HOUR_MS = 60 * 60 * 1000
FOUR_HOUR_MS = 4 * HOUR_MS
SIGNAL_TIME_MS = 1_700_000_000_000


def make_1h_kline(open_time_ms: int, open_price: float) -> Kline:
    return Kline(
        open_time_ms=open_time_ms,
        open=open_price,
        high=open_price,
        low=open_price,
        close=open_price,
        volume=1.0,
        close_time_ms=open_time_ms + HOUR_MS - 1,
    )


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


def hourly_klines(prior_open: float, signal_open: float) -> list[Kline]:
    return [
        make_1h_kline(SIGNAL_TIME_MS - 24 * HOUR_MS, prior_open),
        make_1h_kline(SIGNAL_TIME_MS, signal_open),
    ]


def completed_4h_klines_before_signal() -> list[Kline]:
    start = SIGNAL_TIME_MS - 42 * FOUR_HOUR_MS
    volumes = [1.0] * 36 + [2.0] * 6
    return [make_4h_kline(start + index * FOUR_HOUR_MS, volume) for index, volume in enumerate(volumes)]


def test_gain_24h_uses_1h_open_at_signal_and_24h_before_signal() -> None:
    klines = hourly_klines(prior_open=10.0, signal_open=12.5)

    assert gain_24h_from_1h_klines(klines, SIGNAL_TIME_MS) == 0.25


def test_build_top3_leaderboard_sorts_by_reconstructed_24h_gain() -> None:
    one_hour_by_symbol = {
        "AAAUSDT": hourly_klines(prior_open=10.0, signal_open=11.0),
        "BBBUSDT": hourly_klines(prior_open=10.0, signal_open=13.0),
        "CCCUSDT": hourly_klines(prior_open=10.0, signal_open=12.0),
        "DDDUSDT": hourly_klines(prior_open=10.0, signal_open=10.5),
    }
    four_hour_by_symbol = {
        symbol: completed_4h_klines_before_signal()
        for symbol in one_hour_by_symbol
    }

    leaderboard = build_top3_leaderboard(
        one_hour_by_symbol=one_hour_by_symbol,
        four_hour_by_symbol=four_hour_by_symbol,
        signal_time_ms=SIGNAL_TIME_MS,
    )

    assert [(entry.rank, entry.symbol, entry.gain_24h) for entry in leaderboard] == [
        (1, "BBBUSDT", 0.30),
        (2, "CCCUSDT", 0.20),
        (3, "AAAUSDT", 0.10),
    ]
    assert leaderboard[0].volume_24h_ratio_7d == 12.0 / (48.0 / 7.0)


def test_leaderboard_skips_symbols_without_required_1h_or_4h_history() -> None:
    leaderboard = build_top3_leaderboard(
        one_hour_by_symbol={
            "AAAUSDT": hourly_klines(prior_open=10.0, signal_open=11.0),
            "BBBUSDT": [make_1h_kline(SIGNAL_TIME_MS, 13.0)],
        },
        four_hour_by_symbol={
            "AAAUSDT": completed_4h_klines_before_signal(),
            "BBBUSDT": completed_4h_klines_before_signal(),
        },
        signal_time_ms=SIGNAL_TIME_MS,
    )

    assert [entry.symbol for entry in leaderboard] == ["AAAUSDT"]
