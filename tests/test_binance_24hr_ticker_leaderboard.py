from __future__ import annotations

from src.market.binance_futures import BinanceFuturesMarketClient
from src.market.ticker_leaderboard import build_top3_from_24hr_tickers


def test_fetches_24hr_ticker_stats_from_public_futures_endpoint() -> None:
    def transport(path: str, params: dict[str, str | int]) -> object:
        assert path == "/fapi/v1/ticker/24hr"
        assert params == {}
        return [
            {"symbol": "AAAUSDT", "priceChangePercent": "10.50"},
            {"symbol": "BBBUSDT", "priceChangePercent": "-2.00"},
        ]

    client = BinanceFuturesMarketClient(transport=transport)

    stats = client.ticker_24hr_stats()

    assert stats["AAAUSDT"].price_change_percent == 10.50
    assert stats["BBBUSDT"].price_change_percent == -2.00


def test_build_top3_from_24hr_tickers_sorts_by_price_change_percent() -> None:
    leaderboard = build_top3_from_24hr_tickers(
        ticker_stats_by_symbol={
            "AAAUSDT": 10.0,
            "BBBUSDT": 30.0,
            "CCCUSDT": 20.0,
            "DDDUSDT": 5.0,
        },
        volume_ratio_by_symbol={
            "AAAUSDT": 1.8,
            "BBBUSDT": 2.5,
            "CCCUSDT": 4.2,
            "DDDUSDT": 3.5,
        },
    )

    assert [(entry.rank, entry.symbol, entry.gain_24h) for entry in leaderboard] == [
        (1, "BBBUSDT", 0.30),
        (2, "CCCUSDT", 0.20),
        (3, "AAAUSDT", 0.10),
    ]
    assert leaderboard[1].volume_24h_ratio_7d == 4.2


def test_build_top3_from_24hr_tickers_preserves_rank_when_volume_ratio_is_missing() -> None:
    leaderboard = build_top3_from_24hr_tickers(
        ticker_stats_by_symbol={
            "AAAUSDT": 10.0,
            "BBBUSDT": 30.0,
        },
        volume_ratio_by_symbol={
            "AAAUSDT": 1.8,
        },
    )

    assert [(entry.rank, entry.symbol, entry.volume_24h_ratio_7d) for entry in leaderboard] == [
        (1, "BBBUSDT", None),
        (2, "AAAUSDT", 1.8),
    ]
