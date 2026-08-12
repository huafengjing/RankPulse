from __future__ import annotations

from http.client import IncompleteRead

import src.market.binance_futures as binance_futures
from src.market.binance_futures import BinanceFuturesMarketClient


def test_exchange_info_filters_trading_usdt_perpetual_symbols() -> None:
    def transport(path: str, params: dict[str, str | int]) -> object:
        assert path == "/fapi/v1/exchangeInfo"
        assert params == {}
        return {
            "symbols": [
                {
                    "symbol": "AAAUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "marginAsset": "USDT",
                },
                {
                    "symbol": "BBBUSDT",
                    "contractType": "CURRENT_QUARTER",
                    "status": "TRADING",
                    "quoteAsset": "USDT",
                    "marginAsset": "USDT",
                },
                {
                    "symbol": "CCCUSDT",
                    "contractType": "PERPETUAL",
                    "status": "BREAK",
                    "quoteAsset": "USDT",
                    "marginAsset": "USDT",
                },
                {
                    "symbol": "DDDUSDC",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "quoteAsset": "USDC",
                    "marginAsset": "USDC",
                },
            ]
        }

    client = BinanceFuturesMarketClient(transport=transport)

    assert client.usdt_perpetual_symbols() == ["AAAUSDT"]


def test_fetches_1h_and_4h_klines_from_public_futures_endpoint() -> None:
    calls: list[tuple[str, dict[str, str | int]]] = []

    def transport(path: str, params: dict[str, str | int]) -> object:
        calls.append((path, params))
        return [
            [
                1_700_000_000_000,
                "10.0",
                "11.0",
                "9.0",
                "10.5",
                "100.0",
                1_700_003_599_999,
                "1050.0",
                12,
                "50.0",
                "525.0",
                "0",
            ]
        ]

    client = BinanceFuturesMarketClient(transport=transport)

    one_hour = client.klines("AAAUSDT", "1h", limit=25)
    four_hour = client.klines("AAAUSDT", "4h", limit=42, end_time_ms=1_700_000_000_000)

    assert calls == [
        ("/fapi/v1/klines", {"symbol": "AAAUSDT", "interval": "1h", "limit": 25}),
        (
            "/fapi/v1/klines",
            {
                "symbol": "AAAUSDT",
                "interval": "4h",
                "limit": 42,
                "endTime": 1_700_000_000_000,
            },
        ),
    ]
    assert one_hour[0].open == 10.0
    assert four_hour[0].volume == 100.0


def test_fetches_latest_prices_from_v2_ticker_price_endpoint() -> None:
    def transport(path: str, params: dict[str, str | int]) -> object:
        assert path == "/fapi/v2/ticker/price"
        assert params == {}
        return [
            {"symbol": "AAAUSDT", "price": "1.23", "time": 1_700_000_000_000},
            {"symbol": "BBBUSDT", "price": "4.56", "time": 1_700_000_000_001},
        ]

    client = BinanceFuturesMarketClient(transport=transport)

    assert client.latest_prices() == {"AAAUSDT": 1.23, "BBBUSDT": 4.56}


def test_public_market_client_defaults_to_five_retries_with_longer_backoff() -> None:
    client = BinanceFuturesMarketClient(transport=lambda path, params: {})

    assert client.max_retries == 5
    assert client.retry_sleep_seconds == 1.0


def test_public_market_client_retries_incomplete_read(monkeypatch) -> None:
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b'[{"symbol":"AAAUSDT","priceChangePercent":"12.5"}]'

    def fake_urlopen(request, timeout: float):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IncompleteRead(b"x" * 100, 20)
        return Response()

    monkeypatch.setattr(binance_futures, "urlopen", fake_urlopen)
    monkeypatch.setattr(binance_futures.time, "sleep", lambda seconds: None)
    client = BinanceFuturesMarketClient(max_retries=2)

    stats = client.ticker_24hr_stats()

    assert attempts == 2
    assert stats["AAAUSDT"].price_change_percent == 12.5
