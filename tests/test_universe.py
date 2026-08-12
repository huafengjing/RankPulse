from __future__ import annotations

from src.data.universe import eligible_symbols


def test_eligible_symbols_keeps_unicode_symbols() -> None:
    exchange_info = {
        "symbols": [
            {"symbol": "币安人生USDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ABCUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "ABCUSDC", "contractType": "PERPETUAL", "quoteAsset": "USDC", "status": "TRADING"},
            {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "BREAK"},
        ]
    }
    symbols = eligible_symbols(exchange_info, tickers_24h=None, exclude_symbols=["BTCUSDT"])
    assert symbols == ["ABCUSDT", "币安人生USDT"]


def test_eligible_symbols_only_filters_volume_when_tickers_are_provided() -> None:
    exchange_info = {
        "symbols": [
            {"symbol": "LOWUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
            {"symbol": "HIGHUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING"},
        ]
    }
    tickers = [
        {"symbol": "LOWUSDT", "quoteVolume": "100"},
        {"symbol": "HIGHUSDT", "quoteVolume": "100000000"},
    ]
    assert eligible_symbols(exchange_info, tickers_24h=None, min_quote_volume=20_000_000) == ["HIGHUSDT", "LOWUSDT"]
    assert eligible_symbols(exchange_info, tickers_24h=tickers, min_quote_volume=20_000_000) == ["HIGHUSDT"]
