from __future__ import annotations

from http.client import IncompleteRead
from io import BytesIO
from urllib.error import HTTPError

from src.exchange.binance_live import (
    BinanceLiveAPIError,
    LiveExecutionClient,
    _api_error_from_http_error,
)


def test_live_client_uses_production_fapi_base_url_and_api_key_header() -> None:
    calls: list[tuple[str, str, dict[str, str | int | float | bool], dict[str, str]]] = []

    def transport(method: str, path: str, params: dict[str, str | int | float | bool], headers: dict[str, str]) -> object:
        calls.append((method, path, params, headers))
        return {"orderId": 123}

    client = LiveExecutionClient(
        api_key="key",
        api_secret="secret",
        timestamp_ms=lambda: 1_700_000_000_000,
        transport=transport,
    )

    client.market_open_long("BTCUSDT", quantity="0.001")

    method, path, params, headers = calls[0]
    assert client.base_url == "https://fapi.binance.com"
    assert method == "POST"
    assert path == "/fapi/v1/order"
    assert headers["X-MBX-APIKEY"] == "key"
    assert params["symbol"] == "BTCUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "MARKET"
    assert "signature" in params


def test_live_client_defaults_to_five_retries_with_longer_backoff() -> None:
    client = LiveExecutionClient(api_key="key", api_secret="secret")

    assert client.max_retries == 5
    assert client.retry_sleep_seconds == 1.0


def test_live_execution_client_does_not_expose_market_data_methods() -> None:
    client = LiveExecutionClient(api_key="key", api_secret="secret")

    assert not hasattr(client, "ticker_24hr_stats")
    assert not hasattr(client, "four_hour_klines")
    assert not hasattr(client, "latest_prices")


def test_live_client_supports_read_only_account_position_mode_and_positions() -> None:
    paths: list[tuple[str, str]] = []

    def transport(method: str, path: str, params: dict[str, str | int | float | bool], headers: dict[str, str]) -> object:
        paths.append((method, path))
        if path == "/fapi/v2/account":
            return {"canTrade": True, "assets": [{"asset": "USDT", "walletBalance": "12.5"}]}
        if path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": False}
        if path == "/fapi/v2/positionRisk":
            return [{"symbol": "BTCUSDT", "positionAmt": "0.001", "entryPrice": "60000"}]
        raise AssertionError(path)

    client = LiveExecutionClient(
        api_key="key",
        api_secret="secret",
        timestamp_ms=lambda: 1_700_000_000_000,
        transport=transport,
    )

    account = client.account()
    position_mode = client.position_mode()
    positions = client.open_positions()

    assert account["canTrade"] is True
    assert position_mode["dualSidePosition"] is False
    assert positions["BTCUSDT"].position_amt == 0.001
    assert paths == [
        ("GET", "/fapi/v2/account"),
        ("GET", "/fapi/v1/positionSide/dual"),
        ("GET", "/fapi/v2/positionRisk"),
    ]


def test_live_client_retries_incomplete_read_from_signed_request() -> None:
    attempts = 0

    def transport(method: str, path: str, params: dict[str, str | int | float | bool], headers: dict[str, str]) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise IncompleteRead(b"x" * 100, 20)
        return {"canTrade": True}

    client = LiveExecutionClient(
        api_key="key",
        api_secret="secret",
        timestamp_ms=lambda: 1_700_000_000_000,
        transport=transport,
        max_retries=2,
        retry_sleep_seconds=0,
    )

    assert client.account()["canTrade"] is True
    assert attempts == 2


def test_live_http_error_preserves_api_code_and_message() -> None:
    error = HTTPError(
        url="https://fapi.binance.com/fapi/v1/order",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=BytesIO(b'{"code":-4131,"msg":"Market order rejected."}'),
    )

    api_error = _api_error_from_http_error(error, "/fapi/v1/order")

    assert isinstance(api_error, BinanceLiveAPIError)
    assert api_error.code == -4131
    assert api_error.message == "Market order rejected."
    assert "Binance Live API error" in str(api_error)
