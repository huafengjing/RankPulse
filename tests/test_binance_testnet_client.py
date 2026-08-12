from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from src.exchange.binance_testnet import (
    BinanceTestnetAPIError,
    TestnetExecutionClient,
    _api_error_from_http_error,
)
from src.exchange.signing import sign_query


def test_sign_query_uses_hmac_sha256() -> None:
    signature = sign_query(
        "symbol=BTCUSDT&side=BUY&type=MARKET&quantity=1&timestamp=1700000000000",
        "secret",
    )

    assert signature == "24175f9abcf847730fec764ea3aad689557f21e22eacd6fee1255efc58c5f984"


def test_testnet_client_uses_demo_fapi_base_url_and_api_key_header() -> None:
    calls: list[tuple[str, str, dict[str, str | int | float | bool], dict[str, str]]] = []

    def transport(method: str, path: str, params: dict[str, str | int | float | bool], headers: dict[str, str]) -> object:
        calls.append((method, path, params, headers))
        return {"orderId": 123}

    client = TestnetExecutionClient(
        api_key="key",
        api_secret="secret",
        timestamp_ms=lambda: 1_700_000_000_000,
        transport=transport,
    )

    client.market_open_long("BTCUSDT", quantity="0.001")

    method, path, params, headers = calls[0]
    assert client.base_url == "https://testnet.binancefuture.com"
    assert method == "POST"
    assert path == "/fapi/v1/order"
    assert headers["X-MBX-APIKEY"] == "key"
    assert params["symbol"] == "BTCUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "MARKET"
    assert "signature" in params


def test_testnet_client_supports_leverage_close_position_positions_and_order_query() -> None:
    paths: list[tuple[str, str]] = []

    def transport(method: str, path: str, params: dict[str, str | int | float | bool], headers: dict[str, str]) -> object:
        paths.append((method, path))
        if path == "/fapi/v2/positionRisk":
            return [{"symbol": "BTCUSDT", "positionAmt": "0.001", "entryPrice": "60000"}]
        return {"ok": True}

    client = TestnetExecutionClient(
        api_key="key",
        api_secret="secret",
        timestamp_ms=lambda: 1_700_000_000_000,
        transport=transport,
    )

    client.set_leverage("BTCUSDT", 3)
    client.market_close_long("BTCUSDT", quantity="0.001")
    positions = client.open_positions()
    client.order("BTCUSDT", order_id=123)

    assert paths == [
        ("POST", "/fapi/v1/leverage"),
        ("POST", "/fapi/v1/order"),
        ("GET", "/fapi/v2/positionRisk"),
        ("GET", "/fapi/v1/order"),
    ]
    assert positions["BTCUSDT"].position_amt == 0.001


def test_testnet_execution_client_does_not_expose_market_data_methods() -> None:
    client = TestnetExecutionClient(api_key="key", api_secret="secret")

    assert not hasattr(client, "ticker_24hr_stats")
    assert not hasattr(client, "four_hour_klines")
    assert not hasattr(client, "latest_prices")


def test_binance_http_error_preserves_api_code_and_message() -> None:
    error = HTTPError(
        url="https://testnet.binancefuture.com/fapi/v1/order",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=BytesIO(b'{"code":-2022,"msg":"ReduceOnly Order is rejected."}'),
    )

    api_error = _api_error_from_http_error(error, "/fapi/v1/order")

    assert api_error.code == -2022
    assert api_error.message == "ReduceOnly Order is rejected."
    assert "code=-2022" in str(api_error)


def test_timestamp_error_syncs_server_time_and_retries_signed_request() -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []
    position_attempts = 0

    def transport(method: str, path: str, params: dict, headers: dict) -> object:
        nonlocal position_attempts
        calls.append((method, path, dict(params)))
        if path == "/fapi/v1/time":
            return {"serverTime": 1_700_000_000_000}
        if path == "/fapi/v2/positionRisk":
            position_attempts += 1
            if position_attempts == 1:
                raise BinanceTestnetAPIError(
                    status=400,
                    code=-1021,
                    message="Timestamp ahead",
                    path=path,
                )
            return []
        raise AssertionError(path)

    client = TestnetExecutionClient(
        api_key="key",
        api_secret="secret",
        timestamp_ms=lambda: 1_700_000_001_500,
        transport=transport,
    )

    assert client.open_positions() == {}
    assert [path for _, path, _ in calls] == [
        "/fapi/v2/positionRisk",
        "/fapi/v1/time",
        "/fapi/v2/positionRisk",
    ]
    assert calls[-1][2]["timestamp"] == 1_700_000_000_000
