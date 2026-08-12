from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from http.client import IncompleteRead
from ssl import SSLError
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.exchange.signing import sign_query


logger = logging.getLogger(__name__)


BINANCE_FUTURES_LIVE_BASE_URL = "https://fapi.binance.com"


SignedTransport = Callable[
    [str, str, dict[str, str | int | float | bool], dict[str, str]],
    object,
]


@dataclass(frozen=True)
class FuturesPosition:
    symbol: str
    position_amt: float
    entry_price: float
    unrealized_profit: float = 0.0


class BinanceLiveAPIError(RuntimeError):
    def __init__(self, status: int, code: int | None, message: str, path: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.path = path
        super().__init__(
            f"Binance Live API error on {path}: HTTP {status}, "
            f"code={code}, msg={message}"
        )


class LiveExecutionClient:
    """Signed Binance USD-M Futures live execution client.

    Market data for signals must still come from src.market.binance_futures.
    This client is only for live account state and order execution.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = BINANCE_FUTURES_LIVE_BASE_URL,
        timeout_seconds: float = 10.0,
        max_retries: int = 5,
        retry_sleep_seconds: float = 1.0,
        timestamp_ms: Callable[[], int] | None = None,
        transport: SignedTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self._timestamp_ms = timestamp_ms or (lambda: int(time.time() * 1000))
        self._transport = transport
        self._clock_offset_ms = 0

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, object]:
        return _as_dict(self._signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}))

    def set_isolated_margin(self, symbol: str) -> dict[str, object]:
        return _as_dict(self._signed_request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}))

    def market_open_long(self, symbol: str, quantity: str) -> dict[str, object]:
        return _as_dict(
            self._signed_request(
                "POST",
                "/fapi/v1/order",
                {"symbol": symbol, "side": "BUY", "type": "MARKET", "quantity": quantity},
            )
        )

    def market_close_long(self, symbol: str, quantity: str) -> dict[str, object]:
        return _as_dict(
            self._signed_request(
                "POST",
                "/fapi/v1/order",
                {
                    "symbol": symbol,
                    "side": "SELL",
                    "type": "MARKET",
                    "quantity": quantity,
                    "reduceOnly": "true",
                },
            )
        )

    def open_positions(self) -> dict[str, FuturesPosition]:
        data = self._signed_request("GET", "/fapi/v2/positionRisk", {})
        if not isinstance(data, list):
            return {}

        positions: dict[str, FuturesPosition] = {}
        for raw_position in data:
            if not isinstance(raw_position, dict):
                continue
            symbol = raw_position.get("symbol")
            position_amt = float(raw_position.get("positionAmt", 0))
            if isinstance(symbol, str) and position_amt != 0:
                positions[symbol] = FuturesPosition(
                    symbol=symbol,
                    position_amt=position_amt,
                    entry_price=float(raw_position.get("entryPrice", 0)),
                    unrealized_profit=float(raw_position.get("unRealizedProfit", 0)),
                )
        return positions

    def account(self) -> dict[str, object]:
        return _as_dict(self._signed_request("GET", "/fapi/v2/account", {}))

    def position_mode(self) -> dict[str, object]:
        return _as_dict(self._signed_request("GET", "/fapi/v1/positionSide/dual", {}))

    def order(self, symbol: str, order_id: int) -> dict[str, object]:
        return _as_dict(self._signed_request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}))

    def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, str | int | float | bool],
    ) -> object:
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            signed_params = self._sign_params(params)
            body = urlencode(_normalize_params(signed_params)).encode("utf-8")
            try:
                if self._transport is not None:
                    return self._transport(method, path, signed_params, headers)
                request = Request(
                    url,
                    data=body if method != "GET" else None,
                    headers=headers,
                    method=method,
                )
                if method == "GET":
                    request = Request(
                        f"{url}?{body.decode('utf-8')}",
                        headers=headers,
                        method=method,
                    )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except BinanceLiveAPIError as exc:
                last_error = exc
                if exc.code == -1021 and attempt + 1 < self.max_retries:
                    self._sync_server_time()
                    continue
                raise
            except HTTPError as exc:
                last_error = exc
                api_error = _api_error_from_http_error(exc, path)
                if api_error.code == -1021 and attempt + 1 < self.max_retries:
                    self._sync_server_time()
                    continue
                if exc.code == 418:
                    raise api_error from exc
                if exc.code == 429:
                    retry_after = _parse_retry_after(exc)
                    logger.warning(
                        "Rate limited (429) on %s, retry_after=%s, attempt=%d/%d",
                        path,
                        retry_after,
                        attempt + 1,
                        self.max_retries,
                    )
                    if attempt + 1 >= self.max_retries:
                        break
                    time.sleep(retry_after or 5.0)
                    continue
                if 400 <= exc.code < 500:
                    raise api_error from exc
                if attempt + 1 >= self.max_retries:
                    raise api_error from exc
                logger.warning("HTTP %d on %s, attempt=%d/%d", exc.code, path, attempt + 1, self.max_retries)
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
            except (TimeoutError, URLError, SSLError, IncompleteRead) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                logger.warning("Request failed on %s, attempt=%d/%d: %s", path, attempt + 1, self.max_retries, exc)
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(
            f"Live API request failed after {self.max_retries} retries: {path}"
        ) from last_error

    def _sign_params(
        self,
        params: dict[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        signed_params: dict[str, str | int | float | bool] = {
            **params,
            "recvWindow": 5000,
            "timestamp": self._timestamp_ms() + self._clock_offset_ms,
        }
        query_string = urlencode(_normalize_params(signed_params))
        signed_params["signature"] = sign_query(query_string, self.api_secret)
        return signed_params

    def _sync_server_time(self) -> None:
        local_before = self._timestamp_ms()
        if self._transport is not None:
            payload = self._transport("GET", "/fapi/v1/time", {}, {})
        else:
            request = Request(
                f"{self.base_url}/fapi/v1/time",
                headers={"User-Agent": "Top3MomentumTrader/0.1"},
                method="GET",
            )
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        local_after = self._timestamp_ms()
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise RuntimeError("Invalid Binance live server time response.")
        local_midpoint = (local_before + local_after) // 2
        self._clock_offset_ms = int(payload["serverTime"]) - local_midpoint
        logger.warning("Synchronized Binance live clock offset: %dms", self._clock_offset_ms)


def _parse_retry_after(exc: HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
    if value is not None:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    return None


def _api_error_from_http_error(exc: HTTPError, path: str) -> BinanceLiveAPIError:
    code: int | None = None
    message = str(exc.reason)
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        if isinstance(payload, dict):
            raw_code = payload.get("code")
            code = int(raw_code) if raw_code is not None else None
            message = str(payload.get("msg", message))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return BinanceLiveAPIError(exc.code, code, message, path)


def _normalize_params(params: dict[str, str | int | float | bool]) -> dict[str, str | int | float]:
    normalized: dict[str, str | int | float] = {}
    for key, value in params.items():
        if isinstance(value, bool):
            normalized[key] = str(value).lower()
        else:
            normalized[key] = value
    return normalized


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected Binance response object.")
    return value
