from __future__ import annotations

import json
import logging
import time
from http.client import IncompleteRead
from dataclasses import dataclass
from ssl import SSLError
from typing import Callable
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


BINANCE_USDM_FUTURES_BASE_URL = "https://fapi.binance.com"


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int


@dataclass(frozen=True)
class Ticker24hrStat:
    symbol: str
    price_change_percent: float
    last_price: float | None = None
    open_price: float | None = None
    close_time_ms: int | None = None


Transport = Callable[[str, dict[str, str | int]], object]


class BinanceFuturesMarketClient:
    """Read-only client for Binance USD-M Futures public market data."""

    def __init__(
        self,
        base_url: str = BINANCE_USDM_FUTURES_BASE_URL,
        timeout_seconds: float = 10.0,
        max_retries: int = 5,
        retry_sleep_seconds: float = 1.0,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self._transport = transport
        self._exchange_info_cache: tuple[float, dict[str, object]] | None = None

    def exchange_info(self) -> dict[str, object]:
        now = time.time()
        if self._exchange_info_cache is not None and now - self._exchange_info_cache[0] < 600:
            return self._exchange_info_cache[1]
        data = self._get_json("/fapi/v1/exchangeInfo", {})
        if isinstance(data, dict):
            self._exchange_info_cache = (now, data)
        else:
            raise ValueError("Binance exchangeInfo response must be an object.")
        return data

    def usdt_perpetual_symbols(self) -> list[str]:
        symbols = self.exchange_info().get("symbols", [])
        if not isinstance(symbols, list):
            return []

        result: list[str] = []
        for raw_symbol in symbols:
            if not isinstance(raw_symbol, dict):
                continue
            symbol = raw_symbol.get("symbol")
            if (
                isinstance(symbol, str)
                and raw_symbol.get("contractType") == "PERPETUAL"
                and raw_symbol.get("status") == "TRADING"
                and raw_symbol.get("quoteAsset") == "USDT"
                and raw_symbol.get("marginAsset") == "USDT"
            ):
                result.append(symbol)
        return sorted(result)

    def exchange_symbol(self, symbol: str) -> dict[str, object]:
        symbols = self.exchange_info().get("symbols", [])
        if isinstance(symbols, list):
            for raw_symbol in symbols:
                if isinstance(raw_symbol, dict) and raw_symbol.get("symbol") == symbol:
                    return raw_symbol
        raise ValueError(f"Exchange symbol not found: {symbol}")

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        data = self._get_json("/fapi/v1/klines", params)
        if not isinstance(data, list):
            raise ValueError("Binance klines response must be a list.")
        return [_parse_kline(raw_kline) for raw_kline in data]

    def one_hour_klines(
        self,
        symbol: str,
        limit: int = 25,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        return self.klines(symbol=symbol, interval="1h", limit=limit, end_time_ms=end_time_ms)

    def four_hour_klines(
        self,
        symbol: str,
        limit: int = 42,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        return self.klines(symbol=symbol, interval="4h", limit=limit, end_time_ms=end_time_ms)

    def latest_prices(self) -> dict[str, float]:
        data = self._get_json("/fapi/v2/ticker/price", {})
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("Binance ticker price response must be an object or list.")

        prices: dict[str, float] = {}
        for raw_price in data:
            if not isinstance(raw_price, dict):
                continue
            symbol = raw_price.get("symbol")
            price = raw_price.get("price")
            if isinstance(symbol, str) and isinstance(price, str):
                prices[symbol] = float(price)
        return prices

    def ticker_24hr_stats(self) -> dict[str, Ticker24hrStat]:
        data = self._get_json("/fapi/v1/ticker/24hr", {})
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("Binance 24hr ticker response must be an object or list.")

        stats: dict[str, Ticker24hrStat] = {}
        for raw_stat in data:
            if not isinstance(raw_stat, dict):
                continue
            symbol = raw_stat.get("symbol")
            price_change_percent = raw_stat.get("priceChangePercent")
            if not isinstance(symbol, str) or not isinstance(price_change_percent, str):
                continue

            stats[symbol] = Ticker24hrStat(
                symbol=symbol,
                price_change_percent=float(price_change_percent),
                last_price=_optional_float(raw_stat.get("lastPrice")),
                open_price=_optional_float(raw_stat.get("openPrice")),
                close_time_ms=_optional_int(raw_stat.get("closeTime")),
            )
        return stats

    def _get_json(self, path: str, params: dict[str, str | int]) -> object:
        if self._transport is not None:
            return self._transport(path, params)

        query = urlencode(params)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers={"User-Agent": "Top3MomentumTrader/0.1"})
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = exc
                if exc.code == 418:
                    raise RuntimeError(
                        f"Binance API blocked (HTTP 418) for {path}. Do not retry."
                    ) from exc
                if exc.code == 429:
                    retry_after = _parse_retry_after(exc)
                    logger.warning(
                        "Rate limited (429) on %s, retry_after=%s, attempt=%d/%d",
                        path, retry_after, attempt + 1, self.max_retries,
                    )
                    if attempt + 1 >= self.max_retries:
                        break
                    time.sleep(retry_after or 5.0)
                    continue
                if attempt + 1 >= self.max_retries:
                    break
                logger.warning(
                    "HTTP %d on %s, attempt=%d/%d", exc.code, path,
                    attempt + 1, self.max_retries,
                )
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
            except (TimeoutError, URLError, SSLError, IncompleteRead) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                logger.warning(
                    "Request failed on %s, attempt=%d/%d: %s",
                    path, attempt + 1, self.max_retries, exc,
                )
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(
            f"Market data request failed after {self.max_retries} retries: {path}"
        ) from last_error


def _parse_kline(raw_kline: object) -> Kline:
    if not isinstance(raw_kline, list) or len(raw_kline) < 7:
        raise ValueError("Binance kline item must be a list with at least 7 fields.")

    return Kline(
        open_time_ms=int(raw_kline[0]),
        open=float(raw_kline[1]),
        high=float(raw_kline[2]),
        low=float(raw_kline[3]),
        close=float(raw_kline[4]),
        volume=float(raw_kline[5]),
        close_time_ms=int(raw_kline[6]),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_retry_after(exc: HTTPError) -> float | None:
    """Extract Retry-After header from a rate-limit response."""
    value = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
    if value is not None:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    return None
