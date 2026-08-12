from __future__ import annotations

import time
from dataclasses import asdict
from typing import Protocol

from src.config.schedule import exit_after_ms
from src.config.settings import AppSettings
from src.exchange.binance_filters import SymbolFilters
from src.exchange.binance_testnet import FuturesPosition
from src.execution.quantity import QuantityTooSmallError, format_order_quantity, order_quantity
from src.execution.testnet_state import (
    TestnetClosedPosition,
    TestnetPosition,
    TestnetState,
    TestnetStateStore,
)
from src.paper.store import PaperEventLogger
from src.paper.trading import PaperSignal


class TestnetClientProtocol(Protocol):
    def open_positions(self) -> dict[str, FuturesPosition]:
        ...

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, object]:
        ...

    def set_isolated_margin(self, symbol: str) -> dict[str, object]:
        ...

    def market_open_long(self, symbol: str, quantity: str) -> dict[str, object]:
        ...

    def market_close_long(self, symbol: str, quantity: str) -> dict[str, object]:
        ...

    def order(self, symbol: str, order_id: int) -> dict[str, object]:
        ...


class SymbolFilterProvider(Protocol):
    def filters_for_symbol(self, symbol: str) -> SymbolFilters:
        ...


class PositionPrecheckFailedError(RuntimeError):
    pass


class OrderConfirmationFailedError(RuntimeError):
    pass


class TestnetExecutionEngine:
    __test__ = False

    def __init__(
        self,
        client: TestnetClientProtocol,
        filter_provider: SymbolFilterProvider,
        state_store: TestnetStateStore,
        logger: PaperEventLogger,
        settings: AppSettings,
        event_prefix: str = "testnet",
    ) -> None:
        self.client = client
        self.filter_provider = filter_provider
        self.state_store = state_store
        self.logger = logger
        self.settings = settings
        self.event_prefix = event_prefix

    def open_from_signal(self, signal: PaperSignal, leverage: int) -> TestnetPosition | None:
        state = self.state_store.load()
        if state.open_position(signal.symbol) is not None:
            self._log("testnet_open_skipped", {"symbol": signal.symbol, "reason": "local_position"})
            return None
        if len(state.open_positions) >= self.settings.max_open_positions:
            self._log("testnet_open_skipped", {"symbol": signal.symbol, "reason": "max_open_positions"})
            return None
        try:
            exchange_positions = self.client.open_positions()
        except Exception as exc:
            raise PositionPrecheckFailedError(
                "执行跳过：开仓前读取实盘持仓失败，未下单\n"
                "原因：Binance positionRisk 网络连接中断"
            ) from exc
        if signal.symbol in exchange_positions:
            self._log("testnet_open_skipped", {"symbol": signal.symbol, "reason": "exchange_position"})
            return None

        filters = self.filter_provider.filters_for_symbol(signal.symbol)
        try:
            qty = order_quantity(
                margin_usdt=self.settings.position_margin_usdt,
                leverage=leverage,
                price=signal.fill_price,
                filters=filters,
            )
        except QuantityTooSmallError as exc:
            self._log("testnet_open_skipped", {"symbol": signal.symbol, "reason": str(exc)})
            return None

        try:
            self.client.set_isolated_margin(signal.symbol)
        except Exception as exc:
            if getattr(exc, "code", None) != -4046:
                raise
        self.client.set_leverage(signal.symbol, leverage)
        quantity = format_order_quantity(qty, filters)
        order = self.client.market_open_long(signal.symbol, quantity)
        order_id = int(order.get("orderId", 0))
        entry_price, executed_qty, confirmation_source = self._confirm_open_fill(
            symbol=signal.symbol,
            order_id=order_id,
            fallback_price=signal.fill_price,
            fallback_qty=qty,
        )
        position = TestnetPosition(
            symbol=signal.symbol,
            entry_time_ms=signal.signal_time_ms,
            entry_price=entry_price,
            qty=executed_qty,
            leverage=leverage,
            order_id=order_id,
            planned_exit_time_ms=signal.signal_time_ms + exit_after_ms("planned", self.settings),
            extreme_weak_exit_check_time_ms=signal.signal_time_ms + exit_after_ms("extreme_weak", self.settings),
            weak_exit_check_time_ms=signal.signal_time_ms + exit_after_ms("weak", self.settings),
        )
        self.state_store.save(
            TestnetState(
                open_positions=[*state.open_positions, position],
                closed_positions=state.closed_positions,
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )
        self._log(
            f"{self.event_prefix}_opened",
            {
                **asdict(position),
                "rank": signal.rank,
                "gain_24h": signal.gain_24h,
                "confirmation_source": confirmation_source,
            },
        )
        return position

    def close_position(self, symbol: str, exit_time_ms: int, reason: str) -> TestnetClosedPosition | None:
        state = self.state_store.load()
        position = state.open_position(symbol)
        if position is None:
            return None

        exchange_position = self.client.open_positions().get(symbol)
        if exchange_position is None or exchange_position.position_amt == 0:
            self.state_store.save(
                TestnetState(
                    open_positions=[item for item in state.open_positions if item.symbol != symbol],
                    closed_positions=state.closed_positions,
                    last_signal_time_ms=state.last_signal_time_ms,
                    last_exit_check_time_ms=state.last_exit_check_time_ms,
                    last_information_time_ms=state.last_information_time_ms,
                    last_preflight_time_ms=state.last_preflight_time_ms,
                )
            )
            self._log(
                f"{self.event_prefix}_position_reconciled",
                {
                    "symbol": symbol,
                    "reason": "exchange_position_missing",
                    "local_qty": position.qty,
                },
            )
            return None
        if exchange_position.position_amt < 0:
            self._log(
                f"{self.event_prefix}_close_skipped",
                {
                    "symbol": symbol,
                    "reason": "exchange_position_is_short",
                    "position_amt": exchange_position.position_amt,
                },
            )
            return None

        filters = self.filter_provider.filters_for_symbol(symbol)
        exchange_qty = exchange_position.position_amt
        order = self.client.market_close_long(
            symbol,
            format_order_quantity(exchange_qty, filters),
        )
        exit_order_id = int(order.get("orderId", 0))
        exit_price, exit_qty, confirmation_source = self._confirm_close_fill(
            symbol=symbol,
            order_id=exit_order_id,
            fallback_order=order,
            fallback_price=position.entry_price,
            fallback_qty=exchange_qty,
        )
        closed = TestnetClosedPosition(
            symbol=position.symbol,
            entry_time_ms=position.entry_time_ms,
            exit_time_ms=exit_time_ms,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=exit_qty,
            leverage=position.leverage,
            entry_order_id=position.order_id,
            exit_order_id=exit_order_id,
            realized_pnl=round((exit_price - position.entry_price) * exit_qty, 12),
            exit_reason=reason,
        )
        self.state_store.save(
            TestnetState(
                open_positions=[item for item in state.open_positions if item.symbol != symbol],
                closed_positions=[*state.closed_positions, closed],
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )
        self._log(
            f"{self.event_prefix}_closed",
            {
                **asdict(closed),
                "confirmation_source": confirmation_source,
            },
        )
        return closed

    def _log(self, event: str, payload: dict[str, object]) -> None:
        self.logger.log(
            event,
            {
                **payload,
                "trading_mode": self.settings.trading_mode.value,
                "signal_mode": self.settings.signal_mode.value,
            },
        )

    def _confirm_open_fill(
        self,
        symbol: str,
        order_id: int,
        fallback_price: float,
        fallback_qty: float,
    ) -> tuple[float, float, str]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                filled_order = self.client.order(symbol, order_id)
                return (
                    _order_price(filled_order, fallback=fallback_price),
                    _order_qty(filled_order, fallback=fallback_qty),
                    "order",
                )
            except Exception as exc:
                last_error = exc
                if getattr(exc, "code", None) != -2013:
                    raise
                if attempt < 4:
                    time.sleep(0.25 * (attempt + 1))

        try:
            exchange_position = self.client.open_positions().get(symbol)
        except Exception as exc:
            raise OrderConfirmationFailedError(
                "开仓订单可能已经提交，但查询订单详情和实盘持仓都失败。\n"
                "请立刻运行 live_diag 或在 Binance App 确认真实持仓。"
            ) from exc

        if exchange_position is not None and exchange_position.position_amt > 0:
            self._log(
                f"{self.event_prefix}_open_order_reconciled",
                {
                    "symbol": symbol,
                    "order_id": order_id,
                    "reason": "order_query_not_found_but_exchange_position_exists",
                    "position_amt": exchange_position.position_amt,
                    "entry_price": exchange_position.entry_price,
                    "order_query_error": str(last_error),
                },
            )
            return exchange_position.entry_price, exchange_position.position_amt, "positionRisk"

        raise OrderConfirmationFailedError(
            "开仓订单已提交，但 Binance 暂时查询不到订单详情，也没有查到对应持仓。\n"
            "本次不会写入本地持仓，请运行 live_diag 确认真实账户状态。"
        ) from last_error


    def _confirm_close_fill(
        self,
        symbol: str,
        order_id: int,
        fallback_order: dict[str, object],
        fallback_price: float,
        fallback_qty: float,
    ) -> tuple[float, float, str]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                filled_order = self.client.order(symbol, order_id)
                return (
                    _order_price(filled_order, fallback=_order_price(fallback_order, fallback_price)),
                    _order_qty(filled_order, fallback=_order_qty(fallback_order, fallback_qty)),
                    "order",
                )
            except Exception as exc:
                last_error = exc
                if getattr(exc, "code", None) != -2013:
                    raise
                if attempt < 4:
                    time.sleep(0.25 * (attempt + 1))

        try:
            exchange_position = self.client.open_positions().get(symbol)
        except Exception as exc:
            raise OrderConfirmationFailedError(
                "Close order may have been submitted, but order details and positionRisk both failed.\n"
                "Run live_diag or check Binance App immediately before taking another action."
            ) from exc

        if exchange_position is None or exchange_position.position_amt <= 0:
            exit_price = _order_price(fallback_order, fallback=fallback_price)
            exit_qty = _order_qty(fallback_order, fallback=fallback_qty)
            self._log(
                f"{self.event_prefix}_close_order_reconciled",
                {
                    "symbol": symbol,
                    "order_id": order_id,
                    "reason": "order_query_not_found_but_exchange_position_is_closed",
                    "fallback_exit_price": exit_price,
                    "fallback_exit_qty": exit_qty,
                    "order_query_error": str(last_error),
                },
            )
            return exit_price, exit_qty, "positionRisk"

        raise OrderConfirmationFailedError(
            "Close order was submitted, but Binance still reports an open position after order lookup failed.\n"
            "Local state was not changed. Run live_diag or check Binance App immediately."
        ) from last_error


def _order_price(order: dict[str, object], fallback: float) -> float:
    for key in ("avgPrice", "price"):
        value = order.get(key)
        try:
            f = float(value)  # type: ignore[arg-type]
            if f != 0.0:
                return f
        except (TypeError, ValueError):
            continue
    return fallback


def _order_qty(order: dict[str, object], fallback: float) -> float:
    value = order.get("executedQty")
    try:
        f = float(value)  # type: ignore[arg-type]
        if f != 0.0:
            return f
    except (TypeError, ValueError):
        pass
    return fallback
