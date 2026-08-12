from __future__ import annotations

import math

from src.exchange.binance_filters import SymbolFilters


class QuantityTooSmallError(ValueError):
    pass


def order_quantity(margin_usdt: float, leverage: int, price: float, filters: SymbolFilters) -> float:
    raw_qty = (margin_usdt * leverage) / price
    if filters.step_size <= 0:
        raise QuantityTooSmallError("Invalid stepSize.")

    stepped_qty = math.floor(raw_qty / filters.step_size) * filters.step_size
    stepped_qty = round(stepped_qty, filters.quantity_precision)
    if stepped_qty < filters.min_qty:
        raise QuantityTooSmallError("Quantity is below minQty.")
    if stepped_qty * price < filters.min_notional:
        raise QuantityTooSmallError("Quantity notional is below minNotional.")
    return stepped_qty


def format_order_quantity(quantity: float, filters: SymbolFilters) -> str:
    precision = filters.quantity_precision
    return f"{quantity:.{precision}f}" if precision else str(int(quantity))
