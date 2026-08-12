from __future__ import annotations

import pytest

from src.exchange.binance_filters import SymbolFilters
from src.execution.quantity import QuantityTooSmallError, format_order_quantity, order_quantity


def test_order_quantity_uses_margin_times_leverage_and_rounds_down_to_step_size() -> None:
    filters = SymbolFilters(step_size=0.001, min_qty=0.001, min_notional=5.0)

    qty = order_quantity(margin_usdt=10, leverage=3, price=20000, filters=filters)

    assert qty == 0.001
    assert format_order_quantity(qty, filters) == "0.001"


def test_order_quantity_rejects_below_min_notional_after_rounding() -> None:
    filters = SymbolFilters(step_size=1.0, min_qty=1.0, min_notional=20.0)

    with pytest.raises(QuantityTooSmallError):
        order_quantity(margin_usdt=10, leverage=1, price=10, filters=filters)


def test_order_quantity_respects_min_qty() -> None:
    filters = SymbolFilters(step_size=0.1, min_qty=2.0, min_notional=1.0)

    with pytest.raises(QuantityTooSmallError):
        order_quantity(margin_usdt=10, leverage=1, price=100, filters=filters)
