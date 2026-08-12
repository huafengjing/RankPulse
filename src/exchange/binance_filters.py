from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolFilters:
    step_size: float
    min_qty: float
    min_notional: float

    @property
    def quantity_precision(self) -> int:
        text = f"{self.step_size:.16f}".rstrip("0")
        if "." not in text:
            return 0
        return len(text.split(".", 1)[1])


def filters_from_exchange_symbol(raw_symbol: dict[str, object]) -> SymbolFilters:
    step_size = 0.0
    min_qty = 0.0
    min_notional = 0.0
    filters = raw_symbol.get("filters", [])
    if isinstance(filters, list):
        for raw_filter in filters:
            if not isinstance(raw_filter, dict):
                continue
            if raw_filter.get("filterType") == "LOT_SIZE":
                step_size = float(raw_filter.get("stepSize", 0))
                min_qty = float(raw_filter.get("minQty", 0))
            if raw_filter.get("filterType") == "MIN_NOTIONAL":
                min_notional = float(raw_filter.get("notional", raw_filter.get("minNotional", 0)))
    return SymbolFilters(step_size=step_size, min_qty=min_qty, min_notional=min_notional)
