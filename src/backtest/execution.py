from __future__ import annotations


def apply_long_entry_slippage(raw_entry_price: float, slippage_rate: float) -> float:
    return raw_entry_price * (1.0 + slippage_rate)


def apply_long_exit_slippage(raw_exit_price: float, slippage_rate: float) -> float:
    return raw_exit_price * (1.0 - slippage_rate)


def net_return_pct(entry_effective: float, exit_effective: float, fee_rate: float) -> tuple[float, float]:
    gross = exit_effective / entry_effective - 1.0
    fee_paid = fee_rate + fee_rate * (exit_effective / entry_effective)
    return gross - fee_paid, fee_paid
