from __future__ import annotations

from src.backtest.execution import apply_long_entry_slippage, apply_long_exit_slippage, net_return_pct


def test_fees_and_slippage_are_deducted() -> None:
    entry = apply_long_entry_slippage(100, 0.001)
    exit_price = apply_long_exit_slippage(110, 0.001)
    net, fee = net_return_pct(entry, exit_price, 0.0005)
    assert entry == 100.1
    assert exit_price == 109.89
    assert fee > 0
    assert net < 0.10
