# Top3 Leverage Rules

Last updated: 2026-06-17

## Base Unit

- Backtest margin per trade: 100U.
- Fees: 0.1% buy and 0.1% sell.
- Slippage: 0 in research output.

## Current Tuned Leverage

| Signal condition | Leverage |
|---|---:|
| `10% <= gain_24h < 20%` | 3X |
| `20% <= gain_24h < 40%` and Rank2 | 3X |
| `20% <= gain_24h < 40%` and Rank3 | 5X |
| `40% <= gain_24h < 60%`, Rank2, `3 <= volume_24h_ratio_7d < 6` | 2X |
| `60% <= gain_24h < 80%` | No trade |

## Liquidation Assumptions

| Leverage | Underlying MAE threshold | Backtest result when triggered |
|---:|---:|---:|
| 2X | `MAE <= -50%` | `-100U` |
| 3X | `MAE <= -33%` | `-100U` |
| 5X | `MAE <= -20%` | `-100U` |

## Interpretation

The tuned leverage version improved recent-half backtest PnL and PF, but increased liquidation count.

Current tuned leverage result:

- Trades: 253.
- Liquidations: 33.
- Net PnL: +10629.58U.
- PF: 2.31.
- Max drawdown: -754.11U.

The default all-3X comparison:

- Trades: 253.
- Liquidations: 26.
- Net PnL: +6965.17U.
- PF: 1.95.
- Max drawdown: -696.60U.

The tuned version is more profitable in the tested sample, but not lower risk.
