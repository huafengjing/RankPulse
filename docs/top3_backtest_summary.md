# Top3 Backtest Summary

Last updated: 2026-06-17

This summary reflects the current main strategy research output using local cached Binance USD-M Futures 1H kline data.

## Current Main Strategy

- Signal source: reconstructed Top3 by 24H gain.
- Active ranks: Rank2 and Rank3.
- Signal times: Beijing 00:00 and 08:00.
- Holding period: 6D unless 12H weak-exit condition triggers.
- Excluded: Rank1, `RAVEUSDT`, `gain_24h >= 80%`, and all 60%-80% gain signals.
- Fees: 0.1% buy and 0.1% sell.
- Slippage: 0 in backtest.
- Position rule: same symbol cannot be opened again while position is active.

## Overall Results

| Version | Trades | Liquidations | 12H early exits | Net PnL | PF | Win rate | Median return | Max drawdown | Drop top1 | Drop top3 | Drop top5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default all 3X | 253 | 26 | 76 | +6965.17U | 1.95 | 33.99% | -17.67% | -696.60U | +4267.60U | +2436.43U | +1124.80U |
| Current tuned leverage | 253 | 33 | 76 | +10629.58U | 2.31 | 33.20% | -19.28% | -754.11U | +6133.62U | +3713.36U | +1640.36U |

## Monthly Results: Current Tuned Leverage

| Month | Trades | Liquidations | Net PnL | PF | Win rate | Median return | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-12 | 18 | 1 | +551.63U | 2.59 | 38.89% | -13.42% | -207.00U |
| 2026-01 | 41 | 4 | +1091.25U | 1.94 | 41.46% | -15.47% | -483.37U |
| 2026-02 | 41 | 5 | +631.53U | 1.54 | 34.15% | -18.52% | -428.14U |
| 2026-03 | 54 | 7 | +1672.67U | 1.92 | 29.63% | -23.25% | -417.64U |
| 2026-04 | 33 | 5 | +1228.65U | 2.04 | 27.27% | -23.63% | -660.79U |
| 2026-05 | 48 | 8 | +320.02U | 1.19 | 31.25% | -24.06% | -528.70U |
| 2026-06 | 18 | 3 | +5133.83U | 7.73 | 33.33% | -35.13% | -345.38U |

## Bucket Results: Current Tuned Leverage

| Gain bucket | Trades | Liquidations | Net PnL | PF | Win rate | Median return | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10%-20% | 46 | 2 | +1078.49U | 2.15 | 28.26% | -12.31% | -252.11U |
| 20%-40% | 162 | 27 | +10284.02U | 2.82 | 37.65% | -20.09% | -640.70U |
| 40%-60% | 45 | 4 | -732.93U | 0.51 | 22.22% | -25.81% | -917.36U |

## Rank1 Retest

Rank1 was retested under the current filtering framework.

| Version | Trades | Liquidations | Net PnL | PF | Win rate | Median return | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rank1 1X | 36 | 0 | -105.87U | 0.72 | 30.56% | -9.25% | -241.61U |
| Rank1 3X | 36 | 7 | -617.94U | 0.57 | 30.56% | -32.11% | -994.35U |

Conclusion: Rank1 remains excluded.

## Interpretation

The strongest bucket is 20%-40%, especially Rank3 under the tuned leverage mapping. The 40%-60% bucket remains structurally weak even after stricter filtering, so it is kept small with 2X leverage only. 60%-80% is fully filtered.

This result is still research-grade. The current backtest has no slippage, no funding, no order book simulation, and simplified liquidation handling.
