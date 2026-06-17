# Top3 Strategy Specification

Last updated: 2026-06-17

This document is the current executable research specification for the Binance USD-M Futures Top3 momentum strategy. The project is still in Research mode. This is not an automated live trading system specification.

## 1. Strategy Scope

- Market: Binance USD-M Futures USDT perpetual contracts.
- Direction: Long only.
- Signal source: historical reconstructed 24H gain leaderboard from local 1H kline cache.
- Leaderboard universe: all locally available Binance USD-M USDT perpetual symbols.
- Signal ranking source: Top3 by reconstructed `gain_24h`.
- Active ranks: Rank2 and Rank3 only.
- Rank1 status: excluded. Rank1 was retested under current rules and remained negative EV.
- Excluded symbol: `RAVEUSDT`.
- Capital unit in backtest: 100 USDT margin per trade.
- Time standard in data: UTC millisecond timestamp.
- Signal clock shown to user: Beijing time.

## 2. Signal Timing

Signals are checked twice per day:

- Beijing time 00:00
- Beijing time 08:00

The corresponding UTC timestamps are used internally. The entry price is the 1H kline open at the signal timestamp.

## 3. Rank Selection

At each signal timestamp:

1. Reconstruct all symbols' 24H gain from local 1H klines.
2. Sort by `gain_24h` descending.
3. Keep Top3 ranks.
4. Strategy only trades Rank2 and Rank3.

Rank1 is not traded in the current main strategy.

## 4. `gain_24h` Buckets

`gain_24h` is the signal-time 24H gain used to rank symbols.

Current trading buckets:

| `gain_24h` bucket | Rule |
|---|---|
| `<10%` | Do not trade |
| `10%-20%` | Trade Rank2 and Rank3, no volume filter |
| `20%-40%` | Trade only if `1.5 <= volume_24h_ratio_7d < 5` |
| `40%-60%` | Trade only if Rank2 and `3 <= volume_24h_ratio_7d < 6` |
| `60%-80%` | Fully filtered, no trade |
| `>=80%` | Fully filtered, no trade |

Important: 60%-80% signals are currently filtered entirely. Rank3 is not retained in this bucket.

## 5. `volume_24h_ratio_7d`

`volume_24h_ratio_7d` is a pre-trade factor calculated from completed 4H candles before the signal.

Definition:

```text
volume_24h_ratio_7d
= volume of last 24H
/ average daily volume of the previous 7D
```

Implementation detail:

- Last 24H volume = sum of last 6 completed 4H candle volumes.
- Previous 7D daily average = sum of last 42 completed 4H candle volumes / 7.
- The factor does not use future data.

Current use:

- 20%-40% bucket: keep `1.5 <= volume_24h_ratio_7d < 5`.
- 40%-60% bucket: keep Rank2 only and `3 <= volume_24h_ratio_7d < 6`.

## 6. Entry Rules

A trade is opened only when all conditions pass:

1. Symbol is in reconstructed Top3.
2. Rank is Rank2 or Rank3.
3. Signal time is Beijing 00:00 or 08:00.
4. Symbol is not `RAVEUSDT`.
5. `gain_24h < 80%`.
6. Bucket-specific rules pass.
7. The same symbol is not already in an active position.

If the same symbol already has an open position, the new signal is skipped. Position blocking lasts until that trade's actual exit time, including 12H early exits.

## 7. Exit Rules

Default exit:

- Fixed holding period: 6 days.
- Planned exit time: `entry_time + 6D`.
- Exit price: corresponding 1H kline open.

Early weak exit:

If the first 12H after entry satisfies all three conditions:

```text
12H MFE < 5%
12H close_return < 0%
12H MAE < -5%
```

then exit at 12H using the corresponding 1H kline open.

There is no current take-profit rule. There is no ordinary fixed stop-loss in the 1X research path. Leveraged backtests use liquidation assumptions as described below.

If the exact planned exit kline is unavailable, the current research scripts use the latest available 1H kline after entry as the fallback exit. This is a backtest data-completeness convention, not a live trading instruction.

## 8. Leverage Rules

Current tuned leverage mapping:

| Bucket / condition | Leverage |
|---|---:|
| 10%-20% | 3X |
| 20%-40% Rank2 | 3X |
| 20%-40% Rank3 | 5X |
| 40%-60% Rank2 and `3 <= volume_24h_ratio_7d < 6` | 2X |
| 60%-80% | No trade |

Backtest liquidation assumptions:

| Leverage | Underlying MAE liquidation threshold |
|---:|---:|
| 2X | `MAE <= -50%` |
| 3X | `MAE <= -33%` |
| 5X | `MAE <= -20%` |

When liquidation is triggered, that trade is recorded as `-100U`.

## 9. Fees and Slippage

Backtest assumptions:

- Buy fee: 0.1%.
- Sell fee: 0.1%.
- Slippage: 0 in current research output.

These are backtest assumptions only. They are not sufficient for live trading. A live or paper execution model must separately account for spread, order type, funding, latency, partial fills, exchange outages, and liquidation mechanics.

## 10. Risk Controls

Current research controls:

- Rank1 excluded.
- `RAVEUSDT` excluded.
- `gain_24h >= 80%` excluded.
- 60%-80% bucket fully excluded.
- 40%-60% bucket requires Rank2 and volume ratio 3-6.
- Same symbol cannot be opened twice while an existing position is active.
- 12H weak path exit reduces exposure to failed continuation.
- Backtest reports liquidation count, max drawdown, and tail dependency after removing top winners.

Not yet implemented as live controls:

- Account-level max loss.
- Daily loss stop.
- Max simultaneous positions.
- Exchange-side stop order placement.
- Funding-rate filter.
- BTC/market regime filter.
- Liquidity/spread filter.
- Kill switch.

## 11. Current Backtest Snapshot

Latest current-main result for recent half-year window, holding 6D:

| Version | Trades | Liquidations | Net PnL | PF | Win rate | Median return | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default all 3X | 253 | 26 | +6965.17U | 1.95 | 33.99% | -17.67% | -696.60U |
| Current tuned leverage | 253 | 33 | +10629.58U | 2.31 | 33.20% | -19.28% | -754.11U |

The tuned leverage version improves net PnL and PF, but increases liquidation count and slightly worsens median return and drawdown.

## 12. Deployment Stage

Current stage: research/backtest only.

Recommended startup stage if execution is ever tested:

1. Paper trading / simulation first.
2. Then very small capital live test only after explicit approval and separate execution-risk review.

No automatic live order placement is part of the current project scope.
