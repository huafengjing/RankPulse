# Top3 Signal Rules

Last updated: 2026-06-17

## Signal Source

Signals come from a locally reconstructed Binance USD-M Futures 24H gain leaderboard.

The system does not use the current Binance web leaderboard as historical data. It rebuilds historical rankings from local 1H kline cache.

## Ranking Method

For each snapshot time:

1. For every available USDT perpetual symbol, calculate `gain_24h`.
2. Sort symbols by `gain_24h` descending.
3. Assign ranks starting from 1.
4. Keep Top3 for research.

Current strategy trades only Rank2 and Rank3.

## Time Windows

Signal snapshots:

- Beijing 00:00
- Beijing 08:00

Internally, all timestamps are UTC millisecond timestamps.

## Required Filters

A signal must pass:

- Rank is Rank2 or Rank3.
- Symbol is not `RAVEUSDT`.
- `gain_24h < 80%`.
- Signal time is Beijing 00:00 or 08:00.
- Bucket-specific rules pass.
- Same symbol has no active open position.

## `gain_24h` Buckets

| Bucket | Rule |
|---|---|
| `<10%` | No trade |
| `10%-20%` | Trade Rank2 and Rank3 |
| `20%-40%` | Trade only if `1.5 <= volume_24h_ratio_7d < 5` |
| `40%-60%` | Trade only if Rank2 and `3 <= volume_24h_ratio_7d < 6` |
| `60%-80%` | No trade |
| `>=80%` | No trade |

## Volume Factor

`volume_24h_ratio_7d` is calculated before the signal using completed 4H candles:

```text
sum(last 6 completed 4H volumes)
/
(sum(last 42 completed 4H volumes) / 7)
```

No future candles are used.

## Position Blocking

If a symbol is already in a position, new signals for the same symbol are skipped until that position exits.
