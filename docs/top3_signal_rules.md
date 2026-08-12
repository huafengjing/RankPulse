# Top3 Signal Rules

Last updated: 2026-06-17

## Signal Source

Signals for paper/live trading come from a Binance USD-M Futures 24H ticker ranking.

The system does not scrape the Binance web leaderboard. For paper/live trading, it uses Binance public Futures ticker data and sorts symbols by `/fapi/v1/ticker/24hr` `priceChangePercent`.

The original backtest used a locally reconstructed Binance USD-M Futures 24H gain leaderboard from local 1H kline cache. That reconstructed K-line source is retained as a reference mode for comparing paper/live output against the backtest basis.

For paper/live data ingestion, the system must not scrape the Binance web leaderboard. It should use Binance USD-M Futures public market data endpoints:

- `GET /fapi/v1/exchangeInfo` to discover tradable USDT perpetual symbols.
- `GET /fapi/v1/ticker/24hr` to build the default paper/live 24H ticker ranking.
- `GET /fapi/v1/klines` with `1h` interval to reconstruct reference/backtest `gain_24h`.
- `GET /fapi/v1/klines` with `4h` interval to calculate `volume_24h_ratio_7d`.
- `GET /fapi/v2/ticker/price` to read latest public prices for simulated fill/reference pricing.

## Ranking Source Modes

The default paper/live ranking source is:

```text
RANK_SOURCE=BINANCE_24HR_TICKER
```

This mode sorts Binance USD-M Futures symbols by `/fapi/v1/ticker/24hr` `priceChangePercent`. It is intended to track the real Binance 24H Futures movers ranking used for live/paper observation.

The original backtest-compatible reference source can be enabled explicitly:

```text
RANK_SOURCE=KLINE_RECONSTRUCTED
```

This reference mode reconstructs the 24H gain from 1H K-line opens and can differ from the live Binance ticker ranking.

Paper trading should record both Top3 lists when comparison mode is used:

- Binance 24H ticker Top3.
- Kline reconstructed reference Top3.
- Rank-level symbol differences between the two lists.

## Ranking Method

For each snapshot time:

1. For every available USDT perpetual symbol, calculate `gain_24h`.
2. Sort symbols by `gain_24h` descending.
3. Assign ranks starting from 1.
4. Keep Top3 for research.

Current strategy trades only Rank2 and Rank3 from the configured ranking source. For paper/live defaults, that means the real-time Binance 24H ticker Rank2 and Rank3 at the Beijing 00:00 and 08:00 signal checks.

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
| `20%-40%` | Trade Rank2 only if `1.5 <= volume_24h_ratio_7d < 5`; trade Rank3 only if `1.2 <= volume_24h_ratio_7d < 5` |
| `40%-60%` | Trade only if Rank2 and `3 <= volume_24h_ratio_7d < 5.5` |
| `60%-80%` | No trade |
| `>=80%` | No trade |

## Volume Factor

`volume_24h_ratio_7d` is calculated before the signal using completed 4H candles:

```text
sum(last 6 completed 4H volumes)
/
(sum(last 42 completed 4H volumes) / 7)
```

The 42 completed 4H candles include the last 6 candles used in the recent 24H numerator. This inclusive 7D denominator is the current backtest basis and must be preserved for the first paper/live implementation.

No future candles are used.

## Position Blocking

If a symbol is already in a position, new signals for the same symbol are skipped until that position exits.


