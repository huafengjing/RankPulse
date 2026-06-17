# Top3 Exit Rules

Last updated: 2026-06-17

## Default Exit

- Default holding period: 6 days.
- Exit time: `entry_time + 6D`.
- Exit price: 1H kline open at the exit timestamp.

## 12H Weak Exit

Exit early at 12H if all three conditions are true:

```text
12H MFE < 5%
12H close_return < 0%
12H MAE < -5%
```

Execution in backtest:

- Exit time: `entry_time + 12H`.
- Exit price: corresponding 1H open.

## Take Profit

There is no active take-profit rule in the current main strategy.

Previously researched TP and runner rules are not part of the current main strategy.

## Stop Loss

There is no ordinary fixed stop-loss in the 1X path.

Leveraged backtests use liquidation thresholds:

- 2X: liquidation if underlying MAE reaches -50%.
- 3X: liquidation if underlying MAE reaches -33%.
- 5X: liquidation if underlying MAE reaches -20%.

If liquidation triggers, the trade is recorded as `-100U`.

## Missing Future Data

If the planned exit kline is missing, the current research code can use the latest available 1H kline after entry as fallback. This is only a backtest data-completeness convention.
