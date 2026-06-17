# Top3 Risk Controls

Last updated: 2026-06-17

## Current Research Controls

The current strategy uses these controls:

- Trade only reconstructed Top3 Rank2 and Rank3.
- Exclude Rank1.
- Exclude `RAVEUSDT`.
- Exclude `gain_24h >= 80%`.
- Exclude the full 60%-80% gain bucket.
- For 20%-40%, require `1.5 <= volume_24h_ratio_7d < 5`.
- For 40%-60%, require Rank2 and `3 <= volume_24h_ratio_7d < 6`.
- Same symbol cannot be opened twice while a position is active.
- 12H weak-exit rule exits failed continuation early.
- Leverage is reduced to 2X in the 40%-60% bucket.

## Backtest Risk Metrics Tracked

The research output tracks:

- Trade count.
- Win/loss count.
- Gross profit and gross loss.
- Net PnL.
- Profit factor.
- Win rate.
- Average and median return.
- Max drawdown.
- Liquidation count.
- Net PnL after removing top 1, top 3, and top 5 winners.
- Monthly performance.
- Bucket performance.

## Not Yet Implemented

These controls are not implemented as live trading controls:

- Account-level max drawdown stop.
- Daily loss limit.
- Weekly loss limit.
- Max simultaneous positions.
- Max notional exposure.
- Max exposure per symbol.
- Exchange-side stop order.
- Funding-rate filter.
- Spread/liquidity filter.
- BTC or market-regime filter.
- Manual kill switch.
- API key management.
- Automated order placement.

## Execution Stage

Current project mode is Research.

If execution testing is later approved, the first stage should be paper trading or simulation. A small-capital live test should only happen after a separate execution-risk review and explicit user approval.

The current documents and scripts should not be treated as a complete live trading system.
