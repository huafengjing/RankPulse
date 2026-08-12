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
| `20% <= gain_24h < 40%`, Rank2, `1.5 <= volume_24h_ratio_7d < 5` | 3X |
| `20% <= gain_24h < 40%`, Rank3, `1.2 <= volume_24h_ratio_7d < 5` | 5X |
| `40% <= gain_24h < 60%`, Rank2, `3 <= volume_24h_ratio_7d < 5.5` | 2X |
| `60% <= gain_24h < 80%` | No trade |

## Bucket B FR3/YR1 Regime Overlay

The runtime can optionally apply the `FR_avg_return24_l3_gt_0_fr3_yr1` overlay when an as-of regime context is supplied.

Risk-off sensor:

- `D_b_r3_decay_l15`
- Opportunity set: Bucket B Rank3 eligible opportunities.
- `Decay48 = return_48h - return_24h`.
- `D15 = mean(last15(Decay48))`.
- `GREEN`: `D15 > walk-forward Q10`.
- `YELLOW`: `Q5 < D15 <= Q10`.
- `RED`: `D15 <= Q5`.

Fast recovery sensor:

- `AvgReturn24_L3 = mean(last3(return_24h))`.
- Recovery trigger: `AvgReturn24_L3 > 0`.
- Only opportunities with `opportunity_time + 24h <= signal_time` may be used.

Bucket B leverage overlay:

| Regime | Rank2 | Rank3 |
| --- | ---: | ---: |
| GREEN | 3X | 5X |
| YELLOW | 3X | 3X, or 5X when recovery signal is true |
| RED | 2X, or 3X when recovery signal is true | 1X without recovery; 3X on first recovery signal; 5X after two consecutive recovery signals |

Runtime activation is context-driven. Set `TOP3_REGIME_CONTEXT_PATH` to a JSON file with:

```json
{
  "signal_time_ms": 1783296000000,
  "state": "RED",
  "recovery_signal": true,
  "recovery_streak": 1,
  "model": "FR_avg_return24_l3_gt_0_fr3_yr1"
}
```

If the file is missing, not configured, or for a different signal window, runtime falls back to the base leverage mapping above.

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


