# Regime Adaptive Backup Candidate

Last updated: 2026-08-10

Status: backup candidate, research only. This does not replace the current main strategy.

## Selected Candidate

`5d_count_h0__balanced_y2_r3`

This candidate is saved as the current backup Regime Risk Controller. If later parameter tests do not produce a better risk-adjusted result, this is the fallback candidate to compare against.

## Baseline Scope

- Output source: `output/regime_adaptive_leverage_original_velvet`
- Ranking universe basis: original/VELVET-aligned universe
- Ranking universe exclusion for this run: `BTWUSDT`
- Main strategy rules: unchanged
- `VELVETUSDT`: retained
- `RAVEUSDT`: still filtered

The `BTWUSDT` exclusion is only for preserving the original ranking basis where `VELVETUSDT` is Rank3 on `2026-06-06 00:00 UTC`. This file records the chosen backup candidate under that agreed research scope.

## Regime Indicator

`5d_count_h0` uses Rank4-10 Post24Decay:

```text
Post24Decay = mean(future_return_48h - future_return_24h)
```

Window:

- Fixed count: latest 70 mature Rank4-10 observations
- Approximate calendar length: 5 days
- Hysteresis: none (`h0`)
- Warm-up: at least 30 days or at least 70 mature observations; before warm-up, state is `GREEN`
- Look-ahead rule: only observations whose 48H future window is complete before trade time are eligible

State:

- `GREEN`: normal
- `YELLOW`: current same-window Post24Decay <= walk-forward historical Q10
- `RED`: current same-window Post24Decay <= walk-forward historical Q5

## Response Matrix

| Regime | Bucket A | Bucket B | Bucket C |
|---|---|---|---|
| GREEN | base | base | base |
| YELLOW | cap to 2x | cap to 2x | base |
| RED | cap to 1x | off | base |

Bucket definitions:

- Bucket A: `10% <= gain_24h < 20%`, Rank2/Rank3, no volume filter, original 3x
- Bucket B: `20% <= gain_24h < 40%`, Rank2/Rank3, `1.5 <= volume_24h_ratio_7d < 5`, original Rank2 3x / Rank3 5x
- Bucket C: `40% <= gain_24h < 60%`, Rank2 only, `3 <= volume_24h_ratio_7d < 5.5`, original 2x

## Headline Metrics

Original/VELVET-aligned baseline:

- Evaluated positions: 309
- Net PnL: 10967.08U
- PF: 2.00
- Win rate: 30.42%
- Median return: -23.82%
- Max drawdown: -2539.27U
- Liquidations: 55

Backup candidate:

- Evaluated positions: 297
- Closed trades: 291
- Open mark-to-market: 6
- Net PnL: 11460.16U
- PF: 2.13
- Win rate: 30.98%
- Median return: -22.46%
- Max drawdown: -2308.92U
- Liquidations: 49
- Regime OFF skips: 14
- Average leverage: 3.53x
- July PnL: -1388.66U
- July loss saved vs baseline: 230.35U
- Jan-Jun profit sacrifice: -262.72U, meaning Jan-Jun improved rather than sacrificed

## Monthly Delta Vs Baseline

| Month | Baseline PnL | Candidate PnL | Delta | Baseline Liq | Candidate Liq |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 1089.51 | 1398.34 | +308.83 | 3 | 2 |
| 2026-02 | 836.15 | 695.97 | -140.18 | 4 | 4 |
| 2026-03 | 1889.35 | 1887.00 | -2.35 | 7 | 7 |
| 2026-04 | 1156.66 | 1253.09 | +96.43 | 5 | 4 |
| 2026-05 | 458.72 | 458.72 | +0.00 | 8 | 8 |
| 2026-06 | 4504.23 | 4504.23 | +0.00 | 12 | 12 |
| 2026-07 | -1619.01 | -1388.66 | +230.35 | 16 | 12 |
| 2026-08 | 2651.48 | 2651.48 | +0.00 | 0 | 0 |

## Decision Note

Keep this as the backup candidate. Continue testing other Regime parameters, but compare every new result against this candidate and the original baseline. Do not promote it to the main strategy until later tests fail to find a better candidate and the result remains acceptable out of sample.
