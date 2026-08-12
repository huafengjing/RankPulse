# FR3/YR1 Regime Engine Spec

Last updated: 2026-08-11

## Scope

This spec freezes the engineering behavior for `FR_avg_return24_l3_gt_0_fr3_yr1`.

The engine only controls Bucket B Rank2/Rank3 leverage. It does not change ranking, entry filters, volume ratio, exits, position locks, fees, or live safety guards.

## Opportunity Set

`B_R3 eligible opportunity` means:

- Rank is 3.
- `20% <= gain_24h < 40%`.
- `1.5 <= volume_24h_ratio_7d < 5`.
- Symbol is not `RAVEUSDT`.
- The opportunity belongs to the frozen ranking universe used by the current research output.

Opportunity eligibility is signal-layer eligibility. It is independent of whether the account actually opened the trade.

## Returns

For each eligible opportunity:

- `Return24` uses the same entry reference and 24H reference as the frozen backtest output.
- `Return48` uses the same entry reference and 48H reference as the frozen backtest output.
- `Decay48 = Return48 - Return24`.

An opportunity may enter the 24H recovery sensor only when `opportunity_time + 24H <= evaluation_time`.

An opportunity may enter the 48H risk-off sensor only when `opportunity_time + 48H <= evaluation_time`.

## Risk-Off

Model: `D_b_r3_decay_l15`.

At each evaluation timestamp:

- Sort all matured 48H B_R3 opportunities by `(opportunity_time, symbol, opportunity_id)`.
- Use the latest 15 matured 48H opportunities.
- `Last15_Decay48 = mean(last15 Decay48)`.
- Historical thresholds are walk-forward only: use prior generated `Last15_Decay48` values where `regime_observation_time < evaluation_time`.
- Percentile implementation is `numpy.quantile` default behavior, matching the frozen research script.
- Minimum prior values before Q10/Q5 are used: 10.
- Warm-up rule matches the frozen research script: warm when `evaluation_time >= signal_start + 30D` or at least 15 matured 48H opportunities exist.

State:

- `GREEN`: `Last15_Decay48 > Q10`
- `YELLOW`: `Q5 < Last15_Decay48 <= Q10`
- `RED`: `Last15_Decay48 <= Q5`
- Missing value, missing thresholds, or warm-up incomplete remains `GREEN`.

## Fast Recovery

Model: `avg_return24_l3_gt_0`.

At each evaluation timestamp:

- Sort all matured 24H B_R3 opportunities by `(opportunity_time, symbol, opportunity_id)`.
- Use the latest 3 matured 24H opportunities.
- `Last3_AvgReturn24 = mean(last3 Return24)`.
- `recovery_signal = Last3_AvgReturn24 > 0`.

The threshold is strictly greater than zero.

## Recovery Streak

Streak is updated by evaluation event, not by process run.

Rules matching the frozen backtest script:

- On a new UTC month, `recovery_streak` resets to 0 before evaluating that timestamp.
- If final state is `GREEN`, `recovery_streak` resets to 0.
- If final state is `YELLOW` or `RED` and `recovery_signal=true`, `recovery_streak += 1`.
- If final state is `YELLOW` or `RED` and `recovery_signal=false`, `recovery_streak = 0`.
- Re-running the same timestamp from full rebuild produces the same streak and does not advance it again.

## Leverage

Bucket B overlay:

| Regime | Rank2 | Rank3 |
|---|---:|---:|
| GREEN | 3x | 5x |
| YELLOW, no recovery | 3x | 3x |
| YELLOW, recovery | 3x | 5x |
| RED, no recovery | 2x | 1x |
| RED, first recovery | 3x | 3x |
| RED, second consecutive recovery | 3x | 5x |

The live/testnet/paper execution layer reads this as `Top3RegimeContext`.

## Context JSON

The generated JSON includes:

- `signal_time_ms`
- `generated_at_ms`
- `model`
- `model_version`
- `state`
- `last15_decay48`
- `historical_q10`
- `historical_q5`
- `last3_avg_return24`
- `recovery_signal`
- `recovery_streak`
- `r2_leverage`
- `r3_leverage`
- matured opportunity counts and IDs
- `data_cutoff_ms`
- `max_source_timestamp_used`
- `status`

Writes are atomic: temp file, fsync, then rename.

## Fail-Safe

If `TOP3_REGIME_ENABLED=false`, the overlay is disabled and the base leverage mapping remains active.

If `TOP3_REGIME_ENABLED=true`, live/testnet runtime automatically generates context before each production 00:00/08:00 signal evaluation. The JSON is written atomically to `TOP3_REGIME_CONTEXT_PATH` when configured, otherwise to `data/<trading_mode>/<signal_mode>/regime_context.json`.

The reader fails closed for:

- missing file
- invalid JSON
- non-object JSON
- `status != READY`
- `signal_time_ms` not matching the signal window
- model/model_version mismatch
- invalid state
- future `data_cutoff_ms`
- non-finite numeric fields

This avoids silently treating an unknown or stale regime as GREEN.

The 23:00 observation cycle does not generate official regime context and does not advance recovery streak. Preflight checks at 07:30/22:30/23:30 also do not generate or advance official context.

## CLI

Inspect a historical context:

```bash
python -m src.cli.generate_top3_regime_context --as-of 2026-07-06T00:00:00Z --inspect
```

Write context JSON:

```bash
python -m src.cli.generate_top3_regime_context --as-of 2026-07-06T00:00:00Z --output output/regime_context_engineering/regime_context.json
```

Replay and compare with frozen D15 output:

```bash
python -m src.cli.generate_top3_regime_context --replay
```

Cold-start rebuild of the opportunity file from local kline cache:

```bash
python -m src.cli.generate_top3_regime_context --rebuild-opportunities
```
