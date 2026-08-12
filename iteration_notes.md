# Iteration Notes

This file is a lightweight inbox for thoughts and iteration directions about Trending.
Items here are not research conclusions until they are supported by reproducible backtests.

## 2026-05-13

- Set this conversation as the place to capture thoughts and iteration directions for Trending.
- Save future ideas here as notes first, without treating them as validated strategy edge.
- Candidate filter idea: when a symbol first enters Top10, if the trigger-time 24H gain is already `>= 40%`, consider removing the signal from the tradable set. Initial observation: this bucket has very small sample size (`12` signals) but a high liquidation / stop-out style failure rate (`8`, `66.67%`) and only `4` completed `+10%` (`33.33%`). Treat as a risk-control hypothesis that needs validation across samples and parameter assumptions, not as a confirmed edge improvement.

## 2026-05-14

- Candidate focus bucket: when a symbol first enters the 24H rolling gain Top10, require the trigger-time 24H rolling gain to be in the `20%` to `30%` bucket. Initial observation: this bucket appears to improve multiple metrics, likely because the symbol has enough momentum to reach Top10 but may not yet be extremely overextended. Validate as a parameterized bucket test, not a fixed conclusion.
- Candidate exit module: add partial take-profit plus a runner trailing exit to reduce cases where trades reach high floating profit but finish with low realized return. Proposed rule: at `TP1 = +15%`, close `50%`; keep `50%` as runner. Runner stage 1, from `+15%` to `+50%`: exit on either `20%` trailing stop or break below 4H `MA14`, whichever triggers first. Runner stage 2, after price reaches `+50%`: widen trailing stop to `30%`, or exit on 4H close below `MA14`. If a runner position remains at the end of the backtest / holding window, exit at the latest available price. Needs exact implementation choices for intra-candle trigger order, 4H MA alignment, fees/slippage on partial exits, and whether MA exit uses intrabar price or confirmed 4H close.

## 2026-05-16

- Current final strategy / parameter snapshot:
  - Signal: first entry into Top10 within the last 5 days.
  - Trigger-time 24H rolling gain: `20%` to `30%`.
  - Structure filter: Rule C, 21-day sideways consolidation.
  - Volume filter: `1.2 <= volume_1h_vs_24h_avg <= 5`.
  - Current backtest summary by month:

| Month | Trades | PnL | PF | TP15 | +50% | first -10% | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 55 | +1797.26U | 1.69 | 52.73% | 9.09% | 45.45% | -67.64% |
| 2025-11 | 6 | +195.33U | 1.64 | 50.00% | 16.67% | 50.00% | -22.68% |
| 2025-12 | 6 | -239.54U | 0.37 | 33.33% | 0.00% | 50.00% | -36.42% |
| 2026-01 | 7 | +94.64U | 1.25 | 42.86% | 0.00% | 57.14% | -32.76% |
| 2026-02 | 7 | +348.60U | 1.83 | 57.14% | 0.00% | 42.86% | -16.52% |
| 2026-03 | 4 | -22.54U | 0.89 | 50.00% | 0.00% | 50.00% | -9.44% |
| 2026-04 | 19 | +1089.43U | 2.06 | 52.63% | 15.79% | 47.37% | -59.52% |
| 2026-05 | 6 | +331.34U | 2.65 | 83.33% | 16.67% | 16.67% | -10.18% |

  - Notes: sample size is still limited (`55` trades overall, several months have `4-7` trades). `PF = 1.69` and positive total PnL are promising, but max drawdown is very large (`-67.64%`), and 2025-12 remains clearly negative. Treat this as the current research snapshot, not a deployable live-trading conclusion.
