# RankPulse 杠杆规则

最后更新：2026-08-15

## 1. 基础杠杆

如果没有启用 FR3/YR1 Regime，当前代码使用以下基础杠杆。

| 条件 | 杠杆 |
|---|---:|
| Rank1，`20%-40%`，`2 <= 量比 < 3` | 3x |
| Rank1，`20%-40%`，`5 <= 量比 < 6` | 5x |
| Rank1，`40%-60%`，`2 <= 量比 < 3` | 5x |
| Rank2 / Rank3，`10%-20%` | 3x |
| Rank2，`20%-40%`，`1.5 <= 量比 < 5` | 3x |
| Rank3，`20%-40%`，`1.2 <= 量比 < 5` | 5x |
| Rank2，`40%-60%`，`3 <= 量比 < 5.5` | 2x |

其他情况不交易。

## 2. FR3/YR1 Regime 覆盖范围

FR3/YR1 只覆盖 Bucket B：

```text
20% <= gain_24h < 40%
```

覆盖对象：

- Rank2
- Rank3

不覆盖：

- Rank1
- 10%-20%
- 40%-60%
- 60%-80%
- >=80%

## 3. Regime 开关

`.env`：

```env
RANKPULSE_REGIME_ENABLED=true
RANKPULSE_REGIME_CONTEXT_AUTO_GENERATE=true
```

启用后，live/testnet daemon 会在 00:00 / 08:00 信号评估前自动生成当期 Regime Context。

如果 `RANKPULSE_REGIME_CONTEXT_PATH` 留空，默认写入：

```text
data/<trading_mode>/<signal_mode>/regime_context.json
```

## 4. Risk-Off 指标

模型：

```text
D_b_r3_decay_l15
```

机会集合：

```text
Bucket B Rank3 eligible opportunity
```

即：

- Rank3
- `20% <= gain_24h < 40%`
- `1.5 <= volume_24h_ratio_7d < 5`
- symbol 不在黑名单

注意：当前交易入场规则中 Rank3 20%-40% 是 `1.2 <= 量比 < 5`。但 FR3/YR1 Regime 生成器用于判断市场状态的历史机会集合仍是 `1.5 <= 量比 < 5`。这是当前代码实际口径，不能混淆。

指标：

```text
Decay48 = Return48 - Return24
Last15_Decay48 = 最近 15 个已成熟 48H 机会的 Decay48 均值
```

状态：

| 状态 | 条件 |
|---|---|
| GREEN | `Last15_Decay48 > Q10` |
| YELLOW | `Q5 < Last15_Decay48 <= Q10` |
| RED | `Last15_Decay48 <= Q5` |

Q10 / Q5 使用 walk-forward 历史值，只能使用当前 evaluation time 之前已经生成的值。

## 5. Fast Recovery 指标

模型：

```text
avg_return24_l3_gt_0
```

指标：

```text
Last3_AvgReturn24 = 最近 3 个已成熟 24H Bucket B Rank3 机会的 Return24 均值
```

Recovery 条件：

```text
Last3_AvgReturn24 > 0
```

必须严格大于 0。

## 6. Regime 动态杠杆

| Regime | Rank2 | Rank3 |
|---|---:|---:|
| GREEN | 3x | 5x |
| YELLOW，无 recovery | 3x | 3x |
| YELLOW，有 recovery | 3x | 5x |
| RED，无 recovery | 2x | 1x |
| RED，第 1 次 recovery | 3x | 3x |
| RED，连续第 2 次 recovery | 3x | 5x |

Rank3 RED 状态必须是：

```text
无 recovery: 1x
第一次 recovery: 3x
连续第二次 recovery: 5x
```

## 7. Fail-Closed

当 `RANKPULSE_REGIME_ENABLED=true` 时：

如果 Regime Generator 出错、context 缺失、时间戳不一致、model 不一致、`status != READY`、JSON 损坏、或校验失败：

```text
不生成信号
不下单
不回退到基础杠杆
不允许 Rank3 恢复 5x
```

这是最高优先级安全规则。

## 8. 爆仓假设

历史杠杆回测使用以下简化爆仓阈值：

| 杠杆 | 标的 MAE 阈值 |
|---:|---:|
| 1x | -100% |
| 2x | -50% |
| 3x | -33% |
| 5x | -20% |

触发后该笔交易记为 `-100U`。

这是回测假设，不等同于 Binance 实盘强平精确模型。
