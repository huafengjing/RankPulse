# FR3/YR1 Regime Engine 规格

最后更新：2026-08-15

## 1. 作用范围

FR3/YR1 Regime 只调整 Bucket B 的 Rank2 / Rank3 杠杆。

Bucket B：

```text
20% <= gain_24h < 40%
```

不改变：

- 排名来源
- 入场过滤
- volume ratio 计算
- 退出规则
- 持仓限制
- live 安全开关

## 2. 启用方式

`.env`：

```env
RANKPULSE_REGIME_ENABLED=true
RANKPULSE_REGIME_CONTEXT_AUTO_GENERATE=true
RANKPULSE_REGIME_CONTEXT_PATH=
```

`RANKPULSE_REGIME_CONTEXT_PATH` 留空时，自动写入：

```text
data/<trading_mode>/<signal_mode>/regime_context.json
```

## 3. 自动生成时机

production 模式下，只在正式交易信号前生成：

```text
北京时间 00:00
北京时间 08:00
```

流程：

```text
确定 signal window T
-> generate context as_of=T
-> atomic 写 JSON
-> 校验 context
-> 生成 Rank2 / Rank3 信号
-> 解析杠杆
-> 下单
```

禁止先生成交易信号、再更新 Regime Context。

## 4. 23:00 观察

北京时间 23:00 只做观察推送。

23:00 不会：

- 生成正式 Regime Context
- 推进 recovery streak
- 开仓

## 5. Risk-Off 模型

模型名：

```text
D_b_r3_decay_l15
```

历史机会集合：

```text
Bucket B Rank3 eligible opportunity
```

当前 Regime 生成器口径：

- Rank3
- `20% <= gain_24h < 40%`
- `1.5 <= volume_24h_ratio_7d < 5`
- symbol 不在黑名单
- path_status 为 ok

注意：当前实际交易入场中，Rank3 20%-40% 使用 `1.2 <= 量比 < 5`。Regime 历史状态生成仍使用 `1.5 <= 量比 < 5`。这是当前代码口径。

## 6. Decay48

```text
Decay48 = Return48 - Return24
```

机会必须成熟后才能使用：

```text
24H recovery: opportunity_time + 24H <= evaluation_time
48H risk-off: opportunity_time + 48H <= evaluation_time
```

## 7. Last15 Risk-Off

```text
Last15_Decay48 = 最近 15 个成熟 48H 机会的 Decay48 均值
```

阈值：

- Q10：历史 Last15_Decay48 的 walk-forward 10% 分位
- Q5：历史 Last15_Decay48 的 walk-forward 5% 分位

状态：

| 状态 | 条件 |
|---|---|
| GREEN | `Last15_Decay48 > Q10` |
| YELLOW | `Q5 < Last15_Decay48 <= Q10` |
| RED | `Last15_Decay48 <= Q5` |

warmup、最小历史样本数、分位数计算必须和冻结回测脚本一致。

## 8. Fast Recovery

模型名：

```text
avg_return24_l3_gt_0
```

```text
Last3_AvgReturn24 = 最近 3 个成熟 24H 机会的 Return24 均值
```

Recovery：

```text
Last3_AvgReturn24 > 0
```

必须严格大于 0。

## 9. Recovery Streak

streak 按正式 evaluation event 推进，不按程序运行次数推进。

规则：

- UTC 新月份开始前先重置为 0
- GREEN 状态重置为 0
- YELLOW/RED 且 recovery=true，则 streak + 1
- YELLOW/RED 且 recovery=false，则 streak = 0
- 同一个 evaluation timestamp 重复运行，结果必须完全一致，不得重复推进 streak

## 10. 杠杆表

| Regime | Rank2 | Rank3 |
|---|---:|---:|
| GREEN | 3x | 5x |
| YELLOW，无 recovery | 3x | 3x |
| YELLOW，有 recovery | 3x | 5x |
| RED，无 recovery | 2x | 1x |
| RED，第 1 次 recovery | 3x | 3x |
| RED，连续第 2 次 recovery | 3x | 5x |

## 11. Context JSON

自动生成的 JSON 包含：

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
- 成熟机会数量和 ID
- `data_cutoff_ms`
- `status`

写入方式是 atomic write。

## 12. Fail-Closed

当 `RANKPULSE_REGIME_ENABLED=true` 时，以下情况全部 fail closed：

- Generator exception
- JSON 缺失
- JSON 损坏
- `status != READY`
- `signal_time_ms` 不匹配
- model 不匹配
- model_version 不匹配
- state 非法
- `data_cutoff_ms` 晚于 signal time
- 数值字段不是 finite

fail closed 的含义：

```text
不生成交易信号
不下单
不 fallback 到基础规则
不允许 Rank3 意外恢复 5x
```

## 13. 调试命令

历史 replay：

```powershell
python -m src.cli.generate_rankpulse_regime_context --replay
```

查看某个时间点：

```powershell
python -m src.cli.generate_rankpulse_regime_context --as-of 2026-07-06T00:00:00Z --inspect
```

手动写 JSON，仅用于调试或修复：

```powershell
python -m src.cli.generate_rankpulse_regime_context --as-of 2026-07-06T00:00:00Z --output output/regime_context_engineering/regime_context.json
```

正常 live/testnet daemon 不需要手动执行这一步。
