# Regime Adaptive 备选研究方案

最后更新：2026-08-15

状态：备选研究方案，不是当前主策略。

## 1. 备选方案

```text
5d_count_h0__balanced_y2_r3
```

该方案曾作为 Regime Risk Controller 的备选候选保存。当前主策略已经选择并工程化 FR3/YR1，本文件仅用于保留历史研究记录。

## 2. 研究范围

- 输出目录：`output/regime_adaptive_leverage_original_velvet`
- 排名 universe：original / VELVET 对齐口径
- 本轮排除：`BTWUSDT`
- `VELVETUSDT`：保留
- `RAVEUSDT`：过滤
- 主策略入场规则：不变

`BTWUSDT` 排除只用于复现当时研究口径，使 `VELVETUSDT` 在 `2026-06-06 00:00 UTC` 处于 Rank3。

## 3. Regime 指标

指标：

```text
5d_count_h0
```

使用 Rank4-10 的 Post24Decay：

```text
Post24Decay = mean(future_return_48h - future_return_24h)
```

窗口：

- 最近 70 个成熟 Rank4-10 observation
- 约等于 5 天
- 无 hysteresis
- warm-up：至少 30 天或至少 70 个成熟 observation
- warm-up 前状态为 GREEN

只允许使用 48H future window 已经完成的 observation，不允许未来数据。

## 4. 状态定义

| 状态 | 条件 |
|---|---|
| GREEN | 正常 |
| YELLOW | 当前 Post24Decay <= walk-forward Q10 |
| RED | 当前 Post24Decay <= walk-forward Q5 |

## 5. 响应矩阵

| Regime | Bucket A | Bucket B | Bucket C |
|---|---|---|---|
| GREEN | 基础规则 | 基础规则 | 基础规则 |
| YELLOW | 杠杆封顶 2x | 杠杆封顶 2x | 基础规则 |
| RED | 杠杆封顶 1x | 关闭 | 基础规则 |

当时的 bucket 定义：

- Bucket A：`10% <= gain_24h < 20%`，Rank2/Rank3，无量比过滤，原始 3x
- Bucket B：`20% <= gain_24h < 40%`，Rank2/Rank3，`1.5 <= volume_24h_ratio_7d < 5`，原始 Rank2 3x / Rank3 5x
- Bucket C：`40% <= gain_24h < 60%`，仅 Rank2，`3 <= volume_24h_ratio_7d < 5.5`，原始 2x

## 6. 研究结果

original / VELVET 对齐 baseline：

- 评估仓位：309
- 净收益：10967.08U
- PF：2.00
- 胜率：30.42%
- 中位收益：-23.82%
- 最大回撤：-2539.27U
- 爆仓：55

备选方案：

- 评估仓位：297
- 已关闭交易：291
- 未平仓 mark-to-market：6
- 净收益：11460.16U
- PF：2.13
- 胜率：30.98%
- 中位收益：-22.46%
- 最大回撤：-2308.92U
- 爆仓：49
- Regime OFF 跳过：14
- 平均杠杆：3.53x
- 7 月 PnL：-1388.66U
- 7 月相对 baseline 少亏：230.35U

## 7. 当前决策

保留为历史备选方案。

当前主策略不使用该方案；当前主策略使用 FR3/YR1 Regime。
