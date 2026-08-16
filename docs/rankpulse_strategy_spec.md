# RankPulse 当前主策略规格

最后更新：2026-08-15

本文档描述当前交易端代码实际执行的主策略。

## 1. 交易范围

- 市场：Binance USD-M Futures USDT 永续合约
- 方向：只做多
- 默认排名来源：Binance Futures public API `/fapi/v1/ticker/24hr`
- 合约池过滤：`TRADING` + `PERPETUAL` + `USDT`
- 排除交易对：`RAVEUSDT`
- 交易时间：北京时间 `00:00` / `08:00`
- 观察时间：北京时间 `23:00`，只推送信息，不开仓

## 2. 排名口径

实盘、testnet、paper 默认使用 Binance 24H ticker 排名：

```text
priceChangePercent 从高到低排序
取 Top3
```

系统不抓网页榜单。

历史回测曾使用本地 1H K 线 cache 重建 24H 涨幅榜。该口径只作为历史研究参考，不是当前 live 默认排名来源。

## 3. 当前可交易 Rank

当前代码实际支持：

- Rank1：只在少数 tuned cells 交易
- Rank2：按主规则交易
- Rank3：按主规则交易

注意：旧文档曾写“Rank1 完全过滤”，但当前代码已经不是这个状态。以代码为准。

## 4. Rank1 tuned cells

Rank1 只在以下条件交易：

| 条件 | 杠杆 |
|---|---:|
| `20% <= gain_24h < 40%` 且 `2 <= volume_24h_ratio_7d < 3` | 3x |
| `20% <= gain_24h < 40%` 且 `5 <= volume_24h_ratio_7d < 6` | 5x |
| `40% <= gain_24h < 60%` 且 `2 <= volume_24h_ratio_7d < 3` | 5x |

除此之外 Rank1 不交易。

Rank1 不受 FR3/YR1 Regime 调整。

## 5. Rank2 / Rank3 主规则

| 24H 涨幅区间 | Rank2 | Rank3 |
|---|---|---|
| `<10%` | 不交易 | 不交易 |
| `10%-20%` | 交易，3x，无量比过滤 | 交易，3x，无量比过滤 |
| `20%-40%` | `1.5 <= 量比 < 5`，基础 3x | `1.2 <= 量比 < 5`，基础 5x |
| `40%-60%` | `3 <= 量比 < 5.5`，2x | 不交易 |
| `60%-80%` | 不交易 | 不交易 |
| `>=80%` | 不交易 | 不交易 |

如果启用 FR3/YR1，`20%-40%` Bucket B 的 Rank2 / Rank3 杠杆会被 Regime 覆盖，见 [rankpulse_leverage_rules.md](rankpulse_leverage_rules.md)。

## 6. volume_24h_ratio_7d 口径

量比使用已确认的历史回测口径：

```text
recent_24h_volume = 信号前最近 6 根已完成 4H K线成交量之和
seven_day_avg_daily_volume = 信号前最近 42 根已完成 4H K线成交量之和 / 7
volume_24h_ratio_7d = recent_24h_volume / seven_day_avg_daily_volume
```

注意：42 根 4H K 线包含 recent 24H 的 6 根 K 线。

这不是“最近24H / 前7天日均”的严格定义，但它是当前回测、分桶和杠杆参数的真实依据，交易端必须沿用。

## 7. 入场价格

历史回测使用信号时间对应 1H open。

paper / testnet / live 不假设能按 1H open 成交：

- paper 记录模拟成交价
- testnet 记录 Testnet 实际成交价
- live 记录 Binance 实盘订单成交价

## 8. 持仓与重复开仓

- 同 symbol 已持仓期间不重复开仓
- 开仓前检查本地 state
- testnet/live 开仓前检查 Binance 当前真实持仓
- 周期结束时同步 Binance 真实持仓，用于摘要显示和清理本地 stale position

## 9. 退出规则

当前主策略包含：

- 默认持有 6D
- 4H extreme weak exit
- 12H weak exit

当前主策略不使用：

- TP15
- TP50
- runner
- trailing stop
- 普通固定止损单

详情见 [rankpulse_exit_rules.md](rankpulse_exit_rules.md)。

## 10. 运行模式

| 模式 | 说明 |
|---|---|
| `paper` | 本地模拟，不真实下单 |
| `testnet` | Binance Futures Testnet 模拟盘真实 API 下单 |
| `live` | Binance Futures 实盘下单，必须显式开启安全开关 |

信号模式：

| 模式 | 说明 |
|---|---|
| `production` | 北京时间 00:00 / 08:00 检查交易信号；23:00 观察 |
| `test_fast` | 每 N 分钟检查一次，仅用于工程测试，不用于收益评估 |
