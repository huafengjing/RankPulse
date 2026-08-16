# RankPulse 信号规则

最后更新：2026-08-15

## 1. 信号来源

当前交易端默认使用 Binance USD-M Futures public API：

- `/fapi/v1/exchangeInfo`：获取可交易 USDT 永续合约池
- `/fapi/v1/ticker/24hr`：按 `priceChangePercent` 重建 24H 涨幅排名
- `/fapi/v1/klines`，`4h`：计算 `volume_24h_ratio_7d`
- `/fapi/v2/ticker/price`：读取当前价格，用于展示和模拟/真实成交参考

系统不抓 Binance 网页榜单。

## 2. 合约池过滤

只保留：

```text
status = TRADING
contractType = PERPETUAL
quoteAsset = USDT
```

额外排除：

```text
RAVEUSDT
```

## 3. 排名方法

每个信号窗口：

1. 读取所有 USDT 永续的 24H ticker。
2. 按 `priceChangePercent` 从高到低排序。
3. 得到 Rank1、Rank2、Rank3。
4. 对 Rank1/Rank2/Rank3 分别应用策略过滤。

## 4. 时间窗口

正式交易信号：

```text
北京时间 00:00
北京时间 08:00
```

观察推送：

```text
北京时间 23:00
```

23:00 只发送观察信息，不开仓，不推进 FR3/YR1 recovery streak。

## 5. 交易方向

只做多。

## 6. 入场过滤

信号必须同时满足：

- 符合对应 Rank 规则
- 在允许的北京时间窗口
- symbol 不在黑名单
- `gain_24h < 80%`
- 通过对应涨幅区间的量比过滤
- 同 symbol 当前没有未平仓仓位

## 7. 涨幅区间

| 区间 | 处理 |
|---|---|
| `<10%` | 不交易 |
| `10%-20%` | Rank2 / Rank3 交易，无量比过滤 |
| `20%-40%` | Rank2 / Rank3 按量比过滤 |
| `40%-60%` | 仅 Rank2，按量比过滤 |
| `60%-80%` | 不交易 |
| `>=80%` | 不交易 |

## 8. 量比计算

```text
recent_24h_volume = 最近 6 根已完成 4H K线成交量之和
seven_day_avg_daily_volume = 最近 42 根已完成 4H K线成交量之和 / 7
volume_24h_ratio_7d = recent_24h_volume / seven_day_avg_daily_volume
```

42 根 4H K 线包含最近 6 根。

## 9. 当前入场表

| Rank | 涨幅区间 | 量比要求 | 是否交易 |
|---|---|---|---|
| Rank1 | `20%-40%` | `2 <= 量比 < 3` | 是 |
| Rank1 | `20%-40%` | `5 <= 量比 < 6` | 是 |
| Rank1 | `40%-60%` | `2 <= 量比 < 3` | 是 |
| Rank2 | `10%-20%` | 无 | 是 |
| Rank3 | `10%-20%` | 无 | 是 |
| Rank2 | `20%-40%` | `1.5 <= 量比 < 5` | 是 |
| Rank3 | `20%-40%` | `1.2 <= 量比 < 5` | 是 |
| Rank2 | `40%-60%` | `3 <= 量比 < 5.5` | 是 |
| Rank3 | `40%-60%` | 任意 | 否 |
| Rank2/3 | `60%-80%` | 任意 | 否 |
| Rank1/2/3 | `>=80%` | 任意 | 否 |

## 10. 同 symbol 防重复

如果 Binance 或本地 state 显示该 symbol 已有未平仓仓位，则跳过新开仓。

这条规则用于避免同一交易对重复叠仓。
