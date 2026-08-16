# RankPulse 退出规则

最后更新：2026-08-15

## 1. 默认退出

默认持有：

```text
6D
```

planned exit 时间：

```text
entry_time + 6D
```

testnet/live 平仓方式：

```text
reduce-only 市价平多
```

## 2. 12H weak exit

12H 后检查。如果同时满足：

```text
12H MFE < 5%
12H close_return < 0%
```

则提前平仓。

当前规则已经不再要求：

```text
12H MAE < -5%
```

配置：

```env
ENABLE_12H_WEAK_EXIT=true
```

默认必须为 true。

## 3. 4H extreme weak exit

4H 后检查。如果同时满足：

```text
4H MFE < 2%
4H MAE < -8%
```

则提前平仓。

配置：

```env
ENABLE_4H_EXTREME_WEAK_EXIT=true
```

默认必须为 true。

## 4. test_fast 压缩退出时间

`SIGNAL_MODE=test_fast` 只用于工程测试，不用于收益评估。

在 test_fast 中：

```env
TEST_WEAK_EXIT_AFTER_MINUTES=15
TEST_PLANNED_EXIT_AFTER_MINUTES=60
```

含义：

- weak exit 检查压缩为入场后 15 分钟
- planned exit 压缩为入场后 60 分钟

production 模式不受这些配置影响。

## 5. 当前不使用的退出

当前主策略不使用：

- TP15
- TP50
- runner
- trailing stop
- 普通固定止损单
- 网格
- 马丁
- 补仓

## 6. 手动平仓

testnet：

```powershell
python -m src.cli.testnet_close SYMBOLUSDT
```

live：

```powershell
python -m src.cli.live_close SYMBOLUSDT
```

手动平仓使用 reduce-only 市价单。

## 7. 平仓确认

平仓后系统会：

1. 查询订单成交结果。
2. 如果 Binance 暂时返回 `Order does not exist`，会重试。
3. 如果订单查询仍失败，会查询真实持仓。
4. 如果真实持仓已经为 0，则本地 state 关闭该仓位。

这样可以避免“真实已经平仓，但本地仍显示持仓”的问题。
