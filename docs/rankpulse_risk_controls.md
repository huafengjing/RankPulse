# RankPulse 风控与执行安全

最后更新：2026-08-15

## 1. 策略层过滤

当前策略层已有：

- 只做多
- 黑名单：`RAVEUSDT`
- `gain_24h < 10%` 不交易
- `gain_24h >= 80%` 不交易
- `60%-80%` 整个区间不交易
- Rank2 / Rank3 按涨幅区间和量比过滤
- Rank1 只允许 tuned cells
- 同 symbol 未平仓期间不重复开仓
- 4H extreme weak exit
- 12H weak exit
- 默认 6D planned exit

## 2. 仓位数量限制

配置：

```env
MAX_OPEN_POSITIONS=10
```

开仓前会检查当前本地 state 的未平仓数量。

testnet/live 还会读取 Binance 当前真实持仓，防止同 symbol 重复开仓。

## 3. 单笔保证金

配置：

```env
POSITION_MARGIN_USDT=10
```

下单名义价值：

```text
POSITION_MARGIN_USDT * leverage
```

下单数量会按 Binance symbol 的：

- stepSize
- minQty
- minNotional

进行校验和格式化。

## 4. live 安全开关

live 默认关闭。

要启动 live，必须同时配置：

```env
TRADING_MODE=live
ENFORCE_SAFETY_LOCK=false
ALLOW_LIVE_TRADING=true
LIVE_ORDER_CONFIRMATION=I_UNDERSTAND_THIS_IS_REAL_MONEY
```

缺少任何一项，live 程序会拒绝运行。

## 5. testnet/live 地址隔离

行情数据：

```text
https://fapi.binance.com
```

testnet 下单：

```text
https://testnet.binancefuture.com
```

live 下单：

```text
https://fapi.binance.com
```

testnet 策略信号仍使用正式 public market data，不使用 testnet 行情生成排名。

## 6. 状态隔离

不同模式使用不同 state 路径：

```text
data/paper/production/state.json
data/testnet/production/state.json
data/live/production/state.json
data/testnet/test_fast/state.json
```

Regime Context 默认也隔离：

```text
data/<trading_mode>/<signal_mode>/regime_context.json
```

## 7. Binance 真实持仓同步

live/testnet 周期结束时会读取 Binance 真实持仓。

摘要显示：

```text
当前持仓(Binance): ...
```

如果本地 state 有仓位，但 Binance 已经没有真实持仓，则移除本地 stale position。

如果 Binance 有真实持仓，即使本地 state 没有，摘要也会显示 Binance 真实 symbol。

## 8. Regime fail-closed

当：

```env
RANKPULSE_REGIME_ENABLED=true
```

如果 Regime Context 生成或校验失败：

- 不生成信号
- 不下单
- 不 fallback 到基础杠杆
- 不允许 Rank3 意外恢复 5x

## 9. 诊断命令

live 只读诊断：

```powershell
python -m src.cli.live_diag
```

testnet 诊断：

```powershell
python -m src.cli.testnet_diag
```

常见 live 诊断错误：

```text
code=-2015
Invalid API-key, IP, or permissions for action
```

通常表示：

- API Key / Secret 填错
- API Key 未开启 Futures 权限
- API Key 被禁用
- 当前机器 IP 不在 API 白名单

## 10. 尚未实现的账户级风控

当前仍未实现：

- 日亏损停止
- 周亏损停止
- 总权益回撤停止
- 全局 kill switch
- 最大总名义敞口
- 单 symbol 最大敞口
- 资金费率过滤
- spread / order book 流动性过滤
- BTC 或大盘 regime 过滤
- 交易所侧止损单

这些不属于当前已完成模块。
