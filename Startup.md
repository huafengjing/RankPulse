# 启动方式

## 1. 实盘启动

先确认 `.env` 至少包含：

```env
TRADING_MODE=live
SIGNAL_MODE=production

ENFORCE_SAFETY_LOCK=false
ALLOW_LIVE_TRADING=true
LIVE_ORDER_CONFIRMATION=I_UNDERSTAND_THIS_IS_REAL_MONEY

BINANCE_LIVE_API_KEY=你的key
BINANCE_LIVE_API_SECRET=你的secret

POSITION_MARGIN_USDT=10
MAX_OPEN_POSITIONS=10

RANKPULSE_REGIME_ENABLED=true
RANKPULSE_REGIME_CONTEXT_AUTO_GENERATE=true

ENABLE_12H_WEAK_EXIT=true
```

然后按顺序运行：

```powershell
python -m src.cli.live_diag
python -m src.cli.live_once
python -m src.cli.live_daemon
```

含义：

- `live_diag`：只读诊断，不下单
- `live_once`：运行一次交易周期
- `live_daemon`：持续运行

## 2. 启动后会做什么

`live_daemon` 启动后：

- 第一次运行会回放最近 6 天 00:00 / 08:00 信号，只补 Bootstrap 虚拟持仓状态，不补历史订单
- 北京时间 00:00 / 08:00 检查交易信号
- 北京时间 23:00 只推送观察信息，不开仓
- 自动生成 FR3/YR1 Regime Context
- 自动检查 4H / 12H / 6D 退出
- 周期结束时同步 Binance 真实持仓

如果摘要里显示：

```text
当前持仓(Binance): ...
```

表示这里读取的是 Binance 真实账户持仓，不是单纯本地 state。

Bootstrap 虚拟持仓只用于阻断重复开仓和占用最大持仓数量。它不会向 Binance 下单，到理论退出时间后会自动关闭。

## 3. 查看状态

```powershell
python -m src.cli.live_status
```

## 4. 停止程序

在运行 `live_daemon` 的终端按：

```text
Ctrl+C
```

## 5. Testnet 启动

如果跑模拟盘，把 `.env` 改成：

```env
TRADING_MODE=testnet
SIGNAL_MODE=production

BINANCE_TESTNET_API_KEY=你的testnet_key
BINANCE_TESTNET_API_SECRET=你的testnet_secret
```

然后运行：

```powershell
python -m src.cli.testnet_diag
python -m src.cli.testnet_once
python -m src.cli.testnet_daemon
```

## 6. 常见问题

如果出现：

```text
No module named numpy
```

说明当前 `python` 指到了缺少依赖的环境，可以改用：

```powershell
C:\Users\liu\AppData\Local\Programs\Python\Python312\python.exe -m src.cli.live_once
```

如果 `live_diag` 显示：

```text
code=-2015
Invalid API-key, IP, or permissions
```

优先检查：

- API Key / Secret 是否填对
- API Key 是否开启 Futures 权限
- API Key 是否绑定了当前机器 IP
- API Key 是否被禁用
