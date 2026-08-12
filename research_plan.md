# Research Plan: Binance USDT-M Top Gainers Top10 / Top5

## 1. Strategy Hypothesis

Binance USDT-M Futures 24H rolling 涨幅榜可能集中体现短期散户注意力。首次进入 Top10 的合约可能继续冲入 Top5，也可能已经接近短期高点。本研究只验证这个假设，不开发实盘交易机器人。

## 2. Scope

当前阶段为 Research。

先完成：

1. 项目目录结构
2. 数据下载与缓存
3. eligible universe 筛选
4. 24H rolling gain ranking 重建
5. first Top10 / first Top5 信号识别
6. 最小可运行 backtest
7. README 和单元测试

## 3. Data Source

优先使用 Binance USDT-M Futures 官方公开 REST API：

- `GET /fapi/v1/exchangeInfo`
- `GET /fapi/v1/klines`
- `GET /fapi/v1/ticker/24hr`

基础 URL：`https://fapi.binance.com`

## 4. Universe Selection

默认筛选：

- `contractType = PERPETUAL`
- `quoteAsset = USDT`
- `status = TRADING`
- 排除 `BTCUSDT`、`ETHUSDT`、`BNBUSDT`
- 默认 `min_quote_volume = 20000000`

## 5. Ranking Reconstruction

在每个采样时间点 `t`：

```text
rolling_24h_change_pct(t) = close(t) / close(t - 24h) - 1
```

按 `rolling_24h_change_pct` 降序排名，生成 Top5 / Top10 / Top20。

禁止使用未来 close，禁止用当前 Binance 页面代替历史重建。

## 6. Signal Definition

基础信号：

- 某 symbol 当前进入 Top10；
- 过去 `cooldown_days = 3` 内没有进入过 Top10；
- 信号时间是进入 Top10 的采样点；
- 入场时间必须是下一根 K 线 open。

额外记录 Top10 后是否在观察窗口内进入 Top5，以及用时。

## 7. Backtest Assumptions

默认：

- long only
- entry = next candle open
- TP1 = +10%
- SL = -5% / -7% / -10%
- max holding = 24h / 48h / 72h
- taker fee = 0.0005
- slippage = 0.0005
- 同一根 K 线同时触发 TP 和 SL 时，SL 优先。

## 8. Parameter Grid

首版支持：

- strategy_variant: `top10_immediate`
- min_quote_volume: `10000000`, `20000000`, `50000000`
- sl_pct: `-0.05`, `-0.07`, `-0.10`
- max_holding_hours: `24`, `48`, `72`
- tp1_pct: `0.10`

后续扩展：

- pullback EMA20 / VWAP
- Top10 to Top5 confirmation
- overextended filter
- BTC regime filter

## 9. Outputs

- `outputs/signals.csv`
- `outputs/trades.csv`
- `outputs/summary.csv`
- `outputs/equity_curve.csv`
- `outputs/charts/*.png`

## 10. Edge Criteria

不能只看胜率。至少检查：

- expectancy > 0
- profit_factor > 1
- 样本数量足够
- max drawdown 可接受
- 参数变化后结果不崩
- 手续费滑点后仍有效
- 不是一两笔极端交易撑起来

结论必须谨慎：回测结果不代表未来收益。
