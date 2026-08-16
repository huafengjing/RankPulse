# RankPulse

Binance USD-M Futures 涨幅榜 Top3 动量策略工程项目。

当前交易端已经支持：

- paper 本地模拟
- testnet 模拟盘真实 API 下单
- live 实盘下单
- Binance public API 重建 24H ticker 排名
- Rank2 / Rank3 信号监测
- FR3/YR1 Regime 动态杠杆
- 4H extreme weak exit
- 12H weak exit
- 6D 固定持有退出
- Telegram 推送

实盘默认仍受安全开关保护。启动方式见 [Startup.md](Startup.md)。

核心文档：

- [当前主策略规格](docs/rankpulse_strategy_spec.md)
- [信号规则](docs/rankpulse_signal_rules.md)
- [杠杆与 Regime](docs/rankpulse_leverage_rules.md)
- [退出规则](docs/rankpulse_exit_rules.md)
- [风控与执行安全](docs/rankpulse_risk_controls.md)
- [回测摘要](docs/rankpulse_backtest_summary.md)
- [Regime Engine 规格](docs/regime_engine_spec.md)
