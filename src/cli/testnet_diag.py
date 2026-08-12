from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.config.lock import FileLock
from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.config.state_paths import state_paths_for
from src.exchange.binance_testnet import TestnetExecutionClient
from src.exchange.signing import sign_query
from src.execution.testnet_runner import MarketSymbolFilterProvider, TestnetTradingRunner
from src.execution.testnet_state import TestnetStateStore
from src.market.binance_futures import BinanceFuturesMarketClient
from src.paper.store import PaperEventLogger
from src.paper.trading import PaperSignal


def _signed_get(base_url: str, api_key: str, api_secret: str, path: str) -> object:
    params = {"recvWindow": 5000, "timestamp": int(time.time() * 1000)}
    query = urlencode(params)
    params["signature"] = sign_query(query, api_secret)
    query = urlencode(params)
    url = f"{base_url}{path}?{query}"
    with urlopen(Request(url, headers={"X-MBX-APIKEY": api_key}), timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def show_account_info(settings: AppSettings) -> None:
    """检查测试网账户连接状态和余额。"""
    base_url = "https://testnet.binancefuture.com"
    try:
        data = _signed_get(
            base_url,
            settings.binance_testnet_api_key,
            settings.binance_testnet_api_secret,
            "/fapi/v2/account",
        )
    except Exception as exc:
        print(f"[诊断] API 连接失败: {exc}")
        return

    if not isinstance(data, dict):
        print(f"[诊断] API 返回格式异常: {data}")
        return

    can_trade = data.get("canTrade", False)
    print(f"[诊断] 账户状态: {'[OK] 可交易' if can_trade else '[NO] 不可交易'}")
    if not can_trade:
        print("       请在 Testnet 钱包中划转 USDT 到 合约钱包 ( Futures )")

    assets = data.get("assets", [])
    for asset in assets:
        if isinstance(asset, dict) and asset.get("asset") == "USDT":
            wallet_balance = float(asset.get("walletBalance", 0))
            available_balance = float(asset.get("availableBalance", 0))
            print(f"[诊断] USDT 余额: {wallet_balance:.2f} (可用: {available_balance:.2f})")
            break

    positions = data.get("positions", [])
    active = [p for p in positions if isinstance(p, dict) and float(p.get("positionAmt", 0)) != 0]
    print(f"[诊断] 当前持仓: {len(active)} 个")
    for p in active:
        print(f"       {p.get('symbol')}: {float(p.get('positionAmt', 0))} 张, 开仓价 {float(p.get('entryPrice', 0))}")


def force_open(settings: AppSettings, symbol: str) -> None:
    """强制开一个测试仓位，绕过信号规则直接调 engine。"""
    paths = state_paths_for(TradingMode.TESTNET, settings.signal_mode)
    market_client = BinanceFuturesMarketClient()
    execution_client = TestnetExecutionClient(
        api_key=settings.binance_testnet_api_key,
        api_secret=settings.binance_testnet_api_secret,
    )

    # 检查是否已有持仓
    existing = execution_client.open_positions()
    if symbol in existing:
        print(f"[开单] {symbol} 在交易所已有持仓，跳过")
        return
    if existing:
        print(f"[开单] 交易所已有其他持仓: {list(existing.keys())}，跳过")
        return

    # 获取当前价格
    prices = market_client.latest_prices()
    fill_price = prices.get(symbol)
    if fill_price is None:
        print(f"[开单] 无法获取 {symbol} 当前价格")
        return
    print(f"[开单] {symbol} 当前价格: {fill_price}")

    now_ms = int(time.time() * 1000)
    signal = PaperSignal(
        symbol=symbol,
        rank=2,
        gain_24h=0.20,
        volume_24h_ratio_7d=2.0,
        snapshot_hour_bj="00:00",
        signal_time_ms=now_ms,
        fill_price=fill_price,
    )

    lock = FileLock(paths.state_path.parent / ".lock")
    with lock:
        engine = TestnetTradingRunner(
            market_client=market_client,
            execution_client=execution_client,
            state_store=TestnetStateStore(paths.state_path),
            logger=PaperEventLogger(paths.event_log_path),
            settings=settings,
        )
        position = engine.engine.open_from_signal(signal, leverage=2)
        if position is None:
            print(f"[开单] [FAIL] 开仓失败，请检查上面错误日志")
        else:
            print(f"[开单] [OK] 开仓成功: {position.symbol}")
            print(f"       开仓价: {position.entry_price}")
            print(f"       数量: {position.qty}")
            print(f"       杠杆: {position.leverage}x")
            print(f"       计划退出时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(position.planned_exit_time_ms / 1000))} UTC")
            print(f"\n[开单] 要平仓可运行: python -m src.cli.testnet_close {position.symbol}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Testnet 连接诊断和手动开单工具")
    parser.add_argument("--force-open", metavar="SYMBOL", default=None, help="强制开测试仓位，如 BTCUSDT")
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    if settings.trading_mode != TradingMode.TESTNET:
        raise SystemExit("需要 TRADING_MODE=testnet")
    if not settings.binance_testnet_api_key or not settings.binance_testnet_api_secret:
        raise SystemExit("BINANCE_TESTNET_API_KEY 和 BINANCE_TESTNET_API_SECRET 必须配置")

    show_account_info(settings)

    if args.force_open:
        symbol = args.force_open.upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        print()
        force_open(settings, symbol)


if __name__ == "__main__":
    main()
