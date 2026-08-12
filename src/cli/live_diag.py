from __future__ import annotations

import argparse
import json

from src.config.modes import TradingMode
from src.config.settings import AppSettings
from src.exchange.binance_live import LiveExecutionClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Binance Futures live diagnostics.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    settings = AppSettings.from_env_file()
    if settings.trading_mode != TradingMode.LIVE:
        raise SystemExit("live_diag requires TRADING_MODE=live.")
    if not settings.binance_live_api_key or not settings.binance_live_api_secret:
        raise SystemExit("BINANCE_LIVE_API_KEY and BINANCE_LIVE_API_SECRET are required.")

    client = LiveExecutionClient(
        api_key=settings.binance_live_api_key,
        api_secret=settings.binance_live_api_secret,
    )
    report = build_live_diag_report(client)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(render_live_diag_report(report))


def build_live_diag_report(client: LiveExecutionClient) -> dict[str, object]:
    account = client.account()
    position_mode = client.position_mode()
    positions = client.open_positions()
    usdt_asset = _find_asset(account, "USDT")
    dual_side_position = bool(position_mode.get("dualSidePosition", False))

    return {
        "base_url": client.base_url,
        "can_trade": bool(account.get("canTrade", False)),
        "fee_tier": account.get("feeTier"),
        "position_mode": "hedge" if dual_side_position else "one_way",
        "position_mode_ok": not dual_side_position,
        "usdt_wallet_balance": _optional_float(usdt_asset.get("walletBalance") if usdt_asset else None),
        "usdt_available_balance": _optional_float(usdt_asset.get("availableBalance") if usdt_asset else None),
        "open_positions": [
            {
                "symbol": position.symbol,
                "position_amt": position.position_amt,
                "entry_price": position.entry_price,
                "unrealized_profit": position.unrealized_profit,
            }
            for position in positions.values()
        ],
        "checks": {
            "signed_api_ok": True,
            "live_base_url_ok": client.base_url == "https://fapi.binance.com",
            "can_trade_ok": bool(account.get("canTrade", False)),
            "one_way_position_mode_ok": not dual_side_position,
        },
    }


def render_live_diag_report(report: dict[str, object]) -> str:
    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}

    lines = [
        "Binance Futures Live 只读诊断",
        "",
        f"Base URL: {report.get('base_url')}",
        f"签名 API: {_ok(checks.get('signed_api_ok'))}",
        f"实盘地址: {_ok(checks.get('live_base_url_ok'))}",
        f"账户可交易: {_ok(checks.get('can_trade_ok'))}",
        f"仓位模式: {report.get('position_mode')} ({_ok(checks.get('one_way_position_mode_ok'))} 单向持仓)",
        f"USDT 余额: {_fmt_money(report.get('usdt_wallet_balance'))}",
        f"USDT 可用: {_fmt_money(report.get('usdt_available_balance'))}",
        "",
        "当前实盘持仓",
    ]

    positions = report.get("open_positions", [])
    if not positions:
        lines.append("暂无持仓")
    elif isinstance(positions, list):
        lines.append("SYMBOL | AMT | ENTRY | UPNL")
        lines.append("-------+-----+-------+-----")
        for raw_position in positions:
            if not isinstance(raw_position, dict):
                continue
            lines.append(
                f"{raw_position.get('symbol')} | "
                f"{_fmt_number(raw_position.get('position_amt'))} | "
                f"{_fmt_number(raw_position.get('entry_price'))} | "
                f"{_fmt_number(raw_position.get('unrealized_profit'))}"
            )

    if checks.get("one_way_position_mode_ok") is False:
        lines.extend(
            [
                "",
                "注意: 当前是 Hedge Mode / 双向持仓。当前实盘执行代码按 One-way Mode / 单向持仓设计。",
            ]
        )
    if checks.get("can_trade_ok") is False:
        lines.extend(["", "注意: canTrade=false，API Key 或账户权限还不能进行合约交易。"])

    return "\n".join(lines)


def _find_asset(account: dict[str, object], symbol: str) -> dict[str, object] | None:
    assets = account.get("assets", [])
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if isinstance(asset, dict) and asset.get("asset") == symbol:
            return asset
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _ok(value: object) -> str:
    return "OK" if value is True else "FAIL"


def _fmt_money(value: object) -> str:
    number = _optional_float(value)
    if number is None:
        return "N/A"
    return f"{number:.2f}"


def _fmt_number(value: object) -> str:
    number = _optional_float(value)
    if number is None:
        return "N/A"
    return f"{number:.12f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    main()
