from __future__ import annotations

from src.cli.live_diag import build_live_diag_report, live_diag_error_report, render_live_diag_error, render_live_diag_report
from src.exchange.binance_live import BinanceLiveAPIError, FuturesPosition


def test_live_diag_reports_account_balance_positions_and_one_way_mode() -> None:
    client = FakeLiveDiagClient(dual_side_position=False)

    report = build_live_diag_report(client)  # type: ignore[arg-type]
    output = render_live_diag_report(report)

    assert report["base_url"] == "https://fapi.binance.com"
    assert report["can_trade"] is True
    assert report["position_mode"] == "one_way"
    assert report["position_mode_ok"] is True
    assert report["usdt_wallet_balance"] == 100.5
    assert "Binance Futures Live 只读诊断" in output
    assert "账户可交易: OK" in output
    assert "仓位模式: one_way (OK 单向持仓)" in output
    assert "AAAUSDT" in output


def test_live_diag_warns_when_account_is_hedge_mode() -> None:
    client = FakeLiveDiagClient(dual_side_position=True)

    report = build_live_diag_report(client)  # type: ignore[arg-type]
    output = render_live_diag_report(report)

    assert report["position_mode"] == "hedge"
    assert report["position_mode_ok"] is False
    assert "Hedge Mode / 双向持仓" in output


def test_live_diag_renders_invalid_key_ip_or_permission_error_without_traceback() -> None:
    error = BinanceLiveAPIError(
        status=401,
        code=-2015,
        message="Invalid API-key, IP, or permissions for action, request ip: 203.10.99.12",
        path="/fapi/v2/account",
    )

    report = live_diag_error_report(error)
    output = render_live_diag_error(error)

    assert report["ok"] is False
    assert report["binance_code"] == -2015
    assert "API Key 无效、IP 不在白名单、或 API 权限不足" in output
    assert "request ip: 203.10.99.12" in output
    assert "本次只是诊断失败，没有下单" in output


class FakeLiveDiagClient:
    def __init__(self, dual_side_position: bool) -> None:
        self.base_url = "https://fapi.binance.com"
        self.dual_side_position = dual_side_position

    def account(self) -> dict[str, object]:
        return {
            "canTrade": True,
            "feeTier": 0,
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "100.5",
                    "availableBalance": "90.25",
                }
            ],
        }

    def position_mode(self) -> dict[str, object]:
        return {"dualSidePosition": self.dual_side_position}

    def open_positions(self) -> dict[str, FuturesPosition]:
        return {
            "AAAUSDT": FuturesPosition(
                symbol="AAAUSDT",
                position_amt=2.0,
                entry_price=1.25,
                unrealized_profit=0.5,
            )
        }
