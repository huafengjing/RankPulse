from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import AppSettings
from src.exchange.binance_live import LiveExecutionClient

DEFAULT_STRATEGY = "rank1_mixed_3x5d_v23_5x2d_v56_4060v23"
DEFAULT_TRADE_SOURCE = ROOT / "output" / "rank1_portfolio_variant_comparison" / "trade_details_all_variants.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "dashboard"
BEIJING = timezone(timedelta(hours=8))


def today_bj_start_ms() -> int:
    now_bj = datetime.now(BEIJING)
    start_bj = now_bj.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start_bj.astimezone(timezone.utc).timestamp() * 1000)


def parse_start(value: str) -> int:
    raw = value.strip()
    if raw.isdigit():
        return int(raw)
    if raw.lower() == "today_bj":
        return today_bj_start_ms()
    if len(raw) == 10:
        dt = datetime.fromisoformat(raw).replace(tzinfo=BEIJING)
    else:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BEIJING)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def ms_to_utc(value: Any) -> str:
    ms = to_int(value)
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def ms_to_bj(value: Any) -> str:
    ms = to_int(value)
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=BEIJING).strftime("%Y-%m-%d %H:%M BJ")


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def percent(value: Any) -> float | None:
    number = to_float(value)
    return None if number is None else number * 100.0


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events



def fetch_live_account_positions(enabled: bool) -> tuple[dict[str, dict[str, float]], str]:
    if not enabled:
        return {}, "disabled"
    try:
        settings = AppSettings.from_env_file()
        if not settings.binance_live_api_key or not settings.binance_live_api_secret:
            return {}, "missing_api_key"
        client = LiveExecutionClient(settings.binance_live_api_key, settings.binance_live_api_secret, timeout_seconds=8.0, max_retries=2)
        positions = client.open_positions()
        return {
            symbol: {
                "position_amt": position.position_amt,
                "entry_price": position.entry_price,
                "unrealized_profit": position.unrealized_profit,
            }
            for symbol, position in positions.items()
        }, "ok"
    except Exception as exc:
        return {}, f"failed: {type(exc).__name__}: {exc}"
def latest_close(symbol: str) -> tuple[float | None, int | None]:
    path = ROOT / "data" / "futures_klines_1h" / f"{symbol}_1h.csv"
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path, usecols=["close", "close_time"])
    except Exception:
        return None, None
    if df.empty:
        return None, None
    row = df.iloc[-1]
    return to_float(row.get("close")), to_int(row.get("close_time"))




def max_floating_profit(
    symbol: str,
    entry_ms: int | None,
    exit_ms: int | None,
    entry_price: float | None,
    leverage: float | None,
    qty: float | None,
) -> tuple[float | None, float | None, float | None]:
    if not symbol or entry_ms is None or entry_price is None or entry_price <= 0:
        return None, None, None
    path = ROOT / "data" / "futures_klines_1h" / f"{symbol}_1h.csv"
    if not path.exists():
        return None, None, None
    try:
        df = pd.read_csv(path, usecols=["high", "close_time"])
    except Exception:
        return None, None, None
    if df.empty:
        return None, None, None
    end_ms = exit_ms or to_int(df["close_time"].max())
    if end_ms is None:
        return None, None, None
    window = df[(df["close_time"] >= entry_ms) & (df["close_time"] <= end_ms)].copy()
    if window.empty:
        return None, None, None
    window["high"] = pd.to_numeric(window["high"], errors="coerce")
    window = window.dropna(subset=["high"])
    if window.empty:
        return None, None, None
    idx = window["high"].idxmax()
    high = to_float(window.loc[idx, "high"])
    high_time = to_int(window.loc[idx, "close_time"])
    if high is None or high_time is None:
        return None, None, None
    lev = leverage or 1.0
    max_profit_pct = (high / entry_price - 1.0) * lev * 100.0
    max_profit_u = (high - entry_price) * qty if qty else None
    max_profit_day = (high_time - entry_ms) / 86400000
    return max_profit_pct, max_profit_u, max_profit_day


def pnl_for_open(position: dict[str, Any]) -> tuple[float | None, float | None, float | None, str]:
    symbol = str(position.get("symbol", ""))
    entry_price = to_float(position.get("entry_price"))
    qty = to_float(position.get("qty")) or 0.0
    leverage = to_float(position.get("leverage")) or 1.0
    current_price, close_time = latest_close(symbol)
    if entry_price is None or current_price is None:
        return None, None, current_price, ms_to_utc(close_time)
    underlying_return = (current_price / entry_price - 1.0) * 100.0
    pnl_u = (current_price - entry_price) * qty if qty else None
    net_return = underlying_return * leverage
    return pnl_u, net_return, current_price, ms_to_utc(close_time)


def state_path(mode: str, signal_mode: str) -> Path:
    return ROOT / "data" / mode / signal_mode / "state.json"


def events_path(mode: str, signal_mode: str) -> Path:
    return ROOT / "data" / mode / signal_mode / "events.jsonl"


def load_backtest_rows(source: Path, strategy: str) -> pd.DataFrame:
    if not source.exists():
        return pd.DataFrame()
    df = pd.read_csv(source)
    if "strategy" in df.columns:
        df = df[df["strategy"].eq(strategy)].copy()
    return df


def export_backtest_trades(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    trades = df[df["pnl_u"].notna()].copy() if "pnl_u" in df.columns else df.copy()
    if "entry_time_ms" in trades.columns:
        trades = trades.sort_values("entry_time_ms", ascending=False)
    rows: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        rows.append(
            {
                "source": "backtest",
                "symbol": row.get("symbol", ""),
                "rank": to_int(row.get("rank")),
                "month": row.get("month", ""),
                "status": row.get("status", ""),
                "entry_time_utc": row.get("entry_time_utc") or ms_to_utc(row.get("entry_time_ms")),
                "entry_time_bj": row.get("entry_time_bj") or ms_to_bj(row.get("entry_time_ms")),
                "exit_time_utc": row.get("exit_time_utc") or ms_to_utc(row.get("exit_time_ms")),
                "entry_price": to_float(row.get("entry_price")),
                "exit_price": to_float(row.get("exit_price")),
                "gain_24h_pct": percent(row.get("gain_24h")),
                "volume_24h_ratio_7d": to_float(row.get("volume_24h_ratio_7d")),
                "leverage": to_float(row.get("leverage")),
                "holding_days": to_float(row.get("holding_days")),
                "exit_reason": row.get("exit_reason", ""),
                "pnl_u": to_float(row.get("pnl_u")),
                "net_return_pct": to_float(row.get("net_return_pct")),
                "mfe_pct": to_float(row.get("mfe_pct")),
                "max_profit_pct": to_float(row.get("mfe_pct")),
                "max_profit_u": (to_float(row.get("mfe_pct")) or 0.0) if to_float(row.get("mfe_pct")) is not None else None,
                "max_profit_day": max_floating_profit(str(row.get("symbol", "")), to_int(row.get("entry_time_ms")), to_int(row.get("exit_time_ms")), to_float(row.get("entry_price")), to_float(row.get("leverage")), None)[2],
                "mae_pct": to_float(row.get("mae_pct")),
                "liquidated": bool_value(row.get("liquidated")),
                "is_win": bool_value(row.get("is_win")),
                "strategy_component": row.get("strategy_component", ""),
                "bucket": row.get("bucket", ""),
            }
        )
    return rows


def live_trades_from_events(events: list[dict[str, Any]], state: dict[str, Any], start_ms: int, account_positions: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    opened: dict[tuple[str, int], dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    for item in events:
        event = str(item.get("event", ""))
        payload = item.get("payload", {}) if isinstance(item.get("payload"), dict) else {}
        if not event.startswith("live_"):
            continue
        if event == "live_opened":
            entry_ms = to_int(payload.get("entry_time_ms"))
            if entry_ms is None or entry_ms < start_ms:
                continue
            opened[(str(payload.get("symbol", "")), entry_ms)] = payload
        elif event == "live_closed":
            entry_ms = to_int(payload.get("entry_time_ms"))
            exit_ms = to_int(payload.get("exit_time_ms"))
            if (exit_ms is None or exit_ms < start_ms) and (entry_ms is None or entry_ms < start_ms):
                continue
            entry = opened.get((str(payload.get("symbol", "")), entry_ms or 0), {})
            qty = to_float(payload.get("qty")) or to_float(entry.get("qty")) or 0.0
            entry_price = to_float(payload.get("entry_price")) or to_float(entry.get("entry_price"))
            exit_price = to_float(payload.get("exit_price"))
            leverage = to_float(payload.get("leverage")) or to_float(entry.get("leverage"))
            ret_pct = None
            if entry_price and exit_price and leverage:
                ret_pct = (exit_price / entry_price - 1.0) * leverage * 100.0
            pnl = to_float(payload.get("realized_pnl"))
            max_profit_pct, max_profit_u, max_profit_day = max_floating_profit(
                str(payload.get("symbol", "")), entry_ms, exit_ms, entry_price, leverage, qty
            )
            closed.append(
                {
                    "source": "live_account",
                    "symbol": payload.get("symbol", ""),
                    "rank": to_int(entry.get("rank")),
                    "month": datetime.fromtimestamp((exit_ms or entry_ms or start_ms) / 1000, tz=timezone.utc).strftime("%Y-%m"),
                    "status": "closed",
                    "entry_time_utc": ms_to_utc(entry_ms),
                    "entry_time_bj": ms_to_bj(entry_ms),
                    "exit_time_utc": ms_to_utc(exit_ms),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gain_24h_pct": percent(entry.get("gain_24h")),
                    "volume_24h_ratio_7d": to_float(entry.get("volume_24h_ratio_7d")),
                    "leverage": leverage,
                    "holding_days": ((exit_ms - entry_ms) / 86400000) if entry_ms and exit_ms else None,
                    "exit_reason": payload.get("exit_reason", ""),
                    "pnl_u": pnl,
                    "net_return_pct": ret_pct,
                    "mfe_pct": max_profit_pct,
                    "max_profit_pct": max_profit_pct,
                    "max_profit_u": max_profit_u,
                    "max_profit_day": max_profit_day,
                    "mae_pct": None,
                    "liquidated": str(payload.get("exit_reason", "")).lower() == "liquidation",
                    "is_win": pnl is not None and pnl > 0,
                    "strategy_component": f"Rank{entry.get('rank', '')}" if entry.get("rank") else "",
                    "bucket": "",
                }
            )
    closed_keys = {(row["symbol"], row["entry_time_utc"]) for row in closed}
    open_rows: list[dict[str, Any]] = []
    for raw in state.get("open_positions", []) if isinstance(state.get("open_positions"), list) else []:
        if not isinstance(raw, dict):
            continue
        entry_ms = to_int(raw.get("entry_time_ms"))
        if entry_ms is None or entry_ms < start_ms:
            continue
        if (raw.get("symbol", ""), ms_to_utc(entry_ms)) in closed_keys:
            continue
        pnl_u, ret_pct, current_price, _ = pnl_for_open(raw)
        account_position = account_positions.get(str(raw.get("symbol", "")), {})
        if account_position:
            pnl_u = to_float(account_position.get("unrealized_profit"))
            qty_abs = abs(to_float(account_position.get("position_amt")) or 0.0)
            entry_price_for_account = to_float(raw.get("entry_price"))
            leverage_for_account = to_float(raw.get("leverage")) or 1.0
            if qty_abs and entry_price_for_account:
                current_price = (pnl_u / qty_abs + entry_price_for_account) if pnl_u is not None else current_price
                if current_price is not None:
                    ret_pct = (current_price / entry_price_for_account - 1.0) * leverage_for_account * 100.0
        opened_payload = opened.get((str(raw.get("symbol", "")), entry_ms), {})
        max_profit_pct, max_profit_u, max_profit_day = max_floating_profit(
            str(raw.get("symbol", "")), entry_ms, None, to_float(raw.get("entry_price")), to_float(raw.get("leverage")), to_float(raw.get("qty"))
        )
        if max_profit_u is None and pnl_u is not None and pnl_u > 0:
            max_profit_u = pnl_u
            max_profit_pct = ret_pct
            max_profit_day = ((datetime.now(timezone.utc).timestamp() * 1000) - entry_ms) / 86400000 if entry_ms else None
        open_rows.append(
            {
                "source": "live_account",
                "symbol": raw.get("symbol", ""),
                "rank": to_int(opened_payload.get("rank")),
                "month": datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc).strftime("%Y-%m"),
                "status": "open",
                "entry_time_utc": ms_to_utc(entry_ms),
                "entry_time_bj": ms_to_bj(entry_ms),
                "exit_time_utc": "",
                "entry_price": to_float(raw.get("entry_price")),
                "exit_price": current_price,
                "gain_24h_pct": percent(opened_payload.get("gain_24h")),
                "volume_24h_ratio_7d": to_float(opened_payload.get("volume_24h_ratio_7d")),
                "leverage": to_float(raw.get("leverage")),
                "holding_days": None,
                "exit_reason": "open",
                "pnl_u": pnl_u,
                "net_return_pct": ret_pct,
                "mfe_pct": max_profit_pct,
                "max_profit_pct": max_profit_pct,
                "max_profit_u": max_profit_u,
                "max_profit_day": max_profit_day,
                "mae_pct": None,
                "liquidated": False,
                "is_win": pnl_u is not None and pnl_u > 0,
                "strategy_component": f"Rank{opened_payload.get('rank', '')}" if opened_payload.get("rank") else "",
                "bucket": "",
            }
        )
    return sorted(closed + open_rows, key=lambda row: row.get("entry_time_utc", ""), reverse=True)


def backtest_signal_row(row: pd.Series) -> dict[str, Any]:
    skipped = str(row.get("status", "")).lower() == "skipped"
    return {
        "source": "backtest",
        "signal_time_utc": row.get("entry_time_utc") or ms_to_utc(row.get("entry_time_ms")),
        "signal_time_bj": row.get("entry_time_bj") or ms_to_bj(row.get("entry_time_ms")),
        "symbol": row.get("symbol", ""),
        "rank": to_int(row.get("rank")),
        "price": to_float(row.get("entry_price")),
        "gain_24h_pct": percent(row.get("gain_24h")),
        "volume_24h_ratio_7d": to_float(row.get("volume_24h_ratio_7d")),
        "passed": not skipped,
        "filter_reason": row.get("skip_reason") if skipped else "pass",
        "leverage": to_float(row.get("leverage")) if not skipped else None,
        "planned_hold_days": to_float(row.get("target_hold_days")),
        "regime_state": row.get("regime_state", ""),
        "recovery_signal": bool_value(row.get("recovery_signal")),
        "order_status": "skipped" if skipped else row.get("status", ""),
        "order_error": "",
    }


def signals_from_backtest(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    signals = df.copy()
    if "entry_time_ms" in signals.columns:
        signals = signals.sort_values("entry_time_ms", ascending=False)
    return [backtest_signal_row(row) for _, row in signals.iterrows()]


def signals_from_live_events(events: list[dict[str, Any]], start_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in events:
        event = str(item.get("event", ""))
        payload = item.get("payload", {}) if isinstance(item.get("payload"), dict) else {}
        if not event.startswith("live_"):
            continue
        if event.endswith("_signal_snapshot"):
            signal_ms = to_int(payload.get("signal_time_ms"))
            if signal_ms is None or signal_ms < start_ms:
                continue
            for row in payload.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "source": "live_account",
                        "signal_time_utc": ms_to_utc(row.get("signal_time_ms")),
                        "signal_time_bj": ms_to_bj(row.get("signal_time_ms")),
                        "symbol": row.get("symbol", ""),
                        "rank": to_int(row.get("rank")),
                        "price": to_float(row.get("price")),
                        "gain_24h_pct": percent(row.get("gain_24h")),
                        "volume_24h_ratio_7d": to_float(row.get("volume_24h_ratio_7d")),
                        "passed": bool(row.get("passed")),
                        "filter_reason": row.get("filter_reason", ""),
                        "leverage": to_float(row.get("leverage")),
                        "planned_hold_days": None,
                        "regime_state": row.get("regime_state", ""),
                        "recovery_signal": row.get("recovery_signal"),
                        "order_status": "information" if row.get("information_only") else "candidate",
                        "order_error": "",
                    }
                )
        elif event in {"live_opened", "live_open_failed", "live_closed"}:
            ts = payload.get("entry_time_ms") or payload.get("exit_time_ms") or payload.get("timestamp")
            ts_i = to_int(ts)
            if ts_i is None or ts_i < start_ms:
                continue
            status = "opened" if event == "live_opened" else "failed" if event == "live_open_failed" else "closed"
            rows.append(
                {
                    "source": "live_account",
                    "signal_time_utc": ms_to_utc(ts_i),
                    "signal_time_bj": ms_to_bj(ts_i),
                    "symbol": payload.get("symbol", ""),
                    "rank": to_int(payload.get("rank")),
                    "price": to_float(payload.get("entry_price") or payload.get("exit_price")),
                    "gain_24h_pct": percent(payload.get("gain_24h")),
                    "volume_24h_ratio_7d": to_float(payload.get("volume_24h_ratio_7d")),
                    "passed": event != "live_open_failed",
                    "filter_reason": "pass" if event != "live_open_failed" else "order_failed",
                    "leverage": to_float(payload.get("leverage")),
                    "planned_hold_days": None,
                    "regime_state": "",
                    "recovery_signal": None,
                    "order_status": status,
                    "order_error": payload.get("error", ""),
                }
            )
    return sorted(rows, key=lambda item: item.get("signal_time_utc", ""), reverse=True)


def account_position_rows(account_positions: dict[str, dict[str, float]], state: dict[str, Any]) -> list[dict[str, Any]]:
    local_by_symbol = {
        str(item.get("symbol", "")): item
        for item in state.get("open_positions", [])
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for symbol, account_position in sorted(account_positions.items()):
        local = local_by_symbol.get(symbol, {})
        entry_price = to_float(account_position.get("entry_price")) or to_float(local.get("entry_price"))
        qty = abs(to_float(account_position.get("position_amt")) or 0.0)
        pnl_u = to_float(account_position.get("unrealized_profit"))
        current_price = None
        if entry_price is not None and qty and pnl_u is not None:
            current_price = entry_price + pnl_u / qty
        ret_pct = None
        leverage = to_float(local.get("leverage"))
        if entry_price and current_price and leverage:
            ret_pct = (current_price / entry_price - 1.0) * leverage * 100.0
        rows.append({
            "source": "binance_positionRisk",
            "symbol": symbol,
            "rank": None,
            "entry_time_utc": ms_to_utc(local.get("entry_time_ms")),
            "entry_time_bj": ms_to_bj(local.get("entry_time_ms")),
            "entry_price": entry_price,
            "current_price": current_price,
            "current_price_time_utc": "Binance positionRisk",
            "qty": qty,
            "leverage": leverage,
            "planned_exit_utc": ms_to_utc(local.get("planned_exit_time_ms")),
            "weak_exit_checked": bool(local.get("weak_exit_checked", False)),
            "extreme_weak_exit_checked": bool(local.get("extreme_weak_exit_checked", False)),
            "pnl_u": pnl_u,
            "return_pct": ret_pct,
            "gain_24h_pct": None,
            "volume_24h_ratio_7d": None,
        })
    return rows


def overview_from_trades(
    trades: list[dict[str, Any]],
    state: dict[str, Any],
    events: list[dict[str, Any]],
    mode: str,
    signal_mode: str,
    start_ms: int,
    source: str,
    account_positions: dict[str, dict[str, float]],
    account_refresh_status: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if account_refresh_status == "ok":
        positions = account_position_rows(account_positions, state)
    else:
        positions = []
        for raw in state.get("open_positions", []) if isinstance(state.get("open_positions"), list) else []:
            if not isinstance(raw, dict):
                continue
            entry_ms = to_int(raw.get("entry_time_ms"))
            if source == "live" and (entry_ms is None or entry_ms < start_ms):
                continue
            pnl_u, ret_pct, current_price, price_time = pnl_for_open(raw)
            positions.append({
                "source": "local_state",
                "symbol": raw.get("symbol", ""),
                "rank": None,
                "entry_time_utc": ms_to_utc(entry_ms),
                "entry_time_bj": ms_to_bj(entry_ms),
                "entry_price": to_float(raw.get("entry_price")),
                "current_price": current_price,
                "current_price_time_utc": price_time,
                "qty": to_float(raw.get("qty")),
                "leverage": to_float(raw.get("leverage")),
                "planned_exit_utc": ms_to_utc(raw.get("planned_exit_time_ms")),
                "weak_exit_checked": bool(raw.get("weak_exit_checked", False)),
                "extreme_weak_exit_checked": bool(raw.get("extreme_weak_exit_checked", False)),
                "pnl_u": pnl_u,
                "return_pct": ret_pct,
                "gain_24h_pct": None,
                "volume_24h_ratio_7d": None,
            })
    open_pnl_values = [to_float(row.get("pnl_u")) for row in positions]
    open_pnl_known = sum(1 for value in open_pnl_values if value is not None)
    open_pnl_total = sum(value for value in open_pnl_values if value is not None)
    realized = [row for row in trades if row.get("status") == "closed" and row.get("pnl_u") is not None]
    pnl = [float(row["pnl_u"]) for row in realized]
    gross_profit = sum(x for x in pnl if x > 0)
    gross_loss = -sum(x for x in pnl if x < 0)
    latest_regime = next((e.get("payload", {}) for e in reversed(events) if str(e.get("event", "")).endswith("_regime_context_ready")), {})
    return {
        "generated_at_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
        "strategy_version": DEFAULT_STRATEGY,
        "mode": mode,
        "signal_mode": signal_mode,
        "data_source": source,
        "account_refresh_status": account_refresh_status,
        "start_time_utc": ms_to_utc(start_ms),
        "start_time_bj": ms_to_bj(start_ms),
        "state_path": str(state_path(mode, signal_mode).relative_to(ROOT)),
        "events_path": str(events_path(mode, signal_mode).relative_to(ROOT)),
        "last_signal_time_utc": ms_to_utc(state.get("last_signal_time_ms")),
        "last_information_time_utc": ms_to_utc(state.get("last_information_time_ms")),
        "last_preflight_time_utc": ms_to_utc(state.get("last_preflight_time_ms")),
        "open_positions_count": len(positions),
        "virtual_positions_count": 0 if source == "live" else len(state.get("bootstrap_virtual_positions", []) or []),
        "blocking_positions_count": len(positions),
        "open_pnl_u_known": open_pnl_known,
        "open_pnl_u": open_pnl_total,
        "month": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m"),
        "month_realized_pnl_u": sum(pnl),
        "month_net_pnl_u": sum(pnl) + open_pnl_total if open_pnl_known else sum(pnl),
        "month_pf": gross_profit / gross_loss if gross_loss else None,
        "month_win_rate_pct": (sum(1 for x in pnl if x > 0) / len(pnl) * 100.0) if pnl else None,
        "month_liquidations": sum(1 for row in trades if row.get("liquidated")),
        "month_liq_rate_pct": (sum(1 for row in trades if row.get("liquidated")) / len(trades) * 100.0) if trades else None,
        "month_trades": len(trades),
        "regime_state": latest_regime.get("state", ""),
        "regime_model": latest_regime.get("model", ""),
        "recovery_signal": latest_regime.get("recovery_signal"),
        "recovery_streak": latest_regime.get("recovery_streak"),
        "positions": positions,
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RankPulse dashboard JSON files.")
    parser.add_argument("--source", choices=["live", "backtest"], default="backtest")
    parser.add_argument("--start", default="today_bj", help="Live source start time: today_bj, YYYY-MM-DD in Beijing time, ISO timestamp, or ms.")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--trade-source", type=Path, default=DEFAULT_TRADE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", default="live", choices=["live", "testnet", "paper"])
    parser.add_argument("--signal-mode", default="production", choices=["production", "test_fast"])
    parser.add_argument("--no-account-refresh", action="store_true", help="Do not query Binance live positionRisk for unrealized PnL.")
    args = parser.parse_args()

    start_ms = parse_start(args.start)
    state = read_json(state_path(args.mode, args.signal_mode))
    events = read_events(events_path(args.mode, args.signal_mode))
    account_positions, account_refresh_status = fetch_live_account_positions(args.source == "live" and not args.no_account_refresh)

    if args.source == "live":
        trades = live_trades_from_events(events, state, start_ms, account_positions)
        signals = signals_from_live_events(events, start_ms)
    else:
        df = load_backtest_rows(args.trade_source, args.strategy)
        trades = export_backtest_trades(df)
        signals = signals_from_backtest(df)

    overview = overview_from_trades(trades, state, events, args.mode, args.signal_mode, start_ms, args.source, account_positions, account_refresh_status)
    write_json(args.output_dir / "overview.json", overview)
    write_json(args.output_dir / "trades.json", trades)
    write_json(args.output_dir / "signals.json", signals[:2000])
    write_json(
        args.output_dir / "manifest.json",
        {
            "generated_at_utc": overview["generated_at_utc"],
            "overview": "overview.json",
            "trades": "trades.json",
            "signals": "signals.json",
            "source": args.source,
            "start_time_utc": overview["start_time_utc"],
            "start_time_bj": overview["start_time_bj"],
            "strategy": args.strategy,
        },
    )
    print(f"Exported dashboard data to {args.output_dir}")
    print(f"Source: {args.source} | Start: {overview['start_time_bj']} | Trades: {len(trades)} | Signals: {len(signals[:2000])}")


if __name__ == "__main__":
    main()