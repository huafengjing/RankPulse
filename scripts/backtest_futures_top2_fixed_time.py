from __future__ import annotations

import math
import sys
import time
import json
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import KLINE_COLUMNS, parse_klines


INTERVAL = "1h"
HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
SIGNAL_DAYS = 180
BUY_NOTIONAL_U = 100.0
FEE_RATE = 0.001
HOLDING_DAYS = [3, 7, 14]
ALLOW_DOWNLOADS = False
SNAPSHOT_UTC_HOURS = [0, 4, 8, 12, 16, 20]
TOP_N = 3

CACHE_DIR = ROOT / "data" / "futures_klines_1h"
LOCAL_5M_CACHE_DIR = ROOT / "data" / "raw" / "klines" / "5m"
LOCAL_EXCHANGE_INFO_PATH = ROOT / "data" / "raw" / "exchange_info" / "exchange_info_latest.json"
OUT = ROOT / "output"
SIGNALS_PATH = OUT / "futures_top2_fixed_time_signals.csv"
TRADES_PATH = OUT / "futures_top2_fixed_time_trades.csv"
SUMMARY_PATH = OUT / "futures_top2_fixed_time_summary.csv"
MONTHLY_PATH = OUT / "futures_top2_fixed_time_monthly.csv"
GROUP_PATH = OUT / "futures_top2_fixed_time_group_stats.csv"
TAIL_PATH = OUT / "futures_top2_fixed_time_tail_dependency.csv"


class SimpleBinanceFuturesClient:
    def __init__(self, base_url: str = "https://fapi.binance.com", timeout: int = 20, max_retries: int = 5) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                last_error = exc
                try:
                    return self._get_via_powershell(url)
                except Exception as ps_exc:
                    last_error = ps_exc
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2 ** (attempt - 1)))
        raise RuntimeError(f"Binance request failed after {self.max_retries} attempts: {path}") from last_error

    def _get_via_powershell(self, url: str) -> Any:
        escaped = url.replace("'", "''")
        command = f"Invoke-WebRequest -Uri '{escaped}' -UseBasicParsing | Select-Object -ExpandProperty Content"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout + 20,
        )
        return json.loads(result.stdout)

    def exchange_info(self) -> dict[str, Any]:
        return self._get("/fapi/v1/exchangeInfo")

    def klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return self._get("/fapi/v1/klines", params=params)


def ms_to_utc(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


def ms_to_bj_string(ms: int) -> str:
    return (ms_to_utc(ms) + pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def get_futures_symbols(client: SimpleBinanceFuturesClient) -> list[str]:
    if LOCAL_EXCHANGE_INFO_PATH.exists():
        info = json.loads(LOCAL_EXCHANGE_INFO_PATH.read_text(encoding="utf-8"))
    else:
        info = client.exchange_info()
    rows = []
    for item in info.get("symbols", []):
        if item.get("quoteAsset") != "USDT":
            continue
        if item.get("contractType") != "PERPETUAL":
            continue
        if item.get("status") != "TRADING":
            continue
        if item.get("underlyingType") != "COIN":
            continue
        rows.append(str(item["symbol"]))
    return sorted(rows)


def latest_signal_end_dt(now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    now_bj = now_utc.astimezone(timezone(timedelta(hours=8)))
    schedule_hours_bj = [0, 4, 8, 12, 16, 20]
    eligible_hours = [hour for hour in schedule_hours_bj if hour <= now_bj.hour]
    if eligible_hours:
        end_bj = now_bj.replace(hour=max(eligible_hours), minute=0, second=0, microsecond=0)
    else:
        previous_day = now_bj - timedelta(days=1)
        end_bj = previous_day.replace(hour=20, minute=0, second=0, microsecond=0)
    return end_bj.astimezone(timezone.utc).replace(tzinfo=timezone.utc)


def download_klines(
    client: SimpleBinanceFuturesClient,
    symbol: str,
    start_time: int,
    end_time: int,
    sleep_seconds: float = 0.12,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    cursor = start_time
    while cursor <= end_time:
        raw = client.klines(symbol=symbol, interval=INTERVAL, start_time=cursor, end_time=end_time, limit=1500)
        chunk = parse_klines(raw, symbol=symbol, interval=INTERVAL)
        if chunk.empty:
            break
        chunks.append(chunk)
        last_open = int(chunk["open_time"].max())
        next_cursor = last_open + HOUR_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(sleep_seconds)
    if not chunks:
        return pd.DataFrame(columns=KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"])
    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")
    return frame[(frame["open_time"] >= start_time) & (frame["open_time"] <= end_time)].copy()


def _read_cached_symbol(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for col in ["open_time", "close_time", "trade_count"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64").astype("int64")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["symbol"] = symbol
    frame["interval"] = INTERVAL
    return frame.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")


def load_local_1h_cache(symbols: list[str], start_time: int, end_time: int) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _read_cached_symbol(symbol)
        if frame.empty:
            continue
        scoped = frame[(frame["open_time"] >= start_time) & (frame["open_time"] <= end_time)].copy()
        if scoped.empty:
            continue
        result[symbol] = scoped
    return result


def _write_cached_symbol(symbol: str, frame: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}_1h.csv"
    keep = [col for col in KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"] if col in frame.columns]
    frame[keep].drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time").to_csv(path, index=False)


def _aggregate_5m_to_1h(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce").astype("int64")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "trade_count" in frame.columns:
        frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce").fillna(0).astype("int64")
    else:
        frame["trade_count"] = 0
    frame["open_hour"] = (frame["open_time"] // HOUR_MS) * HOUR_MS
    frame = frame.sort_values(["symbol", "open_time"])
    grouped = frame.groupby(["symbol", "open_hour"], sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        trade_count=("trade_count", "sum"),
    ).reset_index()
    out = out.rename(columns={"open_hour": "open_time"})
    out["close_time"] = out["open_time"] + HOUR_MS - 1
    out["symbol"] = out["symbol"].astype(str)
    out["interval"] = INTERVAL
    out["open_time_utc"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["close_time_utc"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)
    for col in ["taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]:
        out[col] = np.nan
    return out.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def load_local_5m_as_1h(symbols: list[str], start_time: int, end_time: int) -> dict[str, pd.DataFrame]:
    if not LOCAL_5M_CACHE_DIR.exists():
        return {}
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        symbol_dir = LOCAL_5M_CACHE_DIR / symbol
        if not symbol_dir.exists():
            continue
        frames = []
        for path in list(symbol_dir.glob("*.parquet")) + list(symbol_dir.glob("*.csv")):
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame = frame[(frame["open_time"] >= start_time) & (frame["open_time"] <= end_time)].copy()
            if not frame.empty:
                frames.append(frame)
        if not frames:
            continue
        raw = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "open_time"])
        one_hour = _aggregate_5m_to_1h(raw)
        if not one_hour.empty:
            result[symbol] = one_hour
    return result


def has_required_hour_range(frame: pd.DataFrame, start_time: int, end_time: int) -> bool:
    if frame.empty:
        return False
    scoped = frame[(frame["open_time"] >= start_time) & (frame["open_time"] <= end_time)]
    if scoped.empty:
        return False
    expected = set(range(start_time, end_time + 1, HOUR_MS))
    available = set(int(v) for v in scoped["open_time"].dropna().tolist())
    return expected.issubset(available)


def load_or_download_klines(
    client: SimpleBinanceFuturesClient,
    symbol: str,
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    cached = _read_cached_symbol(symbol)
    if not cached.empty:
        scoped = cached[(cached["open_time"] >= start_time) & (cached["open_time"] <= end_time)].copy()
        if has_required_hour_range(scoped, start_time, end_time):
            return scoped

    downloaded = download_klines(client, symbol, start_time, end_time)
    combined = pd.concat([cached, downloaded], ignore_index=True) if not cached.empty else downloaded
    if not combined.empty:
        combined = combined.drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")
        combined["open_time_utc"] = pd.to_datetime(combined["open_time"], unit="ms", utc=True)
        combined["close_time_utc"] = pd.to_datetime(combined["close_time"], unit="ms", utc=True)
        _write_cached_symbol(symbol, combined)
    return combined[(combined["open_time"] >= start_time) & (combined["open_time"] <= end_time)].copy()


def ensure_symbol_range(
    client: SimpleBinanceFuturesClient,
    symbol: str,
    kline_map: dict[str, pd.DataFrame],
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    existing = kline_map.get(symbol, pd.DataFrame())
    cached = _read_cached_symbol(symbol)
    chunks = []
    if not existing.empty:
        chunks.append(existing)
    if not cached.empty:
        chunks.append(cached)
    combined = pd.concat(chunks, ignore_index=True).drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time") if chunks else pd.DataFrame()
    if has_required_hour_range(combined, start_time, end_time):
        scoped = combined[(combined["open_time"] >= start_time) & (combined["open_time"] <= end_time)].copy()
        kline_map[symbol] = combined
        return scoped
    downloaded = download_klines(client, symbol, start_time, end_time)
    chunks.append(downloaded)
    combined = pd.concat(chunks, ignore_index=True).drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time")
    if not combined.empty:
        combined["open_time_utc"] = pd.to_datetime(combined["open_time"], unit="ms", utc=True)
        combined["close_time_utc"] = pd.to_datetime(combined["close_time"], unit="ms", utc=True)
        _write_cached_symbol(symbol, combined)
    kline_map[symbol] = combined
    return combined[(combined["open_time"] >= start_time) & (combined["open_time"] <= end_time)].copy()


def build_snapshot_rankings(snapshot_time: int, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    current_open = snapshot_time - HOUR_MS
    close_24h_open = current_open - 24 * HOUR_MS
    rows = []
    for symbol, frame in kline_map.items():
        if frame.empty:
            continue
        indexed = frame.set_index("open_time", drop=False)
        if current_open not in indexed.index or close_24h_open not in indexed.index:
            continue
        current = indexed.loc[current_open]
        previous = indexed.loc[close_24h_open]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[-1]
        if isinstance(previous, pd.DataFrame):
            previous = previous.iloc[-1]
        prev_close = float(previous["close"])
        curr_close = float(current["close"])
        if not np.isfinite(prev_close) or prev_close <= 0 or not np.isfinite(curr_close):
            continue
        rows.append(
            {
                "signal_time": snapshot_time,
                "symbol": symbol,
                "gain_24h": curr_close / prev_close - 1.0,
                "current_close_time": current_open,
                "close_24h_ago_time": close_24h_open,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_time",
                "signal_time_utc",
                "signal_time_bj",
                "snapshot_hour_bj",
                "symbol",
                "rank",
                "gain_24h",
            ]
        )
    ranked = pd.DataFrame(rows).sort_values(["gain_24h", "symbol"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def generate_signals(signal_start: int, signal_end: int, kline_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    start_dt = ms_to_utc(signal_start).floor("D")
    end_dt = ms_to_utc(signal_end).floor("D")
    for day in pd.date_range(start=start_dt, end=end_dt, freq="D", tz="UTC"):
        for hour in SNAPSHOT_UTC_HOURS:
            snapshot_time = int((day + pd.Timedelta(hours=hour)).timestamp() * 1000)
            if snapshot_time < signal_start or snapshot_time > signal_end:
                continue
            snapshot_hour_bj = (ms_to_utc(snapshot_time) + pd.Timedelta(hours=8)).strftime("%H:%M")
            ranking = build_snapshot_rankings(snapshot_time, kline_map)
            top_n = ranking.head(TOP_N)
            for _, item in top_n.iterrows():
                rows.append(
                    {
                        "signal_time": snapshot_time,
                        "signal_time_utc": ms_to_utc(snapshot_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "signal_time_bj": ms_to_bj_string(snapshot_time),
                        "snapshot_hour_bj": snapshot_hour_bj,
                        "symbol": item["symbol"],
                        "rank": int(item["rank"]),
                        "gain_24h": float(item["gain_24h"]),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_time",
                "signal_time_utc",
                "signal_time_bj",
                "snapshot_hour_bj",
                "symbol",
                "rank",
                "gain_24h",
            ]
        )
    return pd.DataFrame(rows).sort_values(["signal_time", "rank"]).reset_index(drop=True)


def simulate_trade(signal: pd.Series, holding_days: int, kline_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    signal_time = int(signal["signal_time"])
    entry_time = signal_time
    planned_exit_time = entry_time + holding_days * DAY_MS
    exit_time = planned_exit_time
    base = {
        "signal_time_utc": signal["signal_time_utc"],
        "signal_time_bj": signal["signal_time_bj"],
        "signal_time_ms": signal_time,
        "snapshot_hour_bj": signal["snapshot_hour_bj"],
        "symbol": signal["symbol"],
        "rank": int(signal["rank"]),
        "gain_24h": float(signal["gain_24h"]),
        "entry_time_ms": entry_time,
        "entry_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_bj": ms_to_bj_string(entry_time),
        "exit_time_ms": exit_time,
        "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(exit_time),
        "planned_exit_time_ms": planned_exit_time,
        "planned_exit_time_utc": ms_to_utc(planned_exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "planned_exit_time_bj": ms_to_bj_string(planned_exit_time),
        "holding_days": holding_days,
        "month": ms_to_bj_string(entry_time)[:7],
        "status": "completed",
        "skip_reason": "",
    }
    frame = kline_map.get(str(signal["symbol"]), pd.DataFrame())
    if frame.empty:
        return base | {"entry_price": np.nan, "exit_price": np.nan, "gross_return_pct": np.nan, "net_return_pct": np.nan, "pnl_u": np.nan, "is_win": False, "status": "skipped", "skip_reason": "missing_symbol_klines"}
    indexed = frame.set_index("open_time", drop=False)
    if entry_time not in indexed.index:
        return base | {"entry_price": np.nan, "exit_price": np.nan, "gross_return_pct": np.nan, "net_return_pct": np.nan, "pnl_u": np.nan, "is_win": False, "status": "skipped", "skip_reason": "missing_entry_kline"}
    exit_reason = ""
    if exit_time not in indexed.index:
        available_after_entry = frame[frame["open_time"] >= entry_time]
        if available_after_entry.empty:
            entry_row = indexed.loc[entry_time]
            if isinstance(entry_row, pd.DataFrame):
                entry_row = entry_row.iloc[-1]
            return base | {"entry_price": float(entry_row["open"]), "exit_price": np.nan, "gross_return_pct": np.nan, "net_return_pct": np.nan, "pnl_u": np.nan, "is_win": False, "status": "skipped", "skip_reason": "missing_exit_kline"}
        exit_time = int(available_after_entry["open_time"].max())
        exit_reason = "exit_at_latest_available_kline"
        base["exit_time_ms"] = exit_time
        base["exit_time_utc"] = ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S")
        base["exit_time_bj"] = ms_to_bj_string(exit_time)

    entry_row = indexed.loc[entry_time]
    exit_row = indexed.loc[exit_time]
    if isinstance(entry_row, pd.DataFrame):
        entry_row = entry_row.iloc[-1]
    if isinstance(exit_row, pd.DataFrame):
        exit_row = exit_row.iloc[-1]
    entry_price = float(entry_row["open"])
    exit_price = float(exit_row["open"])
    gross_return = exit_price / entry_price - 1.0
    qty = BUY_NOTIONAL_U * (1.0 - FEE_RATE) / entry_price
    exit_value = qty * exit_price * (1.0 - FEE_RATE)
    pnl_u = exit_value - BUY_NOTIONAL_U
    net_return = pnl_u / BUY_NOTIONAL_U
    return base | {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return_pct": gross_return * 100,
        "net_return_pct": net_return * 100,
        "pnl_u": pnl_u,
        "is_win": pnl_u > 0,
        "exit_reason": exit_reason,
    }


def skipped_open_position_trade(signal: pd.Series, holding_days: int, open_until: int) -> dict[str, Any]:
    signal_time = int(signal["signal_time"])
    entry_time = signal_time
    exit_time = entry_time + holding_days * DAY_MS
    return {
        "signal_time_utc": signal["signal_time_utc"],
        "signal_time_bj": signal["signal_time_bj"],
        "signal_time_ms": signal_time,
        "snapshot_hour_bj": signal["snapshot_hour_bj"],
        "symbol": signal["symbol"],
        "rank": int(signal["rank"]),
        "gain_24h": float(signal["gain_24h"]),
        "entry_time_ms": entry_time,
        "entry_time_utc": ms_to_utc(entry_time).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_time_bj": ms_to_bj_string(entry_time),
        "exit_time_ms": exit_time,
        "exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time_bj": ms_to_bj_string(exit_time),
        "planned_exit_time_ms": exit_time,
        "planned_exit_time_utc": ms_to_utc(exit_time).strftime("%Y-%m-%d %H:%M:%S"),
        "planned_exit_time_bj": ms_to_bj_string(exit_time),
        "holding_days": holding_days,
        "month": ms_to_bj_string(entry_time)[:7],
        "entry_price": np.nan,
        "exit_price": np.nan,
        "gross_return_pct": np.nan,
        "net_return_pct": np.nan,
        "pnl_u": np.nan,
        "is_win": False,
        "status": "skipped",
        "skip_reason": "symbol_already_open",
        "exit_reason": "",
        "blocking_position_exit_time_utc": ms_to_utc(open_until).strftime("%Y-%m-%d %H:%M:%S"),
        "blocking_position_exit_time_bj": ms_to_bj_string(open_until),
    }


def simulate_trades_with_position_limit(
    signals: pd.DataFrame,
    holding_days: int,
    kline_map: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    open_until_by_symbol: dict[str, int] = {}
    ordered = signals.sort_values(["signal_time", "rank"]).reset_index(drop=True)
    for _, signal in ordered.iterrows():
        symbol = str(signal["symbol"])
        signal_time = int(signal["signal_time"])
        open_until = open_until_by_symbol.get(symbol)
        if open_until is not None and signal_time < open_until:
            rows.append(skipped_open_position_trade(signal, holding_days, open_until))
            continue

        trade = simulate_trade(signal, holding_days, kline_map)
        rows.append(trade)
        if trade["status"] == "completed":
            open_until_by_symbol[symbol] = int(trade.get("exit_time_ms", signal_time + holding_days * DAY_MS))
    return rows


def calculate_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    equity = pnl_series.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def profit_factor_from_pnl(pnl: pd.Series) -> float:
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses == 0:
        return math.inf
    return gains / losses


def _format_pf(value: float) -> str:
    if value == math.inf or np.isinf(value):
        return "inf"
    if pd.isna(value):
        return "nan"
    return f"{value:.2f}"


def summarize_completed(group: pd.DataFrame) -> dict[str, Any]:
    completed = group[group["status"] == "completed"].sort_values("entry_time_utc").copy()
    skipped = group[group["status"] != "completed"]
    pnl = completed["pnl_u"].astype(float) if not completed.empty else pd.Series(dtype=float)
    returns = completed["net_return_pct"].astype(float) if not completed.empty else pd.Series(dtype=float)
    wins = completed[completed["pnl_u"] > 0]
    losses = completed[completed["pnl_u"] < 0]
    max_dd_u = calculate_drawdown(pnl)
    total_deployed = len(completed) * BUY_NOTIONAL_U
    return {
        "raw_signals": int(len(group)),
        "completed_trades": int(len(completed)),
        "skipped_trades": int(len(skipped)),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate": float(len(wins) / len(completed)) if len(completed) else np.nan,
        "avg_return_pct": float(returns.mean()) if len(returns) else np.nan,
        "median_return_pct": float(returns.median()) if len(returns) else np.nan,
        "avg_win_pct": float(wins["net_return_pct"].mean()) if len(wins) else np.nan,
        "avg_loss_pct": float(losses["net_return_pct"].mean()) if len(losses) else np.nan,
        "max_win_pct": float(returns.max()) if len(returns) else np.nan,
        "max_loss_pct": float(returns.min()) if len(returns) else np.nan,
        "total_net_pnl_u": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl_u": float(pnl.mean()) if len(pnl) else np.nan,
        "profit_factor": profit_factor_from_pnl(pnl),
        "max_drawdown_u": max_dd_u,
        "max_drawdown_pct_on_total_deployed": float(max_dd_u / total_deployed) if total_deployed else np.nan,
    }


def calculate_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for holding_days, group in trades.groupby("holding_days", sort=True):
        row = summarize_completed(group)
        row = {"version": f"hold_{holding_days}d", "holding_days": int(holding_days)} | row
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (holding_days, month), group in trades.groupby(["holding_days", "month"], sort=True):
        row = summarize_completed(group)
        rows.append({"holding_days": int(holding_days), "month": month} | row)
    return pd.DataFrame(rows)


def calculate_group_stats(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for holding_days, hdf in trades.groupby("holding_days", sort=True):
        for value, group in hdf.groupby("snapshot_hour_bj", sort=True):
            rows.append({"holding_days": int(holding_days), "group_type": "snapshot_hour_bj", "group_value": value} | summarize_completed(group))
        for value, group in hdf.groupby("rank", sort=True):
            rows.append({"holding_days": int(holding_days), "group_type": "rank", "group_value": f"Rank {int(value)}"} | summarize_completed(group))
        hdf = hdf.copy()
        hdf["snapshot_rank"] = hdf["snapshot_hour_bj"] + " Rank" + hdf["rank"].astype(int).astype(str)
        for value, group in hdf.groupby("snapshot_rank", sort=True):
            rows.append({"holding_days": int(holding_days), "group_type": "snapshot_hour_plus_rank", "group_value": value} | summarize_completed(group))
    return pd.DataFrame(rows)


def calculate_tail_dependency(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for holding_days, group in trades.groupby("holding_days", sort=True):
        completed = group[group["status"] == "completed"].copy()
        sorted_group = completed.sort_values("pnl_u", ascending=False)
        row: dict[str, Any] = {"holding_days": int(holding_days), "completed_trades": int(len(completed))}
        for drop_n, label in [(0, "original"), (1, "drop_best_1"), (3, "drop_best_3"), (5, "drop_best_5"), (10, "drop_best_10")]:
            scoped = sorted_group.iloc[drop_n:] if drop_n else sorted_group
            pnl = scoped["pnl_u"].astype(float) if not scoped.empty else pd.Series(dtype=float)
            row[f"{label}_net_pnl_u"] = float(pnl.sum()) if len(pnl) else 0.0
            row[f"{label}_profit_factor"] = profit_factor_from_pnl(pnl)
        rows.append(row)
    return pd.DataFrame(rows)


def _print_table(title: str, frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 60) -> None:
    print(f"\n========== {title} ==========", flush=True)
    if frame.empty:
        print("(empty)", flush=True)
        return
    view = frame[columns].copy() if columns else frame.copy()
    print(view.head(max_rows).to_string(index=False), flush=True)


def final_conclusion(summary: pd.DataFrame, monthly: pd.DataFrame, group_stats: pd.DataFrame, tail: pd.DataFrame) -> list[str]:
    ranked = summary.sort_values("profit_factor", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    best_days = int(best["holding_days"])
    best_tail = tail[tail["holding_days"] == best_days].iloc[0]
    best_monthly = monthly[monthly["holding_days"] == best_days].copy()
    profitable_months = int((best_monthly["total_net_pnl_u"] > 0).sum()) if not best_monthly.empty else 0
    losing_months = int((best_monthly["total_net_pnl_u"] < 0).sum()) if not best_monthly.empty else 0
    best_month = best_monthly.sort_values("total_net_pnl_u", ascending=False).iloc[0] if not best_monthly.empty else None
    worst_month = best_monthly.sort_values("total_net_pnl_u", ascending=True).iloc[0] if not best_monthly.empty else None
    positive_after_drop3 = float(best_tail["drop_best_3_net_pnl_u"]) > 0
    pf = float(best["profit_factor"])
    total_pnl = float(best["total_net_pnl_u"])
    if pf > 1.3 and positive_after_drop3:
        edge = "强 Edge"
    elif (1.05 <= pf <= 1.3) or (total_pnl > 0):
        edge = "弱 Edge"
    else:
        edge = "无 Edge"
    if float(best_tail["drop_best_1_net_pnl_u"]) < 0:
        edge = "无 Edge"
    if losing_months > profitable_months:
        edge = "无 Edge" if pf <= 1.05 else edge

    best_groups = group_stats[group_stats["holding_days"] == best_days].copy()
    time_groups = best_groups[best_groups["group_type"] == "snapshot_hour_bj"].set_index("group_value")
    rank_groups = best_groups[best_groups["group_type"] == "rank"].set_index("group_value")
    time_0800 = time_groups.loc["08:00"] if "08:00" in time_groups.index else None
    time_2100 = time_groups.loc["21:00"] if "21:00" in time_groups.index else None
    rank1 = rank_groups.loc["Rank 1"] if "Rank 1" in rank_groups.index else None
    rank2 = rank_groups.loc["Rank 2"] if "Rank 2" in rank_groups.index else None

    def row_pf(row: pd.Series | None) -> float:
        return float(row["profit_factor"]) if row is not None else np.nan

    def row_pnl(row: pd.Series | None) -> float:
        return float(row["total_net_pnl_u"]) if row is not None else np.nan

    better_time = "08:00" if row_pf(time_0800) >= row_pf(time_2100) else "21:00"
    better_rank = "Rank1" if row_pf(rank1) >= row_pf(rank2) else "Rank2"
    month_dependency = "存在明显单月依赖" if best_month is not None and abs(float(best_month["total_net_pnl_u"])) >= max(abs(total_pnl) * 0.5, 1e-9) else "未见明显单月依赖"

    return [
        "========== 最终结论 ==========",
        "",
        "1. PF最高版本：",
        f"持有{best_days}天，PF = {_format_pf(pf)}，总净收益 = {total_pnl:+.2f} U，胜率 = {float(best['win_rate']) * 100:.2f}%。",
        "",
        "2. 是否存在基础 Edge：",
        f"判断为：{edge}。去最大3笔后净收益 = {float(best_tail['drop_best_3_net_pnl_u']):+.2f} U；去最大1笔后净收益 = {float(best_tail['drop_best_1_net_pnl_u']):+.2f} U。",
        "",
        "3. 月度稳定性：",
        f"盈利月份 {profitable_months} 个，亏损月份 {losing_months} 个。"
        + (f"最赚钱月份 {best_month['month']} ({float(best_month['total_net_pnl_u']):+.2f} U)，最亏钱月份 {worst_month['month']} ({float(worst_month['total_net_pnl_u']):+.2f} U)。{month_dependency}。" if best_month is not None and worst_month is not None else ""),
        "",
        "4. 时间点比较：",
        f"北京时间 08:00 的 PF = {_format_pf(row_pf(time_0800))}，净收益 = {row_pnl(time_0800):+.2f} U；北京时间 21:00 的 PF = {_format_pf(row_pf(time_2100))}，净收益 = {row_pnl(time_2100):+.2f} U。更优时间点是：{better_time}。",
        "",
        "5. 排名比较：",
        f"Rank1 的 PF = {_format_pf(row_pf(rank1))}，净收益 = {row_pnl(rank1):+.2f} U；Rank2 的 PF = {_format_pf(row_pf(rank2))}，净收益 = {row_pnl(rank2):+.2f} U。更优排名是：{better_rank}。",
    ]


def _summary_row(summary: pd.DataFrame, holding_days: int) -> pd.Series | None:
    rows = summary[summary["holding_days"] == holding_days]
    if rows.empty:
        return None
    return rows.iloc[0]


def _tail_row(tail: pd.DataFrame, holding_days: int) -> pd.Series | None:
    rows = tail[tail["holding_days"] == holding_days]
    if rows.empty:
        return None
    return rows.iloc[0]


def _group_row(group_stats: pd.DataFrame, holding_days: int, group_type: str, group_value: str) -> pd.Series | None:
    rows = group_stats[
        (group_stats["holding_days"] == holding_days)
        & (group_stats["group_type"] == group_type)
        & (group_stats["group_value"] == group_value)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _month_stability(monthly: pd.DataFrame, holding_days: int) -> dict[str, Any]:
    rows = monthly[monthly["holding_days"] == holding_days].copy()
    if rows.empty:
        return {
            "profitable_months": 0,
            "losing_months": 0,
            "best_month": None,
            "worst_month": None,
            "dependency": "no_monthly_data",
            "stability_score": -1.0,
        }
    profitable = int((rows["total_net_pnl_u"] > 0).sum())
    losing = int((rows["total_net_pnl_u"] < 0).sum())
    best = rows.sort_values("total_net_pnl_u", ascending=False).iloc[0]
    worst = rows.sort_values("total_net_pnl_u", ascending=True).iloc[0]
    total_abs = float(rows["total_net_pnl_u"].abs().sum())
    best_abs_share = abs(float(best["total_net_pnl_u"])) / total_abs if total_abs else 0.0
    dependency = "yes" if best_abs_share >= 0.50 else "no"
    return {
        "profitable_months": profitable,
        "losing_months": losing,
        "best_month": best,
        "worst_month": worst,
        "dependency": dependency,
        "stability_score": profitable - losing - best_abs_share,
    }


def _pf_and_pnl_text(row: pd.Series | None) -> str:
    if row is None:
        return "PF=nan, net=nan U"
    return f"PF={_format_pf(float(row['profit_factor']))}, net={float(row['total_net_pnl_u']):+.2f} U"


def hold14_comparison_conclusion(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    group_stats: pd.DataFrame,
    tail: pd.DataFrame,
) -> list[str]:
    ordered = summary.sort_values("profit_factor", ascending=False).copy()
    lines = [
        "",
        "========== 新增 Hold 14D 对比结论 ==========",
        "",
        "1. 四个版本 PF 排名：",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"   hold_{int(row['holding_days'])}d: PF={_format_pf(float(row['profit_factor']))}, "
            f"净收益={float(row['total_net_pnl_u']):+.2f} U, 胜率={float(row['win_rate']) * 100:.2f}%"
        )

    h14 = _summary_row(summary, 14)
    h7 = _summary_row(summary, 7)
    t14 = _tail_row(tail, 14)
    m14 = _month_stability(monthly, 14)
    m7 = _month_stability(monthly, 7)
    time_0800 = _group_row(group_stats, 14, "snapshot_hour_bj", "08:00")
    time_2100 = _group_row(group_stats, 14, "snapshot_hour_bj", "21:00")
    rank1 = _group_row(group_stats, 14, "rank", "Rank 1")
    rank2 = _group_row(group_stats, 14, "rank", "Rank 2")

    if h14 is None or t14 is None:
        return lines + ["", "hold_14d 没有可用结果，无法比较。"]

    better_than_7d = False
    if h7 is not None:
        better_than_7d = (
            float(h14["profit_factor"]) > float(h7["profit_factor"])
            and float(h14["total_net_pnl_u"]) >= float(h7["total_net_pnl_u"])
            and m14["stability_score"] >= m7["stability_score"]
        )
    stable_edge = (
        float(h14["profit_factor"]) > 1.05
        and float(t14["drop_best_3_net_pnl_u"]) > 0
        and m14["profitable_months"] >= m14["losing_months"]
        and m14["dependency"] == "no"
    )

    better_time = "08:00"
    if time_0800 is not None and time_2100 is not None and float(time_2100["profit_factor"]) > float(time_0800["profit_factor"]):
        better_time = "21:00"
    better_rank = "Rank1"
    if rank1 is not None and rank2 is not None and float(rank2["profit_factor"]) > float(rank1["profit_factor"]):
        better_rank = "Rank2"

    best_month = m14["best_month"]
    worst_month = m14["worst_month"]
    lines.extend(
        [
            "",
            "2. hold_14d 核心结果：",
            f"   总信号数: {int(h14['raw_signals'])}",
            f"   完成交易数: {int(h14['completed_trades'])}",
            f"   净收益: {float(h14['total_net_pnl_u']):+.2f} U",
            f"   PF: {_format_pf(float(h14['profit_factor']))}",
            f"   胜率: {float(h14['win_rate']) * 100:.2f}%",
            f"   平均收益率: {float(h14['avg_return_pct']):+.2f}%",
            f"   中位数收益率: {float(h14['median_return_pct']):+.2f}%",
            f"   最大回撤: {float(h14['max_drawdown_u']):+.2f} U",
            "",
            "3. hold_14d 长尾依赖：",
            f"   原始净收益: {float(t14['original_net_pnl_u']):+.2f} U",
            f"   去最大1笔: {float(t14['drop_best_1_net_pnl_u']):+.2f} U ({'仍为正' if float(t14['drop_best_1_net_pnl_u']) > 0 else '转负或非正'})",
            f"   去最大3笔: {float(t14['drop_best_3_net_pnl_u']):+.2f} U ({'仍为正' if float(t14['drop_best_3_net_pnl_u']) > 0 else '转负或非正'})",
            f"   去最大5笔: {float(t14['drop_best_5_net_pnl_u']):+.2f} U ({'仍为正' if float(t14['drop_best_5_net_pnl_u']) > 0 else '转负或非正'})",
            f"   去最大10笔: {float(t14['drop_best_10_net_pnl_u']):+.2f} U ({'仍为正' if float(t14['drop_best_10_net_pnl_u']) > 0 else '转负或非正'})",
            "",
            "4. hold_14d 月度稳定性：",
            f"   盈利月份: {m14['profitable_months']}",
            f"   亏损月份: {m14['losing_months']}",
            f"   最赚钱月份: {best_month['month']} ({float(best_month['total_net_pnl_u']):+.2f} U)" if best_month is not None else "   最赚钱月份: NA",
            f"   最亏钱月份: {worst_month['month']} ({float(worst_month['total_net_pnl_u']):+.2f} U)" if worst_month is not None else "   最亏钱月份: NA",
            f"   是否依赖单月: {'是' if m14['dependency'] == 'yes' else '否'}",
            f"   是否比 hold_7d 更稳定: {'是' if m14['stability_score'] > m7['stability_score'] else '否'}",
            "",
            "5. hold_14d 分组：",
            f"   08:00 PF / 净收益: {_pf_and_pnl_text(time_0800)}",
            f"   21:00 PF / 净收益: {_pf_and_pnl_text(time_2100)}，更优时间点: {better_time}",
            f"   Rank1 PF / 净收益: {_pf_and_pnl_text(rank1)}",
            f"   Rank2 PF / 净收益: {_pf_and_pnl_text(rank2)}，更优排名: {better_rank}",
            "",
            "6. 最终判断：",
            f"   hold_14d 是否优于 hold_7d？{'是' if better_than_7d else '否'}",
            f"   是否存在稳定基础 Edge？{'是' if stable_edge else '否'}",
        ]
    )
    return lines


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    client = SimpleBinanceFuturesClient()

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    signal_end_dt = latest_signal_end_dt(now)
    signal_start_dt = signal_end_dt - timedelta(days=SIGNAL_DAYS)
    signal_start = int(signal_start_dt.timestamp() * 1000)
    signal_end = int(signal_end_dt.timestamp() * 1000)
    kline_start = signal_start - 25 * HOUR_MS
    kline_end = signal_end

    print("Loading Binance Futures symbols...", flush=True)
    symbols = get_futures_symbols(client)
    print(f"Symbols Count: {len(symbols)}", flush=True)

    kline_map: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []

    print("Trying local 1h csv cache for ranking data...", flush=True)
    kline_map = load_local_1h_cache(symbols, kline_start, signal_end)
    print(f"Loaded local 1h cache for symbols={len(kline_map)}", flush=True)

    print("Trying local 5m parquet/csv cache for remaining ranking data...", flush=True)
    try:
        missing_for_5m = [symbol for symbol in symbols if symbol not in kline_map]
        kline_map.update(load_local_5m_as_1h(missing_for_5m, kline_start, signal_end))
        print(f"Loaded local 1h/5m cache for symbols={len(kline_map)}", flush=True)
    except ImportError as exc:
        print(f"Local 5m cache exists but parquet engine is unavailable: {exc}", flush=True)

    missing_for_ranking = [symbol for symbol in symbols if symbol not in kline_map]
    if missing_for_ranking and ALLOW_DOWNLOADS:
        print(f"Downloading ranking-stage 1h data for missing symbols={len(missing_for_ranking)}", flush=True)
    elif missing_for_ranking:
        print(f"Skipping missing ranking-stage symbols without local cache={len(missing_for_ranking)}", flush=True)
    for idx, symbol in enumerate(missing_for_ranking if ALLOW_DOWNLOADS else [], start=1):
        try:
            frame = load_or_download_klines(client, symbol, kline_start, signal_end)
            if not frame.empty:
                kline_map[symbol] = frame
            print(f"[ranking {idx}/{len(missing_for_ranking)}] {symbol} rows={len(frame)}", flush=True)
        except Exception as exc:
            failures.append({"stage": "ranking", "symbol": symbol, "error": repr(exc)})
            print(f"[ranking {idx}/{len(missing_for_ranking)}] {symbol} failed: {exc}", flush=True)

    if failures:
        pd.DataFrame(failures).to_csv(OUT / "futures_top2_fixed_time_download_failures.csv", index=False)

    print("Generating fixed-time Top2 signals...", flush=True)
    signals = generate_signals(signal_start, signal_end, kline_map)
    signals.to_csv(SIGNALS_PATH, index=False)
    top_symbols = sorted(signals["symbol"].dropna().astype(str).unique().tolist()) if not signals.empty else []
    print(f"Top2 unique symbols needing exit data: {len(top_symbols)}", flush=True)
    for idx, symbol in enumerate(top_symbols, start=1):
        try:
            if ALLOW_DOWNLOADS:
                frame = ensure_symbol_range(client, symbol, kline_map, kline_start, kline_end)
            else:
                frame = kline_map.get(symbol, pd.DataFrame())
            print(f"[exit {idx}/{len(top_symbols)}] {symbol} rows={len(frame)}", flush=True)
        except Exception as exc:
            failures.append({"stage": "exit", "symbol": symbol, "error": repr(exc)})
            print(f"[exit {idx}/{len(top_symbols)}] {symbol} failed: {exc}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(OUT / "futures_top2_fixed_time_download_failures.csv", index=False)

    trade_rows = []
    for holding_days in HOLDING_DAYS:
        trade_rows.extend(simulate_trades_with_position_limit(signals, holding_days, kline_map))
    trades = pd.DataFrame(trade_rows)
    summary = calculate_summary(trades)
    monthly = calculate_monthly_summary(trades)
    group_stats = calculate_group_stats(trades)
    tail = calculate_tail_dependency(trades)

    trades.to_csv(TRADES_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    monthly.to_csv(MONTHLY_PATH, index=False)
    group_stats.to_csv(GROUP_PATH, index=False)
    tail.to_csv(TAIL_PATH, index=False)

    snapshot_count = len(pd.date_range(start=ms_to_utc(signal_start).floor("D"), end=ms_to_utc(signal_end).floor("D"), freq="D", tz="UTC")) * len(SNAPSHOT_UTC_HOURS)
    print("\n========== Binance Futures Top2 Fixed Time Backtest ==========", flush=True)
    print("Backtest Period:", flush=True)
    print(f"Signal Start: {ms_to_utc(signal_start).strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
    print(f"Signal End:   {ms_to_utc(signal_end).strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
    print(f"Symbols Count: {len(symbols)}", flush=True)
    print(f"Downloaded/usable Symbols: {len(kline_map)}", flush=True)
    print(f"Total Snapshots: {snapshot_count}", flush=True)
    print(f"Total Raw Signals: {len(signals)}", flush=True)
    print(f"Fee Assumption: buy {FEE_RATE:.3%}, sell {FEE_RATE:.3%}, no slippage, 1x, 100 USDT per completed entry", flush=True)
    print("Position Rule: same symbol will not be bought again while an existing position is still open for the same holding-days version", flush=True)

    _print_table("Overall Summary", summary)
    _print_table("Monthly Summary", monthly)
    _print_table("Group Stats", group_stats)
    _print_table("Tail Dependency", tail)
    print("\n".join(final_conclusion(summary, monthly, group_stats, tail)), flush=True)
    print("\n".join(hold14_comparison_conclusion(summary, monthly, group_stats, tail)), flush=True)
    print(f"\nWrote files under: {OUT}", flush=True)


if __name__ == "__main__":
    main()
