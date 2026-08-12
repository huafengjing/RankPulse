from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_futures_top2_fixed_time import CACHE_DIR, HOUR_MS


BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL = "1h"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def fetch_klines(symbol: str, start_ms: int, end_ms: int, retries: int = 3) -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1500,
        }
    )
    url = f"{BASE_URL}?{params}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                raw = json.loads(response.read().decode("utf-8"))
            frame = pd.DataFrame(raw, columns=KLINE_COLUMNS)
            if frame.empty:
                return pd.DataFrame(columns=KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"])
            for col in ["open_time", "close_time", "trade_count"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("int64")
            for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["symbol"] = symbol
            frame["interval"] = INTERVAL
            frame["open_time_utc"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).astype(str)
            frame["close_time_utc"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True).astype(str)
            return frame[(frame["open_time"] >= start_ms) & (frame["open_time"] <= end_ms)].copy()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{symbol} fetch failed: {last_error!r}")


def read_symbol(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"])
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce").astype("int64")
    return frame


def update_symbol(symbol: str, target_ms: int) -> dict[str, object]:
    path = CACHE_DIR / f"{symbol}_1h.csv"
    cached = read_symbol(path)
    if cached.empty:
        return {"symbol": symbol, "downloaded": 0, "last_ms": None, "error": "empty_cache"}
    last_ms = int(cached["open_time"].max())
    if last_ms >= target_ms:
        return {"symbol": symbol, "downloaded": 0, "last_ms": last_ms, "error": ""}
    start_ms = last_ms + HOUR_MS
    downloaded = fetch_klines(symbol, start_ms, target_ms)
    if not downloaded.empty:
        combined = pd.concat([cached, downloaded], ignore_index=True)
        keep = [col for col in KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"] if col in combined.columns]
        combined[keep].drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time").to_csv(path, index=False)
    return {"symbol": symbol, "downloaded": int(len(downloaded)), "last_ms": target_ms, "error": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="UTC target hour, e.g. 2026-07-27 15:00:00")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    target_ms = int(pd.Timestamp(args.target, tz="UTC").timestamp() * 1000)
    symbols = sorted(path.stem.removesuffix("_1h") for path in CACHE_DIR.glob("*_1h.csv"))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(update_symbol, symbol, target_ms): symbol for symbol in symbols}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"symbol": futures[future], "downloaded": 0, "last_ms": None, "error": repr(exc)})

    audit = pd.DataFrame(rows).sort_values("symbol")
    out = Path("output") / "fast_update_1h_cache_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False, encoding="utf-8-sig")
    print(
        "symbols",
        len(audit),
        "downloaded_rows",
        int(audit["downloaded"].sum()),
        "errors",
        int(audit["error"].astype(str).ne("").sum()),
    )


if __name__ == "__main__":
    main()
