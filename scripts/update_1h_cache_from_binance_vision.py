from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_futures_top2_fixed_time import CACHE_DIR, HOUR_MS


BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
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


def read_cached_symbol(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"])
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce").astype("int64")
    return frame


def parse_daily_zip(symbol: str, day: pd.Timestamp, retries: int = 3) -> pd.DataFrame:
    date_text = day.strftime("%Y-%m-%d")
    quoted_symbol = urllib.request.quote(symbol, safe="")
    url = f"{BASE_URL}/{quoted_symbol}/{INTERVAL}/{quoted_symbol}-{INTERVAL}-{date_text}.zip"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                name = archive.namelist()[0]
                with archive.open(name) as handle:
                    frame = pd.read_csv(handle, header=None)
            if len(frame.columns) >= len(KLINE_COLUMNS):
                frame = frame.iloc[:, : len(KLINE_COLUMNS)]
                frame.columns = KLINE_COLUMNS
            else:
                return pd.DataFrame(columns=KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"])
            frame = frame[pd.to_numeric(frame["open_time"], errors="coerce").notna()].copy()
            if frame.empty:
                return pd.DataFrame(columns=KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"])
            for col in ["open_time", "close_time", "trade_count"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype("int64")
            for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["symbol"] = symbol
            frame["interval"] = INTERVAL
            frame["open_time_utc"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).astype(str)
            frame["close_time_utc"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True).astype(str)
            return frame
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"{symbol} {date_text} failed: {last_error!r}")


def update_symbol(symbol: str, target_ms: int) -> dict[str, object]:
    path = CACHE_DIR / f"{symbol}_1h.csv"
    cached = read_cached_symbol(path)
    if cached.empty:
        return {"symbol": symbol, "downloaded": 0, "last_ms": None, "error": "empty_cache"}
    last_ms = int(cached["open_time"].max())
    if last_ms >= target_ms:
        return {"symbol": symbol, "downloaded": 0, "last_ms": last_ms, "error": ""}

    start = pd.to_datetime(last_ms + HOUR_MS, unit="ms", utc=True).floor("D")
    end = pd.to_datetime(target_ms, unit="ms", utc=True).floor("D")
    parts = []
    for day in pd.date_range(start=start, end=end, freq="D", tz="UTC"):
        part = parse_daily_zip(symbol, day)
        if not part.empty:
            parts.append(part)

    if parts:
        downloaded = pd.concat(parts, ignore_index=True)
        downloaded = downloaded[(downloaded["open_time"] > last_ms) & (downloaded["open_time"] <= target_ms)].copy()
        combined = pd.concat([cached, downloaded], ignore_index=True)
        keep = [col for col in KLINE_COLUMNS + ["symbol", "interval", "open_time_utc", "close_time_utc"] if col in combined.columns]
        combined[keep].drop_duplicates(["symbol", "interval", "open_time"]).sort_values("open_time").to_csv(path, index=False)
        return {"symbol": symbol, "downloaded": int(len(downloaded)), "last_ms": target_ms, "error": ""}

    return {"symbol": symbol, "downloaded": 0, "last_ms": last_ms, "error": "no_rows"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="UTC target hour, e.g. 2026-07-27 15:00:00")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    target_ms = int(pd.Timestamp(args.target, tz="UTC").timestamp() * 1000)
    symbols = sorted(path.stem.removesuffix("_1h") for path in CACHE_DIR.glob("*_1h.csv"))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(update_symbol, symbol, target_ms): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"symbol": symbol, "downloaded": 0, "last_ms": None, "error": repr(exc)})

    audit = pd.DataFrame(rows).sort_values("symbol")
    out = Path("output") / "binance_vision_1h_cache_audit.csv"
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
