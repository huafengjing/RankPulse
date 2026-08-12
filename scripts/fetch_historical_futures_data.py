from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT_CACHE = ROOT / "data" / "futures_klines_1h"
HISTORY_CACHE = ROOT / "data" / "futures_klines_1h_history_2025h2"
RAW_ROOT = ROOT / "data" / "raw" / "binance_vision" / "futures_um" / "monthly_klines" / "1h"
METADATA_PATH = ROOT / "data" / "raw" / "exchange_info" / "exchange_info_latest.json"
MANIFEST_PATH = ROOT / "data" / "cache_metadata" / "history_2025h2_download_manifest.csv"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
# Two extra days are retained so the first 2025-05-25 warmup snapshot has a
# complete 24H lookback. This is earlier than the task's minimum download date.
START = pd.Timestamp("2025-05-22 00:00:00", tz="UTC")
END = pd.Timestamp("2026-01-05 00:00:00", tz="UTC")
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]


def unix_ms(value: pd.Timestamp) -> int:
    return int(value.timestamp() * 1000)


def eligible_contracts(metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in metadata.get("symbols", []):
        if item.get("contractType") != "PERPETUAL" or item.get("quoteAsset") != "USDT":
            continue
        onboard = int(item.get("onboardDate") or 0)
        delivery = int(item.get("deliveryDate") or 0)
        if onboard >= unix_ms(END) or (delivery and delivery <= unix_ms(START)):
            continue
        rows.append(
            {
                "symbol": item["symbol"],
                "contract_type": item.get("contractType"),
                "quote_asset": item.get("quoteAsset"),
                "current_status": item.get("status"),
                "onboard_time_ms": onboard,
                "onboard_time_utc": pd.to_datetime(onboard, unit="ms", utc=True),
                "delivery_time_ms": delivery,
                "delivery_time_utc": pd.to_datetime(delivery, unit="ms", utc=True),
                "listing_source": "official_exchange_info_snapshot",
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def cached_bounds(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    frame = pd.read_csv(path, usecols=["open_time"])
    if frame.empty:
        return None, None
    values = pd.to_numeric(frame.open_time, errors="coerce").dropna()
    return (int(values.min()), int(values.max())) if len(values) else (None, None)


def months_for_symbol(symbol: str) -> list[pd.Period]:
    current_start, current_end = cached_bounds(CURRENT_CACHE / f"{symbol}_1h.csv")
    all_months = list(pd.period_range(START.strftime("%Y-%m"), END.strftime("%Y-%m"), freq="M"))
    if current_start is None or current_end is None:
        return all_months
    first_cached_month = pd.Period(pd.to_datetime(current_start, unit="ms", utc=True).strftime("%Y-%m"), freq="M")
    last_cached_month = pd.Period(pd.to_datetime(current_end, unit="ms", utc=True).strftime("%Y-%m"), freq="M")
    return [month for month in all_months if month <= first_cached_month or month >= last_cached_month]


def parse_archive(payload: bytes, symbol: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None)
    if frame.shape[1] < len(KLINE_COLUMNS):
        return pd.DataFrame(columns=KLINE_COLUMNS)
    frame = frame.iloc[:, : len(KLINE_COLUMNS)]
    frame.columns = KLINE_COLUMNS
    frame["open_time"] = pd.to_numeric(frame.open_time, errors="coerce")
    frame = frame[frame.open_time.notna()].copy()
    if frame.empty:
        return frame
    for column in ["open_time", "close_time", "trade_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
    for column in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = symbol
    frame["interval"] = "1h"
    frame["open_time_utc"] = pd.to_datetime(frame.open_time, unit="ms", utc=True).astype(str)
    frame["close_time_utc"] = pd.to_datetime(frame.close_time, unit="ms", utc=True).astype(str)
    return frame


def fetch_archive(symbol: str, month: pd.Period, retries: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    month_text = str(month)
    filename = f"{symbol}-1h-{month_text}.zip"
    raw_path = RAW_ROOT / symbol / filename
    url = f"{BASE_URL}/{urllib.parse.quote(symbol, safe='')}/1h/{urllib.parse.quote(filename, safe='')}"
    source = "cache"
    status = "ok"
    error = ""
    http_status: int | None = None
    payload = b""
    if raw_path.exists():
        payload = raw_path.read_bytes()
    else:
        source = "download"
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "frozen-strategy-history-research/1.0"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    http_status = int(response.status)
                    payload = response.read()
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(payload)
                break
            except urllib.error.HTTPError as exc:
                http_status = int(exc.code)
                if exc.code == 404:
                    status = "not_available"
                    last_error = exc
                    break
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.5 * (attempt + 1))
        if not payload and status != "not_available":
            status = "failed"
            error = repr(last_error)
    try:
        frame = parse_archive(payload, symbol) if payload else pd.DataFrame(columns=KLINE_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        frame = pd.DataFrame(columns=KLINE_COLUMNS)
        status = "parse_failed"
        error = repr(exc)
    row = {
        "symbol": symbol,
        "month": month_text,
        "request_url": url,
        "source": source,
        "status": status,
        "http_status": http_status,
        "returned_rows": int(len(frame)),
        "raw_bytes": int(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
        "raw_path": str(raw_path.resolve()),
        "error": error,
    }
    return frame, row


def process_symbol(symbol: str, retries: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = []
    manifest = []
    for month in months_for_symbol(symbol):
        frame, row = fetch_archive(symbol, month, retries)
        manifest.append(row)
        if not frame.empty:
            frames.append(frame)
    history_path = HISTORY_CACHE / f"{symbol}_1h.csv"
    previous = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if len(previous):
        frames.append(previous)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined[(combined.open_time >= unix_ms(START)) & (combined.open_time <= unix_ms(END))]
        combined = combined.drop_duplicates(["symbol", "interval", "open_time"], keep="last").sort_values("open_time")
        HISTORY_CACHE.mkdir(parents=True, exist_ok=True)
        combined.to_csv(history_path, index=False)
    else:
        combined = pd.DataFrame()
    return {
        "symbol": symbol,
        "history_rows": int(len(combined)),
        "history_start": pd.to_datetime(combined.open_time.min(), unit="ms", utc=True) if len(combined) else pd.NaT,
        "history_end": pd.to_datetime(combined.open_time.max(), unit="ms", utc=True) if len(combined) else pd.NaT,
        "failed_archives": sum(row["status"] in {"failed", "parse_failed"} for row in manifest),
        "not_available_archives": sum(row["status"] == "not_available" for row in manifest),
    }, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    contracts = eligible_contracts(metadata)
    results: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_symbol, symbol, args.retries): symbol for symbol in contracts.symbol}
        for number, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result, rows = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"symbol": symbol, "history_rows": 0, "history_start": pd.NaT, "history_end": pd.NaT, "failed_archives": 1, "not_available_archives": 0}
                rows = [{"symbol": symbol, "month": "", "request_url": "", "source": "", "status": "symbol_failed", "http_status": None, "returned_rows": 0, "raw_bytes": 0, "sha256": "", "raw_path": "", "error": repr(exc)}]
            results.append(result)
            manifest.extend(rows)
            if number % 25 == 0 or number == len(futures):
                print(f"processed {number}/{len(futures)} symbols", flush=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).sort_values(["symbol", "month"]).to_csv(MANIFEST_PATH, index=False)
    audit_path = MANIFEST_PATH.with_name("history_2025h2_symbol_download_audit.csv")
    pd.DataFrame(results).sort_values("symbol").to_csv(audit_path, index=False)
    print("historical_contracts", len(contracts))
    print("processed_symbols", len(results))
    print("history_rows", sum(row["history_rows"] for row in results))
    print("failed_archives", sum(row["failed_archives"] for row in results))
    print("manifest", MANIFEST_PATH.resolve())
    print("symbol_audit", audit_path.resolve())


if __name__ == "__main__":
    main()
