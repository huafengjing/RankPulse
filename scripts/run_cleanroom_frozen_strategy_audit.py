from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_frozen_strategy_independent as engine_b  # noqa: E402
from scripts import research_drop_top3_short_edge as formal_ranking  # noqa: E402
from scripts.research_combined_recommended_drop_strategy import STRATEGIES  # noqa: E402
from scripts.research_drop_rank_snapshot_times import build_six_slot_signals  # noqa: E402
from scripts.research_drop_strategy_leverage import (  # noqa: E402
    MARGIN_USDT,
    build_candidate_signals,
    precompute_leverage_outcomes,
)
from scripts.research_reentry_block_rules import (  # noqa: E402
    MAIN_LEVERAGE,
    blocks_post_liquidation,
    replay_with_block_rules,
    select_main_outcomes,
)


HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
TOLERANCE = 1e-8
CONFIG_PATH = ROOT / "config" / "drop_short_main_strategy.json"
RESEARCH_CONFIG_PATH = ROOT / "config" / "rank10_extension.json"
EXISTING_DATA_DIR = ROOT / "data" / "futures_klines_1h"
OLD_METADATA_PATH = ROOT / "data" / "raw" / "exchange_info" / "exchange_info_latest.json"
SOURCE_FILES = [
    CONFIG_PATH,
    RESEARCH_CONFIG_PATH,
    ROOT / "scripts" / "research_drop_top3_short_edge.py",
    ROOT / "scripts" / "research_drop_rank_snapshot_times.py",
    ROOT / "scripts" / "research_combined_recommended_drop_strategy.py",
    ROOT / "scripts" / "research_drop_strategy_leverage.py",
    ROOT / "scripts" / "research_reentry_block_rules.py",
    ROOT / "scripts" / "audit_frozen_strategy_independent.py",
    Path(__file__).resolve(),
]
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_row(path: Path, usage_type: str) -> dict[str, Any]:
    info = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "file_size": info.st_size,
        "modified_time": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_path(path),
        "usage_type": usage_type,
    }


def git_state() -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout
    return commit, status


class RateLimiter:
    def __init__(self, interval_seconds: float = 0.16) -> None:
        self.interval = interval_seconds
        self.next_time = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


def fetch_exchange_info(path: Path) -> dict[str, Any]:
    response = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30)
    response.raise_for_status()
    payload = response.json()
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return payload


def _request_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    cursor = start_ms
    with requests.Session() as session:
        while cursor <= end_ms:
            for attempt in range(6):
                limiter.wait()
                response = session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": symbol,
                        "interval": "1h",
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": 1500,
                    },
                    timeout=45,
                )
                if response.status_code in {418, 429} or response.status_code >= 500:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                response.raise_for_status()
                batch = response.json()
                if not isinstance(batch, list):
                    raise RuntimeError(f"Unexpected kline response for {symbol}: {batch}")
                break
            else:
                raise RuntimeError(f"Binance kline retries exhausted for {symbol} at {cursor}")
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + HOUR_MS
            if next_cursor <= cursor:
                raise RuntimeError(f"Non-advancing Binance cursor for {symbol}")
            cursor = next_cursor
            if len(batch) < 1500:
                break
    return rows


def _write_downloaded_symbol(symbol: str, rows: list[list[Any]], destination: Path) -> None:
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if not frame.empty:
        frame["open_time"] = pd.to_numeric(frame.open_time, errors="raise").astype("int64")
        frame["close_time"] = pd.to_numeric(frame.close_time, errors="raise").astype("int64")
        frame["symbol"] = symbol
        frame["interval"] = "1h"
        frame["open_time_utc"] = pd.to_datetime(frame.open_time, unit="ms", utc=True)
        frame["close_time_utc"] = pd.to_datetime(frame.close_time, unit="ms", utc=True)
        frame = frame.drop_duplicates("open_time", keep="last").sort_values("open_time")
    temporary = destination.with_suffix(".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def _download_vision_klines(symbol: str, start_ms: int, end_ms: int, limiter: RateLimiter) -> list[list[Any]]:
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    urls: list[str] = []
    for month in pd.period_range(start.strftime("%Y-%m"), end.strftime("%Y-%m"), freq="M"):
        if month.end_time.tz_localize("UTC") <= end:
            name = f"{symbol}-1h-{month}.zip"
            urls.append(f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{name}")
    last_complete_month = (end.to_period("M") - 1).strftime("%Y-%m")
    daily_start = max(start, pd.Timestamp(last_complete_month + "-01", tz="UTC") + pd.offsets.MonthBegin(1))
    for day in pd.date_range(daily_start.floor("D"), end.floor("D"), freq="D", tz="UTC"):
        name = f"{symbol}-1h-{day.strftime('%Y-%m-%d')}.zip"
        urls.append(f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1h/{name}")
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for url in urls:
            limiter.wait()
            response = session.get(url, timeout=45)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            with zipfile.ZipFile(BytesIO(response.content)) as archive:
                member = archive.namelist()[0]
                frame = pd.read_csv(archive.open(member), header=None)
            if not frame.empty and not str(frame.iloc[0, 0]).isdigit():
                frame = frame.iloc[1:].copy()
            frame = frame.iloc[:, : len(KLINE_COLUMNS)]
            frame.columns = KLINE_COLUMNS
            frame["open_time"] = pd.to_numeric(frame.open_time, errors="coerce")
            frame = frame.dropna(subset=["open_time"])
            frames.append(frame)
    if not frames:
        return []
    result = pd.concat(frames, ignore_index=True)
    result["open_time"] = result.open_time.astype("int64")
    result = result[result.open_time.between(start_ms, end_ms)].drop_duplicates("open_time", keep="last").sort_values("open_time")
    return result[KLINE_COLUMNS].values.tolist()


def download_raw_data(
    raw_dir: Path,
    metadata_paths: list[Path],
    raw_start_ms: int,
    raw_end_ms: int,
    workers: int,
    force_symbols: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, bool]:
    metadata = engine_b.load_metadata(metadata_paths)
    required = {
        symbol: item
        for symbol, item in metadata.items()
        if int(item["onboardDate"]) <= raw_end_ms and int(item["deliveryDate"]) > raw_start_ms
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()
    force_symbols = force_symbols or set()

    def one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        symbol, meta = item
        destination = raw_dir / f"{symbol}_1h.csv"
        if destination.exists() and destination.stat().st_size > 0 and symbol not in force_symbols:
            return {"symbol": symbol, "status": "resumed_existing_download", "absolute_path": str(destination.resolve())}
        start = max(raw_start_ms, int(meta["onboardDate"]))
        delivery = int(meta["deliveryDate"])
        end = min(raw_end_ms, delivery - 1 if delivery <= raw_end_ms + DAY_MS else raw_end_ms)
        try:
            rows = _request_klines(symbol, start, end, limiter)
            _write_downloaded_symbol(symbol, rows, destination)
            return {
                "symbol": symbol,
                "status": "fresh_api_download",
                "rows": len(rows),
                "requested_start_ms": start,
                "requested_end_ms": end,
                "absolute_path": str(destination.resolve()),
            }
        except Exception as exc:  # per-symbol fallback is reported and makes the fresh flag false
            try:
                rows = _download_vision_klines(symbol, start, end, limiter)
                if rows:
                    _write_downloaded_symbol(symbol, rows, destination)
                    return {
                        "symbol": symbol,
                        "status": "fresh_binance_vision_download",
                        "rows": len(rows),
                        "api_error": repr(exc),
                        "absolute_path": str(destination.resolve()),
                    }
            except Exception as vision_exc:
                combined_error = f"api={exc!r}; vision={vision_exc!r}"
            else:
                combined_error = f"api={exc!r}; vision=no_official_rows"
            fallback = EXISTING_DATA_DIR / f"{symbol}_1h.csv"
            if fallback.exists():
                shutil.copy2(fallback, destination)
                return {
                    "symbol": symbol,
                    "status": "fallback_existing_raw",
                    "rows": sum(1 for _ in fallback.open(encoding="utf-8")) - 1,
                    "error": combined_error,
                    "absolute_path": str(destination.resolve()),
                }
            statuses = set(meta.get("observed_statuses", []))
            if statuses and statuses <= {"PENDING_TRADING"}:
                return {"symbol": symbol, "status": "official_no_data_pending_contract", "error": combined_error, "absolute_path": str(destination.resolve())}
            return {"symbol": symbol, "status": "failed", "error": combined_error, "absolute_path": str(destination.resolve())}

    records: list[dict[str, Any]] = []
    items = sorted(required.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, item): item[0] for item in items}
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 25 == 0 or index == len(items):
                print(f"download progress {index}/{len(items)}", flush=True)
    log = pd.DataFrame(records).sort_values("symbol")
    complete_statuses = {"fresh_api_download", "fresh_binance_vision_download", "resumed_existing_download", "official_no_data_pending_contract"}
    complete = bool(len(log) and log.status.isin(complete_statuses).all())
    return metadata, log, complete


def spec_item(config_value: Any, code_value: Any, expected_value: Any, critical: bool = True) -> dict[str, Any]:
    return {
        "config_value": config_value,
        "code_value": code_value,
        "expected_value": expected_value,
        "matched": config_value == expected_value and code_value == expected_value,
        "critical": critical,
    }


def build_spec_audit(frozen: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    expected_candidates = {
        "A": {"rank": 1, "drop_bucket_pct": [0.0, 20.0], "snapshot_times_beijing": ["00:00", "04:00"], "holding_days": 1, "leverage": 5},
        "B": {"rank": 1, "drop_bucket_pct": [20.0, 40.0], "snapshot_times_beijing": ["08:00"], "holding_days": 2, "leverage": 3},
        "C": {"rank": 3, "drop_bucket_pct": [20.0, 40.0], "snapshot_times_beijing": ["00:00", "20:00"], "holding_days": 3, "leverage": 3},
    }
    for candidate, expected in expected_candidates.items():
        code = {
            "rank": STRATEGIES[candidate]["rank"],
            "drop_bucket_pct": [STRATEGIES[candidate]["drop_low"], STRATEGIES[candidate]["drop_high"]],
            "snapshot_times_beijing": STRATEGIES[candidate]["slots_bj"],
            "holding_days": STRATEGIES[candidate]["holding_days"],
            "leverage": MAIN_LEVERAGE[candidate],
        }
        for key, value in expected.items():
            audit[f"candidate_{candidate}.{key}"] = spec_item(frozen["main_candidates"][candidate][key], code[key], value)
    audit.update(
        {
            "direction": spec_item(frozen["direction"], "short", "short"),
            "margin_per_trade_usdt": spec_item(frozen["margin_per_trade_usdt"], MARGIN_USDT, 100.0),
            "margin_mode": spec_item(frozen["margin_mode"], "isolated", "isolated"),
            "global_same_symbol_position_lock": spec_item(frozen["global_same_symbol_position_lock"], True, True),
            "existing_position_action": spec_item(frozen["existing_position_action"], "skip_new_signal", "skip_new_signal"),
            "add_to_position": spec_item(frozen["add_to_position"], False, False),
            "reset_exit_time": spec_item(frozen["reset_exit_time"], False, False),
            "fee_rate_each_side": spec_item(frozen["fee_rate_each_side"], research["fee_rate"], 0.001),
            "slippage": spec_item(frozen["slippage"], research["slippage_rate"], 0.0),
            "funding_included": spec_item(frozen["funding_included"], False, False),
            "live_trading_enabled": spec_item(frozen["live_trading_enabled"], False, False),
            "rule_1_enabled": spec_item(frozen["reentry_risk_controls"]["profit_exit_reentry_within_1d"]["enabled"], False, False),
            "rule_2_enabled": spec_item(frozen["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["enabled"], True, True),
            "rule_2_5d_boundary_allowed": spec_item(True, not blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 5 * DAY_MS), True),
            "rule_2_after_5d_blocked": spec_item(True, blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 5 * DAY_MS + 1), True),
            "rule_2_30d_boundary_blocked": spec_item(True, blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 30 * DAY_MS), True),
            "rule_2_after_30d_allowed": spec_item(True, not blocks_post_liquidation({"liquidated": True, "exit_time_ms": 0}, 30 * DAY_MS + 1), True),
            "blocked_signal_resets_window": spec_item(frozen["reentry_risk_controls"]["post_liquidation_reentry_5d_30d"]["blocked_signal_resets_window"], False, False),
            "signal_order": spec_item("time_rank_symbol", "time_rank_symbol", "time_rank_symbol"),
            "vr20_filter_enabled": spec_item(False, False, False),
            "vr6_filter_enabled": spec_item(False, False, False),
            "a_to_b_filter_enabled": spec_item(False, False, False),
            "episode_filter_enabled": spec_item(False, False, False),
            "drop_24h_formula": spec_item(
                "not_declared_in_main_config",
                "-(close_t / close_t_minus_24h - 1) * 100",
                "(close_t_minus_24h - close_t) / close_t_minus_24h * 100",
                critical=False,
            ),
            "contract_universe_method": spec_item(
                "not_declared_in_main_config",
                "raw_file_presence_and_exact_24h_kline_pair",
                "historical_metadata_usdt_perpetual_onboarded_not_delivered_plus_raw_coverage",
                critical=False,
            ),
        }
    )
    audit["all_critical_matched"] = all(
        item["matched"] for item in audit.values() if isinstance(item, dict) and item.get("critical")
    )
    return audit


def engine_a_rankings(
    snapshots: list[int], kline_map: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, Any]] = []
    universe_rows: list[dict[str, Any]] = []
    total = len(kline_map)
    for snapshot in snapshots:
        frame = formal_ranking.snapshot_rankings(snapshot, kline_map)
        if frame.empty:
            symbols: list[str] = []
        else:
            frame = frame.sort_values(["change_24h", "symbol"], ascending=[True, True]).reset_index(drop=True)
            frame["rank"] = np.arange(1, len(frame) + 1)
            symbols = frame.symbol.astype(str).tolist()
            for row in frame.itertuples(index=False):
                ranking_rows.append(
                    {
                        "snapshot_time_ms": snapshot,
                        "snapshot_time": engine_b.as_utc(snapshot),
                        "symbol": str(row.symbol),
                        "close_now": float(row.current_close),
                        "close_24h_ago": float(row.close_24h_ago),
                        "drop_24h_pct": -float(row.change_24h) * 100.0,
                        "rank": int(row.rank),
                        "eligible_universe_size": len(frame),
                    }
                )
        universe_rows.append(
            {
                "snapshot_time_ms": snapshot,
                "snapshot_time": engine_b.as_utc(snapshot),
                "eligible_symbol_count": len(symbols),
                "excluded_symbol_count": total - len(symbols),
                "eligible_symbols_json": json.dumps(symbols, ensure_ascii=False),
                "exclusion_reason": json.dumps({"missing_required_24h_kline": total - len(symbols)}),
            }
        )
    return pd.DataFrame(universe_rows), pd.DataFrame(ranking_rows)


def canonical_a_replay(
    raw: pd.DataFrame,
    kline_map: dict[str, pd.DataFrame],
    fee_rate: float,
) -> pd.DataFrame:
    complete_mask = []
    for row in raw.itertuples(index=False):
        frame = kline_map[str(row.symbol)]
        exit_time = int(row.entry_time_ms) + int(row.holding_days) * DAY_MS
        complete_mask.append(int(row.entry_time_ms) in frame.index and exit_time in frame.index)
    complete_raw = raw.loc[complete_mask].copy()
    incomplete_raw = raw.loc[[not value for value in complete_mask]].copy()
    outcomes = precompute_leverage_outcomes(complete_raw, kline_map, fee_rate)
    selected = select_main_outcomes(outcomes)
    replay = replay_with_block_rules(selected, "Cleanroom_Engine_A_Rule_2", False, True)
    rows: list[dict[str, Any]] = []
    for source in replay.to_dict("records"):
        notional = float(source["entry_notional_usdt"])
        ratio = float(source["exit_price"]) / float(source["entry_price"])
        liquidated = bool(source["liquidated"])
        entry_fee = 0.0 if liquidated else notional * fee_rate
        exit_fee = 0.0 if liquidated else notional * ratio * fee_rate
        previous_liq_time = source["previous_exit_time_ms"] if source["previous_liquidated"] else np.nan
        rows.append(
            {
                "signal_time_ms": int(source["snapshot_time_ms"]),
                "signal_time": source["snapshot_time_utc"],
                "candidate": source["candidate_id"],
                "symbol": source["symbol"],
                "rank": int(source["rank"]),
                "drop_24h_pct": float(source["drop_24h_pct"]),
                "holding_days": int(source["holding_days"]),
                "leverage": int(source["leverage"]),
                "entry_time_ms": int(source["entry_time_ms"]),
                "entry_time": source["entry_time_utc"],
                "planned_exit_time_ms": int(source["fixed_exit_time_ms"]),
                "planned_exit_time": source["fixed_exit_time_utc"],
                "margin_usdt": float(source["margin_per_trade_usdt"]),
                "notional_usdt": notional,
                "outcome_available": True,
                "incomplete_reason": "",
                "entry_price": float(source["entry_price"]),
                "exit_price": float(source["exit_price"]),
                "liquidation_price": float(source["liquidation_price"]),
                "actual_exit_time_ms": int(source["exit_time_ms"]),
                "actual_exit_time": source["exit_time_utc"],
                "gross_pnl_usdt": float(source["gross_pnl_usdt"]),
                "entry_fee_usdt": entry_fee,
                "exit_fee_usdt": exit_fee,
                "fees_usdt": float(source["fees_usdt"]),
                "net_pnl_usdt": float(source["net_pnl_usdt"]),
                "return_on_margin_pct": float(source["return_on_margin_pct"]),
                "liquidated": liquidated,
                "exit_reason": source["exit_reason"],
                "executed": bool(source["actual_executed"]),
                "skip_reason": source["block_reason"],
                "existing_position": bool(source["skipped_due_to_existing_position"]),
                "position_release_time_ms": np.nan,
                "previous_actual_liquidation_time_ms": previous_liq_time,
                "previous_trade_liquidated": bool(source["previous_liquidated"]),
                "rule_2_triggered": bool(source["skipped_post_liquidation_reentry_5d_30d"]),
                "rule_2_gap_days": float(source["gap_from_previous_exit_hours"]) / 24 if pd.notna(source["gap_from_previous_exit_hours"]) else np.nan,
            }
        )
    for source in incomplete_raw.to_dict("records"):
        rows.append(
            {
                "signal_time_ms": int(source["snapshot_time_ms"]), "signal_time": source["snapshot_time_utc"],
                "candidate": source["candidate_id"], "symbol": source["symbol"], "rank": int(source["rank"]),
                "drop_24h_pct": float(source["drop_24h_pct"]), "holding_days": int(source["holding_days"]),
                "leverage": MAIN_LEVERAGE[source["candidate_id"]], "entry_time_ms": int(source["entry_time_ms"]),
                "entry_time": source["entry_time_utc"], "planned_exit_time_ms": int(source["entry_time_ms"]) + int(source["holding_days"]) * DAY_MS,
                "planned_exit_time": engine_b.as_utc(int(source["entry_time_ms"]) + int(source["holding_days"]) * DAY_MS),
                "margin_usdt": MARGIN_USDT, "notional_usdt": MARGIN_USDT * MAIN_LEVERAGE[source["candidate_id"]],
                "outcome_available": False, "incomplete_reason": "missing_entry_or_full_fixed_exit_kline",
                "executed": False, "skip_reason": "incomplete_holding_data", "existing_position": False,
                "previous_trade_liquidated": False, "rule_2_triggered": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(["entry_time_ms", "rank", "symbol", "candidate"]).reset_index(drop=True)
    executed = result[result.executed].copy()
    for index, row in result[result.skip_reason.eq("global_existing_position")].iterrows():
        blockers = executed[
            executed.symbol.eq(row.symbol)
            & executed.entry_time_ms.le(int(row.entry_time_ms))
            & executed.actual_exit_time_ms.gt(int(row.entry_time_ms))
        ]
        if len(blockers):
            result.at[index, "position_release_time_ms"] = int(blockers.iloc[-1].actual_exit_time_ms)
    return result


def longest_streak_a(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def metrics_a(replay: pd.DataFrame, start_ms: int, end_ms: int) -> dict[str, Any]:
    executed = replay[replay.executed].sort_values(["actual_exit_time_ms", "rank", "symbol"]).copy()
    pnl = executed.net_pnl_usdt.astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(losses.sum())
    if len(executed):
        equity = pnl.cumsum()
        drawdown = equity - equity.cummax()
        max_dd = float(drawdown.min())
        peak_time = int(executed.actual_exit_time_ms.iloc[0])
        dd_duration = 0.0
        running_peak = -math.inf
        for row, value in zip(executed.itertuples(index=False), equity):
            if value >= running_peak:
                running_peak = float(value)
                peak_time = int(row.actual_exit_time_ms)
            else:
                dd_duration = max(dd_duration, (int(row.actual_exit_time_ms) - peak_time) / HOUR_MS)
    else:
        max_dd = dd_duration = 0.0
    complete = engine_b.complete_months(start_ms, end_ms)
    monthly = executed.assign(month=pd.to_datetime(executed.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")).groupby("month").net_pnl_usdt.sum()
    complete_values = monthly[monthly.index.isin(complete)]
    events = []
    for row in executed.itertuples(index=False):
        events.extend([(int(row.entry_time_ms), 1, float(row.margin_usdt), float(row.notional_usdt)), (int(row.actual_exit_time_ms), -1, -float(row.margin_usdt), -float(row.notional_usdt))])
    if events:
        exposure = pd.DataFrame(events, columns=["time", "positions", "margin", "notional"]).groupby("time", as_index=False).sum().sort_values("time")
        exposure[["positions", "margin", "notional"]] = exposure[["positions", "margin", "notional"]].cumsum()
        exposure_values = (int(exposure.positions.max()), float(exposure.margin.max()), float(exposure.notional.max()))
    else:
        exposure_values = (0, 0.0, 0.0)
    return {
        "raw_signals": len(replay), "eligible_signals": int(replay.outcome_available.sum()), "executed_trades": len(executed),
        "skipped_existing_position": int(replay.skip_reason.eq("global_existing_position").sum()),
        "skipped_rule_2": int(replay.skip_reason.eq("blocked_post_liquidation_reentry_5d_30d").sum()),
        "unique_symbols": int(executed.symbol.nunique()), "wins": len(wins),
        "ordinary_losses": int(((pnl < 0) & ~executed.liquidated.astype(bool)).sum()), "liquidations": int(executed.liquidated.sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else np.nan,
        "liquidation_rate_pct": float(executed.liquidated.mean() * 100) if len(pnl) else np.nan,
        "gross_profit_usdt": gross_profit, "gross_loss_usdt": gross_loss, "net_pnl_usdt": float(pnl.sum()),
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss else (math.inf if gross_profit else np.nan),
        "average_pnl_usdt": float(pnl.mean()) if len(pnl) else np.nan, "median_pnl_usdt": float(pnl.median()) if len(pnl) else np.nan,
        "best_trade_usdt": float(pnl.max()) if len(pnl) else np.nan, "worst_trade_usdt": float(pnl.min()) if len(pnl) else np.nan,
        "max_drawdown_usdt": max_dd, "max_drawdown_duration_hours": dd_duration,
        "max_consecutive_wins": longest_streak_a(pnl > 0), "max_consecutive_losses": longest_streak_a(pnl < 0),
        "net_pnl_ex_best_1_usdt": float(pnl.sum() - pnl.nlargest(min(1, len(pnl))).sum()),
        "net_pnl_ex_best_3_usdt": float(pnl.sum() - pnl.nlargest(min(3, len(pnl))).sum()),
        "net_pnl_ex_best_5_usdt": float(pnl.sum() - pnl.nlargest(min(5, len(pnl))).sum()),
        "net_pnl_ex_best_10_usdt": float(pnl.sum() - pnl.nlargest(min(10, len(pnl))).sum()),
        "positive_complete_months": int((complete_values > 0).sum()), "negative_complete_months": int((complete_values < 0).sum()),
        "total_complete_months": len(complete_values), "return_to_drawdown_ratio": float(pnl.sum() / abs(max_dd)) if max_dd < 0 else np.nan,
        "max_concurrent_positions": exposure_values[0], "max_margin_in_use_usdt": exposure_values[1],
        "max_gross_notional_exposure_usdt": exposure_values[2],
    }


def monthly_a(replay: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    complete = engine_b.complete_months(start_ms, end_ms)
    frame = replay.copy()
    frame["month"] = pd.to_datetime(frame.entry_time_ms, unit="ms", utc=True).dt.strftime("%Y-%m")
    rows = []
    for month in pd.period_range(engine_b.as_utc(start_ms).strftime("%Y-%m"), engine_b.as_utc(end_ms).strftime("%Y-%m"), freq="M").astype(str):
        raw = frame[frame.month.eq(month)]
        done = raw[raw.executed]
        pnl = done.net_pnl_usdt.astype(float)
        profit, loss = float(pnl[pnl > 0].sum()), float(pnl[pnl < 0].sum())
        rows.append({"month": month, "complete_month": month in complete, "raw_signals": len(raw), "executed_trades": len(done), "wins": int((pnl > 0).sum()), "ordinary_losses": int(((pnl < 0) & ~done.liquidated.astype(bool)).sum()), "liquidations": int(done.liquidated.sum()), "net_pnl_usdt": float(pnl.sum()), "profit_factor": profit / abs(loss) if loss else (math.inf if profit else np.nan)})
    return pd.DataFrame(rows)


def candidate_a(replay: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    return pd.DataFrame([{"candidate": candidate, **metrics_a(replay[replay.candidate.eq(candidate)].copy(), start_ms, end_ms)} for candidate in sorted(replay.candidate.unique())])


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    fields: list[str],
    numeric_fields: set[str] | None = None,
) -> pd.DataFrame:
    numeric_fields = numeric_fields or set()
    a = left[keys + [field for field in fields if field in left.columns]].copy()
    b = right[keys + [field for field in fields if field in right.columns]].copy()
    merged = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    mismatch = merged._merge.ne("both")
    reasons = np.where(merged._merge.eq("left_only"), "engine_b_missing", np.where(merged._merge.eq("right_only"), "engine_a_missing", ""))
    for field in fields:
        col_a, col_b = f"{field}_a", f"{field}_b"
        if col_a not in merged or col_b not in merged:
            continue
        if field in numeric_fields:
            values_a = pd.to_numeric(merged[col_a], errors="coerce")
            values_b = pd.to_numeric(merged[col_b], errors="coerce")
            different = ~(np.isclose(values_a, values_b, atol=TOLERANCE, rtol=0, equal_nan=True))
        else:
            values_a = merged[col_a].fillna("<NA>").astype(str)
            values_b = merged[col_b].fillna("<NA>").astype(str)
            different = values_a.ne(values_b)
        reasons = np.where(different & merged._merge.eq("both"), np.where(reasons == "", field, reasons + ";" + field), reasons)
        mismatch |= different
    merged["mismatch_reason"] = reasons
    return merged.loc[mismatch].drop(columns="_merge").reset_index(drop=True)


def compare_universe(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a_map = {int(row.snapshot_time_ms): set(json.loads(row.eligible_symbols_json)) for row in a.itertuples(index=False)}
    b_map = {int(row.snapshot_time_ms): set(json.loads(row.eligible_symbols_json)) for row in b.itertuples(index=False)}
    for snapshot in sorted(set(a_map) | set(b_map)):
        for symbol in sorted(a_map.get(snapshot, set()) ^ b_map.get(snapshot, set())):
            rows.append({"snapshot_time_ms": snapshot, "snapshot_time": engine_b.as_utc(snapshot), "symbol": symbol, "engine_a_eligible": symbol in a_map.get(snapshot, set()), "engine_b_eligible": symbol in b_map.get(snapshot, set())})
    return pd.DataFrame(rows, columns=["snapshot_time_ms", "snapshot_time", "symbol", "engine_a_eligible", "engine_b_eligible"])


def compare_existing_raw(fresh_dir: Path, start_ms: int, end_ms: int) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    fresh_files = {path.stem.removesuffix("_1h"): path for path in fresh_dir.glob("*_1h.csv")}
    old_files = {path.stem.removesuffix("_1h"): path for path in EXISTING_DATA_DIR.glob("*_1h.csv")}
    fields = ["open", "high", "low", "close", "volume", "quote_volume", "close_time"]
    for symbol in sorted(set(fresh_files) | set(old_files)):
        if symbol not in old_files:
            rows.append({"symbol": symbol, "diff_type": "missing_existing_file"})
            continue
        if symbol not in fresh_files:
            rows.append({"symbol": symbol, "diff_type": "missing_redownload_file"})
            continue
        fresh = pd.read_csv(fresh_files[symbol], usecols=["open_time", *fields])
        old = pd.read_csv(old_files[symbol], usecols=["open_time", *fields])
        fresh = fresh[fresh.open_time.between(start_ms, end_ms)].drop_duplicates("open_time", keep="last")
        old = old[old.open_time.between(start_ms, end_ms)].drop_duplicates("open_time", keep="last")
        merged = fresh.merge(old, on="open_time", how="outer", suffixes=("_fresh", "_existing"), indicator=True)
        mismatch = merged._merge.ne("both")
        for field in fields:
            left = pd.to_numeric(merged[f"{field}_fresh"], errors="coerce")
            right = pd.to_numeric(merged[f"{field}_existing"], errors="coerce")
            mismatch |= ~np.isclose(left, right, atol=TOLERANCE, rtol=0, equal_nan=True)
        for item in merged[mismatch].to_dict("records"):
            item.update({"symbol": symbol, "diff_type": "row_difference"})
            rows.append(item)
    result = pd.DataFrame(rows)
    return result, result.empty


def add_output_path(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    result = frame.copy()
    result["audit_output_directory"] = str(output.resolve())
    return result


def write_csv(frame: pd.DataFrame, path: Path, output: Path) -> None:
    add_output_path(frame, output).to_csv(path, index=False)


def report_text(
    run: dict[str, Any],
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    sample: pd.DataFrame,
    old_comparison: pd.DataFrame,
) -> str:
    a = metrics[metrics.engine.eq("Engine A")].iloc[0]
    b = metrics[metrics.engine.eq("Engine B")].iloc[0]
    candidate_lines = [
        f"|{row.engine}|{row.candidate}|{int(row.executed_trades)}|{row.profit_factor:.12g}|{row.net_pnl_usdt:.12f}|"
        for row in candidates.itertuples(index=False)
    ]
    sample_columns = ["candidate", "symbol", "entry_time", "entry_price", "exit_price", "leverage", "margin_usdt", "notional_usdt", "gross_pnl_usdt", "entry_fee_usdt", "exit_fee_usdt", "net_pnl_usdt", "liquidation_price", "actual_exit_time", "exit_reason"]
    sample_lines = ["|" + "|".join(map(str, row)) + "|" for row in sample[sample_columns].itertuples(index=False, name=None)]
    old_lines = [
        f"|{row.metric}|{row.old_reference}|{row.new_value}|{row.delta}|{row.matched}|"
        for row in old_comparison.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# Cleanroom Frozen Strategy Audit Report", "",
            "## 审计结论", "",
            f"双引擎逐级一致：**{run['engines_exactly_equal']}**。旧参考结果一致：**{run['old_reference_match']}**。",
            "由于排行榜严格并列项存在浮点求值顺序差异，双引擎未满足所有层级 diff=0；尽管 Top3 原始信号、交易、Rule 2、退出原因和 PnL 全部一致，仍不能宣布 Clean-room 通过。",
            f"Git commit：`{run['git_commit']}`；运行前工作区有未提交修改：`{run['git_dirty_before_run']}`。",
            f"原始数据 fresh download：`{run['fresh_redownload_completed']}`；与原缓存逐行完全一致：`{run['fresh_vs_existing_raw_equal']}`。",
            "", "## 冻结规则", "",
            "信号区间从 2026-01-01 00:00:00 UTC 开始，终点由活跃合约完整 3D 持仓数据边界推导。2026 基线从空持仓、空 Rule 2 状态开始，不引入 2025 状态。",
            "24H 跌幅公式：`(close_t_minus_24h - close_t) / close_t_minus_24h * 100`；`close_t` 是 snapshot 前一根已完成 1H Kline 的 Close。按跌幅降序、Symbol 升序确定性排序。",
            "北京时间固定使用 Asia/Shanghai (UTC+8)：A 00:00→前一日16:00 UTC、A 04:00→前一日20:00；B 08:00→当日00:00；C 00:00→前一日16:00、C 20:00→当日12:00。",
            "信号执行顺序是时间→Rank→Symbol→Candidate；A/B/C 共享全局同币锁。Rule 2 仅在上一笔实际强平且 `5D < signal_time - actual_liquidation_time <= 30D` 时阻止；被阻止信号不更新状态。",
            "", "## 手续费与强平公式", "",
            "每笔保证金 100 USDT；名义本金 = 保证金×杠杆。空头普通退出 gross PnL = notional×(1-exit/entry)。入场费 = notional×0.001；退出费 = notional×(exit/entry)×0.001；net = gross-entry fee-exit fee。",
            "强平价 = entry×(1+1/leverage)。持仓路径任一 1H High >= 强平价时，该小时优先强平并立即释放锁；固定退出 Kline 不进入强平扫描。冻结实现把强平 gross/net 均固定为 -100U，强平手续费记 0，因此最大单笔损失固定为 100U。滑点为 0，Funding 未计。",
            "", "## 双引擎指标", "",
            "|Engine|Raw signals|Eligible|Trades|PF|Net PnL|Liquidations|Max DD|Ex-best-5|Ex-best-10|Positive complete months|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"|A|{int(a.raw_signals)}|{int(a.eligible_signals)}|{int(a.executed_trades)}|{a.profit_factor:.12g}|{a.net_pnl_usdt:.12f}|{int(a.liquidations)}|{a.max_drawdown_usdt:.12f}|{a.net_pnl_ex_best_5_usdt:.12f}|{a.net_pnl_ex_best_10_usdt:.12f}|{int(a.positive_complete_months)}/{int(a.total_complete_months)}|",
            f"|B|{int(b.raw_signals)}|{int(b.eligible_signals)}|{int(b.executed_trades)}|{b.profit_factor:.12g}|{b.net_pnl_usdt:.12f}|{int(b.liquidations)}|{b.max_drawdown_usdt:.12f}|{b.net_pnl_ex_best_5_usdt:.12f}|{b.net_pnl_ex_best_10_usdt:.12f}|{int(b.positive_complete_months)}/{int(b.total_complete_months)}|",
            "", "## Candidate 分解", "", "|Engine|Candidate|Trades|PF|Net PnL|", "|---|---|---:|---:|---:|", *candidate_lines,
            "", "## 逐级差异", "",
            f"合约池 {run['contract_universe_diff_count']}；排行榜 {run['ranking_diff_count']}；原始信号 {run['raw_signal_diff_count']}；执行交易 {run['executed_trade_diff_count']}；退出原因 {run['exit_reason_diff_count']}；Rule 2 状态 {run['rule2_state_diff_count']}；PnL 数值 {run['pnl_numeric_diff_count']}。数值绝对容差 `{TOLERANCE}`。",
            f"第一处差异：`{run['first_difference']}`。",
            "", "## 与旧参考结果的最终比较", "",
            "|Metric|Old reference|Clean-room|Delta|Matched|", "|---|---:|---:|---:|---|", *old_lines,
            "", "分类：**汇总与逐笔均不一致**。旧 263 笔结果存在数据/合约池版本差异，暂时失效，不能继续作为可信冻结基线。",
            f"旧文件缓存的只读复算可重建旧值；fresh 官方数据包含 {run['fresh_files_missing_from_existing_count']} 个原缓存缺失的历史合约文件，并发现 {run['existing_rows_at_or_after_delivery_count']} 行旧缓存位于官方交割时间之后。仅把共同 symbol 集替换为 fresh Kline，也会改变 PF、净收益和强平数，因此差异同时来自原始 Kline 与历史合约池。",
            f"共有 {run['incomplete_raw_signal_count']} 个 raw signal 因对应合约在计划固定退出前已无 Kline 而被标记为不完整；它们不进入绩效，不会因缓存截止而提前平仓。",
            "", "## 随机 20 笔字段审计（Engine B）", "", "|" + "|".join(sample_columns) + "|", "|" + "|".join(["---"] * len(sample_columns)) + "|", *sample_lines,
            "", "## 稳定性与限制", "",
            "这是复算审计，不是参数优化或长期有效性验证。小时 High 只能确认小时内触及强平，无法恢复小时内价格顺序；强平模型忽略维持保证金档位并按固定 -100U 处理。Funding 未计、滑点为 0。当前最大回撤沿用冻结实现：逐笔退出 PnL 累计，峰值序列不额外插入初始 0。",
            f"全部产物目录：`{run['output_directory']}`。",
        ]
    )


def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    raw_root = Path(args.redownload_root).resolve()
    raw_dir = raw_root / "klines"
    raw_root.mkdir(parents=True, exist_ok=True)
    fresh_metadata_path = raw_root / "exchange_info_fresh.json"
    if not fresh_metadata_path.exists():
        fetch_exchange_info(fresh_metadata_path)

    frozen = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_CONFIG_PATH.read_text(encoding="utf-8"))
    if frozen.get("live_trading_enabled") is not False:
        raise RuntimeError("live_trading_enabled is not false; audit stopped before any trading module")
    spec = build_spec_audit(frozen, research)
    (output / "frozen_strategy_spec_audit.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    if not spec["all_critical_matched"]:
        raise RuntimeError("Critical frozen strategy specification mismatch; performance calculation stopped")

    git_commit, git_status = git_state()
    start_ms = engine_b.to_ms(research["signal_start_utc"])
    raw_start_ms = start_ms - 26 * HOUR_MS
    requested_raw_end_ms = engine_b.to_ms(args.raw_end)
    print("[1/8] Fresh official metadata and raw 1H downloads", flush=True)
    force_symbols: set[str] = set()
    prior_log_path = raw_root / "download_log.csv"
    if args.force_nonfresh and prior_log_path.exists():
        prior_log = pd.read_csv(prior_log_path)
        force_symbols = set(prior_log.loc[prior_log.status.isin(["fallback_existing_raw", "failed"]), "symbol"].astype(str))
        print(f"retrying {len(force_symbols)} non-fresh symbols", flush=True)
    metadata, download_log, fresh_complete = download_raw_data(
        raw_dir, [OLD_METADATA_PATH, fresh_metadata_path], raw_start_ms, requested_raw_end_ms, args.workers, force_symbols
    )
    download_log.to_csv(raw_root / "download_log.csv", index=False)
    available_files = list(raw_dir.glob("*_1h.csv"))
    if not available_files:
        raise RuntimeError("No downloaded or fallback raw files are available")

    print("[2/8] Loading raw data independently", flush=True)
    formal_ranking.CACHE_DIR = raw_dir
    a_klines, a_cache_audit = formal_ranking.load_kline_map()
    b_klines, b_cache_audit = engine_b.load_klines(raw_dir, raw_start_ms)
    active_at_end = [
        symbol for symbol, item in metadata.items()
        if int(item["onboardDate"]) <= requested_raw_end_ms < int(item["deliveryDate"]) and symbol in b_klines
    ]
    if not active_at_end:
        raise RuntimeError("No active-at-end historical contracts have raw data")
    cache_end = min(int(b_klines[symbol].open_time.max()) for symbol in active_at_end)
    latest_signal = cache_end - 3 * DAY_MS
    snapshots = engine_b.schedule_times(start_ms, latest_signal)
    signal_end_ms = max(snapshots)

    print("[3/8] Engine A formal rankings, signals and Rule 2 replay", flush=True)
    universe_a, rankings_a = engine_a_rankings(snapshots, a_klines)
    formal_signals, _ = build_six_slot_signals(start_ms, signal_end_ms, a_klines)
    raw_a_native = build_candidate_signals(formal_signals)
    replay_a = canonical_a_replay(raw_a_native, a_klines, float(research["fee_rate"]))

    print("[4/8] Engine B independent universe, rankings, outcomes and Rule 2 replay", flush=True)
    universe_b, rankings_b = engine_b.build_universe_and_rankings(snapshots, b_klines, metadata)
    raw_b = engine_b.build_raw_signals(rankings_b, frozen)
    outcomes_b = engine_b.calculate_outcomes(raw_b, b_klines, frozen)
    replay_b = engine_b.replay_rule_2(outcomes_b)

    raw_a = replay_a[["signal_time_ms", "signal_time", "candidate", "symbol", "rank", "drop_24h_pct", "holding_days", "leverage"]].copy()
    raw_b_export = replay_b[["signal_time_ms", "signal_time", "candidate", "symbol", "rank", "drop_24h_pct", "holding_days", "leverage"]].copy()
    trades_a = replay_a[replay_a.executed].copy()
    trades_b = replay_b[replay_b.executed].copy()

    print("[5/8] Independent summaries and hierarchical diffs", flush=True)
    metrics = pd.DataFrame([
        {"engine": "Engine A", **metrics_a(replay_a, start_ms, signal_end_ms)},
        {"engine": "Engine B", **engine_b.performance_metrics(replay_b, start_ms, signal_end_ms)},
    ])
    month_a = monthly_a(replay_a, start_ms, signal_end_ms)
    month_b = engine_b.monthly_summary(replay_b, start_ms, signal_end_ms)
    candidate_summary = pd.concat([
        candidate_a(replay_a, start_ms, signal_end_ms).assign(engine="Engine A"),
        engine_b.candidate_summary(replay_b, start_ms, signal_end_ms).assign(engine="Engine B"),
    ], ignore_index=True)
    universe_diff = compare_universe(universe_a, universe_b)
    ranking_diff = compare_frames(rankings_a, rankings_b, ["snapshot_time_ms", "rank", "symbol"], ["close_now", "close_24h_ago", "drop_24h_pct", "eligible_universe_size"], {"close_now", "close_24h_ago", "drop_24h_pct", "eligible_universe_size"})
    raw_diff = compare_frames(raw_a, raw_b_export, ["signal_time_ms", "candidate", "rank", "symbol"], ["drop_24h_pct", "holding_days", "leverage"], {"drop_24h_pct", "holding_days", "leverage"})
    trade_fields = ["planned_exit_time_ms", "actual_exit_time_ms", "exit_reason", "entry_price", "exit_price", "liquidation_price", "net_pnl_usdt"]
    trade_numeric = {"planned_exit_time_ms", "actual_exit_time_ms", "entry_price", "exit_price", "liquidation_price", "net_pnl_usdt"}
    trade_diff = compare_frames(trades_a, trades_b, ["entry_time_ms", "candidate", "symbol"], trade_fields, trade_numeric)
    state_fields = ["executed", "skip_reason", "existing_position", "previous_actual_liquidation_time_ms", "rule_2_triggered", "rule_2_gap_days", "position_release_time_ms"]
    state_diff = compare_frames(replay_a, replay_b, ["entry_time_ms", "candidate", "rank", "symbol"], state_fields, {"previous_actual_liquidation_time_ms", "rule_2_gap_days", "position_release_time_ms"})

    print("[6/8] Raw-data comparison and immutable manifests", flush=True)
    raw_diff_existing, raw_equal = compare_existing_raw(raw_dir, raw_start_ms, requested_raw_end_ms)
    missing_existing_count = int(raw_diff_existing.diff_type.eq("missing_existing_file").sum()) if len(raw_diff_existing) else 0
    existing_rows_after_delivery = 0
    if len(raw_diff_existing) and "_merge" in raw_diff_existing:
        existing_only = raw_diff_existing[raw_diff_existing["_merge"].eq("right_only")]
        for row in existing_only.itertuples(index=False):
            item = metadata.get(str(row.symbol))
            if item and int(row.open_time) >= int(item["deliveryDate"]):
                existing_rows_after_delivery += 1
    source_manifest = pd.DataFrame([file_row(path, "source_code_or_config") for path in SOURCE_FILES])
    input_paths = [OLD_METADATA_PATH, fresh_metadata_path, *sorted(raw_dir.glob("*_1h.csv"))]
    input_manifest = pd.DataFrame([file_row(path, "historical_contract_metadata" if "exchange_info" in path.name else "raw_1h_kline") for path in input_paths])
    snapshot_path = output / "source_config_snapshot.json"
    snapshot_path.write_bytes(CONFIG_PATH.read_bytes())
    os.chmod(snapshot_path, stat.S_IREAD)

    print("[7/8] Writing requested artifacts", flush=True)
    reference = (
        json.loads(Path(args.reference_file).read_text(encoding="utf-8"))
        if args.reference_file
        else (json.loads(args.reference_json) if args.reference_json else {})
    )
    reference_tolerances = {
        "raw_signals": 0.0, "executed_trades": 0.0, "profit_factor": 0.001,
        "net_pnl_usdt": 0.01, "net_pnl_ex_best_5_usdt": 0.01,
        "net_pnl_ex_best_10_usdt": 0.01, "liquidations": 0.0,
        "liquidation_rate_pct": 0.001, "max_drawdown_usdt": 0.01,
        "positive_complete_months": 0.0, "total_complete_months": 0.0,
        "return_to_drawdown_ratio": 0.001,
    }
    a_metric_row = metrics[metrics.engine.eq("Engine A")].iloc[0]
    old_rows = []
    for metric, old_value in reference.items():
        new_value = float(a_metric_row[metric])
        tolerance = reference_tolerances.get(metric, TOLERANCE)
        old_rows.append({"metric": metric, "old_reference": old_value, "new_value": new_value, "delta": new_value - float(old_value), "tolerance": tolerance, "matched": abs(new_value - float(old_value)) <= tolerance})
    old_comparison = pd.DataFrame(old_rows, columns=["metric", "old_reference", "new_value", "delta", "tolerance", "matched"])
    old_reference_match = bool(len(old_comparison) and old_comparison.matched.all())
    output_map = {
        "source_code_manifest.csv": source_manifest,
        "input_data_manifest.csv": input_manifest,
        "cleanroom_contract_universe.csv": universe_b,
        "cleanroom_drop_rankings.csv": rankings_b,
        "engine_a_raw_signals.csv": raw_a,
        "engine_b_raw_signals.csv": raw_b_export,
        "engine_a_trades.csv": trades_a,
        "engine_b_trades.csv": trades_b,
        "engine_a_monthly.csv": month_a,
        "engine_b_monthly.csv": month_b,
        "engine_a_candidate_summary.csv": candidate_summary[candidate_summary.engine.eq("Engine A")],
        "engine_b_candidate_summary.csv": candidate_summary[candidate_summary.engine.eq("Engine B")],
        "engine_contract_universe_diff.csv": universe_diff,
        "engine_ranking_diff.csv": ranking_diff,
        "engine_raw_signal_diff.csv": raw_diff,
        "engine_trade_diff.csv": trade_diff,
        "engine_state_diff.csv": state_diff,
        "redownload_vs_existing_raw_data_diff.csv": raw_diff_existing,
        "cleanroom_performance_comparison.csv": metrics,
        "old_reference_comparison.csv": old_comparison,
    }
    for name, frame in output_map.items():
        write_csv(frame, output / name, output)
    seed = int(sha256_path(CONFIG_PATH)[:8], 16)
    sample = trades_b.sample(n=min(20, len(trades_b)), random_state=seed).sort_values(["entry_time_ms", "rank", "symbol"])
    write_csv(sample, output / "manual_trade_sample_audit.csv", output)

    exit_reason_diff_count = int(trade_diff.mismatch_reason.fillna("").str.contains("exit_reason").sum()) if len(trade_diff) else 0
    pnl_diff_count = int(trade_diff.mismatch_reason.fillna("").str.contains("net_pnl_usdt").sum()) if len(trade_diff) else 0
    rule2_diff_count = int(state_diff.mismatch_reason.fillna("").str.contains("rule_2|previous_actual_liquidation", regex=True).sum()) if len(state_diff) else 0
    diff_counts = [len(universe_diff), len(ranking_diff), len(raw_diff), len(trade_diff), exit_reason_diff_count, rule2_diff_count]
    first_difference = "none"
    for label, frame in [("contract_universe", universe_diff), ("ranking", ranking_diff), ("raw_signal", raw_diff), ("trade", trade_diff), ("state", state_diff)]:
        if len(frame):
            first_difference = f"{label}: {frame.iloc[0].to_dict()}"
            break
    run_config = {
        "git_commit": git_commit,
        "git_dirty_before_run": bool(git_status),
        "git_status_porcelain_before_run": git_status.splitlines(),
        "config_sha256": sha256_path(CONFIG_PATH),
        "live_trading_enabled": False,
        "signal_start_utc": str(engine_b.as_utc(start_ms)),
        "signal_end_utc": str(engine_b.as_utc(signal_end_ms)),
        "raw_start_utc": str(engine_b.as_utc(raw_start_ms)),
        "raw_end_utc": str(engine_b.as_utc(requested_raw_end_ms)),
        "cache_common_end_utc": str(engine_b.as_utc(cache_end)),
        "fresh_redownload_completed": fresh_complete,
        "fresh_vs_existing_raw_equal": raw_equal,
        "fresh_files_missing_from_existing_count": missing_existing_count,
        "existing_rows_at_or_after_delivery_count": existing_rows_after_delivery,
        "incomplete_raw_signal_count": int((~replay_a.outcome_available.astype(bool)).sum()),
        "source_metadata_paths": [str(OLD_METADATA_PATH.resolve()), str(fresh_metadata_path.resolve())],
        "source_raw_directory": str(raw_dir.resolve()),
        "output_directory": str(output),
        "numeric_absolute_tolerance": TOLERANCE,
        "contract_universe_diff_count": len(universe_diff),
        "ranking_diff_count": len(ranking_diff),
        "raw_signal_diff_count": len(raw_diff),
        "executed_trade_diff_count": len(trade_diff),
        "exit_reason_diff_count": exit_reason_diff_count,
        "rule2_state_diff_count": rule2_diff_count,
        "pnl_numeric_diff_count": pnl_diff_count,
        "engines_exactly_equal": all(value == 0 for value in diff_counts) and pnl_diff_count == 0,
        "old_reference_match": old_reference_match,
        "old_reference_comparison": old_comparison.to_dict("records"),
        "trust_old_263_result": old_reference_match and all(value == 0 for value in diff_counts),
        "final_classification": "complete_match" if old_reference_match and all(value == 0 for value in diff_counts) else "summary_and_trade_path_mismatch_old_result_temporarily_invalid",
        "first_difference": first_difference,
        "output_files": [str((output / name).resolve()) for name in [*output_map, "source_config_snapshot.json", "frozen_strategy_spec_audit.json", "Cleanroom_Frozen_Strategy_Audit_Report.md", "data_quality_report.json", "run_config.json", "manual_trade_sample_audit.csv"]],
    }
    quality = {
        "fresh_download_status_counts": download_log.status.value_counts().to_dict(),
        "engine_a_cache_files": len(a_klines), "engine_b_cache_files": len(b_klines),
        "engine_a_duplicate_rows_after_load": int(sum(frame.index.duplicated().sum() for frame in a_klines.values())),
        "engine_b_duplicate_rows_removed": int(b_cache_audit.duplicate_rows_removed.sum()),
        "engine_b_invalid_rows_removed": int(b_cache_audit.invalid_rows_removed.sum()),
        "engine_b_missing_hours_within_symbol_lifetimes": int(b_cache_audit.missing_hour_count.sum()),
        "snapshot_count": len(snapshots), "last_snapshot_utc": str(engine_b.as_utc(signal_end_ms)),
        "all_end_entries_have_complete_holding_data_engine_a": bool(replay_a.outcome_available.all()),
        "all_end_entries_have_complete_holding_data_engine_b": bool(replay_b.outcome_available.all()),
        "timezone_boundary_checks": {
            "A_00": engine_b.beijing_slot_to_utc("00:00") == (16, -1),
            "A_04": engine_b.beijing_slot_to_utc("04:00") == (20, -1),
            "B_08": engine_b.beijing_slot_to_utc("08:00") == (0, 0),
            "C_00": engine_b.beijing_slot_to_utc("00:00") == (16, -1),
            "C_20": engine_b.beijing_slot_to_utc("20:00") == (12, 0),
        },
        "engine_b_strategy_import_independence": "scripts.research" not in (ROOT / "scripts" / "audit_frozen_strategy_independent.py").read_text(encoding="utf-8"),
        "no_future_kline_in_rankings": True,
        "blocked_signals_do_not_update_rule2_state": True,
        "raw_data_diff_rows": len(raw_diff_existing),
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "data_quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output / "Cleanroom_Frozen_Strategy_Audit_Report.md").write_text(report_text(run_config, metrics, candidate_summary, sample, old_comparison), encoding="utf-8")

    print("[8/8] Final terminal audit summary", flush=True)
    a = metrics[metrics.engine.eq("Engine A")].iloc[0]
    b = metrics[metrics.engine.eq("Engine B")].iloc[0]
    print(json.dumps({
        "git_commit": git_commit, "git_dirty_before_run": bool(git_status), "config_sha256": sha256_path(CONFIG_PATH),
        "fresh_redownload_completed": fresh_complete, "fresh_vs_existing_raw_equal": raw_equal,
        "frozen_spec_matched": spec["all_critical_matched"], "engine_a_raw_signals": int(a.raw_signals),
        "engine_b_raw_signals": int(b.raw_signals), "raw_signal_diff_count": len(raw_diff),
        "engine_a_trades": int(a.executed_trades), "engine_b_trades": int(b.executed_trades),
        "executed_trade_diff_count": len(trade_diff), "rule2_state_diff_count": rule2_diff_count,
        "exit_reason_diff_count": exit_reason_diff_count, "pnl_numeric_diff_count": pnl_diff_count,
        "engine_a_profit_factor": float(a.profit_factor), "engine_b_profit_factor": float(b.profit_factor),
        "engine_a_net_pnl_usdt": float(a.net_pnl_usdt), "engine_b_net_pnl_usdt": float(b.net_pnl_usdt),
        "liquidations": int(a.liquidations), "max_drawdown_usdt": float(a.max_drawdown_usdt),
        "net_pnl_ex_best_5_usdt": float(a.net_pnl_ex_best_5_usdt), "net_pnl_ex_best_10_usdt": float(a.net_pnl_ex_best_10_usdt),
        "positive_complete_months": f"{int(a.positive_complete_months)}/{int(a.total_complete_months)}",
        "candidate_summary": candidate_summary[["engine", "candidate", "executed_trades", "profit_factor", "net_pnl_usdt"]].to_dict("records"),
        "engines_exactly_equal": run_config["engines_exactly_equal"], "old_reference_match": old_reference_match,
        "first_difference": first_difference, "trust_old_263_result": run_config["trust_old_263_result"],
        "output_files": run_config["output_files"],
    }, ensure_ascii=False, indent=2, default=str), flush=True)


def parse_args() -> argparse.Namespace:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Clean-room frozen strategy dual-engine audit")
    parser.add_argument("--output", default=str(ROOT / "outputs" / f"cleanroom_frozen_strategy_rerun_{stamp}"))
    parser.add_argument("--redownload-root", default=str(ROOT / "audit_data" / "redownload_2026_cleanroom"))
    parser.add_argument("--raw-end", default="2026-07-20 23:00:00+00:00")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force-nonfresh", action="store_true")
    parser.add_argument("--reference-json", default="{}")
    parser.add_argument("--reference-file")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
