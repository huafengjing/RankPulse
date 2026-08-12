from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.top3_regime_engine import (
    MODEL_NAME,
    RegimeEngineConfig,
    RegimeEvaluation,
    RegimeOpportunity,
    build_regime_timeline,
    evaluation_to_flat_dict,
    replay_context,
    write_context_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPPORTUNITIES_CSV = ROOT / "output" / "bucket_b_rank3_regime_optimization" / "bucket_b_eligible_opportunities.csv"
DEFAULT_TIMELINE_CSV = ROOT / "output" / "rank3_fast_recovery_vs_monthly_reset" / "d15_risk_off_timeline.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "regime_context_engineering"
DEFAULT_SIGNAL_START_MS = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)


@dataclass(frozen=True)
class GenerationResult:
    evaluation: RegimeEvaluation
    context_path: Path | None
    generated_at_ms: int
    opportunity_count: int
    evaluation_count: int


def generate_context(
    evaluation_time_ms: int,
    output_path: str | Path | None = None,
    opportunities_csv: str | Path = DEFAULT_OPPORTUNITIES_CSV,
    timeline_csv: str | Path = DEFAULT_TIMELINE_CSV,
    write_file: bool = True,
) -> GenerationResult:
    opportunities = load_opportunities(opportunities_csv)
    evaluation_times = load_evaluation_times(timeline_csv)
    if evaluation_time_ms not in evaluation_times:
        evaluation_times.append(evaluation_time_ms)
    config = RegimeEngineConfig(signal_start_ms=DEFAULT_SIGNAL_START_MS)
    evaluation = replay_context(evaluation_time_ms, evaluation_times, opportunities, config)
    generated_at_ms = int(time.time() * 1000)
    path = Path(output_path) if output_path else None
    if write_file and path is not None:
        write_context_atomic(path, evaluation.to_context_json(generated_at_ms))
    return GenerationResult(
        evaluation=evaluation,
        context_path=path,
        generated_at_ms=generated_at_ms,
        opportunity_count=len(opportunities),
        evaluation_count=len([t for t in evaluation_times if t <= evaluation_time_ms]),
    )


def full_timeline(
    opportunities_csv: str | Path = DEFAULT_OPPORTUNITIES_CSV,
    timeline_csv: str | Path = DEFAULT_TIMELINE_CSV,
) -> list[RegimeEvaluation]:
    opportunities = load_opportunities(opportunities_csv)
    evaluation_times = load_evaluation_times(timeline_csv)
    return build_regime_timeline(evaluation_times, opportunities, RegimeEngineConfig(signal_start_ms=DEFAULT_SIGNAL_START_MS))


def write_timeline_csv(rows: list[RegimeEvaluation], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [evaluation_to_flat_dict(row) for row in rows]
    if not data:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def load_opportunities(path: str | Path) -> list[RegimeOpportunity]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Regime opportunity CSV not found: {source}")
    frame = pd.read_csv(source)
    required = {
        "signal_time",
        "symbol",
        "rank",
        "gain_24h",
        "volume_24h_ratio_7d",
        "entry_price",
        "return_24h",
        "return_48h",
        "path_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Regime opportunity CSV missing columns: {sorted(missing)}")
    rows: list[RegimeOpportunity] = []
    scoped = frame[
        frame["rank"].astype(int).eq(3)
        & frame["gain_24h"].astype(float).ge(0.20)
        & frame["gain_24h"].astype(float).lt(0.40)
        & frame["volume_24h_ratio_7d"].astype(float).ge(1.5)
        & frame["volume_24h_ratio_7d"].astype(float).lt(5.0)
        & frame["symbol"].astype(str).ne("RAVEUSDT")
    ].copy()
    for row in scoped.sort_values(["signal_time", "symbol"]).itertuples(index=False):
        signal_time = int(getattr(row, "signal_time"))
        symbol = str(getattr(row, "symbol"))
        rank = int(getattr(row, "rank"))
        rows.append(
            RegimeOpportunity(
                opportunity_id=f"{signal_time}:{symbol}:R{rank}",
                signal_time_ms=signal_time,
                symbol=symbol,
                rank=rank,
                gain_24h=float(getattr(row, "gain_24h")),
                volume_24h_ratio_7d=float(getattr(row, "volume_24h_ratio_7d")),
                entry_price=_finite_float(getattr(row, "entry_price")),
                return_24h=_finite_float(getattr(row, "return_24h")),
                return_48h=_finite_float(getattr(row, "return_48h")),
                path_status=str(getattr(row, "path_status", "ok")),
            )
        )
    return rows


def load_evaluation_times(path: str | Path) -> list[int]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Regime timeline CSV not found: {source}")
    frame = pd.read_csv(source, usecols=["signal_time"])
    return sorted(int(v) for v in frame["signal_time"].dropna().astype("int64").unique())


def compare_with_reference(
    generated: list[RegimeEvaluation],
    reference_timeline_csv: str | Path = DEFAULT_TIMELINE_CSV,
    tolerance: float = 1e-10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = pd.read_csv(reference_timeline_csv).set_index("signal_time").to_dict("index")
    mismatches: list[dict[str, Any]] = []
    max_error = 0.0
    for row in generated:
        ref = reference.get(row.signal_time_ms)
        if ref is None:
            mismatches.append({"signal_time": row.signal_time_ms, "field": "reference", "generated": "present", "reference": "missing"})
            continue
        checks = {
            "state": (row.state, str(ref.get("regime_state", ""))),
            "last15_decay48": (row.last15_decay48, _none_if_nan(ref.get("value"))),
            "historical_q10": (row.historical_q10, _none_if_nan(ref.get("hist_yellow_threshold"))),
            "historical_q5": (row.historical_q5, _none_if_nan(ref.get("hist_red_threshold"))),
        }
        for field, (actual, expected) in checks.items():
            if isinstance(actual, str) or isinstance(expected, str):
                if actual != expected:
                    mismatches.append({"signal_time": row.signal_time_ms, "field": field, "generated": actual, "reference": expected})
                continue
            err = _abs_error(actual, expected)
            max_error = max(max_error, err)
            if err > tolerance:
                mismatches.append({"signal_time": row.signal_time_ms, "field": field, "generated": actual, "reference": expected, "abs_error": err})
    summary = {
        "model": MODEL_NAME,
        "total_evaluations": len(generated),
        "mismatch_count": len(mismatches),
        "numeric_max_abs_error": max_error,
    }
    return mismatches, summary


def rebuild_opportunity_csv_from_cache(output_path: str | Path = DEFAULT_OPPORTUNITIES_CSV) -> Path:
    from scripts.backfill_old_half_and_run_main_strategy import DAY_MS, OUT, load_kline_map
    from scripts.backtest_futures_top2_fixed_time import generate_signals, latest_signal_end_dt
    from scripts.bucket_b_rank3_regime_optimization import EXCLUDE_SYMBOLS, opportunity_sets
    from scripts.run_current_main_strategy_2026_jan_jun import SIGNAL_START_MS, cache_common_end_ms, cached_symbols

    symbols = [symbol for symbol in cached_symbols() if symbol not in EXCLUDE_SYMBOLS]
    common_end = cache_common_end_ms(symbols)
    signal_end = min(int(latest_signal_end_dt().timestamp() * 1000), common_end)
    kline_map = load_kline_map(symbols, SIGNAL_START_MS - 10 * DAY_MS, common_end)
    raw = generate_signals(SIGNAL_START_MS, signal_end, kline_map)
    sets = opportunity_sets(raw, kline_map)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sets["B_R3"].to_csv(target, index=False, encoding="utf-8-sig")
    return target


def _finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        return math.nan
    return out


def _none_if_nan(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return None if not math.isfinite(out) else out


def _abs_error(actual: float | None, expected: float | None) -> float:
    if actual is None and expected is None:
        return 0.0
    if actual is None or expected is None:
        return math.inf
    return abs(float(actual) - float(expected))
