from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

import numpy as np

from src.research.rankpulse_regime_constants import (
    MIN_PRIOR_VALUES,
    MODEL_NAME,
    MODEL_VERSION,
    RECOVERY_MODEL_NAME,
    RECOVERY_WINDOW,
    RISK_MODEL_NAME,
    RISK_WINDOW,
    WARMUP_DAYS,
)
from src.research.rankpulse_strategy_rules import Top3RegimeContext, Top3Signal, leverage_for_signal


HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS


@dataclass(frozen=True)
class RegimeOpportunity:
    opportunity_id: str
    signal_time_ms: int
    symbol: str
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float
    entry_price: float
    return_24h: float
    return_48h: float
    path_status: str = "ok"

    @property
    def mature_24h_ms(self) -> int:
        return self.signal_time_ms + 24 * HOUR_MS

    @property
    def mature_48h_ms(self) -> int:
        return self.signal_time_ms + 48 * HOUR_MS

    @property
    def decay48(self) -> float:
        return self.return_48h - self.return_24h


@dataclass(frozen=True)
class RegimeEvaluation:
    signal_time_ms: int
    signal_time_utc: str
    month: str
    state: str
    last15_decay48: float | None
    historical_q10: float | None
    historical_q5: float | None
    hist_count_48h: int
    warmup_complete: bool
    last3_avg_return24: float | None
    recovery_signal: bool
    recovery_streak: int
    r2_leverage: int
    r3_leverage: int
    eligible_opportunity_ids: list[str] = field(default_factory=list)
    matured_24h_ids: list[str] = field(default_factory=list)
    matured_48h_ids: list[str] = field(default_factory=list)
    last_24h_opportunity_time_ms: int | None = None
    last_48h_opportunity_time_ms: int | None = None
    max_source_timestamp_used: int | None = None
    status: str = "READY"
    error: str = ""

    def to_context_json(self, generated_at_ms: int, data_cutoff_ms: int | None = None) -> dict[str, Any]:
        return {
            "signal_time_ms": self.signal_time_ms,
            "generated_at_ms": generated_at_ms,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "state": self.state,
            "last15_decay48": self.last15_decay48,
            "historical_q10": self.historical_q10,
            "historical_q5": self.historical_q5,
            "last3_avg_return24": self.last3_avg_return24,
            "recovery_signal": self.recovery_signal,
            "recovery_streak": self.recovery_streak,
            "r2_leverage": self.r2_leverage,
            "r3_leverage": self.r3_leverage,
            "r3_matured_24h_count": len(self.matured_24h_ids),
            "r3_matured_48h_count": len(self.matured_48h_ids),
            "last_24h_opportunity_time_ms": self.last_24h_opportunity_time_ms,
            "last_48h_opportunity_time_ms": self.last_48h_opportunity_time_ms,
            "data_cutoff_ms": data_cutoff_ms if data_cutoff_ms is not None else self.max_source_timestamp_used,
            "max_source_timestamp_used": self.max_source_timestamp_used,
            "eligible_opportunity_ids": self.eligible_opportunity_ids,
            "last_matured_24h_ids": self.matured_24h_ids[-RECOVERY_WINDOW:],
            "last_matured_48h_ids": self.matured_48h_ids[-RISK_WINDOW:],
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class RegimeEngineConfig:
    signal_start_ms: int
    warmup_days: int = WARMUP_DAYS
    min_prior_values: int = MIN_PRIOR_VALUES
    risk_window: int = RISK_WINDOW
    recovery_window: int = RECOVERY_WINDOW
    yellow_q: float = 0.10
    red_q: float = 0.05


def is_bucket_b_rank3_eligible(signal: Top3Signal) -> bool:
    return (
        signal.symbol != "RAVEUSDT"
        and signal.rank == 3
        and 0.20 <= signal.gain_24h < 0.40
        and signal.volume_24h_ratio_7d is not None
        and 1.5 <= signal.volume_24h_ratio_7d < 5.0
    )


def build_regime_timeline(
    evaluation_times_ms: Iterable[int],
    opportunities: Iterable[RegimeOpportunity],
    config: RegimeEngineConfig,
) -> list[RegimeEvaluation]:
    ordered_times = sorted(int(t) for t in evaluation_times_ms)
    opps = sorted(
        [opp for opp in opportunities if opp.path_status == "ok"],
        key=lambda item: (item.signal_time_ms, item.symbol, item.opportunity_id),
    )
    values_seen: list[float] = []
    recovery_streak = 0
    previous_month: str | None = None
    rows: list[RegimeEvaluation] = []

    for timestamp_ms in ordered_times:
        month = _month_utc(timestamp_ms)
        if month != previous_month:
            recovery_streak = 0
            previous_month = month

        matured_48h = [opp for opp in opps if opp.mature_48h_ms <= timestamp_ms]
        matured_24h = [opp for opp in opps if opp.mature_24h_ms <= timestamp_ms]
        warm = bool(
            timestamp_ms >= config.signal_start_ms + config.warmup_days * DAY_MS
            or len(matured_48h) >= config.risk_window
        )

        last15_decay48 = None
        historical_q10 = None
        historical_q5 = None
        state = "GREEN"
        if warm and len(matured_48h) >= config.risk_window:
            last15 = matured_48h[-config.risk_window :]
            value = _mean([opp.decay48 for opp in last15])
            last15_decay48 = value
            if len(values_seen) >= config.min_prior_values:
                historical_q10 = _quantile(values_seen, config.yellow_q)
                historical_q5 = _quantile(values_seen, config.red_q)
                state = _risk_state(value, historical_q10, historical_q5)

        last3_avg_return24 = None
        recovery_signal = False
        if len(matured_24h) >= config.recovery_window:
            last3 = matured_24h[-config.recovery_window :]
            last3_avg_return24 = _mean([opp.return_24h for opp in last3])
            recovery_signal = bool(last3_avg_return24 > 0)

        if state == "GREEN":
            recovery_streak = 0
        else:
            recovery_streak = recovery_streak + 1 if recovery_signal else 0

        context = Top3RegimeContext(
            state=state,
            recovery_signal=recovery_signal,
            recovery_streak=recovery_streak,
            model=MODEL_NAME,
        )
        r2_signal = Top3Signal("REGIME_R2USDT", 2, 0.25, 2.0, "00:00")
        r3_signal = Top3Signal("REGIME_R3USDT", 3, 0.25, 2.0, "00:00")
        max_source = _max_source_timestamp(timestamp_ms, matured_24h, matured_48h)
        rows.append(
            RegimeEvaluation(
                signal_time_ms=timestamp_ms,
                signal_time_utc=_utc_string(timestamp_ms),
                month=month,
                state=state,
                last15_decay48=last15_decay48,
                historical_q10=historical_q10,
                historical_q5=historical_q5,
                hist_count_48h=len(matured_48h),
                warmup_complete=warm,
                last3_avg_return24=last3_avg_return24,
                recovery_signal=recovery_signal,
                recovery_streak=recovery_streak,
                r2_leverage=int(leverage_for_signal(r2_signal, context) or 3),
                r3_leverage=int(leverage_for_signal(r3_signal, context) or 5),
                eligible_opportunity_ids=[opp.opportunity_id for opp in opps if opp.signal_time_ms <= timestamp_ms],
                matured_24h_ids=[opp.opportunity_id for opp in matured_24h],
                matured_48h_ids=[opp.opportunity_id for opp in matured_48h],
                last_24h_opportunity_time_ms=matured_24h[-1].signal_time_ms if matured_24h else None,
                last_48h_opportunity_time_ms=matured_48h[-1].signal_time_ms if matured_48h else None,
                max_source_timestamp_used=max_source,
            )
        )
        if warm and last15_decay48 is not None and math.isfinite(last15_decay48):
            values_seen.append(last15_decay48)
    return rows


def replay_context(
    evaluation_time_ms: int,
    all_evaluation_times_ms: Iterable[int],
    opportunities: Iterable[RegimeOpportunity],
    config: RegimeEngineConfig,
) -> RegimeEvaluation:
    scoped_times = [t for t in all_evaluation_times_ms if int(t) <= evaluation_time_ms]
    if evaluation_time_ms not in set(int(t) for t in scoped_times):
        scoped_times.append(evaluation_time_ms)
    timeline = build_regime_timeline(scoped_times, opportunities, config)
    for row in reversed(timeline):
        if row.signal_time_ms == evaluation_time_ms:
            if row.max_source_timestamp_used is not None and row.max_source_timestamp_used > evaluation_time_ms:
                raise RuntimeError("Regime replay used future data.")
            return row
    raise RuntimeError("Regime replay did not produce the requested evaluation.")


def write_context_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(target.parent), suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, target)


def evaluation_to_flat_dict(row: RegimeEvaluation) -> dict[str, Any]:
    data = asdict(row)
    data["model"] = MODEL_NAME
    data["model_version"] = MODEL_VERSION
    data["last_matured_24h_ids"] = "|".join(row.matured_24h_ids[-RECOVERY_WINDOW:])
    data["last_matured_48h_ids"] = "|".join(row.matured_48h_ids[-RISK_WINDOW:])
    data["eligible_opportunity_ids"] = "|".join(row.eligible_opportunity_ids)
    data["matured_24h_ids"] = "|".join(row.matured_24h_ids)
    data["matured_48h_ids"] = "|".join(row.matured_48h_ids)
    return data


def _mean(values: Iterable[float]) -> float:
    good = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(good)) if good else math.nan


def _quantile(values: Iterable[float], q: float) -> float:
    good = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.quantile(good, q)) if good else math.nan


def _risk_state(value: float, q10: float, q5: float) -> str:
    if not all(math.isfinite(x) for x in [value, q10, q5]):
        return "GREEN"
    if value <= q5:
        return "RED"
    if value <= q10:
        return "YELLOW"
    return "GREEN"


def _max_source_timestamp(
    evaluation_time_ms: int,
    matured_24h: list[RegimeOpportunity],
    matured_48h: list[RegimeOpportunity],
) -> int | None:
    values = [opp.mature_24h_ms for opp in matured_24h] + [opp.mature_48h_ms for opp in matured_48h]
    if not values:
        return None
    return min(max(values), evaluation_time_ms)


def _utc_string(timestamp_ms: int) -> str:
    return str(np.datetime64(timestamp_ms, "ms")).replace("T", " ")


def _month_utc(timestamp_ms: int) -> str:
    return _utc_string(timestamp_ms)[:7]
