from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from src.research.rankpulse_regime_context_provider import (
    AutoGeneratingRegimeContextProvider,
    JsonRegimeContextProvider,
    regime_context_provider_from_path,
)
from src.research.rankpulse_regime_engine import (
    RegimeEngineConfig,
    RegimeOpportunity,
    build_regime_timeline,
    write_context_atomic,
)
from src.research.rankpulse_regime_generator import compare_with_reference, full_timeline, generate_context


JULY_6_UTC_MS = 1783296000000
TEST_TMP = Path("output") / "test_tmp" / "top3_regime_engine"


def test_full_replay_matches_frozen_d15_risk_off_timeline() -> None:
    rows = full_timeline()

    mismatches, summary = compare_with_reference(rows)

    assert summary["total_evaluations"] == 290
    assert summary["mismatch_count"] == 0
    assert summary["numeric_max_abs_error"] < 1e-10
    assert mismatches == []


def test_july_6_red_first_recovery_context_matches_fr3_yr1_rule() -> None:
    result = generate_context(JULY_6_UTC_MS, write_file=False)
    context = result.evaluation

    assert context.state == "RED"
    assert context.recovery_signal is True
    assert context.recovery_streak == 1
    assert context.r2_leverage == 3
    assert context.r3_leverage == 3
    assert context.last15_decay48 == pytest.approx(-5.67223719647186)
    assert context.historical_q10 == pytest.approx(-3.865080665174891)
    assert context.historical_q5 == pytest.approx(-4.05981067856931)
    assert context.last3_avg_return24 > 0
    assert context.max_source_timestamp_used is not None
    assert context.max_source_timestamp_used <= JULY_6_UTC_MS


def test_regime_engine_duplicate_run_is_idempotent() -> None:
    first = generate_context(JULY_6_UTC_MS, write_file=False).evaluation
    second = generate_context(JULY_6_UTC_MS, write_file=False).evaluation

    assert asdict(first) == asdict(second)


def test_regime_engine_restart_rebuild_keeps_recovery_streak() -> None:
    first = generate_context(JULY_6_UTC_MS, write_file=False).evaluation
    rebuilt = generate_context(JULY_6_UTC_MS, write_file=False).evaluation
    next_context = generate_context(1783468800000, write_file=False).evaluation

    assert first.recovery_streak == 1
    assert rebuilt.recovery_streak == 1
    assert next_context.signal_time_utc == "2026-07-08 00:00:00.000"
    assert next_context.recovery_streak >= 1


def test_regime_engine_recovers_red_rank3_from_1x_to_3x_to_5x() -> None:
    day = 24 * 60 * 60 * 1000
    config = RegimeEngineConfig(signal_start_ms=0, min_prior_values=1)
    opportunities: list[RegimeOpportunity] = []
    for index in range(40):
        signal_time = index * day
        if index < 25:
            return_24h = -1.0
            return_48h = 9.0
        elif index < 28:
            return_24h = -1.0
            return_48h = -21.0
        else:
            return_24h = 10.0
            return_48h = -10.0
        opportunities.append(
            RegimeOpportunity(
                opportunity_id=f"opp-{index}",
                signal_time_ms=signal_time,
                symbol=f"S{index}USDT",
                rank=3,
                gain_24h=0.25,
                volume_24h_ratio_7d=2.0,
                entry_price=1.0,
                return_24h=return_24h,
                return_48h=return_48h,
            )
        )
    evaluation_times = [opp.mature_48h_ms for opp in opportunities]

    rows = build_regime_timeline(evaluation_times, opportunities, config)
    red_no_recovery = next(row for row in rows if row.state == "RED" and not row.recovery_signal)
    red_recovery_rows = [row for row in rows if row.state == "RED" and row.recovery_signal]

    assert red_no_recovery.r2_leverage == 2
    assert red_no_recovery.r3_leverage == 1
    assert red_recovery_rows[0].recovery_streak == 1
    assert red_recovery_rows[0].r3_leverage == 3
    assert red_recovery_rows[1].recovery_streak == 2
    assert red_recovery_rows[1].r3_leverage == 5


def test_context_provider_fails_closed_for_stale_configured_context() -> None:
    workdir = _clean_workdir("provider_stale")
    path = workdir / "context.json"
    path.write_text(json.dumps({"signal_time_ms": 1, "state": "GREEN", "status": "READY"}), encoding="utf-8")

    provider = JsonRegimeContextProvider(path)

    with pytest.raises(RuntimeError, match="timestamp does not match"):
        provider.context_at(2)


def test_context_provider_fails_closed_for_model_mismatch() -> None:
    workdir = _clean_workdir("provider_model_mismatch")
    path = workdir / "context.json"
    payload = generate_context(JULY_6_UTC_MS, write_file=False).evaluation.to_context_json(1)
    payload["model"] = "wrong_model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    provider = JsonRegimeContextProvider(path)

    with pytest.raises(RuntimeError, match="model mismatch"):
        provider.context_at(JULY_6_UTC_MS)


def test_auto_generating_provider_writes_then_validates_context() -> None:
    workdir = _clean_workdir("auto_provider")
    path = workdir / "context.json"
    provider = AutoGeneratingRegimeContextProvider(path)

    context = provider.context_at(JULY_6_UTC_MS)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["signal_time_ms"] == JULY_6_UTC_MS
    assert raw["status"] == "READY"
    assert context.state == "RED"
    assert context.recovery_signal is True
    assert context.recovery_streak == 1


def test_context_provider_empty_path_keeps_overlay_disabled() -> None:
    assert regime_context_provider_from_path("") is None


def test_runtime_context_provider_import_does_not_require_numpy() -> None:
    code = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'numpy' or name.startswith('numpy.'):\n"
        "        raise ModuleNotFoundError('No module named numpy')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n"
        "import src.research.rankpulse_regime_context_provider\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_context_atomic_write_produces_complete_json() -> None:
    workdir = _clean_workdir("atomic")
    path = workdir / "context.json"

    write_context_atomic(path, {"signal_time_ms": 1, "state": "RED", "status": "READY"})

    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "RED"


def _clean_workdir(name: str) -> Path:
    path = TEST_TMP / name
    if path.exists():
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path
