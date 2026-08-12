from __future__ import annotations

import json
import math
from pathlib import Path

from src.config.settings import AppSettings
from src.research.top3_regime_engine import MODEL_NAME, MODEL_VERSION
from src.research.top3_regime_generator import generate_context
from src.research.top3_strategy_rules import Top3RegimeContext


class JsonRegimeContextProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def context_at(self, signal_time_ms: int) -> Top3RegimeContext | None:
        if not self.path.exists():
            raise RuntimeError(f"Configured regime context file does not exist: {self.path}")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Configured regime context file is not valid JSON: {self.path}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"Configured regime context file must contain a JSON object: {self.path}")
        _validate_context_payload(raw, signal_time_ms, source=str(self.path))
        return Top3RegimeContext(
            state=str(raw["state"]).upper(),
            recovery_signal=bool(raw.get("recovery_signal", False)),
            recovery_streak=int(raw.get("recovery_streak", 0)),
            model=str(raw.get("model", MODEL_NAME)),
        )


class AutoGeneratingRegimeContextProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.reader = JsonRegimeContextProvider(self.path)

    def context_at(self, signal_time_ms: int) -> Top3RegimeContext:
        generate_context(
            evaluation_time_ms=signal_time_ms,
            output_path=self.path,
            write_file=True,
        )
        return self.reader.context_at(signal_time_ms)


def regime_context_provider_from_path(path: str) -> JsonRegimeContextProvider | None:
    if not path.strip():
        return None
    return JsonRegimeContextProvider(path)


def regime_context_provider_for_runtime(
    settings: AppSettings,
    default_context_path: str | Path,
) -> JsonRegimeContextProvider | AutoGeneratingRegimeContextProvider | None:
    if not settings.top3_regime_enabled:
        return None
    context_path = settings.top3_regime_context_path.strip() or str(default_context_path)
    if settings.top3_regime_context_auto_generate:
        return AutoGeneratingRegimeContextProvider(context_path)
    return JsonRegimeContextProvider(context_path)


def _validate_context_payload(raw: dict[str, object], signal_time_ms: int, source: str) -> None:
    status = str(raw.get("status", "READY")).upper()
    if status != "READY":
        raise RuntimeError(f"Configured regime context is not READY: status={status}")

    context_signal_time = raw.get("signal_time_ms")
    if context_signal_time is None:
        raise RuntimeError(f"Configured regime context missing signal_time_ms: {source}")
    if int(context_signal_time) != signal_time_ms:
        raise RuntimeError(
            "Configured regime context timestamp does not match signal window: "
            f"context={int(context_signal_time)}, signal={signal_time_ms}"
        )

    model = str(raw.get("model", ""))
    if model != MODEL_NAME:
        raise RuntimeError(f"Configured regime context model mismatch: model={model}, expected={MODEL_NAME}")

    version = int(raw.get("model_version", MODEL_VERSION))
    if version != MODEL_VERSION:
        raise RuntimeError(
            f"Configured regime context model_version mismatch: "
            f"model_version={version}, expected={MODEL_VERSION}"
        )

    state = str(raw.get("state", "")).upper()
    if state not in {"GREEN", "YELLOW", "RED"}:
        raise RuntimeError(f"Configured regime context state is invalid: state={state}")

    recovery_streak = int(raw.get("recovery_streak", 0))
    if recovery_streak < 0:
        raise RuntimeError(f"Configured regime context recovery_streak is invalid: {recovery_streak}")

    data_cutoff = raw.get("data_cutoff_ms")
    if data_cutoff is not None and int(data_cutoff) > signal_time_ms:
        raise RuntimeError(
            "Configured regime context data_cutoff_ms is after signal window: "
            f"data_cutoff={int(data_cutoff)}, signal={signal_time_ms}"
        )

    for field in [
        "last15_decay48",
        "historical_q10",
        "historical_q5",
        "last3_avg_return24",
    ]:
        value = raw.get(field)
        if value is not None and not math.isfinite(float(value)):
            raise RuntimeError(f"Configured regime context {field} is not finite: {value}")

    for field in ["r2_leverage", "r3_leverage"]:
        value = int(raw.get(field, 0))
        if value <= 0:
            raise RuntimeError(f"Configured regime context {field} is invalid: {value}")
