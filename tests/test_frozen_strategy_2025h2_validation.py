import pandas as pd

from scripts.fetch_historical_futures_data import eligible_contracts
from scripts.research_drop_top3_short_edge import HOUR_MS
from scripts.validate_frozen_strategy_2025h2 import (
    HOLDOUT_END,
    HOLDOUT_START,
    complete_months_for,
    precompute_frozen_outcomes,
    window,
)


def test_historical_contract_eligibility_uses_onboard_and_delivery() -> None:
    metadata = {
        "symbols": [
            {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "SETTLING", "onboardDate": 1, "deliveryDate": int(pd.Timestamp("2025-08-01", tz="UTC").timestamp() * 1000)},
            {"symbol": "LATEUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING", "onboardDate": int(pd.Timestamp("2026-02-01", tz="UTC").timestamp() * 1000), "deliveryDate": 4133404800000},
            {"symbol": "QUARTERUSDT", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT", "status": "TRADING", "onboardDate": 1, "deliveryDate": 4133404800000},
        ]
    }
    assert eligible_contracts(metadata).symbol.tolist() == ["OLDUSDT"]


def test_holdout_window_is_left_closed_right_open() -> None:
    frame = pd.DataFrame({"snapshot_time_ms": [int(HOLDOUT_START.timestamp() * 1000), int(HOLDOUT_END.timestamp() * 1000)]})
    result = window(frame, HOLDOUT_START, HOLDOUT_END)
    assert result.snapshot_time_ms.tolist() == [int(HOLDOUT_START.timestamp() * 1000)]


def test_complete_months_marks_all_2025h2_months_complete() -> None:
    assert complete_months_for(HOLDOUT_START, HOLDOUT_END) == {"2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"}


def test_missing_fixed_exit_is_invalid_not_silently_traded() -> None:
    entry = int(pd.Timestamp("2025-07-01", tz="UTC").timestamp() * 1000)
    times = [entry + hour * HOUR_MS for hour in range(24)]
    frame = pd.DataFrame({"open_time": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}).set_index("open_time", drop=False)
    signal = pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT", "snapshot_time_ms": entry, "snapshot_time_utc": pd.to_datetime(entry, unit="ms", utc=True),
                "snapshot_hour_bj": "00:00", "rank": 1, "drop_24h_pct": 10.0, "candidate_id": "A", "holding_days": 1,
                "entry_time_ms": entry, "entry_time_utc": pd.to_datetime(entry, unit="ms", utc=True),
            }
        ]
    )
    valid, invalid = precompute_frozen_outcomes(signal, {"AAAUSDT": frame}, 0.001)
    assert valid.empty
    assert invalid.invalid_data_reason.tolist() == ["fixed_exit_open_missing"]
