import pandas as pd

from scripts.research_2025h2_signal_leverage_separation import decompose_signal_paths
from scripts.research_drop_top3_short_edge import HOUR_MS


def test_liquidated_short_can_finish_profitable_at_fixed_exit() -> None:
    entry = int(pd.Timestamp("2025-07-01", tz="UTC").timestamp() * 1000)
    times = [entry + hour * HOUR_MS for hour in range(25)]
    frame = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * 24 + [90.0],
            "high": [125.0] + [101.0] * 24,
            "low": [99.0] + [85.0] + [89.0] * 23,
            "close": [100.0] * 24 + [90.0],
        }
    ).set_index("open_time", drop=False)
    signal = pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT", "snapshot_time_ms": entry, "snapshot_time_utc": pd.to_datetime(entry, unit="ms", utc=True),
                "snapshot_hour_bj": "00:00", "rank": 1, "drop_24h_pct": 10.0, "candidate_id": "A", "holding_days": 1,
                "entry_time_ms": entry, "entry_time_utc": pd.to_datetime(entry, unit="ms", utc=True),
            }
        ]
    )
    details, invalid = decompose_signal_paths(signal, {"AAAUSDT": frame}, 0.001)
    row = details.iloc[0]
    assert invalid.empty
    assert row.current_leverage_actual_liquidated
    assert row.current_leverage_actual_path_pnl_usdt == -100.0
    assert row.current_leverage_no_liquidation_net_pnl_usdt > 0
    assert row.current_liquidated_then_fixed_exit_theoretical_profitable
    assert not row.one_x_actual_liquidated
    assert row.one_x_actual_path_pnl_usdt > 0
    assert row.mae_underlying_pct == -25.0
    assert row.mfe_underlying_pct == 15.0


def test_mae_mfe_have_short_direction_signs() -> None:
    entry = int(pd.Timestamp("2025-07-01", tz="UTC").timestamp() * 1000)
    times = [entry + hour * HOUR_MS for hour in range(25)]
    frame = pd.DataFrame({"open_time": times, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}).set_index("open_time", drop=False)
    signal = pd.DataFrame([{"symbol": "AAAUSDT", "snapshot_time_ms": entry, "snapshot_time_utc": pd.to_datetime(entry, unit="ms", utc=True), "snapshot_hour_bj": "00:00", "rank": 1, "drop_24h_pct": 10.0, "candidate_id": "A", "holding_days": 1, "entry_time_ms": entry, "entry_time_utc": pd.to_datetime(entry, unit="ms", utc=True)}])
    details, _ = decompose_signal_paths(signal, {"AAAUSDT": frame}, 0.001)
    assert details.iloc[0].mae_underlying_pct == 0.0
    assert details.iloc[0].mfe_underlying_pct == 0.0
