from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research_2026_monthly_equity_and_available_funds import hourly_account_curve, overall_funding_summary
from scripts.research_drop_top3_short_edge import HOUR_MS


def test_hourly_curve_separates_equity_and_available_funds() -> None:
    trades = pd.DataFrame(
        [
            {
                "symbol": "TESTUSDT",
                "rank": 1,
                "entry_time_ms": 0,
                "exit_time_ms": 2 * HOUR_MS,
                "entry_price": 100.0,
                "entry_notional_usdt": 100.0,
                "actual_pnl_usdt": 9.81,
            }
        ]
    )
    frame = pd.DataFrame(
        {
            "open_time": [0, HOUR_MS, 2 * HOUR_MS],
            "close": [100.0, 90.0, 90.0],
        }
    ).set_index("open_time", drop=False)

    curve, missing = hourly_account_curve(trades, {"TESTUSDT": frame}, fee_rate=0.001)

    assert missing == []
    assert np.isclose(curve.loc[0, "equity_delta_from_initial_usdt"], -0.10)
    assert np.isclose(curve.loc[0, "available_funds_delta_from_initial_usdt"], -100.10)
    assert np.isclose(curve.loc[1, "equity_delta_from_initial_usdt"], 9.90)
    assert np.isclose(curve.loc[1, "available_funds_delta_from_initial_usdt"], -100.10)
    assert np.isclose(curve.loc[1, "risk_adjusted_available_delta_from_initial_usdt"], -90.10)
    assert np.isclose(curve.loc[2, "equity_delta_from_initial_usdt"], 9.81)
    assert np.isclose(curve.loc[2, "available_funds_delta_from_initial_usdt"], 9.81)
    assert curve.loc[2, "open_positions"] == 0


def test_funding_floor_reports_required_initial_equity_without_inventing_balance() -> None:
    curve = pd.DataFrame(
        {
            "valuation_time_utc": ["2026-01-01 01:00:00+00:00", "2026-01-01 02:00:00+00:00"],
            "equity_delta_from_initial_usdt": [-25.0, 10.0],
            "available_funds_delta_from_initial_usdt": [-120.0, 10.0],
            "risk_adjusted_available_delta_from_initial_usdt": [-145.0, 10.0],
        }
    )

    summary = overall_funding_summary(curve)

    assert summary["initial_account_equity_usdt"] is None
    assert summary["actual_minimum_equity_usdt"] is None
    assert summary["actual_minimum_available_funds_usdt"] is None
    assert summary["minimum_initial_equity_for_nonnegative_available_usdt"] == 120.0
    assert summary["minimum_initial_equity_for_nonnegative_risk_adjusted_available_usdt"] == 145.0
