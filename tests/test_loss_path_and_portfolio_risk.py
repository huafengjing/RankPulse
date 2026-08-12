from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.research_loss_path_and_portfolio_risk import (
    concurrency_bucket,
    liquidation_timing_bucket,
    normal_loss_class,
    path_metrics_for_trade,
)


def test_fixed_buckets_have_expected_boundaries() -> None:
    assert [liquidation_timing_bucket(value) for value in [6, 7, 12, 13, 24, 25, 48, 49]] == [
        "<=6H", ">6H-12H", ">6H-12H", ">12H-24H", ">12H-24H", ">1D-2D", ">1D-2D", ">2D"
    ]
    assert [concurrency_bucket(value) for value in [0, 1, 2, 3, 4, 5, 6]] == ["0-1", "0-1", "2-3", "2-3", "4-5", "4-5", "6+"]


def test_normal_loss_classification_is_mutually_exclusive_by_precedence() -> None:
    assert normal_loss_class(0.5, -5, 0.6) == "never_profitable"
    assert normal_loss_class(20, -5, 0.6) == "profit_giveback"
    assert normal_loss_class(5, -5, 0.6) == "small_loss"
    assert normal_loss_class(5, -20, 0.6) == "other_normal_loss"


def test_liquidation_bar_low_is_not_used_for_mfe() -> None:
    hour = 3_600_000
    frame = pd.DataFrame(
        [
            {"open_time": 0, "open": 100.0, "high": 105.0, "low": 90.0, "close": 100.0},
            {"open_time": hour, "open": 100.0, "high": 125.0, "low": 50.0, "close": 100.0},
        ]
    ).set_index("open_time", drop=False)
    row = SimpleNamespace(
        entry_time_ms=0,
        exit_time_ms=hour,
        entry_price=100.0,
        entry_notional_usdt=500.0,
        liquidated=True,
        pnl_usdt=-100.0,
    )
    result = path_metrics_for_trade(row, frame, 0.001)
    assert result["mfe_usdt"] == pytest.approx(50.0)
    assert result["mae_usdt"] == -100.0
    assert result["mfe_time_ms"] == 0
    assert result["mae_time_ms"] == hour


def test_fixed_exit_open_is_included_but_exit_bar_range_is_not() -> None:
    hour = 3_600_000
    frame = pd.DataFrame(
        [
            {"open_time": 0, "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
            {"open_time": hour, "open": 80.0, "high": 150.0, "low": 50.0, "close": 100.0},
        ]
    ).set_index("open_time", drop=False)
    row = SimpleNamespace(
        entry_time_ms=0,
        exit_time_ms=hour,
        entry_price=100.0,
        exit_price=80.0,
        entry_notional_usdt=300.0,
        liquidated=False,
        pnl_usdt=59.46,
    )
    result = path_metrics_for_trade(row, frame, 0.001)
    assert result["mfe_usdt"] == pytest.approx(60.0)
    assert result["mfe_time_ms"] == hour
    assert result["mae_usdt"] == pytest.approx(-15.0)
