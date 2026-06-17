from __future__ import annotations

from src.research.top3_strategy_rules import (
    Top3Signal,
    leverage_for_signal,
    liquidation_threshold_for_leverage,
)


def test_10_20_bucket_uses_3x() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.15, volume_24h_ratio_7d=0.8)

    assert leverage_for_signal(signal) == 3


def test_20_40_rank2_uses_3x() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.25, volume_24h_ratio_7d=2.5)

    assert leverage_for_signal(signal) == 3


def test_20_40_rank3_can_use_higher_5x_leverage() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.35, volume_24h_ratio_7d=4.2)

    assert leverage_for_signal(signal) == 5


def test_40_60_rank2_volume_3_to_6_uses_2x() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.50, volume_24h_ratio_7d=5.5)

    assert leverage_for_signal(signal) == 2


def test_filtered_signal_has_no_leverage() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.65, volume_24h_ratio_7d=4.2)

    assert leverage_for_signal(signal) is None


def test_liquidation_thresholds_match_strategy_spec() -> None:
    assert liquidation_threshold_for_leverage(2) == -0.50
    assert liquidation_threshold_for_leverage(3) == -0.33
    assert liquidation_threshold_for_leverage(5) == -0.20
