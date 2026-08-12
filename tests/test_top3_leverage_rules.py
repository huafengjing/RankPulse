from __future__ import annotations

from src.research.top3_strategy_rules import (
    Top3Signal,
    Top3RegimeContext,
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


def test_20_40_rank3_volume_1_2_to_1_5_is_now_allowed() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.35, volume_24h_ratio_7d=1.2)

    assert leverage_for_signal(signal) == 5


def test_20_40_rank2_still_requires_volume_1_5() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.35, volume_24h_ratio_7d=1.49)

    assert leverage_for_signal(signal) is None


def test_20_40_rank3_red_regime_uses_1x_until_recovery() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.35, volume_24h_ratio_7d=4.2)

    assert leverage_for_signal(signal, Top3RegimeContext(state="RED")) == 1
    assert leverage_for_signal(signal, Top3RegimeContext(state="RED", recovery_signal=True, recovery_streak=1)) == 3
    assert leverage_for_signal(signal, Top3RegimeContext(state="RED", recovery_signal=True, recovery_streak=2)) == 5


def test_20_40_rank3_yellow_regime_recovers_to_5x_on_first_recovery_signal() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.35, volume_24h_ratio_7d=4.2)

    assert leverage_for_signal(signal, Top3RegimeContext(state="YELLOW")) == 3
    assert leverage_for_signal(signal, Top3RegimeContext(state="YELLOW", recovery_signal=True)) == 5


def test_20_40_rank2_red_regime_uses_2x_until_recovery() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.25, volume_24h_ratio_7d=2.5)

    assert leverage_for_signal(signal, Top3RegimeContext(state="RED")) == 2
    assert leverage_for_signal(signal, Top3RegimeContext(state="RED", recovery_signal=True)) == 3


def test_40_60_rank2_volume_3_to_5_5_uses_2x() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.50, volume_24h_ratio_7d=5.49)

    assert leverage_for_signal(signal) == 2


def test_40_60_rank2_volume_5_5_has_no_leverage() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.50, volume_24h_ratio_7d=5.5)

    assert leverage_for_signal(signal) is None


def test_filtered_signal_has_no_leverage() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.65, volume_24h_ratio_7d=4.2)

    assert leverage_for_signal(signal) is None


def test_liquidation_thresholds_match_strategy_spec() -> None:
    assert liquidation_threshold_for_leverage(1) == -1.00
    assert liquidation_threshold_for_leverage(2) == -0.50
    assert liquidation_threshold_for_leverage(3) == -0.33
    assert liquidation_threshold_for_leverage(5) == -0.20
