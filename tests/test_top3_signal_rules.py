from __future__ import annotations

from src.research.top3_strategy_rules import Top3Signal, is_trade_signal


def test_rank1_signal_is_not_traded() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=1, gain_24h=0.25, volume_24h_ratio_7d=2.5)

    assert is_trade_signal(signal) is False


def test_rank2_gain_25_volume_2_5_is_traded() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.25, volume_24h_ratio_7d=2.5)

    assert is_trade_signal(signal) is True


def test_rank3_gain_35_volume_4_2_is_traded() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.35, volume_24h_ratio_7d=4.2)

    assert is_trade_signal(signal) is True


def test_gain_65_percent_is_filtered() -> None:
    signal = Top3Signal(symbol="AAAUSDT", rank=3, gain_24h=0.65, volume_24h_ratio_7d=4.2)

    assert is_trade_signal(signal) is False


def test_40_60_bucket_requires_rank2_and_volume_3_to_6() -> None:
    rank2_signal = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.45, volume_24h_ratio_7d=4.5)
    rank3_signal = Top3Signal(symbol="BBBUSDT", rank=3, gain_24h=0.45, volume_24h_ratio_7d=4.5)
    low_volume = Top3Signal(symbol="CCCUSDT", rank=2, gain_24h=0.45, volume_24h_ratio_7d=2.9)
    high_volume = Top3Signal(symbol="DDDUSDT", rank=2, gain_24h=0.45, volume_24h_ratio_7d=6.0)

    assert is_trade_signal(rank2_signal) is True
    assert is_trade_signal(rank3_signal) is False
    assert is_trade_signal(low_volume) is False
    assert is_trade_signal(high_volume) is False


def test_excluded_symbol_and_non_signal_hour_are_filtered() -> None:
    excluded = Top3Signal(symbol="RAVEUSDT", rank=2, gain_24h=0.25, volume_24h_ratio_7d=2.5)
    wrong_hour = Top3Signal(symbol="AAAUSDT", rank=2, gain_24h=0.25, volume_24h_ratio_7d=2.5, snapshot_hour_bj="12:00")

    assert is_trade_signal(excluded) is False
    assert is_trade_signal(wrong_hour) is False
