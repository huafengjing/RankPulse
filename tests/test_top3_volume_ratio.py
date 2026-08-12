from __future__ import annotations

from src.research.top3_strategy_rules import volume_24h_ratio_7d


def test_volume_ratio_uses_latest_42_completed_4h_candles_including_recent_24h() -> None:
    completed_4h_volumes = [1.0] * 36 + [2.0] * 6

    assert volume_24h_ratio_7d(completed_4h_volumes) == 12.0 / (48.0 / 7.0)


def test_volume_ratio_requires_42_completed_4h_candles() -> None:
    completed_4h_volumes = [1.0] * 41

    assert volume_24h_ratio_7d(completed_4h_volumes) is None


def test_volume_ratio_uses_most_recent_42_completed_4h_candles() -> None:
    older_volume = [100.0]
    completed_4h_volumes = older_volume + [1.0] * 36 + [2.0] * 6

    assert volume_24h_ratio_7d(completed_4h_volumes) == 12.0 / (48.0 / 7.0)
