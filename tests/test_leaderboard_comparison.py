from __future__ import annotations

from src.market.leaderboard_compare import compare_leaderboards
from src.market.rankpulse_leaderboard import LeaderboardEntry


def test_compare_leaderboards_records_rank_level_symbol_differences() -> None:
    kline_top3 = [
        LeaderboardEntry(symbol="AAAUSDT", rank=1, gain_24h=0.30, volume_24h_ratio_7d=2.0),
        LeaderboardEntry(symbol="BBBUSDT", rank=2, gain_24h=0.20, volume_24h_ratio_7d=3.0),
        LeaderboardEntry(symbol="CCCUSDT", rank=3, gain_24h=0.10, volume_24h_ratio_7d=4.0),
    ]
    ticker_top3 = [
        LeaderboardEntry(symbol="AAAUSDT", rank=1, gain_24h=0.29, volume_24h_ratio_7d=2.0),
        LeaderboardEntry(symbol="DDDUSDT", rank=2, gain_24h=0.25, volume_24h_ratio_7d=1.8),
        LeaderboardEntry(symbol="CCCUSDT", rank=3, gain_24h=0.11, volume_24h_ratio_7d=4.0),
    ]

    comparison = compare_leaderboards(
        signal_time_ms=1_700_000_000_000,
        kline_top3=kline_top3,
        ticker_top3=ticker_top3,
    )

    assert comparison.signal_time_ms == 1_700_000_000_000
    assert comparison.kline_symbols == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert comparison.ticker_symbols == ["AAAUSDT", "DDDUSDT", "CCCUSDT"]
    assert [(diff.rank, diff.kline_symbol, diff.ticker_symbol, diff.symbol_matches) for diff in comparison.rank_diffs] == [
        (1, "AAAUSDT", "AAAUSDT", True),
        (2, "BBBUSDT", "DDDUSDT", False),
        (3, "CCCUSDT", "CCCUSDT", True),
    ]
