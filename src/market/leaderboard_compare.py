from __future__ import annotations

from dataclasses import dataclass

from src.market.top3_leaderboard import LeaderboardEntry


@dataclass(frozen=True)
class RankDiff:
    rank: int
    kline_symbol: str | None
    ticker_symbol: str | None
    kline_gain_24h: float | None
    ticker_gain_24h: float | None
    symbol_matches: bool


@dataclass(frozen=True)
class LeaderboardComparison:
    signal_time_ms: int
    kline_symbols: list[str]
    ticker_symbols: list[str]
    rank_diffs: list[RankDiff]


def compare_leaderboards(
    signal_time_ms: int,
    kline_top3: list[LeaderboardEntry],
    ticker_top3: list[LeaderboardEntry],
) -> LeaderboardComparison:
    kline_by_rank = {entry.rank: entry for entry in kline_top3}
    ticker_by_rank = {entry.rank: entry for entry in ticker_top3}
    max_rank = max(kline_by_rank.keys() | ticker_by_rank.keys(), default=0)

    rank_diffs: list[RankDiff] = []
    for rank in range(1, max_rank + 1):
        kline_entry = kline_by_rank.get(rank)
        ticker_entry = ticker_by_rank.get(rank)
        kline_symbol = kline_entry.symbol if kline_entry is not None else None
        ticker_symbol = ticker_entry.symbol if ticker_entry is not None else None
        rank_diffs.append(
            RankDiff(
                rank=rank,
                kline_symbol=kline_symbol,
                ticker_symbol=ticker_symbol,
                kline_gain_24h=kline_entry.gain_24h if kline_entry is not None else None,
                ticker_gain_24h=ticker_entry.gain_24h if ticker_entry is not None else None,
                symbol_matches=kline_symbol == ticker_symbol,
            )
        )

    return LeaderboardComparison(
        signal_time_ms=signal_time_ms,
        kline_symbols=[entry.symbol for entry in sorted(kline_top3, key=lambda item: item.rank)],
        ticker_symbols=[entry.symbol for entry in sorted(ticker_top3, key=lambda item: item.rank)],
        rank_diffs=rank_diffs,
    )
