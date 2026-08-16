from __future__ import annotations

from src.market.binance_futures import Ticker24hrStat
from src.market.rankpulse_leaderboard import LeaderboardEntry


def build_top3_from_24hr_tickers(
    ticker_stats_by_symbol: dict[str, float | Ticker24hrStat],
    volume_ratio_by_symbol: dict[str, float],
) -> list[LeaderboardEntry]:
    ranked_candidates: list[tuple[str, float]] = []

    for symbol, stat in ticker_stats_by_symbol.items():
        price_change_percent = (
            stat.price_change_percent
            if isinstance(stat, Ticker24hrStat)
            else stat
        )
        ranked_candidates.append((symbol, round(price_change_percent / 100, 12)))

    ranked_candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        LeaderboardEntry(
            symbol=symbol,
            rank=index + 1,
            gain_24h=gain_24h,
            volume_24h_ratio_7d=volume_ratio_by_symbol.get(symbol),
        )
        for index, (symbol, gain_24h) in enumerate(ranked_candidates[:3])
    ]
