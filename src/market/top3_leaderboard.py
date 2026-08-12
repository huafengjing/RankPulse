from __future__ import annotations

from dataclasses import dataclass

from src.market.binance_futures import Kline
from src.research.top3_strategy_rules import HOUR_MS, Top3Signal, volume_24h_ratio_7d


@dataclass(frozen=True)
class LeaderboardEntry:
    symbol: str
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float | None

    def to_signal(self, snapshot_hour_bj: str) -> Top3Signal:
        return Top3Signal(
            symbol=self.symbol,
            rank=self.rank,
            gain_24h=self.gain_24h,
            volume_24h_ratio_7d=self.volume_24h_ratio_7d,
            snapshot_hour_bj=snapshot_hour_bj,
        )


def gain_24h_from_1h_klines(klines: list[Kline], signal_time_ms: int) -> float | None:
    open_by_time = {kline.open_time_ms: kline.open for kline in klines}
    current_open = open_by_time.get(signal_time_ms)
    prior_open = open_by_time.get(signal_time_ms - 24 * HOUR_MS)
    if current_open is None or prior_open is None or prior_open == 0:
        return None
    return round((current_open / prior_open) - 1, 12)


def build_top3_leaderboard(
    one_hour_by_symbol: dict[str, list[Kline]],
    four_hour_by_symbol: dict[str, list[Kline]],
    signal_time_ms: int,
) -> list[LeaderboardEntry]:
    ranked_candidates: list[tuple[str, float, float | None]] = []

    for symbol, one_hour_klines in one_hour_by_symbol.items():
        gain_24h = gain_24h_from_1h_klines(one_hour_klines, signal_time_ms)
        if gain_24h is None:
            continue

        completed_4h_volumes = [
            kline.volume
            for kline in four_hour_by_symbol.get(symbol, [])
            if kline.close_time_ms < signal_time_ms
        ]
        volume_ratio = volume_24h_ratio_7d(completed_4h_volumes)
        if volume_ratio is None:
            continue

        ranked_candidates.append((symbol, gain_24h, volume_ratio))

    ranked_candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        LeaderboardEntry(
            symbol=symbol,
            rank=index + 1,
            gain_24h=gain_24h,
            volume_24h_ratio_7d=volume_ratio,
        )
        for index, (symbol, gain_24h, volume_ratio) in enumerate(ranked_candidates[:3])
    ]
