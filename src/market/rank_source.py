from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RankingSource(StrEnum):
    KLINE_RECONSTRUCTED = "KLINE_RECONSTRUCTED"
    BINANCE_24HR_TICKER = "BINANCE_24HR_TICKER"


@dataclass(frozen=True)
class RankingConfig:
    rank_source: RankingSource = RankingSource.BINANCE_24HR_TICKER
