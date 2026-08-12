from __future__ import annotations

from src.market.rank_source import RankingConfig, RankingSource


def test_default_rank_source_is_binance_24hr_ticker_for_live_paper() -> None:
    assert RankingConfig().rank_source == RankingSource.BINANCE_24HR_TICKER


def test_kline_reconstructed_rank_source_is_explicit_reference_mode() -> None:
    config = RankingConfig(rank_source=RankingSource.KLINE_RECONSTRUCTED)

    assert config.rank_source == RankingSource.KLINE_RECONSTRUCTED
