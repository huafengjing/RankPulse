from __future__ import annotations

from datetime import datetime, timezone, timedelta

from src.market.binance_futures import Kline, Ticker24hrStat
from src.market.ticker_leaderboard import build_top3_from_24hr_tickers
from src.paper.trading import PaperSignal
from src.config.schedule import is_entry_signal_time
from src.config.settings import AppSettings
from src.research.rankpulse_strategy_rules import SIGNAL_HOURS_BJ, Top3RegimeContext, Top3Signal, rank1_leverage_for_signal, volume_24h_ratio_7d


BEIJING_TZ = timezone(timedelta(hours=8))


def bj_snapshot_hour(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
    return f"{dt.hour:02d}:{dt.minute:02d}"


def is_signal_time_bj(timestamp_ms: int) -> bool:
    return bj_snapshot_hour(timestamp_ms) in SIGNAL_HOURS_BJ


def generate_binance_ticker_rank_signals(
    signal_time_ms: int,
    ticker_stats_by_symbol: dict[str, Ticker24hrStat],
    four_hour_klines_by_symbol: dict[str, list[Kline]],
    latest_prices_by_symbol: dict[str, float],
    settings: AppSettings | None = None,
    regime_context: Top3RegimeContext | None = None,
) -> list[PaperSignal]:
    snapshot_hour_bj = bj_snapshot_hour(signal_time_ms)
    if settings is not None:
        if not is_entry_signal_time(signal_time_ms, settings):
            return []
    elif snapshot_hour_bj not in SIGNAL_HOURS_BJ:
        return []

    volume_ratio_by_symbol: dict[str, float] = {}
    for symbol, klines in four_hour_klines_by_symbol.items():
        completed_volumes = [
            kline.volume
            for kline in klines
            if kline.close_time_ms < signal_time_ms
        ]
        volume_ratio = volume_24h_ratio_7d(completed_volumes)
        if volume_ratio is not None:
            volume_ratio_by_symbol[symbol] = volume_ratio

    top3 = build_top3_from_24hr_tickers(
        ticker_stats_by_symbol=ticker_stats_by_symbol,
        volume_ratio_by_symbol=volume_ratio_by_symbol,
    )

    signals: list[PaperSignal] = []
    for entry in top3:
        if entry.rank not in {1, 2, 3}:
            continue
        if entry.rank == 1 and rank1_leverage_for_signal(
            Top3Signal(
                symbol=entry.symbol,
                rank=entry.rank,
                gain_24h=entry.gain_24h,
                volume_24h_ratio_7d=entry.volume_24h_ratio_7d,
                snapshot_hour_bj="00:00",
            )
        ) is None:
            continue
        fill_price = latest_prices_by_symbol.get(entry.symbol)
        if fill_price is None:
            continue
        signals.append(
            PaperSignal(
                symbol=entry.symbol,
                rank=entry.rank,
                gain_24h=entry.gain_24h,
                volume_24h_ratio_7d=entry.volume_24h_ratio_7d,
                snapshot_hour_bj=snapshot_hour_bj,
                signal_time_ms=signal_time_ms,
                fill_price=fill_price,
                regime_context=regime_context,
            )
        )
    return signals
