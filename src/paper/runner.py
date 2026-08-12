from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

from src.market.binance_futures import Kline, Ticker24hrStat
from src.market.ticker_leaderboard import build_top3_from_24hr_tickers
from src.paper.signals import generate_binance_ticker_rank_signals
from src.paper.store import PaperEventLogger, PaperStateStore
from src.paper.trading import (
    PaperExtremeWeakExitCheck,
    PaperPosition,
    PaperTradeExit,
    PaperTradingEngine,
    PaperWeakExitCheck,
)
from src.research.top3_strategy_rules import HOUR_MS, signal_rejection_reason
from src.research.top3_strategy_rules import Top3RegimeContext


class PaperMarketClient(Protocol):
    def usdt_perpetual_symbols(self) -> list[str]:
        ...

    def ticker_24hr_stats(self) -> dict[str, Ticker24hrStat]:
        ...

    def four_hour_klines(self, symbol: str, limit: int = 42, end_time_ms: int | None = None) -> list[Kline]:
        ...

    def one_hour_klines(self, symbol: str, limit: int = 13, end_time_ms: int | None = None) -> list[Kline]:
        ...

    def latest_prices(self) -> dict[str, float]:
        ...


class RegimeContextProvider(Protocol):
    def context_at(self, signal_time_ms: int) -> Top3RegimeContext | None:
        ...


class PaperTradingRunner:
    def __init__(
        self,
        market_client: PaperMarketClient,
        engine: PaperTradingEngine,
        store: PaperStateStore,
        logger: PaperEventLogger,
        regime_context_provider: RegimeContextProvider | None = None,
    ) -> None:
        self.market_client = market_client
        self.engine = engine
        self.store = store
        self.logger = logger
        self.regime_context_provider = regime_context_provider

    def run_signal_cycle(self, signal_time_ms: int) -> list[PaperPosition]:
        tradable_symbols = set(self.market_client.usdt_perpetual_symbols())
        ticker_stats = {
            symbol: stat
            for symbol, stat in self.market_client.ticker_24hr_stats().items()
            if symbol in tradable_symbols
        }
        top3_symbols = [
            entry.symbol
            for entry in build_top3_from_24hr_tickers(
                ticker_stats_by_symbol=ticker_stats,
                volume_ratio_by_symbol={},
            )
        ]
        four_hour_klines = {
            symbol: self.market_client.four_hour_klines(symbol, limit=43, end_time_ms=signal_time_ms)
            for symbol in top3_symbols
        }
        latest_prices = self.market_client.latest_prices()
        regime_context = (
            self.regime_context_provider.context_at(signal_time_ms)
            if self.regime_context_provider is not None
            else None
        )

        signals = generate_binance_ticker_rank_signals(
            signal_time_ms=signal_time_ms,
            ticker_stats_by_symbol=ticker_stats,
            four_hour_klines_by_symbol=four_hour_klines,
            latest_prices_by_symbol=latest_prices,
            regime_context=regime_context,
        )

        opened: list[PaperPosition] = []
        for signal in signals:
            position = self.engine.on_signal(signal)
            if position is None:
                self.logger.log("signal_skipped", asdict(signal))
                continue
            opened.append(position)
            self.logger.log("signal_opened", asdict(position))

        self.store.save(self.engine)
        return opened

    def run_planned_exits(self, now_ms: int) -> list[PaperTradeExit]:
        latest_prices = self.market_client.latest_prices()
        exits: list[PaperTradeExit] = []

        for position in list(self.engine.open_positions()):
            fill_price = latest_prices.get(position.symbol)
            if fill_price is None:
                continue
            trade_exit = self.engine.on_planned_exit(position.symbol, now_ms, fill_price)
            if trade_exit is None:
                continue
            exits.append(trade_exit)
            self.logger.log("planned_exit", asdict(trade_exit))

        self.store.save(self.engine)
        return exits

    def run_weak_exit_checks(self, now_ms: int) -> list[PaperTradeExit]:
        latest_prices = self.market_client.latest_prices()
        exits: list[PaperTradeExit] = []

        for position in list(self.engine.open_positions()):
            if (
                not position.extreme_weak_exit_checked
                and now_ms >= position.extreme_weak_exit_check_time_ms
            ):
                klines = self.market_client.one_hour_klines(position.symbol, limit=5, end_time_ms=now_ms)
                check = _extreme_weak_exit_check_from_klines(position, klines, now_ms, latest_prices)
                if check is not None:
                    trade_exit = self.engine.on_extreme_weak_exit_check(check)
                    if trade_exit is not None:
                        exits.append(trade_exit)
                        self.logger.log("extreme_weak_4h_exit", asdict(trade_exit))
                        continue
                    self.logger.log(
                        "extreme_weak_4h_checked_no_exit",
                        {"symbol": position.symbol, "check_time_ms": now_ms},
                    )

            current_position = self.engine.open_position(position.symbol)
            if (
                current_position is None
                or current_position.weak_exit_checked
                or now_ms < current_position.entry_time_ms + 12 * HOUR_MS
            ):
                continue

            klines = self.market_client.one_hour_klines(current_position.symbol, limit=13, end_time_ms=now_ms)
            check = _weak_exit_check_from_klines(current_position, klines, now_ms, latest_prices)
            if check is None:
                continue

            trade_exit = self.engine.on_weak_exit_check(check)
            if trade_exit is None:
                self.logger.log("weak_12h_checked_no_exit", {"symbol": position.symbol, "check_time_ms": now_ms})
                continue
            exits.append(trade_exit)
            self.logger.log("weak_12h_exit", asdict(trade_exit))

        self.store.save(self.engine)
        return exits


def _extreme_weak_exit_check_from_klines(
    position: PaperPosition,
    klines: list[Kline],
    now_ms: int,
    latest_prices: dict[str, float],
) -> PaperExtremeWeakExitCheck | None:
    completed = [
        kline
        for kline in klines
        if position.entry_time_ms <= kline.open_time_ms
        and kline.close_time_ms < now_ms
    ]
    if len(completed) < 4:
        return None

    first_4h = sorted(completed, key=lambda item: item.open_time_ms)[:4]
    entry_price = position.entry_price
    mfe_4h = max(kline.high for kline in first_4h) / entry_price - 1
    mae_4h = min(kline.low for kline in first_4h) / entry_price - 1
    fill_price = latest_prices.get(position.symbol, first_4h[-1].close)

    return PaperExtremeWeakExitCheck(
        symbol=position.symbol,
        check_time_ms=now_ms,
        fill_price=fill_price,
        mfe_4h=mfe_4h,
        mae_4h=mae_4h,
    )


def _weak_exit_check_from_klines(
    position: PaperPosition,
    klines: list[Kline],
    now_ms: int,
    latest_prices: dict[str, float],
) -> PaperWeakExitCheck | None:
    completed = [
        kline
        for kline in klines
        if position.entry_time_ms <= kline.open_time_ms
        and kline.close_time_ms < now_ms
    ]
    if len(completed) < 12:
        return None

    first_12h = sorted(completed, key=lambda item: item.open_time_ms)[:12]
    entry_price = position.entry_price
    mfe_12h = max(kline.high for kline in first_12h) / entry_price - 1
    mae_12h = min(kline.low for kline in first_12h) / entry_price - 1
    close_return_12h = first_12h[-1].close / entry_price - 1
    fill_price = latest_prices.get(position.symbol, first_12h[-1].close)

    return PaperWeakExitCheck(
        symbol=position.symbol,
        check_time_ms=now_ms,
        fill_price=fill_price,
        mfe_12h=mfe_12h,
        close_return_12h=close_return_12h,
        mae_12h=mae_12h,
    )
