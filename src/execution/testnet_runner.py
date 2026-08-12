from __future__ import annotations

import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Protocol

from src.config.settings import AppSettings
from src.config.modes import SignalMode
from src.config.schedule import (
    hourly_window_time_ms,
    information_window_time_ms,
    market_preflight_window_time_ms,
    signal_window_time_ms,
)
from src.exchange.binance_filters import SymbolFilters, filters_from_exchange_symbol
from src.execution.testnet_engine import (
    PositionPrecheckFailedError,
    TestnetClientProtocol,
    TestnetExecutionEngine,
)
from src.execution.testnet_state import TestnetPosition, TestnetState, TestnetStateStore
from src.market.binance_futures import Kline, Ticker24hrStat
from src.market.ticker_leaderboard import build_top3_from_24hr_tickers
from src.paper.signals import generate_binance_ticker_rank_signals
from src.paper.store import PaperEventLogger
from src.research.top3_strategy_rules import (
    Top3Signal,
    Top3RegimeContext,
    leverage_for_signal,
    should_exit_extreme_weak_4h,
    should_exit_early_12h,
    signal_rejection_reason,
    volume_24h_ratio_7d,
)


class TestnetMarketClientProtocol:
    def usdt_perpetual_symbols(self) -> list[str]:
        ...

    def ticker_24hr_stats(self) -> dict[str, Ticker24hrStat]:
        ...

    def four_hour_klines(self, symbol: str, limit: int = 42, end_time_ms: int | None = None) -> list[Kline]:
        ...

    def latest_prices(self) -> dict[str, float]:
        ...

    def klines(self, symbol: str, interval: str, limit: int, end_time_ms: int | None = None) -> list[Kline]:
        ...

    def exchange_symbol(self, symbol: str) -> dict[str, object]:
        ...


class SignalNotifier(Protocol):
    def send_signal_table(self, table: str) -> bool:
        ...


class RegimeContextProvider(Protocol):
    def context_at(self, signal_time_ms: int) -> Top3RegimeContext | None:
        ...


class MarketSymbolFilterProvider:
    def __init__(self, market_client: TestnetMarketClientProtocol) -> None:
        self.market_client = market_client

    def filters_for_symbol(self, symbol: str) -> SymbolFilters:
        return filters_from_exchange_symbol(self.market_client.exchange_symbol(symbol))


class TestnetTradingRunner:
    __test__ = False

    def __init__(
        self,
        market_client: TestnetMarketClientProtocol,
        execution_client: TestnetClientProtocol,
        state_store: TestnetStateStore,
        logger: PaperEventLogger,
        settings: AppSettings,
        signal_notifier: SignalNotifier | None = None,
        regime_context_provider: RegimeContextProvider | None = None,
        event_prefix: str = "testnet",
    ) -> None:
        self.market_client = market_client
        self.event_prefix = event_prefix
        self.engine = TestnetExecutionEngine(
            client=execution_client,
            filter_provider=MarketSymbolFilterProvider(market_client),
            state_store=state_store,
            logger=logger,
            settings=settings,
            event_prefix=event_prefix,
        )
        self.state_store = state_store
        self.settings = settings
        self.signal_notifier = signal_notifier
        self.regime_context_provider = regime_context_provider
        self.logger = logger

    def run_signal_cycle(self, signal_time_ms: int) -> list[TestnetPosition]:
        signal_window_ms = signal_window_time_ms(signal_time_ms, self.settings)
        if signal_window_ms is None:
            return []

        state = self.state_store.load()
        if state.last_signal_time_ms == signal_window_ms:
            return []

        regime_context = self._regime_context(signal_window_ms)
        if regime_context is not None:
            self.logger.log(
                f"{self.event_prefix}_regime_context_ready",
                {
                    "signal_time_ms": signal_window_ms,
                    "model": regime_context.model,
                    "state": regime_context.state,
                    "recovery_signal": regime_context.recovery_signal,
                    "recovery_streak": regime_context.recovery_streak,
                    "bucket_b_rank2_leverage": leverage_for_signal(
                        Top3Signal("REGIME_R2USDT", 2, 0.25, 2.0, "00:00"),
                        regime_context,
                    ),
                    "bucket_b_rank3_leverage": leverage_for_signal(
                        Top3Signal("REGIME_R3USDT", 3, 0.25, 2.0, "00:00"),
                        regime_context,
                    ),
                    "trading_mode": self.settings.trading_mode.value,
                    "signal_mode": self.settings.signal_mode.value,
                },
            )
        ticker_stats, four_hour_klines, latest_prices = self._publish_signal_snapshot(
            signal_time_ms,
            information_only=False,
            regime_context=regime_context,
        )

        signals = generate_binance_ticker_rank_signals(
            signal_time_ms=signal_time_ms,
            ticker_stats_by_symbol=ticker_stats,
            four_hour_klines_by_symbol=four_hour_klines,
            latest_prices_by_symbol=latest_prices,
            settings=self.settings,
            regime_context=regime_context,
        )

        opened: list[TestnetPosition] = []
        for signal in signals:
            strategy_signal = Top3Signal(
                symbol=signal.symbol,
                rank=signal.rank,
                gain_24h=signal.gain_24h,
                volume_24h_ratio_7d=signal.volume_24h_ratio_7d,
                snapshot_hour_bj="00:00",
            )
            reason = signal_rejection_reason(strategy_signal)
            if reason is not None:
                continue
            leverage = leverage_for_signal(strategy_signal, signal.regime_context)
            if leverage is None:
                continue
            try:
                position = self.engine.open_from_signal(signal, leverage=leverage)
            except Exception as exc:
                self.logger.log(
                    f"{self.event_prefix}_open_failed",
                    {
                        "symbol": signal.symbol,
                        "rank": signal.rank,
                        "leverage": leverage,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "trading_mode": self.settings.trading_mode.value,
                        "signal_mode": self.settings.signal_mode.value,
                    },
                )
                if isinstance(exc, PositionPrecheckFailedError):
                    print(f"[执行跳过] {signal.symbol} | Rank {signal.rank}\n{exc}")
                else:
                    print(
                        f"[执行失败] {signal.symbol} | Rank {signal.rank} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue
            if position is not None:
                opened.append(position)
        latest_state = self.state_store.load()
        self.state_store.save(
            TestnetState(
                open_positions=latest_state.open_positions,
                closed_positions=latest_state.closed_positions,
                last_signal_time_ms=signal_window_ms,
                last_exit_check_time_ms=latest_state.last_exit_check_time_ms,
                last_information_time_ms=latest_state.last_information_time_ms,
                last_preflight_time_ms=latest_state.last_preflight_time_ms,
            )
        )
        return opened

    def run_information_cycle(self, signal_time_ms: int) -> bool:
        information_window_ms = information_window_time_ms(
            signal_time_ms,
            self.settings,
        )
        if information_window_ms is None:
            return False

        state = self.state_store.load()
        if state.last_information_time_ms == information_window_ms:
            return False

        self._publish_signal_snapshot(
            signal_time_ms,
            information_only=True,
            regime_context=None,
        )
        latest_state = self.state_store.load()
        self.state_store.save(
            TestnetState(
                open_positions=latest_state.open_positions,
                closed_positions=latest_state.closed_positions,
                last_signal_time_ms=latest_state.last_signal_time_ms,
                last_exit_check_time_ms=latest_state.last_exit_check_time_ms,
                last_information_time_ms=information_window_ms,
                last_preflight_time_ms=latest_state.last_preflight_time_ms,
            )
        )
        return True

    def run_market_preflight_cycle(self, now_ms: int) -> bool:
        preflight_window_ms = market_preflight_window_time_ms(now_ms, self.settings)
        if preflight_window_ms is None:
            return False

        state = self.state_store.load()
        if state.last_preflight_time_ms == preflight_window_ms:
            return False

        preflight_time = _format_beijing_time(preflight_window_ms)
        target_time = _format_beijing_time(preflight_window_ms + 30 * 60 * 1000)
        try:
            symbols = self.market_client.usdt_perpetual_symbols()
            ticker_stats = self.market_client.ticker_24hr_stats()
            message = (
                f"{preflight_time} | MARKET PREFLIGHT\n"
                "STATUS: OK\n"
                f"TARGET: {target_time}\n"
                f"CHECKED: exchangeInfo OK, 24hr ticker OK\n"
                f"USDT_PERPETUAL_SYMBOLS: {len(symbols)}\n"
                f"TICKER_ROWS: {len(ticker_stats)}\n"
                "ACTION: No action needed."
            )
            event_payload = {
                "status": "ok",
                "preflight_time_ms": preflight_window_ms,
                "target_time_ms": preflight_window_ms + 30 * 60 * 1000,
                "usdt_perpetual_symbols": len(symbols),
                "ticker_rows": len(ticker_stats),
            }
        except Exception as exc:
            message = (
                f"{preflight_time} | MARKET PREFLIGHT\n"
                "STATUS: FAILED\n"
                f"TARGET: {target_time}\n"
                "CHECKED: Binance public market data\n"
                "RESULT: Upcoming Rank2/Rank3 signal may fail if this continues.\n"
                "ACTION: Check network/proxy/IP. If Binance returns HTTP 418, change IP before target time.\n"
                f"DETAIL: {type(exc).__name__}: {exc}"
            )
            event_payload = {
                "status": "failed",
                "preflight_time_ms": preflight_window_ms,
                "target_time_ms": preflight_window_ms + 30 * 60 * 1000,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        if self.signal_notifier is not None:
            self.signal_notifier.send_signal_table(message)
        self.logger.log(
            f"{self.event_prefix}_market_preflight",
            {
                **event_payload,
                "trading_mode": self.settings.trading_mode.value,
                "signal_mode": self.settings.signal_mode.value,
            },
        )

        latest_state = self.state_store.load()
        self.state_store.save(
            TestnetState(
                open_positions=latest_state.open_positions,
                closed_positions=latest_state.closed_positions,
                last_signal_time_ms=latest_state.last_signal_time_ms,
                last_exit_check_time_ms=latest_state.last_exit_check_time_ms,
                last_information_time_ms=latest_state.last_information_time_ms,
                last_preflight_time_ms=preflight_window_ms,
            )
        )
        return True

    def _publish_signal_snapshot(
        self,
        signal_time_ms: int,
        information_only: bool,
        regime_context: Top3RegimeContext | None = None,
    ) -> tuple[
        dict[str, Ticker24hrStat],
        dict[str, list[Kline]],
        dict[str, float],
    ]:
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

        # 计算量比并展示所有 Rank2/3 候选信号（无论是否在信号时间窗口）
        volume_ratio_by_symbol: dict[str, float] = {}
        for klines_symbol, klines in four_hour_klines.items():
            completed_volumes = [k.volume for k in klines if k.close_time_ms < signal_time_ms]
            ratio = volume_24h_ratio_7d(completed_volumes)
            if ratio is not None:
                volume_ratio_by_symbol[klines_symbol] = ratio

        ranked_entries = build_top3_from_24hr_tickers(
            ticker_stats_by_symbol=ticker_stats,
            volume_ratio_by_symbol=volume_ratio_by_symbol,
        )
        candidate_entries = [entry for entry in ranked_entries if entry.rank in {2, 3}]
        table_rows: list[list[str]] = []
        telegram_rows: list[list[str]] = []
        telegram_notes: list[str] = []
        for entry in candidate_entries:
            sig = Top3Signal(
                symbol=entry.symbol,
                rank=entry.rank,
                gain_24h=entry.gain_24h,
                volume_24h_ratio_7d=entry.volume_24h_ratio_7d,
            )
            reason = signal_rejection_reason(sig)
            vol_str = f"{entry.volume_24h_ratio_7d:.2f}" if entry.volume_24h_ratio_7d is not None else "N/A"
            price_str = _format_price(latest_prices.get(entry.symbol))
            if reason:
                result = f"✗ {reason}"
                telegram_status = "SKIP"
                telegram_notes.append(f"{entry.symbol}: {reason}")
            else:
                lev = leverage_for_signal(sig, regime_context)
                result = f"✓ 符合规则 {lev}x"
                telegram_status = f"PASS {lev}x"
                if information_only:
                    telegram_notes.append(
                        f"{entry.symbol}: 符合规则，仅信息观察，不开仓"
                    )
            table_rows.append(
                [
                    entry.symbol,
                    str(entry.rank),
                    price_str,
                    f"{entry.gain_24h:.1%}",
                    vol_str,
                    result,
                ]
            )
            telegram_rows.append(
                [
                    entry.symbol,
                    str(entry.rank),
                    price_str,
                    f"{entry.gain_24h:.1%}",
                    vol_str,
                    telegram_status,
                ]
            )
        table_text = _render_signal_table(table_rows)
        snapshot_time = _format_beijing_time(signal_time_ms)
        terminal_context = (
            "信息观察（不交易）"
            if information_only
            else "交易信号"
        )
        terminal_header = (
            f"{'=' * 16} {snapshot_time} | {terminal_context} {'=' * 16}"
        )
        _safe_print(f"\n{terminal_header}\n{table_text}\n")
        if self.signal_notifier is not None:
            telegram_text = _render_telegram_signal_table(
                telegram_rows,
                telegram_notes,
                regime_context if not information_only else None,
            )
            telegram_context = (
                "INFO ONLY - NO ORDER"
                if information_only
                else "TRADE SIGNAL"
            )
            self.signal_notifier.send_signal_table(
                f"{snapshot_time} | {telegram_context}\n{telegram_text}"
            )
        return ticker_stats, four_hour_klines, latest_prices

    def _regime_context(self, signal_time_ms: int) -> Top3RegimeContext | None:
        if self.regime_context_provider is None:
            return None
        return self.regime_context_provider.context_at(signal_time_ms)

    def run_hourly_exit_cycle(self, now_ms: int):
        exit_window_ms = hourly_window_time_ms(now_ms)
        if exit_window_ms is None:
            return [], []

        state = self.state_store.load()
        if state.last_exit_check_time_ms == exit_window_ms:
            return [], []

        weak_exits = self.run_weak_exit_checks(now_ms)
        planned_exits = self.run_planned_exits(now_ms)
        latest_state = self.state_store.load()
        self.state_store.save(
            TestnetState(
                open_positions=latest_state.open_positions,
                closed_positions=latest_state.closed_positions,
                last_signal_time_ms=latest_state.last_signal_time_ms,
                last_exit_check_time_ms=exit_window_ms,
                last_information_time_ms=latest_state.last_information_time_ms,
                last_preflight_time_ms=latest_state.last_preflight_time_ms,
            )
        )
        return weak_exits, planned_exits

    def run_planned_exits(self, now_ms: int):
        closed = []
        for position in list(self.state_store.load().open_positions):
            if now_ms >= position.planned_exit_time_ms:
                closed_position = self.engine.close_position(position.symbol, now_ms, "planned")
                if closed_position is not None:
                    closed.append(closed_position)
        return closed
    def run_weak_exit_checks(self, now_ms: int):
        if not self.settings.enable_12h_weak_exit and not self.settings.enable_4h_extreme_weak_exit:
            return []

        closed = []
        for position in list(self.state_store.load().open_positions):
            if (
                self.settings.enable_4h_extreme_weak_exit
                and not position.extreme_weak_exit_checked
                and now_ms >= position.extreme_weak_exit_check_time_ms
            ):
                interval = "1m" if self.settings.signal_mode == SignalMode.TEST_FAST else "1h"
                limit = self.settings.test_extreme_weak_exit_after_minutes if interval == "1m" else 4
                metrics = self._exit_metrics(position, interval, limit, now_ms)
                if metrics is not None:
                    mfe, mae, _close_return = metrics
                    if should_exit_extreme_weak_4h(mfe, mae, enabled=True):
                        closed_position = self.engine.close_position(position.symbol, now_ms, "extreme_weak_4h")
                        if closed_position is not None:
                            closed.append(closed_position)
                            continue
                    else:
                        self.state_store.mark_extreme_weak_exit_checked(position.symbol)
                        self.logger.log(
                            f"{self.event_prefix}_extreme_weak_4h_checked_no_exit",
                            {
                                "symbol": position.symbol,
                                "check_time_ms": now_ms,
                                "mfe_4h": mfe,
                                "mae_4h": mae,
                                "trading_mode": self.settings.trading_mode.value,
                                "signal_mode": self.settings.signal_mode.value,
                            },
                        )

            position = self.state_store.load().open_position(position.symbol)
            if position is None:
                continue
            if (
                not self.settings.enable_12h_weak_exit
                or position.weak_exit_checked
                or now_ms < position.weak_exit_check_time_ms
            ):
                continue

            interval = "1m" if self.settings.signal_mode == SignalMode.TEST_FAST else "1h"
            limit = self.settings.test_weak_exit_after_minutes if interval == "1m" else 12
            metrics = self._exit_metrics(position, interval, limit, now_ms)
            if metrics is None:
                continue

            mfe, mae, close_return = metrics
            if should_exit_early_12h(mfe, close_return, mae, enabled=True):
                closed_position = self.engine.close_position(position.symbol, now_ms, "weak_12h")
                if closed_position is not None:
                    closed.append(closed_position)
            else:
                self.state_store.mark_weak_exit_checked(position.symbol)
                self.logger.log(
                    f"{self.event_prefix}_weak_12h_checked_no_exit",
                    {
                        "symbol": position.symbol,
                        "check_time_ms": now_ms,
                        "mfe_12h": mfe,
                        "close_return_12h": close_return,
                        "mae_12h": mae,
                        "trading_mode": self.settings.trading_mode.value,
                        "signal_mode": self.settings.signal_mode.value,
                    },
                )
        return closed

    def _exit_metrics(
        self,
        position: TestnetPosition,
        interval: str,
        limit: int,
        now_ms: int,
    ) -> tuple[float, float, float] | None:
        klines = self.market_client.klines(position.symbol, interval=interval, limit=limit, end_time_ms=now_ms)
        completed = [
            kline
            for kline in klines
            if position.entry_time_ms <= kline.open_time_ms and kline.close_time_ms < now_ms
        ]
        min_required = max(1, int(limit * 0.8))
        if len(completed) < min_required:
            return None

        sorted_completed = sorted(completed, key=lambda item: item.open_time_ms)
        mfe = max(kline.high for kline in sorted_completed) / position.entry_price - 1
        mae = min(kline.low for kline in sorted_completed) / position.entry_price - 1
        close_return = sorted_completed[-1].close / position.entry_price - 1
        return mfe, mae, close_return


def _format_price(price: float | None) -> str:
    if price is None:
        return "N/A"
    return f"{price:.12f}".rstrip("0").rstrip(".")


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _render_signal_table(rows: list[list[str]]) -> str:
    headers = ["信号", "排名", "价格", "涨幅", "量比", "结果"]
    alignments = ["left", "right", "right", "right", "right", "left"]
    widths = [
        max(_display_width(header), *(_display_width(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        cells = [
            _pad_display(value, widths[index], alignments[index])
            for index, value in enumerate(values)
        ]
        return " | ".join(cells)

    header = render_row(headers)
    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([header, separator, *(render_row(row) for row in rows)])


def _render_telegram_signal_table(
    rows: list[list[str]],
    notes: list[str],
    regime_context: Top3RegimeContext | None = None,
) -> str:
    headers = ["SYMBOL", "RANK", "PRICE", "GAIN", "V/R", "STATUS"]
    alignments = ["left", "right", "right", "right", "right", "left"]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        cells = [
            value.rjust(widths[index])
            if alignments[index] == "right"
            else value.ljust(widths[index])
            for index, value in enumerate(values)
        ]
        return " | ".join(cells)

    lines = [
        render_row(headers),
        "-+-".join("-" * width for width in widths),
        *(render_row(row) for row in rows),
    ]
    if regime_context is not None:
        r2_leverage = leverage_for_signal(
            Top3Signal("REGIME_R2USDT", 2, 0.25, 2.0, "00:00"),
            regime_context,
        )
        r3_leverage = leverage_for_signal(
            Top3Signal("REGIME_R3USDT", 3, 0.25, 2.0, "00:00"),
            regime_context,
        )
        lines.extend(
            [
                "",
                f"Regime: {regime_context.state}",
                f"Recovery: {'YES' if regime_context.recovery_signal else 'NO'} / streak {regime_context.recovery_streak}",
                f"Bucket B R2: {r2_leverage}x",
                f"Bucket B R3: {r3_leverage}x",
            ]
        )
    if notes:
        lines.extend(["", "DETAILS", *notes])
    return "\n".join(lines)


def _pad_display(value: str, width: int, alignment: str) -> str:
    padding = max(0, width - _display_width(value))
    if alignment == "right":
        return (" " * padding) + value
    return value + (" " * padding)


def _display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        for char in value
    )


def _format_beijing_time(timestamp_ms: int) -> str:
    beijing = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=beijing,
    ).strftime("%Y-%m-%d %H:%M:%S")
