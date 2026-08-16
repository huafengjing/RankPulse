from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from src.config.settings import AppSettings
from src.exchange.binance_testnet import FuturesPosition
from src.execution.testnet_state import (
    BootstrapMetadata,
    BootstrapVirtualPosition,
    TestnetState,
    TestnetStateStore,
)
from src.market.binance_futures import Kline
from src.paper.store import PaperEventLogger
from src.paper.trading import PaperSignal
from src.research.rankpulse_strategy_rules import (
    DAY_MS,
    Top3Signal,
    leverage_for_signal,
    planned_exit_time_ms,
    should_exit_early_12h,
    should_exit_extreme_weak_4h,
    signal_rejection_reason,
    volume_24h_ratio_7d,
)


BEIJING_TZ = timezone(timedelta(hours=8))
RANK1_BOOTSTRAP_HOLD_MS = 5 * DAY_MS


class BootstrapMarketProtocol(Protocol):
    def usdt_perpetual_symbols(self) -> list[str]:
        ...

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[Kline]:
        ...


class BootstrapSignalProvider(Protocol):
    def signals_between(self, start_time_ms: int, end_time_ms: int) -> list[PaperSignal]:
        ...


class HistoricalKlineBootstrapSignalProvider:
    def __init__(self, market_client: BootstrapMarketProtocol) -> None:
        self.market_client = market_client

    def signals_between(self, start_time_ms: int, end_time_ms: int) -> list[PaperSignal]:
        signal_times = bootstrap_signal_times(start_time_ms, end_time_ms)
        if not signal_times:
            return []

        symbols = self.market_client.usdt_perpetual_symbols()
        one_hour_by_symbol = {
            symbol: self.market_client.klines(
                symbol,
                interval="1h",
                limit=200,
                start_time_ms=start_time_ms - DAY_MS,
                end_time_ms=end_time_ms,
            )
            for symbol in symbols
        }
        four_hour_by_symbol = {
            symbol: self.market_client.klines(
                symbol,
                interval="4h",
                limit=90,
                start_time_ms=start_time_ms - 42 * 4 * 60 * 60 * 1000,
                end_time_ms=end_time_ms,
            )
            for symbol in symbols
        }

        signals: list[PaperSignal] = []
        for signal_time_ms in signal_times:
            entries = _rank_entries_at(
                signal_time_ms=signal_time_ms,
                one_hour_by_symbol=one_hour_by_symbol,
                four_hour_by_symbol=four_hour_by_symbol,
            )
            snapshot_hour = _bj_snapshot_hour(signal_time_ms)
            for entry in entries:
                signals.append(
                    PaperSignal(
                        symbol=entry.symbol,
                        rank=entry.rank,
                        gain_24h=entry.gain_24h,
                        volume_24h_ratio_7d=entry.volume_24h_ratio_7d,
                        snapshot_hour_bj=snapshot_hour,
                        signal_time_ms=signal_time_ms,
                        fill_price=entry.fill_price,
                    )
                )
        return signals


class BootstrapManager:
    def __init__(
        self,
        state_store: TestnetStateStore,
        logger: PaperEventLogger,
        settings: AppSettings,
        market_client: BootstrapMarketProtocol,
        signal_provider: BootstrapSignalProvider | None = None,
        event_prefix: str = "testnet",
    ) -> None:
        self.state_store = state_store
        self.logger = logger
        self.settings = settings
        self.market_client = market_client
        self.signal_provider = signal_provider or HistoricalKlineBootstrapSignalProvider(market_client)
        self.event_prefix = event_prefix

    def ensure_bootstrap(
        self,
        now_ms: int,
        exchange_positions: dict[str, FuturesPosition],
    ) -> list[BootstrapVirtualPosition]:
        if not self.settings.rank1_bootstrap_enabled:
            return []

        state = self.state_store.load()
        metadata = state.bootstrap_metadata
        activation_time_ms = metadata.activation_time_ms or self.settings.rank1_activation_time_ms or now_ms
        strategy_version = self.settings.rank1_bootstrap_strategy_version
        if metadata.bootstrap_completed and metadata.strategy_version == strategy_version:
            return state.bootstrap_virtual_positions

        start_time_ms = activation_time_ms - 6 * DAY_MS
        real_symbols = {
            symbol
            for symbol, position in exchange_positions.items()
            if position.position_amt != 0
        }
        replay_state = TestnetState(
            open_positions=state.open_positions,
            closed_positions=state.closed_positions,
            bootstrap_metadata=BootstrapMetadata(
                bootstrap_completed=False,
                activation_time_ms=activation_time_ms,
                strategy_version=strategy_version,
            ),
            bootstrap_virtual_positions=[],
            closed_bootstrap_virtual_positions=state.closed_bootstrap_virtual_positions,
            last_signal_time_ms=state.last_signal_time_ms,
            last_exit_check_time_ms=state.last_exit_check_time_ms,
            last_information_time_ms=state.last_information_time_ms,
            last_preflight_time_ms=state.last_preflight_time_ms,
        )
        self.state_store.save(replay_state)

        signals_by_time: dict[int, list[PaperSignal]] = {}
        for signal in self.signal_provider.signals_between(start_time_ms, activation_time_ms):
            signals_by_time.setdefault(signal.signal_time_ms, []).append(signal)

        event_times = sorted({*signals_by_time.keys(), activation_time_ms})
        processed_times: set[int] = set()
        while event_times:
            event_time_ms = event_times.pop(0)
            if event_time_ms in processed_times or event_time_ms > activation_time_ms:
                continue
            processed_times.add(event_time_ms)
            self.close_due_virtual_positions(event_time_ms)
            for signal in sorted(signals_by_time.get(event_time_ms, []), key=lambda item: item.rank):
                position = self._maybe_open_virtual(signal, real_symbols, strategy_version)
                if position is not None:
                    for due_time in (
                        position.extreme_weak_exit_check_time_ms,
                        position.weak_exit_check_time_ms,
                        position.planned_exit_time_ms,
                    ):
                        if due_time <= activation_time_ms and due_time not in processed_times:
                            event_times.append(due_time)
                    event_times.sort()

        self.close_due_virtual_positions(activation_time_ms)
        latest_state = self.state_store.load()
        final_virtuals = [
            position
            for position in latest_state.bootstrap_virtual_positions
            if position.symbol not in real_symbols
        ]
        completed_state = TestnetState(
            open_positions=latest_state.open_positions,
            closed_positions=latest_state.closed_positions,
            bootstrap_metadata=BootstrapMetadata(
                bootstrap_completed=True,
                activation_time_ms=activation_time_ms,
                strategy_version=strategy_version,
            ),
            bootstrap_virtual_positions=final_virtuals,
            closed_bootstrap_virtual_positions=latest_state.closed_bootstrap_virtual_positions,
            last_signal_time_ms=latest_state.last_signal_time_ms,
            last_exit_check_time_ms=latest_state.last_exit_check_time_ms,
            last_information_time_ms=latest_state.last_information_time_ms,
            last_preflight_time_ms=latest_state.last_preflight_time_ms,
        )
        self.state_store.save(completed_state)
        self.logger.log(
            f"{self.event_prefix}_bootstrap_completed",
            {
                "activation_time_ms": activation_time_ms,
                "strategy_version": strategy_version,
                "virtual_positions": [asdict(position) for position in final_virtuals],
                "trading_mode": self.settings.trading_mode.value,
                "signal_mode": self.settings.signal_mode.value,
            },
        )
        return final_virtuals

    def close_due_virtual_positions(
        self,
        now_ms: int,
        include_planned: bool = True,
    ) -> list[BootstrapVirtualPosition]:
        closed: list[BootstrapVirtualPosition] = []
        for position in list(self.state_store.load().bootstrap_virtual_positions):
            exit_reason = self._virtual_exit_reason(position, now_ms, include_planned=include_planned)
            if exit_reason is None:
                continue
            exit_price = self._latest_close_or_entry(position.symbol, position.entry_price, now_ms)
            self.state_store.close_bootstrap_virtual_position(
                position.symbol,
                exit_time_ms=now_ms,
                exit_price=exit_price,
                reason=exit_reason,
            )
            self.logger.log(
                f"{self.event_prefix}_bootstrap_virtual_closed",
                {
                    "symbol": position.symbol,
                    "rank": position.rank,
                    "entry_time_ms": position.entry_time_ms,
                    "exit_time_ms": now_ms,
                    "exit_reason": exit_reason,
                    "trading_mode": self.settings.trading_mode.value,
                    "signal_mode": self.settings.signal_mode.value,
                },
            )
            closed.append(position)
        return closed

    def _maybe_open_virtual(
        self,
        signal: PaperSignal,
        real_symbols: set[str],
        strategy_version: str,
    ) -> BootstrapVirtualPosition | None:
        strategy_signal = Top3Signal(
            symbol=signal.symbol,
            rank=signal.rank,
            gain_24h=signal.gain_24h,
            volume_24h_ratio_7d=signal.volume_24h_ratio_7d,
            snapshot_hour_bj=signal.snapshot_hour_bj,
        )
        if signal_rejection_reason(strategy_signal) is not None:
            return None
        state = self.state_store.load()
        if signal.symbol in real_symbols:
            return None
        if state.open_position(signal.symbol) is not None or state.bootstrap_virtual_position(signal.symbol) is not None:
            return None

        leverage = 3 if signal.rank == 1 else leverage_for_signal(strategy_signal)
        if leverage is None:
            return None
        planned_exit = (
            signal.signal_time_ms + RANK1_BOOTSTRAP_HOLD_MS
            if signal.rank == 1
            else planned_exit_time_ms(signal.signal_time_ms, strategy_signal)
        )
        unique_key = f"{strategy_version}-{signal.signal_time_ms}-{signal.symbol}"
        position = BootstrapVirtualPosition(
            source="BOOTSTRAP_VIRTUAL_POSITION",
            strategy_version=strategy_version,
            unique_key=unique_key,
            symbol=signal.symbol,
            rank=signal.rank,
            entry_time_ms=signal.signal_time_ms,
            entry_price=signal.fill_price,
            qty=0.0,
            leverage=leverage,
            planned_exit_time_ms=planned_exit,
            weak_exit_check_time_ms=signal.signal_time_ms + 12 * 60 * 60 * 1000,
            extreme_weak_exit_check_time_ms=signal.signal_time_ms + 4 * 60 * 60 * 1000,
            gain_24h=signal.gain_24h,
            volume_24h_ratio_7d=signal.volume_24h_ratio_7d,
        )
        self.state_store.save(
            TestnetState(
                open_positions=state.open_positions,
                closed_positions=state.closed_positions,
                bootstrap_metadata=state.bootstrap_metadata,
                bootstrap_virtual_positions=[*state.bootstrap_virtual_positions, position],
                closed_bootstrap_virtual_positions=state.closed_bootstrap_virtual_positions,
                last_signal_time_ms=state.last_signal_time_ms,
                last_exit_check_time_ms=state.last_exit_check_time_ms,
                last_information_time_ms=state.last_information_time_ms,
                last_preflight_time_ms=state.last_preflight_time_ms,
            )
        )
        self.logger.log(
            f"{self.event_prefix}_bootstrap_virtual_opened",
            {
                **asdict(position),
                "trading_mode": self.settings.trading_mode.value,
                "signal_mode": self.settings.signal_mode.value,
            },
        )
        return position

    def _virtual_exit_reason(
        self,
        position: BootstrapVirtualPosition,
        now_ms: int,
        include_planned: bool,
    ) -> str | None:
        if (
            self.settings.enable_4h_extreme_weak_exit
            and not position.extreme_weak_exit_checked
            and now_ms >= position.extreme_weak_exit_check_time_ms
        ):
            metrics = self._exit_metrics(position, interval="1h", limit=4, now_ms=now_ms)
            if metrics is not None:
                mfe, mae, _close_return = metrics
                if should_exit_extreme_weak_4h(mfe, mae, enabled=True):
                    return "bootstrap_extreme_weak_4h"
                self.state_store.mark_extreme_weak_exit_checked(position.symbol)

        state_position = self.state_store.load().bootstrap_virtual_position(position.symbol)
        if state_position is None:
            return None
        if (
            self.settings.enable_12h_weak_exit
            and not state_position.weak_exit_checked
            and now_ms >= state_position.weak_exit_check_time_ms
        ):
            metrics = self._exit_metrics(state_position, interval="1h", limit=12, now_ms=now_ms)
            if metrics is not None:
                mfe, mae, close_return = metrics
                if should_exit_early_12h(mfe, close_return, mae, enabled=True):
                    return "bootstrap_weak_12h"
                self.state_store.mark_weak_exit_checked(position.symbol)

        latest_position = self.state_store.load().bootstrap_virtual_position(position.symbol)
        if include_planned and latest_position is not None and now_ms >= latest_position.planned_exit_time_ms:
            return "bootstrap_planned"
        return None

    def _exit_metrics(
        self,
        position: BootstrapVirtualPosition,
        interval: str,
        limit: int,
        now_ms: int,
    ) -> tuple[float, float, float] | None:
        klines = self.market_client.klines(
            position.symbol,
            interval=interval,
            limit=limit,
            end_time_ms=now_ms,
        )
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

    def _latest_close_or_entry(self, symbol: str, fallback: float, now_ms: int) -> float:
        try:
            klines = self.market_client.klines(symbol, interval="1h", limit=1, end_time_ms=now_ms)
        except Exception:
            return fallback
        completed = [kline for kline in klines if kline.close_time_ms < now_ms]
        return completed[-1].close if completed else fallback


def bootstrap_signal_times(start_time_ms: int, end_time_ms: int) -> list[int]:
    start_dt = datetime.fromtimestamp(start_time_ms / 1000, tz=BEIJING_TZ)
    end_dt = datetime.fromtimestamp(end_time_ms / 1000, tz=BEIJING_TZ)
    day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    times: list[int] = []
    while day < end_dt:
        for hour in (0, 8):
            candidate = day.replace(hour=hour)
            candidate_ms = int(candidate.timestamp() * 1000)
            if start_time_ms <= candidate_ms < end_time_ms:
                times.append(candidate_ms)
        day += timedelta(days=1)
    return sorted(times)


def _rank_entries_at(
    signal_time_ms: int,
    one_hour_by_symbol: dict[str, list[Kline]],
    four_hour_by_symbol: dict[str, list[Kline]],
) -> list[_HistoricalEntry]:
    rows: list[_HistoricalEntry] = []
    for symbol, klines in one_hour_by_symbol.items():
        gain = _rolling_24h_gain(klines, signal_time_ms)
        if gain is None:
            continue
        price = _latest_completed_close(klines, signal_time_ms)
        if price is None:
            continue
        volume_ratio = _volume_ratio_at(four_hour_by_symbol.get(symbol, []), signal_time_ms)
        rows.append(
            _HistoricalEntry(
                symbol=symbol,
                rank=0,
                gain_24h=gain,
                volume_24h_ratio_7d=volume_ratio,
                fill_price=price,
            )
        )
    rows.sort(key=lambda item: item.gain_24h, reverse=True)
    return [
        _HistoricalEntry(
            symbol=row.symbol,
            gain_24h=row.gain_24h,
            volume_24h_ratio_7d=row.volume_24h_ratio_7d,
            fill_price=row.fill_price,
            rank=index + 1,
        )
        for index, row in enumerate(rows[:3])
    ]


@dataclass(frozen=True)
class _HistoricalEntry:
    symbol: str
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float | None
    fill_price: float


def _rolling_24h_gain(klines: list[Kline], signal_time_ms: int) -> float | None:
    completed = sorted(
        [kline for kline in klines if kline.close_time_ms < signal_time_ms],
        key=lambda item: item.open_time_ms,
    )
    if len(completed) < 24:
        return None
    latest_24 = completed[-24:]
    first_open = latest_24[0].open
    if first_open == 0:
        return None
    return latest_24[-1].close / first_open - 1


def _latest_completed_close(klines: list[Kline], signal_time_ms: int) -> float | None:
    completed = sorted(
        [kline for kline in klines if kline.close_time_ms < signal_time_ms],
        key=lambda item: item.open_time_ms,
    )
    return completed[-1].close if completed else None


def _volume_ratio_at(klines: list[Kline], signal_time_ms: int) -> float | None:
    completed_volumes = [
        kline.volume
        for kline in sorted(klines, key=lambda item: item.open_time_ms)
        if kline.close_time_ms < signal_time_ms
    ]
    return volume_24h_ratio_7d(completed_volumes)


def _bj_snapshot_hour(timestamp_ms: int) -> str:
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
    return f"{dt.hour:02d}:{dt.minute:02d}"
