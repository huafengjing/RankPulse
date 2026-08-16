from __future__ import annotations

from dataclasses import asdict, dataclass

from src.research.rankpulse_strategy_rules import (
    ENABLE_12H_WEAK_EXIT,
    ENABLE_4H_EXTREME_WEAK_EXIT,
    Top3Signal,
    Top3RegimeContext,
    extreme_weak_exit_time_ms,
    is_duplicate_position,
    is_trade_signal,
    leverage_for_signal,
    planned_exit_time_ms,
    planned_hold_days_for_signal,
    should_exit_extreme_weak_4h,
    should_exit_early_12h,
    signal_rejection_reason,
)


LIVE_ORDER_CONFIRMATION_PHRASE = "I_UNDERSTAND_LIVE_ORDERS"


class LiveTradingDisabledError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperTradingConfig:
    margin_usdt_per_trade: float = 100.0
    enable_12h_weak_exit: bool = ENABLE_12H_WEAK_EXIT
    enable_4h_extreme_weak_exit: bool = ENABLE_4H_EXTREME_WEAK_EXIT
    live_trading_enabled: bool = False
    live_order_confirmation: str | None = None


@dataclass(frozen=True)
class PaperSignal:
    symbol: str
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float | None
    snapshot_hour_bj: str
    signal_time_ms: int
    fill_price: float
    regime_context: Top3RegimeContext | None = None


@dataclass(frozen=True)
class PaperPosition:
    symbol: str
    side: str
    entry_time_ms: int
    entry_price: float
    rank: int
    gain_24h: float
    volume_24h_ratio_7d: float | None
    leverage: int
    margin_usdt: float
    planned_exit_time_ms: int
    extreme_weak_exit_check_time_ms: int
    extreme_weak_exit_checked: bool = False
    weak_exit_checked: bool = False


@dataclass(frozen=True)
class PaperWeakExitCheck:
    symbol: str
    check_time_ms: int
    fill_price: float
    mfe_12h: float
    close_return_12h: float
    mae_12h: float


@dataclass(frozen=True)
class PaperExtremeWeakExitCheck:
    symbol: str
    check_time_ms: int
    fill_price: float
    mfe_4h: float
    mae_4h: float


@dataclass(frozen=True)
class PaperTradeExit:
    symbol: str
    entry_time_ms: int
    exit_time_ms: int
    exit_price: float
    exit_reason: str


class PaperTradingEngine:
    def __init__(self, config: PaperTradingConfig | None = None) -> None:
        self.config = config or PaperTradingConfig()
        self._open_positions: dict[str, PaperPosition] = {}
        self.closed_trades: list[PaperTradeExit] = []

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, object],
        config: PaperTradingConfig | None = None,
    ) -> PaperTradingEngine:
        engine = cls(config=config)
        open_positions = snapshot.get("open_positions", [])
        if isinstance(open_positions, list):
            for raw_position in open_positions:
                if isinstance(raw_position, dict):
                    raw_position.setdefault(
                        "extreme_weak_exit_check_time_ms",
                        extreme_weak_exit_time_ms(int(raw_position["entry_time_ms"])),
                    )
                    raw_position.setdefault("extreme_weak_exit_checked", False)
                    position = PaperPosition(**raw_position)
                    engine._open_positions[position.symbol] = position

        closed_trades = snapshot.get("closed_trades", [])
        if isinstance(closed_trades, list):
            for raw_trade in closed_trades:
                if isinstance(raw_trade, dict):
                    engine.closed_trades.append(PaperTradeExit(**raw_trade))
        return engine

    def snapshot(self) -> dict[str, object]:
        return {
            "open_positions": [asdict(position) for position in self.open_positions()],
            "closed_trades": [asdict(trade_exit) for trade_exit in self.closed_trades],
        }

    def on_signal(self, signal: PaperSignal) -> PaperPosition | None:
        strategy_signal = Top3Signal(
            symbol=signal.symbol,
            rank=signal.rank,
            gain_24h=signal.gain_24h,
            volume_24h_ratio_7d=signal.volume_24h_ratio_7d,
            snapshot_hour_bj=signal.snapshot_hour_bj,
        )
        reason = signal_rejection_reason(strategy_signal)
        if reason is not None:
            print(f"[跳过] {signal.symbol}: {reason}")
            return None

        open_until_by_symbol = {
            symbol: position.planned_exit_time_ms
            for symbol, position in self._open_positions.items()
        }
        if is_duplicate_position(signal.symbol, signal.signal_time_ms, open_until_by_symbol):
            return None

        leverage = leverage_for_signal(strategy_signal, signal.regime_context)
        if leverage is None:
            return None

        position = PaperPosition(
            symbol=signal.symbol,
            side="LONG",
            entry_time_ms=signal.signal_time_ms,
            entry_price=signal.fill_price,
            rank=signal.rank,
            gain_24h=signal.gain_24h,
            volume_24h_ratio_7d=signal.volume_24h_ratio_7d,
            leverage=leverage,
            margin_usdt=self.config.margin_usdt_per_trade,
            planned_exit_time_ms=planned_exit_time_ms(signal.signal_time_ms, strategy_signal),
            extreme_weak_exit_check_time_ms=extreme_weak_exit_time_ms(signal.signal_time_ms),
        )
        self._open_positions[position.symbol] = position
        return position

    def on_extreme_weak_exit_check(self, check: PaperExtremeWeakExitCheck) -> PaperTradeExit | None:
        position = self._open_positions.get(check.symbol)
        if position is None:
            return None

        should_exit = should_exit_extreme_weak_4h(
            mfe_4h=check.mfe_4h,
            mae_4h=check.mae_4h,
            enabled=self.config.enable_4h_extreme_weak_exit,
        )
        if not should_exit:
            self._open_positions[position.symbol] = PaperPosition(
                **{**asdict(position), "extreme_weak_exit_checked": True}
            )
            return None

        return self._close_position(
            position=position,
            exit_time_ms=check.check_time_ms,
            exit_price=check.fill_price,
            exit_reason="extreme_weak_4h",
        )

    def on_weak_exit_check(self, check: PaperWeakExitCheck) -> PaperTradeExit | None:
        position = self._open_positions.get(check.symbol)
        if position is None:
            return None

        should_exit = should_exit_early_12h(
            mfe_12h=check.mfe_12h,
            close_return_12h=check.close_return_12h,
            mae_12h=check.mae_12h,
            enabled=self.config.enable_12h_weak_exit,
        )
        if not should_exit:
            self._open_positions[position.symbol] = PaperPosition(
                **{**asdict(position), "weak_exit_checked": True}
            )
            return None

        return self._close_position(
            position=position,
            exit_time_ms=check.check_time_ms,
            exit_price=check.fill_price,
            exit_reason="weak_12h",
        )

    def on_planned_exit(self, symbol: str, exit_time_ms: int, fill_price: float) -> PaperTradeExit | None:
        position = self._open_positions.get(symbol)
        if position is None or exit_time_ms < position.planned_exit_time_ms:
            return None

        return self._close_position(
            position=position,
            exit_time_ms=exit_time_ms,
            exit_price=fill_price,
            exit_reason=f"planned_{planned_hold_days_for_signal(position_strategy_signal(position))}d",
        )

    def open_position(self, symbol: str) -> PaperPosition | None:
        return self._open_positions.get(symbol)

    def open_positions(self) -> list[PaperPosition]:
        return list(self._open_positions.values())

    def assert_live_orders_allowed(self) -> bool:
        if not self.config.live_trading_enabled:
            raise LiveTradingDisabledError("Live order placement is disabled by default.")
        if self.config.live_order_confirmation != LIVE_ORDER_CONFIRMATION_PHRASE:
            raise LiveTradingDisabledError("Live order placement requires second confirmation.")
        return True

    def _close_position(
        self,
        position: PaperPosition,
        exit_time_ms: int,
        exit_price: float,
        exit_reason: str,
    ) -> PaperTradeExit:
        trade_exit = PaperTradeExit(
            symbol=position.symbol,
            entry_time_ms=position.entry_time_ms,
            exit_time_ms=exit_time_ms,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        self.closed_trades.append(trade_exit)
        del self._open_positions[position.symbol]
        return trade_exit


def position_strategy_signal(position: PaperPosition) -> Top3Signal:
    return Top3Signal(
        symbol=position.symbol,
        rank=position.rank,
        gain_24h=position.gain_24h,
        volume_24h_ratio_7d=position.volume_24h_ratio_7d,
    )
