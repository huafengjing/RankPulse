from __future__ import annotations

from dataclasses import asdict, dataclass

from src.research.rankpulse_strategy_rules import (
    ENABLE_12H_WEAK_EXIT,
    ENABLE_4H_EXTREME_WEAK_EXIT,
    ENABLE_MFE_AGING_PARTIAL_TP,
    MFE_AGING_PARTIAL_TP_RATIO,
    ENABLE_RANK1_24H_WEAK_EXIT,
    Top3Signal,
    Top3RegimeContext,
    extreme_weak_exit_time_ms,
    is_duplicate_position,
    is_trade_signal,
    leverage_for_signal,
    planned_exit_time_ms,
    planned_hold_days_for_signal,
    rank1_weak_exit_time_ms,
    same_symbol_reentry_block_reason,
    should_exit_extreme_weak_4h,
    should_exit_early_12h,
    should_exit_rank1_weak_24h,
    should_trigger_mfe_aging_partial_tp,
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
    enable_rank1_24h_weak_exit: bool = ENABLE_RANK1_24H_WEAK_EXIT
    enable_mfe_aging_partial_tp: bool = ENABLE_MFE_AGING_PARTIAL_TP
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
    rank1_weak_24h_exit_check_time_ms: int
    extreme_weak_exit_checked: bool = False
    weak_exit_checked: bool = False
    rank1_weak_24h_exit_checked: bool = False
    mfe_aging_partial_tp_done: bool = False
    mfe_aging_running_mfe_u: float = 0.0
    mfe_aging_last_high_time_ms: int | None = None
    mfe_aging_partial_tp_time_ms: int | None = None
    mfe_aging_partial_tp_price: float | None = None
    mfe_aging_partial_tp_realized_pnl: float = 0.0


@dataclass(frozen=True)
class PaperWeakExitCheck:
    symbol: str
    check_time_ms: int
    fill_price: float
    mfe_12h: float
    close_return_12h: float
    mae_12h: float


@dataclass(frozen=True)
class PaperRank1Weak24hExitCheck:
    symbol: str
    check_time_ms: int
    fill_price: float
    mfe_24h: float
    close_return_24h: float
    mae_24h: float


@dataclass(frozen=True)
class PaperExtremeWeakExitCheck:
    symbol: str
    check_time_ms: int
    fill_price: float
    mfe_4h: float
    mae_4h: float


@dataclass(frozen=True)
class PaperMfeAgingPartialTpCheck:
    symbol: str
    check_time_ms: int
    fill_price: float
    running_mfe_u: float
    last_high_time_ms: int | None


@dataclass(frozen=True)
class PaperTradeExit:
    symbol: str
    entry_time_ms: int
    exit_time_ms: int
    exit_price: float
    exit_reason: str
    realized_pnl: float = 0.0


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
                    raw_position.setdefault(
                        "rank1_weak_24h_exit_check_time_ms",
                        rank1_weak_exit_time_ms(int(raw_position["entry_time_ms"])),
                    )
                    raw_position.setdefault("extreme_weak_exit_checked", False)
                    raw_position.setdefault("rank1_weak_24h_exit_checked", False)
                    raw_position.setdefault("mfe_aging_partial_tp_done", False)
                    raw_position.setdefault("mfe_aging_running_mfe_u", 0.0)
                    raw_position.setdefault("mfe_aging_last_high_time_ms", None)
                    raw_position.setdefault("mfe_aging_partial_tp_time_ms", None)
                    raw_position.setdefault("mfe_aging_partial_tp_price", None)
                    raw_position.setdefault("mfe_aging_partial_tp_realized_pnl", 0.0)
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
        if same_symbol_reentry_block_reason(
            signal.symbol,
            signal.signal_time_ms,
            self._last_entry_time_by_symbol(),
            self._last_pnl_by_symbol(),
        ):
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
            rank1_weak_24h_exit_check_time_ms=rank1_weak_exit_time_ms(signal.signal_time_ms),
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

    def on_rank1_weak_24h_exit_check(self, check: PaperRank1Weak24hExitCheck) -> PaperTradeExit | None:
        position = self._open_positions.get(check.symbol)
        if position is None:
            return None

        should_exit = should_exit_rank1_weak_24h(
            rank=position.rank,
            mfe_24h=check.mfe_24h,
            close_return_24h=check.close_return_24h,
            enabled=self.config.enable_rank1_24h_weak_exit,
        )
        if not should_exit:
            self._open_positions[position.symbol] = PaperPosition(
                **{**asdict(position), "rank1_weak_24h_exit_checked": True}
            )
            return None

        return self._close_position(
            position=position,
            exit_time_ms=check.check_time_ms,
            exit_price=check.fill_price,
            exit_reason="weak_24h_rank1_rank2",
        )

    def on_mfe_aging_partial_tp_check(self, check: PaperMfeAgingPartialTpCheck) -> PaperPosition | None:
        position = self._open_positions.get(check.symbol)
        if position is None or position.mfe_aging_partial_tp_done:
            return None

        updated_fields = {
            **asdict(position),
            "mfe_aging_running_mfe_u": check.running_mfe_u,
            "mfe_aging_last_high_time_ms": check.last_high_time_ms,
        }
        if not should_trigger_mfe_aging_partial_tp(
            running_mfe_u=check.running_mfe_u,
            last_mfe_high_time_ms=check.last_high_time_ms,
            now_ms=check.check_time_ms,
            enabled=self.config.enable_mfe_aging_partial_tp,
        ):
            self._open_positions[position.symbol] = PaperPosition(**updated_fields)
            return None

        partial_pnl = self._position_pnl(position, check.fill_price) * MFE_AGING_PARTIAL_TP_RATIO
        updated = PaperPosition(
            **{
                **updated_fields,
                "margin_usdt": position.margin_usdt * (1.0 - MFE_AGING_PARTIAL_TP_RATIO),
                "mfe_aging_partial_tp_done": True,
                "mfe_aging_partial_tp_time_ms": check.check_time_ms,
                "mfe_aging_partial_tp_price": check.fill_price,
                "mfe_aging_partial_tp_realized_pnl": position.mfe_aging_partial_tp_realized_pnl + partial_pnl,
            }
        )
        self._open_positions[position.symbol] = updated
        return updated

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

    def _last_entry_time_by_symbol(self) -> dict[str, int]:
        last_entry: dict[str, int] = {}
        for trade_exit in self.closed_trades:
            last_entry[trade_exit.symbol] = max(
                last_entry.get(trade_exit.symbol, 0),
                trade_exit.entry_time_ms,
            )
        return last_entry

    def _last_pnl_by_symbol(self) -> dict[str, float]:
        last: dict[str, tuple[int, float]] = {}
        for trade_exit in self.closed_trades:
            current = last.get(trade_exit.symbol)
            if current is None or trade_exit.entry_time_ms > current[0]:
                last[trade_exit.symbol] = (trade_exit.entry_time_ms, trade_exit.realized_pnl)
        return {symbol: pnl for symbol, (_entry_time, pnl) in last.items()}

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
            realized_pnl=position.mfe_aging_partial_tp_realized_pnl + self._position_pnl(position, exit_price),
        )
        self.closed_trades.append(trade_exit)
        del self._open_positions[position.symbol]
        return trade_exit

    def _position_pnl(self, position: PaperPosition, exit_price: float) -> float:
        return (exit_price - position.entry_price) * position.margin_usdt * position.leverage / position.entry_price


def position_strategy_signal(position: PaperPosition) -> Top3Signal:
    return Top3Signal(
        symbol=position.symbol,
        rank=position.rank,
        gain_24h=position.gain_24h,
        volume_24h_ratio_7d=position.volume_24h_ratio_7d,
    )
