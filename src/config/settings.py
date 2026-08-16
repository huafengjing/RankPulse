from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from src.config.env import load_env_file
from src.config.modes import SignalMode, TradingMode


class LiveTradingDisabledError(RuntimeError):
    pass


LIVE_TRADING_CONFIRMATION_PHRASE = "I_UNDERSTAND_THIS_IS_REAL_MONEY"


@dataclass(frozen=True)
class AppSettings:
    trading_mode: TradingMode = TradingMode.PAPER
    signal_mode: SignalMode = SignalMode.PRODUCTION
    position_margin_usdt: float = 100.0
    max_open_positions: int = 10
    signal_test_interval_minutes: int = 5
    enable_12h_weak_exit: bool = True
    enable_4h_extreme_weak_exit: bool = True
    test_extreme_weak_exit_after_minutes: int = 5
    test_weak_exit_after_minutes: int = 15
    test_planned_exit_after_minutes: int = 60
    enforce_safety_lock: bool = True
    allow_live_trading: bool = False
    live_order_confirmation: str = ""
    binance_testnet_api_key: str = ""
    binance_testnet_api_secret: str = ""
    binance_live_api_key: str = ""
    binance_live_api_secret: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    top3_regime_enabled: bool = False
    top3_regime_context_auto_generate: bool = True
    top3_regime_context_path: str = ""
    rank1_bootstrap_enabled: bool = True
    rank1_activation_time_ms: int | None = None
    rank1_bootstrap_strategy_version: str = "rank1_candidate_v1"

    @classmethod
    def from_env_file(cls, path: str = ".env") -> AppSettings:
        values = {**load_env_file(path), **os.environ}
        return cls.from_env(values)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> AppSettings:
        return cls(
            trading_mode=TradingMode(env.get("TRADING_MODE", TradingMode.PAPER.value).lower()),
            signal_mode=SignalMode(env.get("SIGNAL_MODE", SignalMode.PRODUCTION.value).lower()),
            position_margin_usdt=float(env.get("POSITION_MARGIN_USDT", "100")),
            max_open_positions=int(env.get("MAX_OPEN_POSITIONS", "10")),
            signal_test_interval_minutes=int(env.get("SIGNAL_TEST_INTERVAL_MINUTES", "5")),
            enable_12h_weak_exit=_bool(env.get("ENABLE_12H_WEAK_EXIT", "true")),
            enable_4h_extreme_weak_exit=_bool(env.get("ENABLE_4H_EXTREME_WEAK_EXIT", "true")),
            test_extreme_weak_exit_after_minutes=int(env.get("TEST_EXTREME_WEAK_EXIT_AFTER_MINUTES", "5")),
            test_weak_exit_after_minutes=int(env.get("TEST_WEAK_EXIT_AFTER_MINUTES", "15")),
            test_planned_exit_after_minutes=int(env.get("TEST_PLANNED_EXIT_AFTER_MINUTES", "60")),
            enforce_safety_lock=_bool(env.get("ENFORCE_SAFETY_LOCK", "true")),
            allow_live_trading=_bool(env.get("ALLOW_LIVE_TRADING", "false")),
            live_order_confirmation=env.get("LIVE_ORDER_CONFIRMATION", ""),
            binance_testnet_api_key=env.get("BINANCE_TESTNET_API_KEY", ""),
            binance_testnet_api_secret=env.get("BINANCE_TESTNET_API_SECRET", ""),
            binance_live_api_key=env.get("BINANCE_LIVE_API_KEY", ""),
            binance_live_api_secret=env.get("BINANCE_LIVE_API_SECRET", ""),
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
            top3_regime_enabled=_bool(env.get("RANKPULSE_REGIME_ENABLED", env.get("TOP3_REGIME_ENABLED", "false"))),
            top3_regime_context_auto_generate=_bool(
                env.get("RANKPULSE_REGIME_CONTEXT_AUTO_GENERATE", env.get("TOP3_REGIME_CONTEXT_AUTO_GENERATE", "true"))
            ),
            top3_regime_context_path=env.get("RANKPULSE_REGIME_CONTEXT_PATH", env.get("TOP3_REGIME_CONTEXT_PATH", "")),
            rank1_bootstrap_enabled=_bool(env.get("RANK1_BOOTSTRAP_ENABLED", "true")),
            rank1_activation_time_ms=_optional_int(env.get("RANK1_ACTIVATION_TIME_MS")),
            rank1_bootstrap_strategy_version=env.get("RANK1_BOOTSTRAP_STRATEGY_VERSION", "rank1_candidate_v1"),
        )

    def assert_can_run(self) -> None:
        if self.trading_mode == TradingMode.LIVE:
            if not self.allow_live_trading:
                raise LiveTradingDisabledError("Live trading is disabled. Set ALLOW_LIVE_TRADING=true only for approved small-capital live runs.")
            if self.live_order_confirmation != LIVE_TRADING_CONFIRMATION_PHRASE:
                raise LiveTradingDisabledError(
                    "LIVE_ORDER_CONFIRMATION must exactly match "
                    f"{LIVE_TRADING_CONFIRMATION_PHRASE!r}."
                )
        if self.enforce_safety_lock:
            raise LiveTradingDisabledError(
                "ENFORCE_SAFETY_LOCK is active. Set ENFORCE_SAFETY_LOCK=false "
                "in .env to confirm you understand the risks."
            )


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)
