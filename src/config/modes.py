from __future__ import annotations

from enum import StrEnum


class TradingMode(StrEnum):
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


class SignalMode(StrEnum):
    PRODUCTION = "production"
    TEST_FAST = "test_fast"
