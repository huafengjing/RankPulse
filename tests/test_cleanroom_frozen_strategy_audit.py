from __future__ import annotations

import pandas as pd

from scripts.audit_frozen_strategy_independent import (
    DAY_MS,
    beijing_slot_to_utc,
    build_universe_and_rankings,
    replay_rule_2,
)


def test_beijing_snapshot_boundaries_use_explicit_utc_plus_8() -> None:
    assert beijing_slot_to_utc("00:00") == (16, -1)
    assert beijing_slot_to_utc("04:00") == (20, -1)
    assert beijing_slot_to_utc("08:00") == (0, 0)
    assert beijing_slot_to_utc("20:00") == (12, 0)


def outcome(entry_time: int, liquidated: bool, exit_time: int, candidate: str = "A") -> dict:
    return {
        "signal_time_ms": entry_time,
        "entry_time_ms": entry_time,
        "candidate": candidate,
        "symbol": "TESTUSDT",
        "rank": 1,
        "outcome_available": True,
        "actual_exit_time_ms": exit_time,
        "net_pnl_usdt": -100.0 if liquidated else 1.0,
        "liquidated": liquidated,
        "exit_reason": "liquidation_5x_short" if liquidated else "fixed_exit",
    }


def test_rule_2_boundaries_and_blocked_signal_does_not_reset_state() -> None:
    first = outcome(0, True, DAY_MS)
    at_5d = outcome(6 * DAY_MS, False, 7 * DAY_MS)
    just_over_5d = outcome(6 * DAY_MS + 1, False, 7 * DAY_MS + 1)
    at_30d = outcome(31 * DAY_MS, False, 32 * DAY_MS)
    over_30d = outcome(31 * DAY_MS + 1, False, 32 * DAY_MS + 1)

    replay = replay_rule_2(pd.DataFrame([first, at_5d]))
    assert replay.iloc[1].executed

    replay = replay_rule_2(pd.DataFrame([first, just_over_5d, at_30d, over_30d]))
    assert replay.iloc[1].rule_2_triggered
    assert replay.iloc[2].rule_2_triggered
    assert replay.iloc[3].executed
    assert replay.iloc[2].previous_actual_liquidation_time_ms == DAY_MS


def test_ranking_uses_only_completed_snapshot_and_strict_24h_closes() -> None:
    snapshot = 30 * DAY_MS
    current_open = snapshot - 3_600_000
    prior_open = current_open - DAY_MS
    future_open = snapshot
    frame = pd.DataFrame(
        [
            {"open_time": prior_open, "close": 100.0},
            {"open_time": current_open, "close": 80.0},
            {"open_time": future_open, "close": 1.0},
        ]
    ).set_index("open_time", drop=False)
    metadata = {
        "TESTUSDT": {
            "symbol": "TESTUSDT",
            "contractType": "PERPETUAL",
            "quoteAsset": "USDT",
            "onboardDate": 0,
            "deliveryDate": 4_000_000_000_000,
        }
    }
    _, rankings = build_universe_and_rankings([snapshot], {"TESTUSDT": frame}, metadata)
    assert rankings.iloc[0].close_now == 80.0
    assert rankings.iloc[0].close_24h_ago == 100.0
    assert rankings.iloc[0].drop_24h_pct == 20.0


def test_tied_drop_ranking_uses_symbol_ascending() -> None:
    snapshot = 30 * DAY_MS
    current_open = snapshot - 3_600_000
    prior_open = current_open - DAY_MS
    frames = {
        symbol: pd.DataFrame(
            [{"open_time": prior_open, "close": 100.0}, {"open_time": current_open, "close": 90.0}]
        ).set_index("open_time", drop=False)
        for symbol in ["ZZZUSDT", "AAAUSDT"]
    }
    metadata = {
        symbol: {"symbol": symbol, "contractType": "PERPETUAL", "quoteAsset": "USDT", "onboardDate": 0, "deliveryDate": 4_000_000_000_000}
        for symbol in frames
    }
    _, rankings = build_universe_and_rankings([snapshot], frames, metadata)
    assert rankings.symbol.tolist() == ["AAAUSDT", "ZZZUSDT"]

