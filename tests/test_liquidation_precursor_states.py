import pandas as pd

from scripts.research_liquidation_precursor_states import add_episode_states, prior_pnl_bucket


def test_prior_pnl_buckets_are_fixed() -> None:
    assert [prior_pnl_bucket(value) for value in [-101, -100, -0.01, 0, 99.99, 100, 200, 200.01]] == [
        "< -100", "-100 to 0", "-100 to 0", "0 to +100", "0 to +100", "+100 to +200", "+100 to +200", "> +200"
    ]


def test_episode_prior_state_excludes_current_and_future_trades() -> None:
    day = 86_400_000
    trades = pd.DataFrame(
        [
            {"signal_key": "1", "symbol": "X", "candidate_id": "A", "entry_time_ms": 0, "exit_time_ms": day, "reentry_gap_days": float("nan"), "pnl_usdt": 40.0, "liquidated": False},
            {"signal_key": "2", "symbol": "X", "candidate_id": "B", "entry_time_ms": 2 * day, "exit_time_ms": 3 * day, "reentry_gap_days": 1.0, "pnl_usdt": -100.0, "liquidated": True},
            {"signal_key": "3", "symbol": "X", "candidate_id": "C", "entry_time_ms": 4 * day, "exit_time_ms": 5 * day, "reentry_gap_days": 1.0, "pnl_usdt": 70.0, "liquidated": False},
        ]
    )
    enriched, assigned = add_episode_states(trades)
    ten_day = assigned[10].sort_values("entry_time_ms")

    assert ten_day.episode_prior_pnl.tolist() == [0.0, 40.0, -60.0]
    assert ten_day.prior_episode_liquidations.tolist() == [0, 0, 1]
    assert enriched.episode_prior_pnl_10d.tolist() == [0.0, 40.0, -60.0]
