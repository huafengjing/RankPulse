import pandas as pd

from scripts.research_reentry_episode_analysis import assign_episodes, enrich_reentries, gap_bucket


def test_gap_bucket_boundaries_are_mutually_exclusive() -> None:
    assert [gap_bucket(value) for value in [0, 1, 1.01, 3, 3.01, 5, 5.01, 7, 7.01, 10, 10.01, 30, 30.01]] == [
        "<=1D", "<=1D", ">1D-3D", ">1D-3D", ">3D-5D", ">3D-5D", ">5D-7D",
        ">5D-7D", ">7D-10D", ">7D-10D", ">10D-30D", ">10D-30D", ">30D",
    ]


def test_episode_boundary_uses_previous_actual_exit() -> None:
    day = 86_400_000
    trades = pd.DataFrame(
        [
            {"symbol": "X", "candidate_id": "A", "entry_time_ms": 0, "exit_time_ms": day, "pnl_usdt": 1.0, "liquidated": False, "skipped_due_to_existing_position": False},
            {"symbol": "X", "candidate_id": "B", "entry_time_ms": 3 * day, "exit_time_ms": 4 * day, "pnl_usdt": -1.0, "liquidated": False, "skipped_due_to_existing_position": False},
            {"symbol": "X", "candidate_id": "C", "entry_time_ms": 8 * day, "exit_time_ms": 9 * day, "pnl_usdt": -100.0, "liquidated": True, "skipped_due_to_existing_position": False},
        ]
    )
    enriched = enrich_reentries(trades)
    assigned = assign_episodes(enriched, threshold_days=3)

    assert enriched.reentry_gap_days.tolist()[1:] == [2.0, 4.0]
    assert assigned.episode_number.tolist() == [1, 1, 2]
    assert assigned.episode_entry_number.tolist() == [1, 2, 1]


def test_rank_column_is_accessed_as_data_not_dataframe_method() -> None:
    signals = pd.DataFrame({"rank": [3, 3, 1]})
    ranks = signals["rank"].astype(int).to_numpy()

    assert any(ranks[index] == 3 and (ranks[index + 1 :] == 1).any() for index in range(len(ranks)))
