import pandas as pd

from scripts.research_locked_leverage_holding_days import select_configuration


def test_select_configuration_uses_one_frozen_path_per_candidate() -> None:
    rows = []
    for candidate_id in ["A", "B", "C"]:
        for holding_days in [1, 2, 3]:
            for leverage in [3, 5]:
                rows.append({"candidate_id": candidate_id, "holding_days": holding_days, "leverage": leverage})
    outcomes = pd.DataFrame(rows)

    selected = select_configuration(
        outcomes,
        {"A": 1, "B": 2, "C": 3},
        {"A": 5, "B": 3, "C": 3},
    )

    assert selected[["candidate_id", "holding_days", "leverage"]].to_dict("records") == [
        {"candidate_id": "A", "holding_days": 1, "leverage": 5},
        {"candidate_id": "B", "holding_days": 2, "leverage": 3},
        {"candidate_id": "C", "holding_days": 3, "leverage": 3},
    ]
