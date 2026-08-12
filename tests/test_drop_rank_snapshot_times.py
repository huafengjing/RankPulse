from scripts.research_drop_rank_snapshot_times import BJ_SLOTS, time_configurations, utc_hour_for_beijing


def test_beijing_snapshot_hour_mapping() -> None:
    assert [utc_hour_for_beijing(slot) for slot in BJ_SLOTS] == [16, 20, 0, 4, 8, 12]


def test_time_configuration_counts_and_uniqueness() -> None:
    configurations = time_configurations()
    singles = [item for item in configurations if item[2] == "single"]
    pairs = [item for item in configurations if item[2] == "pair"]
    assert len(configurations) == 21
    assert len(singles) == 6
    assert len(pairs) == 15
    assert len({item[0] for item in pairs}) == 15
