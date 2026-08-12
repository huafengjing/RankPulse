import numpy as np
import pandas as pd

from scripts.research_vr20_volume_buckets import FOUR_HOUR_MS, HOUR_MS, aggregate_4h, feature_at_signal, filter_blocks, spearman_correlation, vr_bucket


def make_hourly(rows: int = 88) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": np.arange(rows) * HOUR_MS,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.0),
            "quote_volume": np.arange(1, rows + 1, dtype=float),
        }
    )


def test_vr_bucket_boundaries_are_fixed() -> None:
    assert [vr_bucket(value) for value in [np.nan, 0.7499, 0.75, 1.25, 2.0, 3.0, 5.0]] == ["MISSING", "B1", "B2", "B3", "B4", "B5", "B6"]


def test_four_hour_aggregation_uses_four_contiguous_hours_and_quote_volume() -> None:
    frame = make_hourly(8)
    bars, audit = aggregate_4h(frame, 0, 7 * HOUR_MS)
    assert audit["valid_four_hour_rows"] == 2
    assert bars.loc[0, "quote_asset_volume_4h"] == 10
    assert bars.loc[FOUR_HOUR_MS, "quote_asset_volume_4h"] == 26
    broken = frame[frame.open_time.ne(2 * HOUR_MS)]
    broken_bars, _ = aggregate_4h(broken, 0, 7 * HOUR_MS)
    assert not bool(broken_bars.loc[0, "valid_4h"])


def test_signal_uses_previous_completed_bar_and_excludes_numerator_from_median() -> None:
    frame = make_hourly(88)
    bars, _ = aggregate_4h(frame, 0, 87 * HOUR_MS)
    signal_time = 22 * FOUR_HOUR_MS
    feature = feature_at_signal("AAAUSDT", signal_time, {"AAAUSDT": bars})
    current_start = 21 * FOUR_HOUR_MS
    expected_previous = bars.loc[FOUR_HOUR_MS:20 * FOUR_HOUR_MS, "quote_asset_volume_4h"]
    assert feature["latest_completed_4h_start_ms"] == current_start
    assert feature["latest_completed_4h_end_ms"] == signal_time
    assert feature["median_previous_20_4h_quote_volume"] == expected_previous.median()
    assert feature["current_4h_quote_volume"] == bars.loc[current_start, "quote_asset_volume_4h"]


def test_missing_vr20_is_never_filtered() -> None:
    for version in ["Exclude_B1", "Exclude_VR20_GE_1_25", "Exclude_VR20_GE_5"]:
        assert not filter_blocks(version, np.nan, "MISSING", "unavailable")


def test_spearman_correlation_uses_average_ranks_without_scipy() -> None:
    assert spearman_correlation(pd.Series([1.0, 2.0, 3.0]), pd.Series([10.0, 20.0, 30.0])) == 1.0
    assert spearman_correlation(pd.Series([1.0, 2.0, 3.0]), pd.Series([30.0, 20.0, 10.0])) == -1.0
