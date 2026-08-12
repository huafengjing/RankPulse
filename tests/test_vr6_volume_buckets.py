import numpy as np
import pandas as pd

from scripts.research_vr20_volume_buckets import FOUR_HOUR_MS, HOUR_MS, aggregate_4h, feature_at_signal
from scripts.research_vr6_volume_buckets import as_vr6_research_frame


def test_vr6_denominator_uses_previous_six_and_excludes_numerator() -> None:
    rows = 32
    frame = pd.DataFrame(
        {
            "open_time": np.arange(rows) * HOUR_MS,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "quote_volume": np.arange(1, rows + 1, dtype=float),
        }
    )
    bars, _ = aggregate_4h(frame, 0, (rows - 1) * HOUR_MS)
    signal_time = 8 * FOUR_HOUR_MS
    feature = feature_at_signal("AAAUSDT", signal_time, {"AAAUSDT": bars})
    current_start = 7 * FOUR_HOUR_MS
    previous_six = bars.loc[FOUR_HOUR_MS:6 * FOUR_HOUR_MS, "quote_asset_volume_4h"]
    assert feature["latest_completed_4h_start_ms"] == current_start
    assert feature["median_previous_6_4h_quote_volume"] == previous_six.median()
    assert feature["volume_ratio_4h_6"] == bars.loc[current_start, "quote_asset_volume_4h"] / previous_six.median()


def test_vr6_alias_does_not_mutate_source_vr20_columns() -> None:
    source = pd.DataFrame(
        {
            "vr20_bucket": ["B6"], "vr20_status": ["available"], "volume_ratio_4h_20": [6.0],
            "vr6_bucket": ["B1"], "vr6_status": ["available"], "volume_ratio_4h_6": [0.5],
        }
    )
    alias = as_vr6_research_frame(source)
    assert alias.loc[0, "vr20_bucket"] == "B1"
    assert alias.loc[0, "volume_ratio_4h_20"] == 0.5
    assert source.loc[0, "vr20_bucket"] == "B6"
    assert source.loc[0, "volume_ratio_4h_20"] == 6.0
