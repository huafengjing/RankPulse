from scripts.research_rank1_rank3_drop_buckets import focused_drop_bucket


def test_focused_drop_bucket_boundaries() -> None:
    assert focused_drop_bucket(-0.01) is None
    assert focused_drop_bucket(0.0) == "0~20%"
    assert focused_drop_bucket(19.999) == "0~20%"
    assert focused_drop_bucket(20.0) == "20~40%"
    assert focused_drop_bucket(39.999) == "20~40%"
    assert focused_drop_bucket(40.0) == "40~60%"
    assert focused_drop_bucket(59.999) == "40~60%"
    assert focused_drop_bucket(60.0) is None
