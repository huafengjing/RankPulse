from __future__ import annotations

import pandas as pd
import pytest

from scripts.research_drop_top3_short_edge import (
    FEE_RATE,
    HOUR_MS,
    build_signals,
    drop_bucket,
    path_excursions,
    simulate_trades,
    snapshot_rankings,
    trade_return,
)


def frame(rows: list[tuple[int, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close"]).set_index("open_time", drop=False)


def test_snapshot_uses_only_completed_bar_before_entry() -> None:
    t = pd.Timestamp("2026-01-02 00:00:00", tz="UTC")
    signal = int(t.timestamp() * 1000)
    prior = signal - 25 * HOUR_MS
    current = signal - HOUR_MS
    future_entry_bar = signal
    klines = {
        "AUSDT": frame([(prior, 100, 100, 100, 100), (current, 80, 80, 80, 80), (future_entry_bar, 1, 1, 1, 1)]),
        "BUSDT": frame([(prior, 100, 100, 100, 100), (current, 90, 90, 90, 90), (future_entry_bar, 1000, 1000, 1000, 1000)]),
    }
    ranked = snapshot_rankings(signal, klines).sort_values("change_24h")
    assert ranked.iloc[0]["symbol"] == "AUSDT"
    assert ranked.iloc[0]["change_24h"] == pytest.approx(-0.20)


def test_short_return_charges_both_side_fees() -> None:
    gross, fees, net = trade_return("short", 100, 80)
    assert gross == pytest.approx(0.20)
    assert fees == pytest.approx(FEE_RATE + FEE_RATE * 0.8)
    assert net == pytest.approx(0.1982)


def test_short_mfe_and_mae_signs() -> None:
    path = frame([(0, 100, 110, 70, 90)])
    mfe, mae = path_excursions("short", path, 100)
    assert mfe == pytest.approx(30)
    assert mae == pytest.approx(-10)


def test_one_x_short_is_capped_at_margin_loss_when_price_doubles() -> None:
    start = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)
    rows = [(start + hour * HOUR_MS, 100, 210 if hour == 1 else 100, 90, 100) for hour in range(25)]
    klines = {"AUSDT": frame(rows)}
    signals = pd.DataFrame(
        [{"direction": "short", "signal_time_ms": start, "symbol": "AUSDT", "rank": 1, "drop_bucket": "10~20%"}]
    )
    trades, _ = simulate_trades(signals, 1, klines, "short")
    assert len(trades) == 1
    assert bool(trades.iloc[0]["liquidated"])
    assert trades.iloc[0]["pnl_usdt"] == pytest.approx(-100)


@pytest.mark.parametrize(
    ("drop", "expected"),
    [(0.0, "no_drop_or_gain"), (0.05, "0~10%"), (0.10, "10~20%"), (0.40, "40~60%"), (0.80, ">=80%")],
)
def test_drop_buckets_keep_unfiltered_tail(drop: float, expected: str) -> None:
    assert drop_bucket(drop) == expected
