from __future__ import annotations

import pandas as pd


def add_ema20_and_vwap(klines: pd.DataFrame) -> pd.DataFrame:
    frame = klines.sort_values(["symbol", "open_time"]).copy()
    frame["ema20"] = frame.groupby("symbol")["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    pv = typical * frame["volume"]
    frame["session_pv"] = pv.groupby([frame["symbol"], pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.date]).cumsum()
    frame["session_volume"] = frame["volume"].groupby([frame["symbol"], pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.date]).cumsum()
    frame["vwap"] = frame["session_pv"] / frame["session_volume"]
    frame["distance_to_ema20_pct"] = frame["close"] / frame["ema20"] - 1.0
    frame["distance_to_vwap_pct"] = frame["close"] / frame["vwap"] - 1.0
    return frame.drop(columns=["session_pv", "session_volume"])
