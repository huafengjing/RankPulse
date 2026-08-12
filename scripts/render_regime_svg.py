from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def main() -> None:
    out = Path("output/july_regime_analysis")
    frame = pd.read_csv(out / "rolling_7d_regime.csv")
    frame["dt"] = pd.to_datetime(frame["signal_time_utc"], utc=True)
    columns = [
        ("top3_24h_positive_rate_7d_lagged", "Top3 24H positive"),
        ("top3_cont10_24h_7d_lagged", "Top3 cont10"),
        ("top3_cq24_7d_lagged", "CQ24"),
        ("top10_retention_7d", "Top10 retention"),
        ("market_breadth_7d", "Breadth"),
        ("gainer_concentration_7d", "Concentration"),
    ]
    width, height = 1200, 900
    left, top, right, bottom = 90, 30, 30, 40
    row_height = (height - top - bottom) / len(columns)
    t0 = frame["dt"].min().timestamp()
    t1 = frame["dt"].max().timestamp()

    def xcoord(value: pd.Timestamp) -> float:
        return left + (value.timestamp() - t0) / (t1 - t0) * (width - left - right)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    july_x = xcoord(pd.Timestamp("2026-07-01", tz="UTC"))
    aug_x = xcoord(pd.Timestamp("2026-08-01", tz="UTC"))
    svg.append(f'<rect x="{july_x:.1f}" y="{top}" width="{aug_x - july_x:.1f}" height="{height - top - bottom}" fill="#ef4444" opacity="0.08"/>')

    for index, (column, title) in enumerate(columns):
        y0 = top + index * row_height
        y1 = y0 + row_height - 18
        values = pd.to_numeric(frame[column], errors="coerce")
        vmin = float(values.quantile(0.02))
        vmax = float(values.quantile(0.98))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
            vmin = float(values.min())
            vmax = float(values.max()) + 1e-9
        points = []
        for dt, value in zip(frame["dt"], values):
            if pd.isna(value):
                continue
            x = xcoord(dt)
            y = y1 - (float(value) - vmin) / (vmax - vmin) * (row_height - 48)
            points.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<text x="10" y="{y0 + 20:.1f}" font-size="14" font-family="Arial">{title}</text>')
        svg.append(f'<text x="10" y="{y0 + 40:.1f}" font-size="11" fill="#666" font-family="Arial">{vmin:.2f}..{vmax:.2f}</text>')
        svg.append(f'<line x1="{left}" y1="{y1:.1f}" x2="{width - right}" y2="{y1:.1f}" stroke="#ddd"/>')
        svg.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="1.5" points="{" ".join(points)}"/>')

    for date_text, label in [("2026-01-01", "Jan"), ("2026-03-01", "Mar"), ("2026-05-01", "May"), ("2026-07-01", "Jul")]:
        x = xcoord(pd.Timestamp(date_text, tz="UTC"))
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#eee"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - 15}" font-size="12" font-family="Arial">{label}</text>')

    svg.append("</svg>")
    path = out / "rolling_7d_regime_chart.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
