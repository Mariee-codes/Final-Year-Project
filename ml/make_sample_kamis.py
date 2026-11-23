"""Generate a SYNTHETIC KAMIS-style price file for development and tests only.

This is NOT real market data. Replace it with a real KAMIS export
(kamis.kilimo.go.ke) before reporting any forecasting results.

Usage: python ml/make_sample_kamis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_kamis_SYNTHETIC.csv"

rng = np.random.default_rng(7)
dates = pd.date_range("2023-01-02", "2026-10-26", freq="W-MON")
rows = []
base = {"maize": 45, "tomato": 70, "potato": 50}  # KSh per kg, illustrative only
phase = {"maize": 0.0, "tomato": 2.0, "potato": 4.0}
for commodity, b in base.items():
    for county, shift in [("Nakuru", 0.0), ("Nairobi", 0.1), ("Kisumu", 0.05)]:
        level = b * (1 + shift)
        for i, d in enumerate(dates):
            season = 0.15 * np.sin(2 * np.pi * (d.dayofyear / 365.25) + phase[commodity])
            trend = 0.04 * i / 52
            noise = rng.normal(0, 0.05)
            price = level * (1 + season + trend + noise)
            rows.append({"date": d.date(), "commodity": commodity, "county": county,
                         "market": f"{county} Market", "wholesale_price": round(price, 1)})

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {len(rows)} synthetic rows to {OUT}")
