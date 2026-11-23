"""Compare price forecasting models on KAMIS data (time-ordered test).

Usage:
  python ml/evaluate_prices.py --csv data/your_kamis_export.csv --crop maize --county Nakuru

With the bundled synthetic file the numbers only prove the code runs;
they are NOT results to report.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from backend.app.modules import price_forecaster as pf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/sample_kamis_SYNTHETIC.csv")
ap.add_argument("--crop", default="maize")
ap.add_argument("--county")
ap.add_argument("--test-weeks", type=int, default=12)
args = ap.parse_args()

series = pf.to_weekly(pd.read_csv(args.csv), args.crop, args.county)
print(f"{args.crop} {args.county or 'all counties'}: {len(series)} weeks of data")
if "SYNTHETIC" in args.csv:
    print("WARNING: synthetic data - for testing the code only.\n")
for r in pf.evaluate(series, args.test_weeks):
    print(f"{r.model:22s}  MAPE {r.mape:6.2f}%   direction accuracy {r.direction_accuracy:5.1f}%")
print("\n4-week forecast:", pf.forecast(series))
