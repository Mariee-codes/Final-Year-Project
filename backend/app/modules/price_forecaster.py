"""Price Forecaster: predict short-term price direction from KAMIS records.

Expected input: a CSV exported from KAMIS with at least these columns
(rename yours to match): date, commodity, county, market, wholesale_price.

Three models are compared honestly on the most recent weeks:
  1. naive baseline  - next week's price = this week's price
  2. seasonal naive  - price from the same week last year (when available)
  3. gradient boosting on lag and seasonal features
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

LAGS = (1, 2, 3, 4, 8, 12)


def to_weekly(df: pd.DataFrame, commodity: str, county: str | None = None) -> pd.Series:
    """Average prices per week for one commodity (and optionally one county)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["commodity"].str.lower() == commodity.lower()]
    if county:
        d = d[d["county"].str.lower() == county.lower()]
    if d.empty:
        raise ValueError(f"No records for {commodity}{' in ' + county if county else ''}.")
    # Drop extreme outliers (data-entry errors) before averaging.
    q1, q3 = d["wholesale_price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    d = d[d["wholesale_price"].between(q1 - 3 * iqr, q3 + 3 * iqr)]
    s = d.set_index("date")["wholesale_price"].resample("W").mean()
    return s.interpolate(limit=3).dropna()


def make_features(s: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=s.index)
    for lag in LAGS:
        X[f"lag_{lag}"] = s.shift(lag)
    X["rolling_mean_4"] = s.shift(1).rolling(4).mean()
    X["week_sin"] = np.sin(2 * np.pi * s.index.isocalendar().week.astype(int) / 52)
    X["week_cos"] = np.cos(2 * np.pi * s.index.isocalendar().week.astype(int) / 52)
    X["y"] = s
    return X.dropna()


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def direction_accuracy(prev, y_true, y_pred) -> float:
    prev, y_true, y_pred = map(np.asarray, (prev, y_true, y_pred))
    return float(np.mean(np.sign(y_true - prev) == np.sign(y_pred - prev)) * 100)


@dataclass
class EvalResult:
    model: str
    mape: float
    direction_accuracy: float


def evaluate(s: pd.Series, test_weeks: int = 12) -> list[EvalResult]:
    """Time-ordered evaluation: train on the past, test on the last `test_weeks`."""
    feats = make_features(s)
    if len(feats) <= test_weeks + 20:
        raise ValueError("Not enough history; need at least ~40 weeks of prices.")
    train, test = feats.iloc[:-test_weeks], feats.iloc[-test_weeks:]
    y_test, prev = test["y"], test["lag_1"]

    results = [EvalResult("naive (last week)", mape(y_test, prev), 0.0)]
    # The naive model always predicts "no change", so its direction accuracy is not meaningful.

    seasonal = s.shift(52).reindex(test.index)
    if seasonal.notna().all():
        results.append(EvalResult("seasonal naive", mape(y_test, seasonal),
                                  direction_accuracy(prev, y_test, seasonal)))

    model = GradientBoostingRegressor(random_state=42, n_estimators=300, max_depth=3, learning_rate=0.05)
    model.fit(train.drop(columns="y"), train["y"])
    pred = model.predict(test.drop(columns="y"))
    results.append(EvalResult("gradient boosting", mape(y_test, pred),
                              direction_accuracy(prev, y_test, pred)))
    return results


def forecast(s: pd.Series, weeks_ahead: int = 4) -> dict:
    """Fit on all history and roll the forecast forward week by week."""
    feats = make_features(s)
    model = GradientBoostingRegressor(random_state=42, n_estimators=300, max_depth=3, learning_rate=0.05)
    model.fit(feats.drop(columns="y"), feats["y"])
    # In-sample residuals understate real uncertainty; treat the range as a rough guide
    # and replace it with errors from evaluate() once you have real KAMIS data.
    residual_sd = float(np.std(feats["y"] - model.predict(feats.drop(columns="y"))))

    history = s.copy()
    preds = []
    for _ in range(weeks_ahead):
        nxt = history.index[-1] + pd.Timedelta(weeks=1)
        extended = pd.concat([history, pd.Series([np.nan], index=[nxt])])
        row = make_features(extended.ffill()).iloc[[-1]].drop(columns="y")
        p = float(model.predict(row)[0])
        preds.append({"week": nxt.date().isoformat(), "price": round(p, 1),
                      "low": round(p - 1.96 * residual_sd, 1), "high": round(p + 1.96 * residual_sd, 1)})
        history = pd.concat([history, pd.Series([p], index=[nxt])])

    last = float(s.iloc[-1])
    change = (preds[-1]["price"] - last) / last * 100
    trend = "up" if change > 3 else "down" if change < -3 else "flat"
    return {"last_price": round(last, 1), "trend": trend, "change_pct": round(change, 1), "weeks": preds}
