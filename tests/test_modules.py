import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.modules import buyer_matcher as bm  # noqa: E402
from backend.app.modules import crop_doctor as cd  # noqa: E402
from backend.app.modules import dashboard, planting_advisor as pa, price_forecaster as pf  # noqa: E402

LABELS = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
          "Tomato___Late_blight", "Tomato___healthy"]


# ---------- Crop Doctor ----------
def test_split_label():
    assert cd.split_label("Corn_(maize)___Common_rust_") == ("Corn (maize)", "Common rust")
    assert cd.split_label("Tomato___healthy") == ("Tomato", "healthy")


def test_confident_disease_gets_advice_and_yield_impact():
    probs = np.array([0.02, 0.9, 0.03, 0.03, 0.02])
    d = cd.interpret(probs, LABELS, cd.load_treatments(), "en")
    assert d.label == "Potato___Late_blight" and d.is_confident and not d.healthy
    assert d.yield_impact > 0 and len(d.advice) >= 2
    assert d.top3[0][0] == "Potato___Late_blight"


def test_low_confidence_says_unsure_in_swahili():
    probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    d = cd.interpret(probs, LABELS, cd.load_treatments(), "sw")
    assert not d.is_confident and d.yield_impact == 0
    assert "uhakika" in d.advice[0]


def test_healthy_plant():
    probs = np.array([0.01, 0.01, 0.95, 0.02, 0.01])
    d = cd.interpret(probs, LABELS, cd.load_treatments())
    assert d.healthy and d.yield_impact == 0


def test_softmax_sums_to_one():
    assert cd.softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


# ---------- Planting Advisor ----------
def weather(rain_per_day, tmax=26, tmin=14, days=16, start=date(2026, 3, 20)):
    return [pa.DailyWeather(start + timedelta(i), rain_per_day, tmax, tmin) for i in range(days)]


def test_plant_now_when_rains_are_enough():
    a = pa.advise("maize", date(2026, 3, 20), weather(5))
    assert a.recommendation == "plant_now"
    assert a.expected_harvest == date(2026, 3, 20) + timedelta(days=120)


def test_wait_when_rain_is_low():
    a = pa.advise("maize", date(2026, 3, 20), weather(0.5))
    assert a.recommendation == "wait" and a.warnings


def test_not_in_season_points_to_next_season():
    a = pa.advise("maize", date(2026, 6, 15), weather(5))
    assert a.recommendation == "not_in_season" and a.window_start == date(2026, 10, 1)


def test_soil_ph_and_heat_warnings():
    a = pa.advise("potato", date(2026, 3, 20), weather(5, tmax=30), soil_ph=4.5)
    assert any("pH" in w for w in a.warnings)
    assert any("hotter" in w for w in a.warnings)


def test_unknown_crop():
    with pytest.raises(ValueError):
        pa.advise("coffee", date(2026, 3, 20), weather(5))


# ---------- Price Forecaster ----------
@pytest.fixture(scope="module")
def series():
    df = pd.read_csv(ROOT / "data" / "sample_kamis_SYNTHETIC.csv")
    return pf.to_weekly(df, "maize", "Nakuru")


def test_weekly_series(series):
    assert len(series) > 150 and series.isna().sum() == 0


def test_evaluate_returns_models(series):
    names = [r.model for r in pf.evaluate(series)]
    assert "naive (last week)" in names and "gradient boosting" in names


def test_forecast_shape(series):
    f = pf.forecast(series, 4)
    assert len(f["weeks"]) == 4 and f["trend"] in {"up", "down", "flat"}
    assert all(w["low"] <= w["price"] <= w["high"] for w in f["weeks"])


def test_unknown_commodity():
    df = pd.read_csv(ROOT / "data" / "sample_kamis_SYNTHETIC.csv")
    with pytest.raises(ValueError):
        pf.to_weekly(df, "avocado")


# ---------- Buyer Matcher ----------
def test_haversine_nairobi_nakuru():
    km = bm.haversine_km(-1.2864, 36.8172, -0.3031, 36.0800)
    assert 120 < km < 140


def test_closer_better_offer_ranks_first():
    listing = bm.Listing("L1", "F1", "tomato", 500, 60, -0.30, 36.08)
    near = bm.BuyerRequest("R1", "B1", "tomato", 500, 65, -0.31, 36.10)
    far = bm.BuyerRequest("R2", "B2", "tomato", 500, 65, -1.29, 36.82)
    other_crop = bm.BuyerRequest("R3", "B3", "maize", 500, 65, -0.31, 36.10)
    ranked = bm.match_listing(listing, [far, near, other_crop])
    assert [m["request_id"] for m in ranked] == ["R1", "R2"]


def test_lowball_offer_penalised_against_market():
    assert bm.price_fit(60, 30, market=60) < bm.price_fit(60, 30)


# ---------- Dashboard (modules feeding each other) ----------
def test_disease_reduces_listing_and_price_advice():
    diag = {"healthy": False, "is_confident": True, "yield_impact": 0.5,
            "condition": "Late blight", "advice": ["Remove infected plants."]}
    price = {"trend": "down", "change_pct": -8.0}
    plan = dashboard.weekly_plan("potato", 1000, "en", diag, price)
    assert plan["expected_yield_kg"] == 500
    assert plan["suggested_listing_kg"] == 400
    assert any("selling soon" in m for m in plan["messages"])


def test_dashboard_swahili():
    plan = dashboard.weekly_plan("maize", 800, "sw", price={"trend": "up", "change_pct": 6})
    assert any("kupanda" in m for m in plan["messages"])
