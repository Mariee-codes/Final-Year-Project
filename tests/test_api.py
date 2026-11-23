import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and "crop_doctor_model" in r.json()


def test_diagnose_without_model_is_clear():
    r = client.post("/diagnose", files={"image": ("leaf.jpg", b"fake", "image/jpeg")})
    assert r.status_code == 503 and "not trained" in r.json()["detail"]


def test_price_forecast():
    r = client.get("/price-forecast", params={"crop": "tomato", "county": "Nairobi"})
    assert r.status_code == 200 and len(r.json()["weeks"]) == 4


def test_listing_request_and_matching():
    lid = client.post("/listings", json={"farmer_id": "F1", "crop": "potato", "quantity_kg": 800,
                                         "asking_price": 50, "lat": -0.30, "lon": 36.08}).json()["id"]
    client.post("/buyer-requests", json={"buyer_id": "B1", "crop": "potato", "quantity_kg": 600,
                                         "offer_price": 52, "lat": -0.28, "lon": 36.07})
    r = client.get(f"/matches/{lid}", params={"county": "Nakuru"})
    assert r.status_code == 200 and r.json()[0]["score"] > 0.5


def test_dashboard():
    r = client.post("/dashboard", json={"crop": "maize", "base_yield_kg": 900, "county": "Kisumu", "language": "sw"})
    assert r.status_code == 200 and r.json()["suggested_listing_kg"] == 720
