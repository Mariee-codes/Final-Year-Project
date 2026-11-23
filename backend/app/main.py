"""ShambaIQ API.

Run:  uvicorn backend.app.main:app --reload   (from the project root)
Docs: http://127.0.0.1:8000/docs

Storage is in memory for the prototype. Swap the `store` dict for
PostgreSQL tables (see the data model in the proposal) before the pilot.
"""
from __future__ import annotations

import os
import uuid
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .modules import buyer_matcher as bm
from .modules import dashboard, planting_advisor, price_forecaster
from .modules.crop_doctor import CropDoctor

ROOT = Path(__file__).resolve().parents[2]
KAMIS_CSV = Path(os.getenv("KAMIS_CSV", ROOT / "data" / "sample_kamis_SYNTHETIC.csv"))

app = FastAPI(title="ShambaIQ API", version="0.1.0",
              description="AI advisor for small-scale Kenyan farmers")
doctor = CropDoctor()
store: dict[str, dict] = {"listings": {}, "requests": {}}


@lru_cache
def kamis() -> pd.DataFrame:
    if not KAMIS_CSV.exists():
        raise HTTPException(503, f"Price data not found at {KAMIS_CSV}. Run ml/make_sample_kamis.py or set KAMIS_CSV.")
    return pd.read_csv(KAMIS_CSV)


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "crop_doctor_model": doctor.available, "price_data": str(KAMIS_CSV)}


@app.post("/diagnose")
async def diagnose(image: UploadFile = File(...), language: str = Form("en")):
    if not doctor.available:
        raise HTTPException(503, "Crop Doctor model not trained yet. See ml/train_crop_doctor.py.")
    data = await image.read()
    return doctor.diagnose(data, language).to_dict()


@app.get("/planting-advice")
def planting_advice(crop: str, lat: float, lon: float, soil_ph: float | None = None):
    try:
        weather = planting_advisor.fetch_forecast(lat, lon)
    except Exception as e:  # network or API problem
        raise HTTPException(502, f"Could not reach the weather service: {e}")
    try:
        return planting_advisor.advise(crop, date.today(), weather, soil_ph).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/price-forecast")
def price_forecast(crop: str, county: str | None = None, weeks: int = 4):
    try:
        series = price_forecaster.to_weekly(kamis(), crop, county)
        return price_forecaster.forecast(series, weeks)
    except ValueError as e:
        raise HTTPException(404, str(e))


class ListingIn(BaseModel):
    farmer_id: str
    crop: str
    quantity_kg: float = Field(gt=0)
    asking_price: float = Field(gt=0)
    lat: float
    lon: float


class RequestIn(BaseModel):
    buyer_id: str
    crop: str
    quantity_kg: float = Field(gt=0)
    offer_price: float = Field(gt=0)
    lat: float
    lon: float


@app.post("/listings")
def create_listing(item: ListingIn):
    lid = uuid.uuid4().hex[:8]
    store["listings"][lid] = bm.Listing(id=lid, **item.model_dump())
    return {"id": lid}


@app.post("/buyer-requests")
def create_request(item: RequestIn):
    rid = uuid.uuid4().hex[:8]
    store["requests"][rid] = bm.BuyerRequest(id=rid, **item.model_dump())
    return {"id": rid}


@app.get("/matches/{listing_id}")
def matches(listing_id: str, county: str | None = None):
    listing = store["listings"].get(listing_id)
    if not listing:
        raise HTTPException(404, "Listing not found")
    market = None
    try:
        market = float(price_forecaster.to_weekly(kamis(), listing.crop, county).iloc[-1])
    except Exception:
        pass
    return bm.match_listing(listing, list(store["requests"].values()), market)


class DashboardIn(BaseModel):
    crop: str
    base_yield_kg: float = Field(gt=0)
    county: str | None = None
    language: str = "en"
    diagnosis: dict | None = None


@app.post("/dashboard")
def weekly_dashboard(body: DashboardIn):
    price = None
    try:
        price = price_forecaster.forecast(price_forecaster.to_weekly(kamis(), body.crop, body.county))
    except Exception:
        pass
    return dashboard.weekly_plan(body.crop, body.base_yield_kg, body.language, body.diagnosis, price)
