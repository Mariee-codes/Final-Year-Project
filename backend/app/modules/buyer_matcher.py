"""Buyer Matcher: rank buyer requests for a farmer's listing (or vice versa).

Score (0-1) = weighted mix of
  distance fit  - closer is better, zero beyond MAX_DISTANCE_KM
  quantity fit  - how much of the buyer's need the listing covers
  price fit     - how close the asking price is to the buyer's offer,
                  sanity-checked against the current market price
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_DISTANCE_KM = 150.0
WEIGHTS = {"distance": 0.4, "quantity": 0.3, "price": 0.3}


@dataclass
class Listing:
    id: str
    farmer_id: str
    crop: str
    quantity_kg: float
    asking_price: float  # KSh per kg
    lat: float
    lon: float


@dataclass
class BuyerRequest:
    id: str
    buyer_id: str
    crop: str
    quantity_kg: float
    offer_price: float  # KSh per kg
    lat: float
    lon: float


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_fit(km: float) -> float:
    return max(0.0, 1 - km / MAX_DISTANCE_KM)


def quantity_fit(supply: float, demand: float) -> float:
    if supply <= 0 or demand <= 0:
        return 0.0
    return min(supply, demand) / max(supply, demand)


def price_fit(asking: float, offer: float, market: float | None = None) -> float:
    if offer >= asking:
        fit = 1.0
    else:
        fit = max(0.0, 1 - (asking - offer) / asking)
    if market:
        # Penalise deals far below market (bad for the farmer).
        if offer < 0.7 * market:
            fit *= 0.5
    return fit


def score(listing: Listing, req: BuyerRequest, market_price: float | None = None) -> dict:
    km = haversine_km(listing.lat, listing.lon, req.lat, req.lon)
    parts = {
        "distance": distance_fit(km),
        "quantity": quantity_fit(listing.quantity_kg, req.quantity_kg),
        "price": price_fit(listing.asking_price, req.offer_price, market_price),
    }
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return {"listing_id": listing.id, "request_id": req.id, "score": round(total, 3),
            "distance_km": round(km, 1), "parts": {k: round(v, 3) for k, v in parts.items()}}


def match_listing(listing: Listing, requests: list[BuyerRequest],
                  market_price: float | None = None, top_n: int = 5) -> list[dict]:
    same_crop = [r for r in requests if r.crop.lower() == listing.crop.lower()]
    scored = [score(listing, r, market_price) for r in same_crop]
    scored = [s for s in scored if s["parts"]["distance"] > 0]
    return sorted(scored, key=lambda s: s["score"], reverse=True)[:top_n]
