"""Planting Advisor: recommend a planting window and flag risks.

Rule-based on purpose so every recommendation can be explained to a farmer.
Weather comes from Open-Meteo (free, no API key for non-commercial use).
Soil pH can come from iSDAsoil or be entered by the farmer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def load_requirements() -> dict:
    with open(DATA_DIR / "crop_requirements.json", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_comment", None)
    return data


@dataclass
class DailyWeather:
    day: date
    rain_mm: float
    temp_max_c: float
    temp_min_c: float


@dataclass
class PlantingAdvice:
    crop: str
    recommendation: str  # "plant_now", "wait", "not_in_season"
    window_start: date | None
    window_end: date | None
    expected_harvest: date | None
    forecast_rain_mm: float
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("window_start", "window_end", "expected_harvest"):
            d[k] = d[k].isoformat() if d[k] else None
        return d


def fetch_forecast(lat: float, lon: float, client=None) -> list[DailyWeather]:
    """Get a 16-day daily forecast from Open-Meteo."""
    import httpx

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "forecast_days": 16,
        "timezone": "Africa/Nairobi",
    }
    client = client or httpx.Client(timeout=15)
    r = client.get(OPEN_METEO_URL, params=params)
    r.raise_for_status()
    d = r.json()["daily"]
    return [
        DailyWeather(date.fromisoformat(t), p or 0.0, hi, lo)
        for t, p, hi, lo in zip(d["time"], d["precipitation_sum"],
                                d["temperature_2m_max"], d["temperature_2m_min"])
    ]


def in_season(today: date, seasons: list[list[int]]) -> tuple[bool, tuple[int, int] | None]:
    for start, end in seasons:
        if start <= today.month <= end:
            return True, (start, end)
    return False, None


def next_season(today: date, seasons: list[list[int]]) -> tuple[int, int]:
    future = sorted(seasons, key=lambda s: (s[0] - today.month) % 12)
    return tuple(future[0])


def advise(crop: str, today: date, forecast: list[DailyWeather],
           soil_ph: float | None = None, requirements: dict | None = None) -> PlantingAdvice:
    reqs = (requirements or load_requirements())
    crop = crop.lower()
    if crop not in reqs:
        raise ValueError(f"Unknown crop '{crop}'. Known crops: {', '.join(sorted(reqs))}")
    r = reqs[crop]

    rain = sum(d.rain_mm for d in forecast)
    days = len(forecast) or 1
    # Scale the forecast rain to a 30-day estimate so it compares with the requirement.
    rain_30 = rain * 30 / days
    warnings, reasons = [], []

    t_lo, t_hi = r["temp_c"]
    hot_days = sum(1 for d in forecast if d.temp_max_c > t_hi)
    cold_days = sum(1 for d in forecast if d.temp_min_c < t_lo - 5)
    if hot_days >= 3:
        warnings.append(f"{hot_days} forecast days hotter than {t_hi}°C, which can stress young {crop}.")
    if cold_days >= 3:
        warnings.append(f"{cold_days} forecast nights much colder than {t_lo}°C.")

    if soil_ph is not None:
        lo, hi = r["soil_ph"]
        if soil_ph < lo:
            warnings.append(f"Soil pH {soil_ph} is below the {lo}-{hi} range for {crop}; consider liming.")
        elif soil_ph > hi:
            warnings.append(f"Soil pH {soil_ph} is above the {lo}-{hi} range for {crop}.")

    ok, season = in_season(today, r["seasons"])
    if not ok:
        start_m, end_m = next_season(today, r["seasons"])
        year = today.year if start_m > today.month else today.year + 1
        ws = date(year, start_m, 1)
        return PlantingAdvice(crop, "not_in_season", ws, None, None, round(rain, 1), warnings,
                              [f"{crop.title()} is usually planted from month {start_m} to {end_m}."])

    need = r["min_rain_first_30_days_mm"]
    if rain_30 >= need:
        rec = "plant_now"
        ws = today
        reasons.append(f"Expected rain (about {rain_30:.0f} mm over 30 days) meets the {need} mm {crop} needs to establish.")
    else:
        rec = "wait"
        ws = today + timedelta(days=7)
        reasons.append(f"Expected rain (about {rain_30:.0f} mm over 30 days) is below the {need} mm {crop} needs. Wait for the rains to settle.")
        warnings.append("Risk of poor germination if planted now without irrigation.")

    _, end_m = season
    we = date(today.year, end_m, 28)
    harvest = ws + timedelta(days=r["days_to_harvest"])
    return PlantingAdvice(crop, rec, ws, we, harvest, round(rain, 1), warnings, reasons)
