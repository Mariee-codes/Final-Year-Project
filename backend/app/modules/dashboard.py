"""Farmer Dashboard: combine all modules into one weekly plan.

This is where the modules feed each other:
  Crop Doctor  -> yield impact -> expected harvest
  expected harvest + Price Forecaster trend -> sell-or-hold advice
  expected harvest -> suggested quantity for the buyer listing
"""
from __future__ import annotations

TEXT = {
    "en": {
        "sell_now": "Prices for {crop} are expected to fall about {pct}% in the next 4 weeks. Consider selling soon.",
        "hold": "Prices for {crop} are expected to rise about {pct}% in the next 4 weeks. If you can store safely, consider holding.",
        "flat": "Prices for {crop} look stable for the next 4 weeks.",
        "yield_cut": "Because of {disease}, expected harvest drops from {base} kg to about {adj} kg if untreated.",
        "list": "You could offer about {qty} kg of {crop} to buyers.",
    },
    "sw": {
        "sell_now": "Bei ya {crop} inatarajiwa kushuka kwa takriban {pct}% katika wiki 4 zijazo. Fikiria kuuza mapema.",
        "hold": "Bei ya {crop} inatarajiwa kupanda kwa takriban {pct}% katika wiki 4 zijazo. Ukiweza kuhifadhi salama, fikiria kusubiri.",
        "flat": "Bei ya {crop} inaonekana kuwa thabiti kwa wiki 4 zijazo.",
        "yield_cut": "Kwa sababu ya {disease}, mavuno yanayotarajiwa yanashuka kutoka kilo {base} hadi takriban kilo {adj} bila tiba.",
        "list": "Unaweza kuwapa wanunuzi takriban kilo {qty} za {crop}.",
    },
}


def adjusted_yield(base_kg: float, yield_impact: float) -> float:
    return round(base_kg * (1 - max(0.0, min(yield_impact, 1.0))), 1)


def weekly_plan(crop: str, base_yield_kg: float, language: str = "en",
                diagnosis: dict | None = None, price: dict | None = None,
                planting: dict | None = None, keep_for_home_pct: float = 0.2) -> dict:
    t = TEXT["sw" if language == "sw" else "en"]
    messages = []
    expected = base_yield_kg

    if diagnosis and not diagnosis.get("healthy") and diagnosis.get("is_confident"):
        expected = adjusted_yield(base_yield_kg, diagnosis.get("yield_impact", 0.0))
        messages.append(t["yield_cut"].format(disease=diagnosis["condition"], base=base_yield_kg, adj=expected))
        messages.extend(diagnosis.get("advice", []))

    if price:
        pct = abs(price["change_pct"])
        key = {"down": "sell_now", "up": "hold"}.get(price["trend"], "flat")
        messages.append(t[key].format(crop=crop, pct=pct))

    listing_qty = round(expected * (1 - keep_for_home_pct))
    messages.append(t["list"].format(qty=listing_qty, crop=crop))

    if planting:
        messages.extend(planting.get("reasons", []))
        messages.extend(planting.get("warnings", []))

    return {"crop": crop, "expected_yield_kg": expected, "suggested_listing_kg": listing_qty,
            "messages": messages}
