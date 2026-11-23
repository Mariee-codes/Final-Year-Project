"""Crop Doctor: identify crop diseases from a leaf photo.

The model itself is trained separately with ``ml/train_crop_doctor.py`` and
exported to ONNX. This module loads that file, runs inference, applies the
confidence threshold and attaches treatment advice in English or Swahili.

If no trained model is present yet, ``CropDoctor.available`` is False and the
API returns a clear "model not trained yet" message instead of guessing.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

# Below this confidence the app says it is unsure and refers the farmer to an expert.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# ImageNet normalisation, matching the pretrained backbone used in training.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224


@dataclass
class Diagnosis:
    label: str
    crop: str
    condition: str
    confidence: float
    is_confident: bool
    healthy: bool
    yield_impact: float  # expected fraction of yield lost if untreated (0-1)
    advice: list[str] = field(default_factory=list)
    top3: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "crop": self.crop,
            "condition": self.condition,
            "confidence": round(self.confidence, 4),
            "is_confident": self.is_confident,
            "healthy": self.healthy,
            "yield_impact": self.yield_impact,
            "advice": self.advice,
            "top3": [{"label": l, "confidence": round(c, 4)} for l, c in self.top3],
        }


def load_treatments() -> dict:
    with open(DATA_DIR / "treatments.json", encoding="utf-8") as f:
        return json.load(f)


def split_label(label: str) -> tuple[str, str]:
    """PlantVillage folder names look like 'Tomato___Late_blight'."""
    if "___" in label:
        crop, condition = label.split("___", 1)
    else:
        crop, condition = label, "unknown"
    crop = crop.replace("_", " ").replace(",", "").strip()
    condition = condition.replace("_", " ").strip()
    return crop, condition


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def interpret(probs: np.ndarray, labels: list[str], treatments: dict,
              language: str = "en",
              threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Diagnosis:
    """Turn class probabilities into a farmer-facing diagnosis."""
    order = np.argsort(probs)[::-1]
    best = int(order[0])
    label = labels[best]
    confidence = float(probs[best])
    crop, condition = split_label(label)
    healthy = condition.lower() == "healthy"
    info = treatments.get(label, {})
    is_confident = confidence >= threshold

    lang = "sw" if language == "sw" else "en"
    if not is_confident:
        advice = treatments["_unsure"][lang]
    elif healthy:
        advice = treatments["_healthy"][lang]
    else:
        advice = info.get(lang) or treatments["_no_advice"][lang]

    return Diagnosis(
        label=label,
        crop=crop,
        condition=condition,
        confidence=confidence,
        is_confident=is_confident,
        healthy=healthy,
        yield_impact=float(info.get("yield_impact", 0.0)) if is_confident else 0.0,
        advice=advice,
        top3=[(labels[int(i)], float(probs[int(i)])) for i in order[:3]],
    )


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Resize, centre-crop and normalise an image to a 1x3x224x224 array."""
    from PIL import Image  # imported lazily so tests run without Pillow

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    short = min(img.size)
    left = (img.width - short) // 2
    top = (img.height - short) // 2
    img = img.crop((left, top, left + short, top + short)).resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


class CropDoctor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_path = model_dir / "crop_doctor.onnx"
        self.labels_path = model_dir / "labels.json"
        self.treatments = load_treatments()
        self.session = None
        self.labels: list[str] = []
        if self.model_path.exists() and self.labels_path.exists():
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(self.model_path))
            with open(self.labels_path, encoding="utf-8") as f:
                self.labels = json.load(f)

    @property
    def available(self) -> bool:
        return self.session is not None

    def diagnose(self, image_bytes: bytes, language: str = "en",
                 threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Diagnosis:
        if not self.available:
            raise RuntimeError("Crop Doctor model not trained yet. Run ml/train_crop_doctor.py.")
        x = preprocess(image_bytes)
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: x})[0][0]
        return interpret(softmax(logits), self.labels, self.treatments, language, threshold)
