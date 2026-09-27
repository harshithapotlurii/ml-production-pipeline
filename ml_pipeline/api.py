from __future__ import annotations

import os
from functools import lru_cache

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_pipeline.train import FEATURES

app = FastAPI(title="Synthetic Classification Demo")


class Features(BaseModel):
    tenure_months: int = Field(ge=0, le=1200)
    monthly_spend: float = Field(ge=0, le=1_000_000)
    support_tickets: int = Field(ge=0, le=100_000)


@lru_cache(maxsize=1)
def load_model():
    path = os.getenv("MODEL_PATH", "model.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    artifact = joblib.load(path)
    if tuple(artifact["features"]) != FEATURES:
        raise ValueError("Model feature schema does not match API schema")
    return artifact["model"]


@app.post("/predict")
def predict(data: Features) -> dict:
    try:
        model = load_model()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Model artifact unavailable or invalid") from exc
    row = [[data.tenure_months, data.monthly_spend, data.support_tickets]]
    probability = float(model.predict_proba(row)[0, 1])
    return {"probability": probability, "predicted_class": int(probability >= 0.5), "dataset": "synthetic"}
