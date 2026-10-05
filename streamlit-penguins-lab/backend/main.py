"""FastAPI backend that serves the penguin species model."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model" / "penguin_model.joblib"

app = FastAPI(title="Penguin Species API")
model = joblib.load(MODEL_PATH)


class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., gt=0)
    bill_depth_mm: float = Field(..., gt=0)
    flipper_length_mm: float = Field(..., gt=0)
    body_mass_g: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    species: str
    probabilities: dict[str, float]


@app.get("/")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PenguinFeatures):
    X = pd.DataFrame([features.model_dump()])
    probs = model.predict_proba(X)[0]
    prob_map = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probs)}
    species = max(prob_map, key=prob_map.get)
    return PredictionResponse(species=species, probabilities=prob_map)
