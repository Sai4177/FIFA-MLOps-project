"""EXPERIMENT 2a - Model deployment as a REST API (MLflow Model Registry + FastAPI).

Loads the registered champion model:  models:/FIFA_Tier_Classifier@champion
Run (from project root):   uvicorn api.fastapi_app:app --reload --port 8000
Swagger UI:                http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fifa_mlops.config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI, RAW_FEATURES, REGISTERED_MODEL_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    champion = mlflow.tracking.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model_loaded, load_error = True, None
    print("Model loaded successfully:", MODEL_URI, "| version", champion.version)
except Exception as exc:                                    # API still starts; /health reports 503
    model, champion, model_loaded, load_error = None, None, False, str(exc)
    print("Model loading failed:", exc)

app = FastAPI(
    title="FIFA Tier Prediction API",
    description="REST API serving the champion model registered in the MLflow Model Registry",
    version="1.0",
)


class TeamFeatures(BaseModel):
    previous_rank: int = Field(..., ge=1, le=250, examples=[3])
    previous_points: float = Field(..., gt=0, examples=[1770.65])
    association: Literal["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"] = Field(..., examples=["CONMEBOL"])


@app.get("/")
def home():
    return {"message": "FIFA Tier Prediction API is running"}


@app.get("/health")
def health():
    if model_loaded:
        return {"status": "healthy", "model_loaded": True, "model_name": REGISTERED_MODEL_NAME,
                "model_uri": MODEL_URI, "model_version": champion.version}
    raise HTTPException(status_code=503, detail=f"Model is not available: {load_error}")


@app.get("/features")
def features():
    return {"number_of_features": len(RAW_FEATURES), "required_features": RAW_FEATURES,
            "classes": list(model.classes_) if model_loaded else []}


@app.post("/predict")
def predict(team: TeamFeatures):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        df = pd.DataFrame([team.model_dump()])[RAW_FEATURES]
        prediction = str(model.predict(df)[0])
        proba = model.predict_proba(df)[0]
        return {"prediction": prediction,
                "probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
