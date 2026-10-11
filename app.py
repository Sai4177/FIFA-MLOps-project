"""CI/CD Practical-5/6/7: Flask prediction application (FIFA tier classifier).

Endpoints:  GET /  ,  GET /health  ,  POST /predict
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "src").exists():                       # repo checkout or Docker image with /app/src
    sys.path.insert(0, str(HERE / "src"))

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = HERE / "fifa_tier_model.pkl"
FEATURES = ["previous_rank", "previous_points", "association"]
ASSOCIATIONS = {"AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"}
TIERS = ["Elite", "Strong", "Mid", "Developing"]
_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("fifa_tier_model.pkl was not found. Run: python train_model.py")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/")
def home():
    return jsonify({"status": "ok", "service": "fifa-tier-prediction"})


@app.get("/health")
def health():
    try:
        load_model()
    except FileNotFoundError as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 503
    return jsonify({"status": "healthy", "model_loaded": True})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required"}), 400
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400
    try:
        rank, points = int(data["previous_rank"]), float(data["previous_points"])
    except (TypeError, ValueError):
        return jsonify({"error": "previous_rank must be an integer and previous_points a number"}), 400
    association = str(data["association"]).upper()
    if rank < 1 or points <= 0 or association not in ASSOCIATIONS:
        return jsonify({"error": "Invalid values", "allowed_associations": sorted(ASSOCIATIONS)}), 400

    model = load_model()
    sample = pd.DataFrame([{"previous_rank": rank, "previous_points": points, "association": association}])
    tier = str(model.predict(sample)[0])
    proba = model.predict_proba(sample)[0]
    return jsonify({
        "prediction": tier,
        "prediction_code": TIERS.index(tier),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
