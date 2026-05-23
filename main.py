import pathlib
# Jika artifact dibuat di Colab (Linux) dan API jalan di Windows:
pathlib.PosixPath = pathlib.WindowsPath  # type: ignore

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# load artifact (pastikan nama file sama)
artifact = joblib.load("final_assessment_artifact.pkl")
model = artifact["model"]
feature_cols = artifact["feature_cols"]

app = FastAPI(
    title="90-Day Pre-Diabetes Monitoring API",
    description="API prediksi status akhir pasien (Membaik vs Memburuk/Stagnan)",
    version="1.0.0",
)

class PatientFeatures(BaseModel):
    glucose_month1_mean: float
    glucose_month1_std: float
    glucose_slope_month1: float
    baseline_glucose: float
    steps_month1_mean: float
    steps_consistency_90d: float
    sleep_month1_mean: float
    sleep_consistency_90d: float
    carbs_month1_mean: float
    sleep_adherence_m1: float
    steps_adherence_m1: float
    max_streak: float
    m1: float
    corr_steps_glucose: float
    corr_carbs_glucose: float

@app.get("/")
def root():
    return {"message": "Glucare 90-day final assessment API is running"}

@app.post("/predict")
def predict_status(features: PatientFeatures):
    data_dict = features.dict()
    df = pd.DataFrame([data_dict])

    # cek kolom
    missing = set(feature_cols) - set(df.columns)
    if missing:
        return {
            "error": "Missing features in request",
            "missing_features": sorted(list(missing)),
        }

    df = df[feature_cols]

    y_pred = model.predict(df)[0]

    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        y_proba = {
            "prob_memburuk_stagnan": float(proba[0]),
            "prob_membaik": float(proba[1]),
        }

    label_map = {0: "Memburuk/Stagnan", 1: "Membaik"}
    status = label_map.get(int(y_pred), "Unknown")

    return {
        "status": status,
        "raw_prediction": int(y_pred),
        "probabilities": y_proba,
    }