from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("model_final_assessment.pkl")

FEATURE_COLS = [
    "glucose_month1_mean", "glucose_month1_std", "glucose_slope_month1",
    "baseline_glucose",
    "steps_month1_mean", "steps_consistency_m1",
    "sleep_month1_mean", "sleep_consistency_m1",
    "carbs_month1_mean",
    "sleep_adherence_m1", "steps_adherence_m1", "max_streak_m1",
    "corr_sleep_glucose_m1", "corr_steps_glucose_m1", "corr_carbs_glucose_m1",
]

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(x, y)[0])

def aggregate_first_30_days(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "day_idx", "glucose_mean", "steps", "sleep_hours", "carbs_g",
        "target_sleep_met", "target_steps_met", "streak", "baseline_glucose"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df.sort_values("day_idx").copy()

    if len(work) < 30:
        raise ValueError("At least 30 daily rows are required for prediction.")

    m1 = work.iloc[:30]

    glucose = m1["glucose_mean"].astype(float).to_numpy()
    steps = m1["steps"].astype(float).to_numpy()
    sleep = m1["sleep_hours"].astype(float).to_numpy()
    carbs = m1["carbs_g"].astype(float).to_numpy()
    tgt_sleep = m1["target_sleep_met"].astype(float).to_numpy()
    tgt_steps = m1["target_steps_met"].astype(float).to_numpy()
    streak = m1["streak"].astype(float).to_numpy()

    features = {
        "glucose_month1_mean": float(glucose.mean()),
        "glucose_month1_std": float(glucose.std()),
        "glucose_slope_month1": float(linregress(np.arange(30), glucose).slope),
        "baseline_glucose": float(m1["baseline_glucose"].iloc[0]),

        "steps_month1_mean": float(steps.mean()),
        "steps_consistency_m1": float(1 / (steps.std() + 1)),

        "sleep_month1_mean": float(sleep.mean()),
        "sleep_consistency_m1": float(1 / (sleep.std() + 1)),

        "carbs_month1_mean": float(carbs.mean()),

        "sleep_adherence_m1": float(tgt_sleep.mean()),
        "steps_adherence_m1": float(tgt_steps.mean()),
        "max_streak_m1": float(streak.max()),

        "corr_sleep_glucose_m1": safe_corr(sleep, glucose),
        "corr_steps_glucose_m1": safe_corr(steps, glucose),
        "corr_carbs_glucose_m1": safe_corr(carbs, glucose),
    }

    return pd.DataFrame([features], columns=FEATURE_COLS)

class DailyRecord(BaseModel):
    day_idx: int = Field(..., ge=0, le=29)
    glucose_mean: float
    steps: float
    sleep_hours: float
    carbs_g: float
    target_sleep_met: float = Field(..., ge=0, le=1)
    target_steps_met: float = Field(..., ge=0, le=1)
    streak: float
    baseline_glucose: float

class PredictRequest(BaseModel):
    patient_id: Optional[str] = None
    records: List[DailyRecord]

app = FastAPI(
    title="Glucare Day-30 Early Warning API",
    version="1.0.0",
    description="Predicts day-90 risk using the first 30 days of monitoring data."
)

@app.on_event("startup")
def load_artifacts():
    global artifacts
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    artifacts = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Glucare 90-day final assessment API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": artifacts.get("model_name"),
        "feature_count": len(artifacts.get("feature_cols", [])),
    }

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([r.model_dump() for r in req.records])

        feats = aggregate_first_30_days(df)

        scaler = artifacts.get("scaler")
        X = scaler.transform(feats) if scaler is not None else feats

        model = artifacts["model"]
        pred = int(model.predict(X)[0])
        proba_improve = float(model.predict_proba(X)[0, 1])
        proba_deteriorate = float(1 - proba_improve)

        risk_label = "HIGH_RISK_DAY90" if pred == 0 else "LOW_RISK_DAY90"
        predicted_status = artifacts.get("label_map", {}).get(pred, str(pred))

        return {
            "patient_id": req.patient_id,
            "model_name": artifacts.get("model_name"),
            "prediction_day": 30,
            "predicted_class": pred,
            "predicted_status": predicted_status,
            "early_warning_risk": risk_label,
            "probability_membaik": round(proba_improve, 4),
            "probability_memburuk_stagnan": round(proba_deteriorate, 4),
            "features_used": feats.iloc[0].to_dict(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")