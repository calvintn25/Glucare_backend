# Glucare 90-Day Final Assessment API

Lightweight FastAPI service for predicting final patient status after a 90-day monitoring period.

## Description

This repository contains a small API that loads a serialized model artifact and exposes a `/predict` endpoint to classify patients' final status as either `Membaik` (improved) or `Memburuk/Stagnan` (worse/stagnant).

The API expects a JSON payload with aggregated patient features (glucose statistics, steps, sleep, and derived correlations) and returns a predicted label and probabilities (when available).

## Features

- Load model artifact: `final_assessment_artifact.pkl`
- Predict endpoint: `POST /predict`
- Minimal, dependency-light FastAPI app implemented in `main.py`

## Requirements

- Python 3.8+
- See `requirements.txt` for exact package versions.

## Installation

1. Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

2. Ensure the trained artifact `final_assessment_artifact.pkl` is placed in the project root. The app expects this file and loads two keys from it: `model` and `feature_cols`.

## Running the API

Start the app with uvicorn (development):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

By default the interactive docs will be available at `http://localhost:8000/docs`.

## API Usage

POST /predict

- Content-Type: `application/json`
- Body: JSON object with the following numeric features:

```
glucose_month1_mean
glucose_month1_std
glucose_slope_month1
baseline_glucose
steps_month1_mean
steps_consistency_90d
sleep_month1_mean
sleep_consistency_90d
carbs_month1_mean
sleep_adherence_m1
steps_adherence_m1
max_streak
m1
corr_steps_glucose
corr_carbs_glucose
```

Example curl request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "glucose_month1_mean": 110.5,
    "glucose_month1_std": 12.3,
    "glucose_slope_month1": -0.5,
    "baseline_glucose": 115.0,
    "steps_month1_mean": 4500,
    "steps_consistency_90d": 0.72,
    "sleep_month1_mean": 6.8,
    "sleep_consistency_90d": 0.85,
    "carbs_month1_mean": 180.0,
    "sleep_adherence_m1": 0.9,
    "steps_adherence_m1": 0.8,
    "max_streak": 14,
    "m1": 1.0,
    "corr_steps_glucose": -0.15,
    "corr_carbs_glucose": 0.12
  }'
```

Sample successful response:

```json
{
  "status": "Membaik",
  "raw_prediction": 1,
  "probabilities": {
    "prob_memburuk_stagnan": 0.12,
    "prob_membaik": 0.88
  }
}
```

## Artifact creation / Notes

- The server expects `final_assessment_artifact.pkl` to be a joblib dump containing at least two keys: `model` (a scikit-learn-like estimator) and `feature_cols` (list of column names in the model input order).
- If you trained the model elsewhere (e.g., Colab), ensure the artifact is saved with `joblib.dump({"model": clf, "feature_cols": feature_cols}, "final_assessment_artifact.pkl")`.
- The repository contains a small compatibility shim at the top of `main.py` to allow artifacts created on POSIX paths to work on Windows.

## Contributing

Feel free to open issues or create pull requests for improvements. Keep changes small and focused.

## License

Choose and add a license file if you plan to open-source this project.
