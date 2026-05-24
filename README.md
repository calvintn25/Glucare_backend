# Glucare Day-30 Early Warning API

FastAPI backend that predicts day-90 outcome risk using the first 30 days of patient monitoring data.

## Description

This API receives daily records (day 0 to day 29), performs feature engineering in the backend, and runs a trained model to produce:

- Predicted class and status label
- Early warning risk label (`HIGH_RISK_DAY90` or `LOW_RISK_DAY90`)
- Probabilities for `membaik` and `memburuk/stagnan`
- The engineered features used for inference

## Features

- Model artifact loading from `model_final_assessment.pkl`
- Health check endpoint: `GET /health`
- Prediction endpoint: `POST /predict`
- Automatic first-30-days feature aggregation inside `main.py`

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Create and activate virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

2. Put the trained artifact file `model_final_assessment.pkl` in the project root.

Minimum required artifact keys:

- `model` (required)
- `feature_cols` (recommended)

Optional keys used by the API response:

- `model_name`
- `label_map`
- `scaler`

## Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs: `http://localhost:8000/docs`

## Endpoints

### `GET /`

Returns a simple service message.

### `GET /health`

Returns model metadata:

- `status`
- `model_name`
- `feature_count`

### `POST /predict`

Request body schema:

```json
{
  "patient_id": "P001",
  "records": [
    {
      "day_idx": 0,
      "glucose_mean": 120.5,
      "steps": 4500,
      "sleep_hours": 6.8,
      "carbs_g": 180,
      "target_sleep_met": 1,
      "target_steps_met": 0,
      "streak": 3,
      "baseline_glucose": 130
    }
  ]
}
```

Notes:

- `records` must contain at least 30 rows.
- `day_idx` is validated in range `0..29`.
- The API uses the first 30 records after sorting by `day_idx`.

Example request (PowerShell-friendly body shortened for readability):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "records": [
      {
        "day_idx": 0,
        "glucose_mean": 120.5,
        "steps": 4500,
        "sleep_hours": 6.8,
        "carbs_g": 180,
        "target_sleep_met": 1,
        "target_steps_met": 0,
        "streak": 3,
        "baseline_glucose": 130
      }
    ]
  }'
```

Sample response:

```json
{
  "patient_id": "P001",
  "model_name": "final_assessment_v1",
  "prediction_day": 30,
  "predicted_class": 1,
  "predicted_status": "Membaik",
  "early_warning_risk": "LOW_RISK_DAY90",
  "probability_membaik": 0.8821,
  "probability_memburuk_stagnan": 0.1179,
  "features_used": {
    "glucose_month1_mean": 118.42,
    "glucose_month1_std": 10.31,
    "glucose_slope_month1": -0.22,
    "baseline_glucose": 130.0,
    "steps_month1_mean": 5123.0,
    "steps_consistency_m1": 0.0031,
    "sleep_month1_mean": 6.95,
    "sleep_consistency_m1": 0.61,
    "carbs_month1_mean": 172.0,
    "sleep_adherence_m1": 0.8,
    "steps_adherence_m1": 0.63,
    "max_streak_m1": 9.0,
    "corr_sleep_glucose_m1": -0.12,
    "corr_steps_glucose_m1": -0.2,
    "corr_carbs_glucose_m1": 0.11
  }
}
```

## Inference Pipeline

On each `/predict` call, the API:

1. Validates request schema with Pydantic.
2. Converts records to DataFrame.
3. Builds month-1 aggregated features:
   - central tendency/dispersion (`mean`, `std`)
   - trend (`linregress` slope)
   - adherence ratios
   - max streak
   - Pearson correlations (safe fallback to `0.0`)
4. Applies scaler if provided in artifact.
5. Runs model prediction and probabilities.

## Error Behavior

- `400 Bad Request` for invalid input, missing required columns, or fewer than 30 daily rows.
- `500 Internal Server Error` for model/artifact inference failures.

## Project Files

- `main.py`: FastAPI app, feature engineering, model inference
- `requirements.txt`: Python dependencies
- `model_final_assessment.pkl`: model artifact loaded at startup

## License

Add a license file if this project will be distributed publicly.
