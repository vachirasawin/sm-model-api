import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(title="SM Grade Prediction API")

bundle = joblib.load("SM.pkl")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
subjects = bundle["subjects"]
target_columns = bundle["target_columns"]


class PredictRequest(BaseModel):
    recordData: Dict[str, Optional[float]]
    targetTerm: int


def calc_trend(record, subject, up_to_term):
    xs, ys = [], []
    for term in range(1, up_to_term):
        grade = record.get(f"{subject}_{term}_Grade")
        if grade is not None and grade != "":
            xs.append(term)
            ys.append(float(grade))
    if len(xs) < 2:
        return 0.0
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


def build_features(record, target_term):
    features = {}
    for past_term in range(1, 6):
        for subject in subjects:
            credit_col = f"{subject}_{past_term}_Credit"
            grade_col = f"{subject}_{past_term}_Grade"
            if past_term < target_term:
                credit_val = record.get(credit_col, 0)
                grade_val = record.get(grade_col, 0)
                features[credit_col] = float(credit_val) if credit_val not in (None, "") else 0.0
                features[grade_col] = float(grade_val) if grade_val not in (None, "") else 0.0
            else:
                features[credit_col] = 0.0
                features[grade_col] = 0.0

    for subject in subjects:
        features[f"{subject}_Trend"] = calc_trend(record, subject, target_term)

    return features


def round_to_grade_scale(value, min_grade=0.0, max_grade=4.0, step=0.5):
    value = np.clip(value, min_grade, max_grade)
    return np.round(value / step) * step


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        features = build_features(payload.recordData, payload.targetTerm)

        X_new = pd.DataFrame([features])
        X_new = X_new.reindex(columns=feature_columns, fill_value=0.0)

        y_predicted = model.predict(X_new)
        y_predicted = round_to_grade_scale(y_predicted)[0]

        result = {
            subject: float(grade)
            for subject, grade in zip(target_columns, y_predicted)
        }

        return {"success": True, "targetTerm": payload.targetTerm, "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))