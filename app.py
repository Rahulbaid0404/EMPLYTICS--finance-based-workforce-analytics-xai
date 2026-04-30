# api/app.py  (FIXED VERSION)

import os, sys, pickle
from typing import Optional
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# safer path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.config import (
    MODEL_PATH, SCALER_PATH, ENCODER_PATH,
    TARGET_COLUMN
)

from preprocessing.preprocess import engineer_features, encode_and_scale
from training.train_model import load_model

from explainability.shap_explainer import (
    build_shap_explainer, compute_shap_values, shap_feature_summary_df
)

from explainability.lime_explainer import (
    build_lime_explainer, explain_instance_lime, lime_explanation_df
)

# Globals
_model = None
_scaler = None
_label_encoders = None
_shap_explainer = None
_lime_explainer = None
_feature_names = None
_X_train_sample = None


# ────────────────────────────────────────────
# LOAD ARTIFACTS (SAFE)
# ────────────────────────────────────────────
def _load_artifacts():
    global _model, _scaler, _label_encoders
    global _feature_names, _shap_explainer, _lime_explainer, _X_train_sample

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run training first.")

    try:
        _model = load_model()
        _scaler = pickle.load(open(SCALER_PATH, "rb"))
        _label_encoders = pickle.load(open(ENCODER_PATH, "rb"))

        from utils.config import PROC_DATA_PATH
        df = pd.read_csv(PROC_DATA_PATH)

        X = df.drop(columns=[TARGET_COLUMN, "employee_id"], errors="ignore")

        X, _, _ = encode_and_scale(
            X, fit=False,
            scaler=_scaler,
            label_encoders=_label_encoders
        )

        _feature_names = list(X.columns)
        _X_train_sample = X.sample(min(300, len(X)), random_state=42)

        _shap_explainer = build_shap_explainer(_model, _X_train_sample)
        _lime_explainer = build_lime_explainer(_X_train_sample, _feature_names)

    except Exception as e:
        raise RuntimeError(f"Artifact loading failed: {str(e)}")


# ────────────────────────────────────────────
# APP CONFIG
# ────────────────────────────────────────────
app = FastAPI(
    title="Workforce Analytics API",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────
# INPUT SCHEMA
# ────────────────────────────────────────────
class EmployeeInput(BaseModel):
    age: int = Field(..., ge=18, le=70)
    years_at_company: int = Field(..., ge=0, le=45)
    monthly_salary: float
    performance_score: int = Field(..., ge=1, le=5)
    overtime_hours: float
    training_hours: float
    projects_completed: int
    absenteeism_days: int
    bonus_pct: float
    engagement_score: float

    department: str = "Finance"
    job_level: str = "Mid"
    education: str = "Bachelors"
    gender: str = "Male"
    marital_status: str = "Single"

    explainer: Optional[str] = "shap"


# ────────────────────────────────────────────
# PREPROCESS
# ────────────────────────────────────────────
def _preprocess_input(data: EmployeeInput):
    try:
        row = data.dict()
        df = pd.DataFrame([row])

        df = engineer_features(df)

        df, _, _ = encode_and_scale(
            df,
            fit=False,
            scaler=_scaler,
            label_encoders=_label_encoders
        )

        df = df.reindex(columns=_feature_names, fill_value=0)

        return df

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")


# ────────────────────────────────────────────
# ENDPOINTS
# ────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        _load_artifacts()
        return {"status": "ok", "model_loaded": True}
    except:
        return {"status": "error", "model_loaded": False}


@app.post("/predict")
def predict(data: EmployeeInput):

    try:
        _load_artifacts()
        X = _preprocess_input(data)

        prob = float(_model.predict_proba(X)[0, 1])
        label = int(prob >= 0.5)

        result = {
            "probability": round(prob, 4),
            "prediction": label,
            "verdict": "At-Risk" if label else "Safe"
        }

        # SHAP
        if data.explainer in ["shap", "both"]:
            try:
                shap_exp = compute_shap_values(_shap_explainer, X)
                result["shap"] = shap_feature_summary_df(shap_exp).head(10).to_dict("records")
            except Exception:
                result["shap"] = "SHAP failed"

        # LIME (FIXED)
        if data.explainer in ["lime", "both"]:
            try:
                exp = explain_instance_lime(_lime_explainer, _model, X.values[0])
                label_safe = list(exp.local_exp.keys())[0]
                lime_df = lime_explanation_df(exp)
                result["lime"] = lime_df.head(10).to_dict("records")
            except Exception:
                result["lime"] = "LIME failed"

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────
# RUN SERVER
# ────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)