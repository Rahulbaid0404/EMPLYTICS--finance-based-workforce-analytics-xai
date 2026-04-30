# utils/config.py
# ─────────────────────────────────────────────
# Central configuration for the Finance XAI Tool
# ─────────────────────────────────────────────

import os

# ── Data paths ──────────────────────────────
DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DATA_PATH  = os.path.join(DATA_DIR, "raw_workforce.csv")
PROC_DATA_PATH = os.path.join(DATA_DIR, "processed_workforce.csv")
MODEL_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH     = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH    = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH   = os.path.join(MODEL_DIR, "label_encoders.pkl")

# ── Target & Features ───────────────────────
TARGET_COLUMN = "attrition_risk"   # binary: 0 = stays, 1 = at-risk

NUMERIC_FEATURES = [
    "age",
    "years_at_company",
    "monthly_salary",
    "performance_score",    # 1-5
    "overtime_hours",
    "training_hours",
    "projects_completed",
    "absenteeism_days",
    "bonus_pct",            # % of base salary
    "engagement_score",     # 1-10
]

CATEGORICAL_FEATURES = [
    "department",
    "job_level",            # Junior / Mid / Senior / Lead
    "education",            # Bachelors / Masters / PhD / Other
    "gender",
    "marital_status",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Model hyper-params ───────────────────────
XGB_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 5,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric"     : "logloss",
    "random_state"    : 42,
}

TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── SHAP settings ───────────────────────────
SHAP_MAX_DISPLAY = 15    # features shown in summary plot

# ── LIME settings ───────────────────────────
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES  = 3000

# ── Streamlit page config ───────────────────
PAGE_TITLE = "Finance Workforce Analytics – XAI"
PAGE_ICON  = "💼"
LAYOUT     = "wide"
