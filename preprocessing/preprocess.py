# preprocessing/preprocess.py
# ─────────────────────────────────────────────
# Data ingestion, cleaning & feature engineering
# for Finance Workforce Analytics XAI Tool
# ─────────────────────────────────────────────

import os, pickle
import numpy  as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── allow running standalone OR as imported module ──
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import (
    RAW_DATA_PATH, PROC_DATA_PATH,
    MODEL_DIR, SCALER_PATH, ENCODER_PATH,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    ALL_FEATURES, TARGET_COLUMN,
    TEST_SIZE, RANDOM_SEED,
)


# ════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATOR (for demo / CI)
# ════════════════════════════════════════════
def generate_synthetic_data(n_samples: int = 2000, save: bool = True) -> pd.DataFrame:
    """
    Create a realistic-looking finance workforce dataset.
    Call this once if you don't have real data.
    """
    np.random.seed(RANDOM_SEED)

    departments   = ["Finance", "Risk", "Compliance", "Operations",
                     "Treasury", "Audit", "Investment Banking"]
    job_levels    = ["Junior", "Mid", "Senior", "Lead"]
    educations    = ["Bachelors", "Masters", "PhD", "Other"]
    genders       = ["Male", "Female", "Non-binary"]
    marital_stats = ["Single", "Married", "Divorced"]

    df = pd.DataFrame({
        "employee_id"       : range(1, n_samples + 1),
        "age"               : np.random.randint(22, 62, n_samples),
        "years_at_company"  : np.random.randint(0, 30, n_samples),
        "monthly_salary"    : np.random.normal(8000, 3000, n_samples).clip(2500, 30000).astype(int),
        "performance_score" : np.random.choice([1, 2, 3, 4, 5], n_samples,
                                                p=[0.05, 0.10, 0.35, 0.35, 0.15]),
        "overtime_hours"    : np.random.exponential(8, n_samples).clip(0, 60).astype(int),
        "training_hours"    : np.random.randint(0, 80, n_samples),
        "projects_completed": np.random.randint(1, 20, n_samples),
        "absenteeism_days"  : np.random.poisson(3, n_samples).clip(0, 20),
        "bonus_pct"         : np.random.uniform(0, 30, n_samples).round(1),
        "engagement_score"  : np.random.uniform(1, 10, n_samples).round(1),
        "department"        : np.random.choice(departments, n_samples),
        "job_level"         : np.random.choice(job_levels, n_samples),
        "education"         : np.random.choice(educations, n_samples),
        "gender"            : np.random.choice(genders, n_samples),
        "marital_status"    : np.random.choice(marital_stats, n_samples),
    })

    # ── Simulate target with domain-realistic logic ──
    risk_score = (
        (df["overtime_hours"] > 20).astype(int) * 2
        + (df["engagement_score"] < 5).astype(int) * 3
        + (df["performance_score"] < 3).astype(int) * 2
        + (df["absenteeism_days"] > 7).astype(int) * 2
        + (df["bonus_pct"] < 5).astype(int) * 1
        + (df["years_at_company"] < 2).astype(int) * 1
        + np.random.randint(0, 3, n_samples)
    )
    df[TARGET_COLUMN] = (risk_score >= 5).astype(int)

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    if save:
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"[preprocess] Synthetic data saved → {RAW_DATA_PATH}")

    return df


# ════════════════════════════════════════════
# 2. LOAD DATA
# ════════════════════════════════════════════
def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print("[preprocess] Raw data not found – generating synthetic dataset …")
        return generate_synthetic_data()
    return pd.read_csv(path)


# ════════════════════════════════════════════
# 3. CLEAN & VALIDATE
# ════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill numeric NaN with median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical NaN with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Clip outliers (IQR × 3)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    return df


# ════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Salary-to-age ratio (proxy for career growth)
    if "monthly_salary" in df.columns and "age" in df.columns:
        df["salary_age_ratio"] = (df["monthly_salary"] / df["age"]).round(2)

    # Workload index
    if "overtime_hours" in df.columns and "projects_completed" in df.columns:
        df["workload_index"] = (df["overtime_hours"] * 0.4 +
                                df["projects_completed"] * 0.6).round(2)

    # Loyalty tier
    if "years_at_company" in df.columns:
        df["loyalty_tier"] = pd.cut(
            df["years_at_company"],
            bins=[-1, 2, 5, 10, 100],
            labels=["New", "Growing", "Established", "Veteran"]
        ).astype(str)

    return df


# ════════════════════════════════════════════
# 5. ENCODE & SCALE
# ════════════════════════════════════════════
def encode_and_scale(
    df            : pd.DataFrame,
    fit           : bool = True,
    scaler        : StandardScaler | None = None,
    label_encoders: dict | None = None,
) -> tuple[pd.DataFrame, StandardScaler, dict]:
    """
    Encode categoricals with LabelEncoder, scale numerics with StandardScaler.
    Pass fit=False + existing scaler/encoders at inference time.
    """
    df = df.copy()

    # Collect all categorical cols (original + engineered)
    cat_cols = [c for c in CATEGORICAL_FEATURES + ["loyalty_tier"] if c in df.columns]
    num_cols = [c for c in NUMERIC_FEATURES + ["salary_age_ratio", "workload_index"]
                if c in df.columns]

    # ── Label Encoding ──────────────────────
    if fit:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            if col in label_encoders:
                known = set(label_encoders[col].classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else label_encoders[col].classes_[0]
                )
                df[col] = label_encoders[col].transform(df[col])

    # ── Standard Scaling ────────────────────
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler, label_encoders


# ════════════════════════════════════════════
# 6. FULL PIPELINE  (train-time)
# ════════════════════════════════════════════
def run_preprocessing_pipeline() -> tuple:
    """
    End-to-end: load → clean → engineer → encode/scale → split → save artifacts.
    Returns (X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders)
    """
    print("[preprocess] Loading data …")
    df = load_data()

    print("[preprocess] Cleaning …")
    df = clean_data(df)

    print("[preprocess] Engineering features …")
    df = engineer_features(df)

    # Save processed data
    os.makedirs(os.path.dirname(PROC_DATA_PATH), exist_ok=True)
    df.to_csv(PROC_DATA_PATH, index=False)

    # Separate target
    X = df.drop(columns=[TARGET_COLUMN, "employee_id"], errors="ignore")
    y = df[TARGET_COLUMN]

    # Keep only model features
    feature_cols = [c for c in X.columns if c in
                    NUMERIC_FEATURES + CATEGORICAL_FEATURES +
                    ["salary_age_ratio", "workload_index", "loyalty_tier"]]
    X = X[feature_cols]

    # Encode & scale (fit mode)
    X_enc, scaler, label_encoders = encode_and_scale(X, fit=True)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Persist scaler & encoders
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH,  "wb") as f: pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(label_encoders, f)
    print(f"[preprocess] Scaler  → {SCALER_PATH}")
    print(f"[preprocess] Encoders → {ENCODER_PATH}")

    return X_train, X_test, y_train, y_test, list(X_enc.columns), scaler, label_encoders


# ════════════════════════════════════════════
# CLI entry-point
# ════════════════════════════════════════════
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, *_ = run_preprocessing_pipeline()
    print(f"\n✔  Preprocessing complete")
    print(f"   Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"   Features ({len(features)}): {features}")
    print(f"   Target balance:\n{y_train.value_counts(normalize=True).round(3)}")
