# 💼 Finance Workforce Analytics – Explainable AI Tool

A production-grade **XAI (Explainable AI)** system for finance-sector workforce
attrition analysis. Combines **XGBoost** predictions with **SHAP** global explanations
and **LIME** local explanations, served through a **Streamlit** dashboard and a
**FastAPI** REST backend.

---

## 📁 Project Structure

```
finance_xai/
│
├── data/                          ← auto-created on first run
│   ├── raw_workforce.csv
│   └── processed_workforce.csv
│
├── models/                        ← auto-created after training
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── eval_plots.png
│   └── feature_importance.png
│
├── preprocessing/
│   └── preprocess.py              ← data load · clean · engineer · encode
│
├── training/
│   └── train_model.py             ← XGBoost train · evaluate · save
│
├── explainability/
│   ├── shap_explainer.py          ← SHAP TreeExplainer · global & local plots
│   └── lime_explainer.py          ← LIME LimeTabularExplainer · local plots
│
├── api/
│   └── app.py                     ← FastAPI REST API (/predict, /explain/*)
│
├── dashboard/
│   └── dashboard.py               ← Streamlit UI (5 tabs)
│
├── utils/
│   └── config.py                  ← all paths · hyper-params · constants
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Recommended Path)

### 1 · Create & activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 · Install dependencies
```bash
pip install -r requirements.txt
```

### 3 · (Optional) Train model manually
The dashboard auto-trains on first launch, but you can run it explicitly:
```bash
python preprocessing/preprocess.py   # generates synthetic data + preprocesses
python training/train_model.py        # trains XGBoost + saves model artifacts
```

### 4 · Launch Streamlit dashboard
```bash
streamlit run dashboard/dashboard.py
```
Open http://localhost:8501 in your browser.

### 5 · (Optional) Start FastAPI backend
```bash
uvicorn api.app:app --reload --port 8000
```
Swagger UI available at http://localhost:8000/docs

---

## 🗂️ Module Guide

| File | Purpose |
|------|---------|
| `utils/config.py` | Central config – all paths, feature lists, hyper-params |
| `preprocessing/preprocess.py` | Load → clean → feature engineering → encode/scale → split |
| `training/train_model.py` | Train XGBClassifier, evaluate (AUC, F1), save to disk |
| `explainability/shap_explainer.py` | SHAP TreeExplainer: summary bar, beeswarm, waterfall, dependence, force |
| `explainability/lime_explainer.py` | LIME: per-employee explanation bar + DataFrame |
| `api/app.py` | FastAPI: `/predict`, `/feature-importance/shap`, `/health` |
| `dashboard/dashboard.py` | Streamlit: 5-tab UI covering overview → prediction → global SHAP → LIME → risk report |

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | KPIs, attrition by department, salary vs engagement scatter, feature histograms |
| **Predict Employee** | Form input → risk probability gauge → SHAP waterfall or LIME bar |
| **SHAP Global** | Global feature importance bar, beeswarm, dependence plots |
| **LIME Local** | Select any employee index → local LIME explanation |
| **Risk Report** | Filterable at-risk employee table with CSV download |

---

## 🔌 API Usage (FastAPI)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "years_at_company": 3,
    "monthly_salary": 6500,
    "performance_score": 2,
    "overtime_hours": 25,
    "training_hours": 10,
    "projects_completed": 4,
    "absenteeism_days": 9,
    "bonus_pct": 3.5,
    "engagement_score": 3.5,
    "department": "Risk",
    "job_level": "Mid",
    "education": "Bachelors",
    "gender": "Male",
    "marital_status": "Single",
    "explainer": "both"
  }'
```

---

## 🧠 XAI Concepts Used

- **SHAP (SHapley Additive exPlanations)** – game-theory based, model-wide feature
  attribution; precise global importances and per-employee waterfall plots.
- **LIME (Local Interpretable Model-agnostic Explanations)** – fits a local linear
  approximation around each prediction; great for individual employee stories.
- **XGBoost built-in importance** – fast F-score importance as a baseline cross-check.

---

## 🔧 Bring Your Own Data

1. Place your CSV at `data/raw_workforce.csv`
2. Update `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, and `TARGET_COLUMN` in `utils/config.py`
3. Re-run preprocessing + training:
   ```bash
   python preprocessing/preprocess.py
   python training/train_model.py
   streamlit run dashboard/dashboard.py
   ```
