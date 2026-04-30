# training/train_model.py
# ─────────────────────────────────────────────
# Model training, evaluation & persistence
# for Finance Workforce Analytics XAI Tool
# ─────────────────────────────────────────────

import os, sys, pickle
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, RocCurveDisplay,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(__file__))
from utils.config import MODEL_PATH, XGB_PARAMS, MODEL_DIR
from preprocessing.preprocess import run_preprocessing_pipeline


# ════════════════════════════════════════════
# 1. TRAIN
# ════════════════════════════════════════════
def train_model(X_train, y_train) -> XGBClassifier:
    """Fit XGBoost classifier."""
    print("[train] Fitting XGBClassifier …")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )
    return model


# ════════════════════════════════════════════
# 2. EVALUATE
# ════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, feature_names: list) -> dict:
    """Compute & print evaluation metrics; return as dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "─" * 50)
    print(f"  ROC-AUC : {auc:.4f}")
    print("─" * 50)
    print(classification_report(y_test, y_pred,
                                  target_names=["Stays (0)", "At-Risk (1)"]))

    # ── Confusion Matrix plot ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Stays", "At-Risk"],
        ax=axes[0], colorbar=False,
    )
    axes[0].set_title("Confusion Matrix", fontweight="bold")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve  (AUC = {auc:.3f})", fontweight="bold")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_path = os.path.join(MODEL_DIR, "eval_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"[train] Evaluation plots → {plot_path}")

    # ── Feature Importance plot ─────────────
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#0ea5e9")
    ax.invert_yaxis()
    ax.set_title("XGBoost Feature Importances (Top 15)", fontweight="bold")
    ax.set_xlabel("F-score")
    fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=120)
    plt.close()
    print(f"[train] Feature importance → {fi_path}")

    return {
        "roc_auc"          : round(auc, 4),
        "precision_at_risk": round(report["1"]["precision"], 4),
        "recall_at_risk"   : round(report["1"]["recall"], 4),
        "f1_at_risk"       : round(report["1"]["f1-score"], 4),
        "accuracy"         : round(report["accuracy"], 4),
    }


# ════════════════════════════════════════════
# 3. SAVE MODEL
# ════════════════════════════════════════════
def save_model(model) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved → {MODEL_PATH}")


# ════════════════════════════════════════════
# 4. LOAD MODEL
# ════════════════════════════════════════════
def load_model() -> XGBClassifier:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ════════════════════════════════════════════
# CLI entry-point
# ════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1: Preprocess
    X_train, X_test, y_train, y_test, feature_names, *_ = run_preprocessing_pipeline()

    # Step 2: Train
    model = train_model(X_train, y_train)

    # Step 3: Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # Step 4: Save
    save_model(model)
    print("\n✔  Training pipeline complete!")
