# explainability/lime_explainer.py
# ─────────────────────────────────────────────
# LIME-based local explanations
# for Finance Workforce Analytics XAI Tool
# ─────────────────────────────────────────────

from cProfile import label
import os, sys
from matplotlib.pylab import exp
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from lime import lime_tabular

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    LIME_NUM_FEATURES, LIME_NUM_SAMPLES,
)


# ════════════════════════════════════════════
# 1. BUILD LIME EXPLAINER
# ════════════════════════════════════════════
#This function builds a LIME explainer for a machine learning classification model.
#LIME (Local Interpretable Model-Agnostic Explanations) helps explain why a model made a particular prediction by showing the contribution of each feature.

def build_lime_explainer(
    X_train          : pd.DataFrame,
    feature_names    : list[str],
    categorical_names: dict | None = None,
) -> lime_tabular.LimeTabularExplainer:
    """
    Build a LIME LimeTabularExplainer fitted on training data.

    categorical_names: dict {col_index: [class_name, ...]} for encoded cats.
    """
    # Identify categorical indices in the feature list
    cat_indices = [
        i for i, f in enumerate(feature_names)
        if f in CATEGORICAL_FEATURES + ["loyalty_tier"]
    ]

    explainer = lime_tabular.LimeTabularExplainer(
        training_data      = X_train.values,
        feature_names      = feature_names,
        class_names        = ["Stays (0)", "At-Risk (1)"],
        categorical_features = cat_indices,
        categorical_names  = categorical_names or {},
        mode               = "classification",
        discretize_continuous = True,
        random_state       = 42,
    )
    return explainer


# ════════════════════════════════════════════
# 2. EXPLAIN SINGLE INSTANCE
# ════════════════════════════════════════════
def explain_instance_lime(
    explainer    : lime_tabular.LimeTabularExplainer,
    model        ,
    instance     : np.ndarray,
    num_features : int = LIME_NUM_FEATURES,
    num_samples  : int = LIME_NUM_SAMPLES,
):
    """Return a LIME explanation object for one employee row."""
    exp = explainer.explain_instance(
        data_row         = instance,
        predict_fn       = model.predict_proba,
        num_features     = num_features,
        num_samples      = num_samples,
        top_labels       = 1,
    )
    return exp

# ════════════════════════════════════════════
# 3. LIME PLOT (matplotlib Figure)-👉 This function visualizes how each feature contributes to a model’s prediction using LIME, helping us understand why a particular employee is classified as high or low attrition risk.
# ════════════════════════════════════════════
def lime_plot(
    exp: lime_tabular.LimeTabularExplanation,
    employee_idx : int  = 0,
    risk_prob    : float = 0.5,
) -> plt.Figure:
    """Return a matplotlib Figure of the LIME local explanation."""
    label = 1   # class "At-Risk"
    label = list(exp.local_exp.keys())[0]
    features_weights = exp.as_list(label=label)



    feats  = [fw[0] for fw in features_weights]
    weights = [fw[1] for fw in features_weights]

    colors = ["#ef4444" if w > 0 else "#22c55e" for w in weights]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feats) * 0.55)))
    ax.barh(feats, weights, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME Weight  (+ = increases risk,  − = decreases risk)", fontsize=9)
    ax.set_title(
        f"LIME – Employee #{employee_idx}  |  Attrition Risk Prob: {risk_prob:.1%}",
        fontweight="bold",
    )
    ax.invert_yaxis()

    # Add value labels
    for i, (feat, w) in enumerate(zip(feats, weights)):
        ax.text(w + (0.002 if w >= 0 else -0.002), i,
                f"{w:+.3f}", va="center",
                ha="left" if w >= 0 else "right", fontsize=8)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════
# 4. LIME TABLE (for dashboard)
#This function converts LIME explanation output into a structured DataFrame, making it easy to interpret which features increase or decrease the prediction risk.
# ════════════════════════════════════════════
def lime_explanation_df(exp) -> pd.DataFrame:
    """Return explanation as a tidy DataFrame."""
    label = 1
    label = list(exp.local_exp.keys())[0]
    rows = exp.as_list(label=label)
    df = pd.DataFrame(rows, columns=["Condition / Feature", "LIME Weight"])
    df["Impact"] = df["LIME Weight"].apply(
        lambda w: "↑ Increases Risk" if w > 0 else "↓ Decreases Risk"
    )
    df["LIME Weight"] = df["LIME Weight"].round(4)
    return df.sort_values("LIME Weight", key=abs, ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════
# 5. BATCH LIME (multiple employees)
# ════════════════════════════════════════════
def batch_lime_explanations(
    explainer    : lime_tabular.LimeTabularExplainer,
    model        ,
    X            : pd.DataFrame,
    indices      : list[int],
    num_features : int = LIME_NUM_FEATURES,
) -> dict[int, pd.DataFrame]:
    """
    Return dict {employee_index: lime_df} for a list of row indices.
    Useful for risk-report generation.
    """
    results = {}
    for idx in indices:
        exp = explain_instance_lime(
            explainer, model,
            X.iloc[idx].values,
            num_features=num_features,
        )
        results[idx] = lime_explanation_df(exp)
    return results
