# explainability/shap_explainer.py
# ─────────────────────────────────────────────
# SHAP-based global & local explanations
# for Finance Workforce Analytics XAI Tool
# ─────────────────────────────────────────────

import os, sys
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless backend for Streamlit

import shap

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.config import SHAP_MAX_DISPLAY


# ════════════════════════════════════════════
# 1. BUILD EXPLAINER
# ════════════════════════════════════════════
def build_shap_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Create a TreeExplainer (fast, exact SHAP values for tree-based models).
    X_background: a representative sample of training data (≥100 rows recommended).
    """
    explainer = shap.TreeExplainer(
        model,
        data              = shap.sample(X_background, min(200, len(X_background))),
        feature_perturbation = "interventional",
    )
    return explainer


# ════════════════════════════════════════════
# 2. COMPUTE SHAP VALUES
# ════════════════════════════════════════════
def compute_shap_values(
    explainer : shap.TreeExplainer,
    X         : pd.DataFrame,
) -> shap.Explanation:
    """Return a shap.Explanation object for the positive class (attrition = 1)."""
    sv = explainer(X, check_additivity=False)
    # For binary models TreeExplainer returns shape (n, features, 2) – take class-1
    if len(sv.values.shape) == 3:
        return shap.Explanation(
            values        = sv.values[:, :, 1],
            base_values   = sv.base_values[:, 1]
                            if len(sv.base_values.shape) > 1
                            else sv.base_values,
            data          = sv.data,
            feature_names = sv.feature_names,
        )
    return sv


# ════════════════════════════════════════════
# 3. GLOBAL PLOTS (return matplotlib Figure)
# ════════════════════════════════════════════
def shap_summary_bar(shap_explanation: shap.Explanation) -> plt.Figure:
    """Global feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_explanation, max_display=SHAP_MAX_DISPLAY,
                   show=False, ax=ax)
    ax.set_title("SHAP – Global Feature Importance", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig

#feature impact distribution
def shap_beeswarm(shap_explanation: shap.Explanation) -> plt.Figure:
    """Beeswarm / summary plot showing direction of feature effects."""
    import matplotlib.pyplot as plt

def shap_beeswarm(shap_explanation):

    shap.plots.beeswarm(
        shap_explanation,
        max_display=SHAP_MAX_DISPLAY,
        show=False
    )

    fig = plt.gcf()
    fig.suptitle(
        "SHAP – Feature Impact Distribution (Beeswarm)",
        fontweight="bold"
    )

    fig.tight_layout()
    return fig


def shap_dependence(
    shap_explanation : shap.Explanation,
    feature          : str,
    interaction_feat : str | None = None,
) -> plt.Figure:
    """Dependence plot for one feature (optionally coloured by an interaction)."""
    feat_idx = list(shap_explanation.feature_names).index(feature)
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.plots.scatter(
        shap_explanation[:, feat_idx],
        color   = shap_explanation[:, interaction_feat]
                  if interaction_feat else shap_explanation,
        ax      = ax,
        show    = False,
    )
    ax.set_title(f"SHAP Dependence – {feature}", fontweight="bold")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════
# 4. LOCAL (single-employee) WATERFALL PLOT
# ════════════════════════════════════════════
def shap_waterfall_single(
    shap_explanation : shap.Explanation,
    idx              : int = 0,
) -> plt.Figure:
    """Waterfall plot explaining one employee prediction."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_explanation[idx], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP – Employee #{idx} Local Explanation", fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════
# 5. FORCE PLOT (HTML) – for Streamlit embed
# ════════════════════════════════════════════
def shap_force_plot_html(
    explainer        : shap.TreeExplainer,
    shap_explanation : shap.Explanation,
    idx              : int = 0,
) -> str:
    """Return an HTML string of the SHAP force plot for one employee."""
    shap.initjs()
    plot = shap.force_plot(
        base_value   = float(shap_explanation.base_values[idx]),
        shap_values  = shap_explanation.values[idx],
        features     = shap_explanation.data[idx],
        feature_names= shap_explanation.feature_names,
        matplotlib   = False,
    )
    return f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"


# ════════════════════════════════════════════
# 6. SHAP SUMMARY TABLE (for dashboard table)
# ════════════════════════════════════════════
def shap_feature_summary_df(shap_explanation: shap.Explanation) -> pd.DataFrame:
    """
    Return a DataFrame: feature | mean_abs_shap | direction
    sorted by importance descending.
    """
    mean_abs = np.abs(shap_explanation.values).mean(axis=0)
    mean_val = shap_explanation.values.mean(axis=0)

    df = pd.DataFrame({
        "Feature"        : shap_explanation.feature_names,
        "Mean |SHAP|"    : mean_abs.round(4),
        "Avg Effect"     : mean_val.round(4),
        "Direction"      : ["↑ Increases Risk" if v > 0
                            else "↓ Decreases Risk" for v in mean_val],
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    return df
