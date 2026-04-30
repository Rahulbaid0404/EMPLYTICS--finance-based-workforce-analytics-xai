# dashboard/dashboard.py
# ─────────────────────────────────────────────
# Streamlit Dashboard – Finance Workforce XAI
# Run: streamlit run dashboard/dashboard.py
# ─────────────────────────────────────────────

import os, sys, pickle, warnings
import plotly.express as px
warnings.filterwarnings("ignore")
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express     as px
import plotly.graph_objects as go
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT,
    PROC_DATA_PATH, MODEL_PATH, SCALER_PATH, ENCODER_PATH,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
)

# ════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════
st.set_page_config(
    page_title = PAGE_TITLE,
    page_icon  = PAGE_ICON,
    layout     = LAYOUT,
    initial_sidebar_state = "expanded",
)


# ════════════════════════════════════════════
# CUSTOM CSS – finance dark-glass aesthetic
# ════════════════════════════════════════════
st.markdown("""
<style>
/* ── base ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e2e8f0;
}
.stApp { background: #0a0f1e; }

/* ── sidebar ──────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #0a1020 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.8rem !important; }

/* ── metric cards ─────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d2035 0%, #112944 100%);
    border: 1px solid #1e4976;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(0,100,200,0.15);
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.8rem !important;
    color: #38bdf8 !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── section headings ─────────────────────── */
h1 { font-size: 1.8rem !important; color: #e0f2fe !important; letter-spacing: -0.5px; }
h2 { font-size: 1.2rem !important; color: #bae6fd !important; border-bottom: 1px solid #1e3a5f; padding-bottom: 0.4rem; }
h3 { font-size: 1rem !important; color: #7dd3fc !important; }

/* ── tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: #0d1a2e; gap: 4px; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    color: #64748b; font-weight: 600; font-size: 0.82rem;
    border-radius: 6px; padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #0ea5e9) !important;
    color: white !important;
}

/* ── expander ─────────────────────────────── */
details { background: #0d1a2e !important; border: 1px solid #1e3a5f !important; border-radius: 8px; }

/* ── dataframe ────────────────────────────── */
.dataframe { background: #0d1a2e !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }
thead th { background: #1e3a5f !important; color: #7dd3fc !important; }

/* ── risk badge ───────────────────────────── */
.risk-high {
    display:inline-block; background:#7f1d1d; color:#fca5a5;
    border:1px solid #ef4444; border-radius:6px; padding:4px 14px;
    font-weight:700; font-size:1rem; letter-spacing:1px;
}
.risk-low {
    display:inline-block; background:#052e16; color:#86efac;
    border:1px solid #22c55e; border-radius:6px; padding:4px 14px;
    font-weight:700; font-size:1rem; letter-spacing:1px;
}
/* ── button ───────────────────────────────── */
.stButton>button {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.85; transform: translateY(-1px); }

/* ── progress bar ─────────────────────────── */
.stProgress > div > div { background: linear-gradient(90deg, #0ea5e9, #6366f1) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# ARTIFACT LOADERS  (cached)
# ════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model & artifacts …")
def load_all_artifacts():
    """Load model, scaler, encoders, processed data. Train if missing."""
    if not os.path.exists(MODEL_PATH):
        st.info("🔧 Model not found — running training pipeline … (first run only)")
        from preprocessing.preprocess import run_preprocessing_pipeline
        from training.train_model     import train_model, save_model
        X_tr, X_te, y_tr, y_te, feats, scaler, le = run_preprocessing_pipeline()
        model = train_model(X_tr, y_tr)
        save_model(model)

    with open(MODEL_PATH,  "rb") as f: model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(ENCODER_PATH,"rb") as f: le = pickle.load(f)

    df = pd.read_csv(PROC_DATA_PATH)
    from preprocessing.preprocess import engineer_features, encode_and_scale
    df = engineer_features(df)
    y  = df[TARGET_COLUMN]
    X  = df.drop(columns=[TARGET_COLUMN, "employee_id"], errors="ignore")
    X_enc, _, _ = encode_and_scale(X, fit=False, scaler=scaler, label_encoders=le)

    return model, scaler, le, df, X_enc, y


@st.cache_resource(show_spinner="Building SHAP explainer …")
def get_shap_explainer(_model, _X_enc):
    from explainability.shap_explainer import build_shap_explainer
    sample = _X_enc.sample(min(300, len(_X_enc)), random_state=42)
    return build_shap_explainer(_model, sample)


@st.cache_resource(show_spinner="Building LIME explainer …")
def get_lime_explainer(_X_enc):
    from explainability.lime_explainer import build_lime_explainer
    return build_lime_explainer(_X_enc, list(_X_enc.columns))


# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════
def _mpl_to_st(fig):
    """Render matplotlib figure in Streamlit with dark background."""
    fig.patch.set_facecolor("#0a0f1e")
    for ax in fig.axes:
        ax.set_facecolor("#0d1a2e")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def risk_gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(prob * 100, 1),
        delta = {"reference": 50, "increasing": {"color": "#ef4444"},
                 "decreasing": {"color": "#22c55e"}},
        gauge = {
            "axis"      : {"range": [0, 100], "tickcolor": "#64748b"},
            "bar"       : {"color": "#ef4444" if prob >= 0.5 else "#22c55e"},
            "bgcolor"   : "#0d1a2e",
            "bordercolor": "#1e3a5f",
            "steps"     : [
                {"range": [0,  30], "color": "#052e16"},
                {"range": [30, 60], "color": "#1c3b1e"},
                {"range": [60,100], "color": "#3b0d0d"},
            ],
            "threshold" : {"line": {"color": "#fbbf24", "width": 3},
                           "thickness": 0.75, "value": 50},
        },
        title  = {"text": "Attrition Risk %", "font": {"color": "#94a3b8", "size": 14}},
        number = {"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
    ))
    fig.update_layout(
        height=230, margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="#0a0f1e", font_color="#e2e8f0",
    )
    return fig


# ════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════
def main():
    # ── load ───────────────────────────────
    model, scaler, le, df_raw, X_enc, y = load_all_artifacts()
    shap_exp_obj  = get_shap_explainer(model, X_enc)
    lime_exp_obj  = get_lime_explainer(X_enc)
    feature_names = list(X_enc.columns)

    # ── header ──────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("# 💼 Finance Workforce Analytics")
        st.markdown(
            "<span style='color:#64748b;font-size:0.9rem'>"
            "Explainable AI · Attrition Risk · SHAP & LIME"
            "</span>", unsafe_allow_html=True
        )
    with col_h2:
        at_risk_pct = (y == 1).mean() * 100
        st.metric("Dataset At-Risk %", f"{at_risk_pct:.1f}%")

    st.markdown("---")

    # ════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Predict Employee",
        "🌐 SHAP Global",
        "🧪 LIME Local",
        "📋 Risk Report",
    ])
    
    # ────────────────────────────────────────
    # TAB 1 · OVERVIEW
    # ────────────────────────────────────────
    with tab1:
        st.header("Workforce Overview")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Employees",  f"{len(df_raw):,}")
        k2.metric("At-Risk Employees",f"{int(y.sum()):,}", delta=f"{at_risk_pct:.1f}%", delta_color="inverse")
        k3.metric("Avg Salary",       f"${df_raw['monthly_salary'].mean():,.0f}")
        k4.metric("Avg Engagement",   f"{df_raw['engagement_score'].mean():.1f} / 10")

        st.markdown("###")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Attrition by Department")
            dept_risk = (
                df_raw.groupby("department")[TARGET_COLUMN]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={"sum": "At-Risk", "count": "Total"})
            )
            dept_risk["Risk %"] = (dept_risk["At-Risk"] / dept_risk["Total"] * 100).round(1)
            fig = px.bar(
                dept_risk.sort_values("Risk %", ascending=False),
                x="department", y="Risk %", color="Risk %",
                color_continuous_scale=["#1e40af", "#ef4444"],
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1a2e",
                               coloraxis_showscale=False, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Salary vs Engagement (coloured by Risk)")
            fig2 = px.scatter(
                df_raw.sample(min(600, len(df_raw))),
                x="monthly_salary", y="engagement_score",
                color=TARGET_COLUMN,
                color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                opacity=0.6, template="plotly_dark",
                labels={TARGET_COLUMN: "Attrition Risk"},
            )
            fig2.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1a2e",
                                margin=dict(t=20))
            st.plotly_chart(fig2, use_container_width=True)

        # Distribution
        st.subheader("Feature Distributions")
        feat_choice = st.selectbox("Select numeric feature", NUMERIC_FEATURES)
        fig3 = px.histogram(
            df_raw, x=feat_choice, color=TARGET_COLUMN,
            barmode="overlay", nbins=40, opacity=0.75,
            color_discrete_map={0: "#0ea5e9", 1: "#ef4444"},
            template="plotly_dark",
            labels={TARGET_COLUMN: "Attrition Risk"},
        )
        fig3.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1a2e", margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)


    # ────────────────────────────────────────
    # TAB 2 · PREDICT EMPLOYEE
    # ────────────────────────────────────────
    with tab2:
        st.header("Single Employee Risk Prediction")
        st.caption("Fill in employee details → get risk probability + SHAP / LIME explanation")

        with st.form("employee_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**📌 Demographics**")
                age             = st.slider("Age", 18, 65, 35)
                gender          = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
                marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                education       = st.selectbox("Education", ["Bachelors", "Masters", "PhD", "Other"])

            with c2:
                st.markdown("**🏢 Job Details**")
                department      = st.selectbox("Department",
                                                ["Finance", "Risk", "Compliance", "Operations",
                                                 "Treasury", "Audit", "Investment Banking"])
                job_level       = st.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead"])
                years_at_company= st.slider("Years at Company", 0, 40, 5)
                monthly_salary  = st.number_input("Monthly Salary ($)", 2500, 30000, 7000, step=500)

            with c3:
                st.markdown("**📈 Performance & Wellbeing**")
                performance_score  = st.slider("Performance Score (1-5)", 1, 5, 3)
                engagement_score   = st.slider("Engagement Score (1-10)", 1.0, 10.0, 6.0, step=0.5)
                overtime_hours     = st.slider("Weekly Overtime Hours", 0, 60, 10)
                training_hours     = st.slider("Annual Training Hours", 0, 100, 20)
                projects_completed = st.slider("Projects Completed", 0, 20, 5)
                absenteeism_days   = st.slider("Absenteeism Days / Year", 0, 20, 3)
                bonus_pct          = st.slider("Bonus % of Salary", 0.0, 30.0, 10.0, step=0.5)

            explainer_choice = st.radio(
                "Explanation Method", ["SHAP", "LIME", "Both"],
                horizontal=True,
            )
            submitted = st.form_submit_button("🔮  Predict Attrition Risk", use_container_width=True)

        if submitted:
            # Build raw row
            from preprocessing.preprocess import engineer_features, encode_and_scale
            row = {
                "age": age, "years_at_company": years_at_company,
                "monthly_salary": monthly_salary, "performance_score": performance_score,
                "overtime_hours": overtime_hours, "training_hours": training_hours,
                "projects_completed": projects_completed, "absenteeism_days": absenteeism_days,
                "bonus_pct": bonus_pct, "engagement_score": engagement_score,
                "department": department, "job_level": job_level, "education": education,
                "gender": gender, "marital_status": marital_status,
            }
            df_input = pd.DataFrame([row])
            df_input = engineer_features(df_input)
            df_enc, _, _ = encode_and_scale(
                df_input, fit=False, scaler=scaler, label_encoders=le
            )
            df_enc = df_enc.reindex(columns=feature_names, fill_value=0)

            prob  = float(model.predict_proba(df_enc)[0, 1])
            label = int(prob >= 0.5)

            # ── Result row ───────────────────
            r1, r2, r3 = st.columns([1.5, 1.5, 1])
            with r1:
                st.plotly_chart(risk_gauge(prob), use_container_width=True)
            with r2:
                st.markdown("### Prediction")
                badge = (
                    '<span class="risk-high">⚠ AT-RISK</span>'
                    if label == 1
                    else '<span class="risk-low">✔ LIKELY STAYS</span>'
                )
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(f"**Probability:** `{prob:.1%}`")
                st.markdown(f"**Threshold:** `50%`")
            with r3:
                st.metric("Risk Score", f"{prob:.1%}",
                           delta="High" if label == 1 else "Low",
                           delta_color="inverse")

            # ── SHAP ─────────────────────────
            if explainer_choice in ("SHAP", "Both"):
                st.markdown("---")
                st.subheader("🔵 SHAP Explanation")
                from explainability.shap_explainer import (
                    compute_shap_values, shap_waterfall_single, shap_feature_summary_df
                )
                shap_exp = compute_shap_values(shap_exp_obj, df_enc)
                fig_wf   = shap_waterfall_single(shap_exp, idx=0)
                _mpl_to_st(fig_wf)

                with st.expander("SHAP Feature Table"):
                    st.dataframe(
                        shap_feature_summary_df(shap_exp).head(10),
                        use_container_width=True,
                    )

            # ── LIME ─────────────────────────
            if explainer_choice in ("LIME", "Both"):
                st.markdown("---")
                st.subheader("🟠 LIME Explanation")
                from explainability.lime_explainer import (
                    explain_instance_lime, lime_plot, lime_explanation_df
                )
                exp_lime = explain_instance_lime(
                    lime_exp_obj, model, df_enc.values[0]
                )
                fig_lime = lime_plot(exp_lime, employee_idx=0, risk_prob=prob)
                _mpl_to_st(fig_lime)

                with st.expander("LIME Feature Table"):
                    st.dataframe(
                        lime_explanation_df(exp_lime).head(10),
                        use_container_width=True,
                    )

    # ────────────────────────────────────────
    # TAB 3 · SHAP GLOBAL
    # ────────────────────────────────────────
    with tab3:
        st.header("SHAP – Global Model Explanations")
        st.caption("Computed on a random sample of the training data")

        from explainability.shap_explainer import (
            compute_shap_values, shap_summary_bar, shap_beeswarm,
            shap_dependence, shap_feature_summary_df
        )

        n_sample = st.slider("Sample size for SHAP computation", 50, 500, 200, 50)
        if st.button("Compute Global SHAP"):
            with st.spinner("Computing SHAP values …"):
                sample_X = X_enc.sample(n_sample, random_state=42)
                shap_glob = compute_shap_values(shap_exp_obj, sample_X)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Feature Importance (Bar)")
                _mpl_to_st(shap_summary_bar(shap_glob))
            with col2:
                st.subheader("Feature Impact (Beeswarm)")
                _mpl_to_st(shap_beeswarm(shap_glob))

            st.subheader("SHAP Summary Table")
            st.dataframe(shap_feature_summary_df(shap_glob), use_container_width=True)

            st.subheader("Dependence Plot")
            feat_dep = st.selectbox("Feature", feature_names, key="dep_feat")
            _mpl_to_st(shap_dependence(shap_glob, feat_dep))
        else:
            st.info("👆 Click **Compute Global SHAP** to run the analysis.")


    # ────────────────────────────────────────
    # TAB 4 · LIME LOCAL
    # ────────────────────────────────────────
    with tab4:
        st.header("LIME – Local Explanations")
        st.caption("Explain any specific employee from the dataset")

        idx = st.number_input("Employee row index", 0, len(X_enc) - 1, 0, step=1)

        if st.button("Explain with LIME"):
            from explainability.lime_explainer import (
                explain_instance_lime, lime_plot, lime_explanation_df
            )
            with st.spinner("Running LIME …"):
                prob  = float(model.predict_proba(X_enc.iloc[[idx]])[0, 1])
                exp_l = explain_instance_lime(
                    lime_exp_obj, model, X_enc.iloc[idx].values
                )

            st.markdown(f"**Employee #{idx}** — Attrition Risk: `{prob:.1%}`")
            _mpl_to_st(lime_plot(exp_l, employee_idx=idx, risk_prob=prob))

            with st.expander("Detailed LIME Table"):
                st.dataframe(lime_explanation_df(exp_l), use_container_width=True)

            # Raw features
            with st.expander("Raw Input Features"):
                raw = df_raw.iloc[idx][NUMERIC_FEATURES + CATEGORICAL_FEATURES]
                st.dataframe(raw.to_frame("Value"), use_container_width=True)


    # ────────────────────────────────────────
    # TAB 5 · RISK REPORT
    # ────────────────────────────────────────
    with tab5:
        st.header("📋 At-Risk Employee Report")

        proba_all = model.predict_proba(X_enc)[:, 1]
        df_report = df_raw.copy()
        df_report["risk_score"]  = proba_all.round(3)
        df_report["risk_label"]  = (proba_all >= 0.5).astype(int)
        df_report["risk_bucket"] = pd.cut(
            proba_all,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["Low (< 30%)", "Medium (30–50%)", "High (50–70%)", "Critical (> 70%)"]
        )

        # Filters
        f1, f2, f3 = st.columns(3)
        dept_filter  = f1.multiselect("Department", df_raw["department"].unique().tolist(),
                                       default=df_raw["department"].unique().tolist())
        level_filter = f2.multiselect("Job Level", df_raw["job_level"].unique().tolist(),
                                       default=df_raw["job_level"].unique().tolist())
        min_risk     = f3.slider("Min Risk Score", 0.0, 1.0, 0.5, 0.05)

        filtered = df_report[
            (df_report["department"].isin(dept_filter)) &
            (df_report["job_level"].isin(level_filter)) &
            (df_report["risk_score"] >= min_risk)
        ].sort_values("risk_score", ascending=False)

        st.markdown(f"**{len(filtered):,} employees** matching filters")

        # Risk distribution
        bucket_counts = (
            filtered["risk_bucket"].value_counts()
            .reindex(["Low (< 30%)", "Medium (30–50%)", "High (50–70%)", "Critical (> 70%)"])
            .fillna(0)
            )
        fig_b = px.bar(
            x=bucket_counts.index, y=bucket_counts.values,
            color=bucket_counts.index,
            color_discrete_map={
                "Low (< 30%)": "#22c55e",
                "Medium (30–50%)": "#f59e0b",
                "High (50–70%)": "#f97316",
                "Critical (> 70%)": "#ef4444",
            },
            template="plotly_dark",
            labels={"x": "Risk Bucket", "y": "Count"},
        )
        fig_b.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1a2e",
                             showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_b, use_container_width=True)

        # Table
        display_cols = [
            "employee_id", "department", "job_level", "age",
            "monthly_salary", "performance_score", "engagement_score",
            "risk_score", "risk_bucket",
        ]
        avail = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[avail].head(100)
            .style
            .background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
            use_container_width=True,
        )
        # CSV Download
        csv = filtered[avail].to_csv(index=False)
        st.download_button(
            "⬇  Download Report (CSV)",
            data=csv.encode(), file_name="risk_report.csv", mime="text/csv",
        )
        
# ════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════
if __name__ == "__main__":
    main()
