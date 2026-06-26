"""
Cotiviti GenAI Intern Assessment — POC Demo
Clinical Decision Support: Risk Stratification + SHAP Explainability
Author: Eric Thompson | UAB MSHI Candidate
Topic: Clinical Decision Making and Pattern Recognition (TPO Framework)
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotiviti | Clinical Risk Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #BDD7EE; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border: 1px solid #D6E4F0;
        border-left: 4px solid #2E75B6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1F4E79; }

    .risk-high   { background: #FEE2E2; border-left: 4px solid #DC2626; border-radius: 8px; padding: 1rem; }
    .risk-medium { background: #FEF9C3; border-left: 4px solid #CA8A04; border-radius: 8px; padding: 1rem; }
    .risk-low    { background: #DCFCE7; border-left: 4px solid #16A34A; border-radius: 8px; padding: 1rem; }

    .sbar-box {
        background: #F0F7FF;
        border: 1px solid #BDD7EE;
        border-radius: 8px;
        padding: 1.2rem;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .section-title { color: #1F4E79; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    div[data-testid="stSidebar"] { background: #F0F4F8; }
</style>
""", unsafe_allow_html=True)


# ── DATA GENERATION ──────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1200, seed=42):
    """
    Synthetic clinical dataset mimicking a payment integrity / care management use case.
    Features represent clinical + claims signals used for risk stratification (TPO).
    """
    rng = np.random.default_rng(seed)

    age               = rng.integers(18, 90, n).astype(float)
    los               = rng.integers(1, 21, n).astype(float)          # length of stay
    prior_admits      = rng.integers(0, 8, n).astype(float)
    comorbidity_idx   = rng.integers(0, 6, n).astype(float)           # Charlson proxy
    num_procedures    = rng.integers(0, 15, n).astype(float)
    claim_amount      = rng.normal(15000, 8000, n).clip(500, 80000)
    ed_visits_12m     = rng.integers(0, 10, n).astype(float)
    readmit_14d_prior = rng.integers(0, 2, n).astype(float)
    polypharmacy      = rng.integers(0, 12, n).astype(float)          # # meds
    drg_weight        = rng.uniform(0.5, 5.0, n)                      # CMS DRG weight
    specialist_flag   = rng.integers(0, 2, n).astype(float)
    rural_flag        = rng.integers(0, 2, n).astype(float)
    payer_type        = rng.integers(0, 3, n).astype(float)           # 0=commercial, 1=Medicare, 2=Medicaid

    # Outcome: high-risk (escalation / payment review flag)
    log_odds = (
        -10.5
        + 0.03  * age
        + 0.12  * los
        + 0.35  * prior_admits
        + 0.40  * comorbidity_idx
        + 0.08  * num_procedures
        + 0.000012 * claim_amount
        + 0.30  * ed_visits_12m
        + 1.10  * readmit_14d_prior
        + 0.15  * polypharmacy
        + 0.50  * drg_weight
        + 0.25  * specialist_flag
        + 0.10  * rural_flag
        + 0.20  * payer_type
        + rng.normal(0, 0.6, n)
    )
    prob  = 1 / (1 + np.exp(-log_odds))
    label = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "Age":                  age,
        "Length_of_Stay":       los,
        "Prior_Admissions_12m": prior_admits,
        "Comorbidity_Index":    comorbidity_idx,
        "Num_Procedures":       num_procedures,
        "Claim_Amount_USD":     claim_amount.round(2),
        "ED_Visits_12m":        ed_visits_12m,
        "Readmit_14d_Prior":    readmit_14d_prior,
        "Polypharmacy_Count":   polypharmacy,
        "DRG_Weight":           drg_weight.round(3),
        "Specialist_Involved":  specialist_flag,
        "Rural_Patient":        rural_flag,
        "Payer_Type":           payer_type,
        "High_Risk":            label
    })
    return df


# ── MODEL TRAINING ───────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df.drop("High_Risk", axis=1)
    y = df["High_Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Logistic Regression (fast, interpretable)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

    # Gradient Boosting (higher performance)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

    # SHAP explainer on GBM
    explainer = shap.TreeExplainer(gb)

    return {
        "lr": lr, "gb": gb, "scaler": scaler,
        "explainer": explainer,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "lr_auc": lr_auc, "gb_auc": gb_auc,
        "feature_names": list(X.columns)
    }


# ── RISK LABEL ───────────────────────────────────────────────────────────────
def risk_label(prob):
    if prob >= 0.70:
        return "HIGH", "risk-high", "🔴"
    elif prob >= 0.40:
        return "MODERATE", "risk-medium", "🟡"
    else:
        return "LOW", "risk-low", "🟢"


# ── SBAR GENERATOR ───────────────────────────────────────────────────────────
def generate_sbar(inputs, prob, top_drivers):
    risk, _, icon = risk_label(prob)
    payer_map = {0: "Commercial", 1: "Medicare", 2: "Medicaid"}
    payer = payer_map.get(int(inputs["Payer_Type"]), "Unknown")
    drivers_str = "\n    ".join([f"- {d}" for d in top_drivers[:3]])
    return f"""{icon} CLINICAL RISK ESCALATION — SBAR SUMMARY
══════════════════════════════════════════════

SITUATION
  Patient flagged as {risk} RISK by Cotiviti Clinical Risk Classifier.
  Risk Score: {prob:.1%}  |  Payer: {payer}

BACKGROUND
  Age: {int(inputs['Age'])} yrs  |  LOS: {int(inputs['Length_of_Stay'])} days
  Prior Admissions (12m): {int(inputs['Prior_Admissions_12m'])}
  Comorbidity Index: {int(inputs['Comorbidity_Index'])}
  Claim Amount: ${inputs['Claim_Amount_USD']:,.0f}
  DRG Weight: {inputs['DRG_Weight']:.3f}

ASSESSMENT
  Top model-identified risk drivers (SHAP):
    {drivers_str}

RECOMMENDATION
  {"⚠️  Flag for immediate clinical review and payment integrity audit." if risk == "HIGH"
   else "📋  Route to standard utilization management review queue."
   if risk == "MODERATE"
   else "✅  No escalation required. Routine claims processing."}

  Generated by: Cotiviti Clinical Risk Classifier v1.0 (POC)
  Model: Gradient Boosting  |  AUROC: N/A (see Model Performance tab)
"""


# ── LOAD DATA & MODELS ───────────────────────────────────────────────────────
df     = generate_dataset()
models = train_models(df)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 Cotiviti Risk Classifier")
    st.markdown("**Clinical Decision Support POC**")
    st.caption("Eric Thompson | UAB MSHI | June 2025")
    st.divider()
    page = st.radio("Navigate", [
        "🏠  Overview",
        "👤  Patient Risk Scorer",
        "📊  SHAP Explainability",
        "📈  Model Performance",
    ])
    st.divider()
    st.caption("⚠️ Synthetic data only. Not for clinical use.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Cotiviti Clinical Risk Classifier</h1>
        <p>Proof-of-Concept | Topic 2: Clinical Decision Making & Pattern Recognition (TPO Framework)</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    high  = (df["High_Risk"] == 1).sum()
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Patients</div><div class="value">{total:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">High-Risk Flags</div><div class="value">{high:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">LR AUROC</div><div class="value">{models["lr_auc"]:.3f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="label">GBM AUROC</div><div class="value">{models["gb_auc"]:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("#### What This Demonstrates")
        st.markdown("""
**Clinical Decision Making + Pattern Recognition** applied to Cotiviti's TPO framework:

| Capability | Application |
|---|---|
| **Classification** | Binary risk flag for escalation / audit routing |
| **Prediction** | Patient-level risk score (0–100%) |
| **Explainability** | SHAP waterfall shows *why* each patient scored high |
| **SBAR Generation** | Structured clinical handoff auto-generated from model output |
| **Anomaly Signal** | High claim amount + high DRG weight = payment integrity flag |

**Architecture:**
- Synthetic dataset: 1,200 patients, 13 clinical + claims features
- Models: Logistic Regression (interpretable baseline) + Gradient Boosting (performance)
- Explainability: SHAP TreeExplainer on GBM
- Framework: Streamlit + scikit-learn + SHAP
        """)

    with col2:
        st.markdown("#### TPO Coverage")
        st.markdown("""
🏥 **Treatment**
- Risk stratification for care escalation
- Comorbidity + LOS + readmission signals

💳 **Payment**
- Claim amount anomaly detection
- DRG weight + procedure count audit flags

⚙️ **Operations**
- Automated SBAR generation
- Queue routing (high / moderate / low)
        """)

    st.markdown("---")
    st.markdown("#### Sample Dataset (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Patient" in page:
    st.markdown("""
    <div class="main-header">
        <h1>👤 Patient Risk Scorer</h1>
        <p>Enter patient and claim attributes to generate a real-time risk score with SHAP explanation and SBAR output.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown("**Clinical & Demographic Inputs**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age       = st.slider("Age", 18, 89, 65)
            los       = st.slider("Length of Stay (days)", 1, 20, 5)
            prior_adm = st.slider("Prior Admissions (12m)", 0, 7, 1)
            comorbid  = st.slider("Comorbidity Index (0–5)", 0, 5, 2)
        with c2:
            n_proc    = st.slider("# Procedures", 0, 14, 3)
            claim_amt = st.number_input("Claim Amount ($)", 500, 80000, 18000, step=500)
            ed_visits = st.slider("ED Visits (12m)", 0, 9, 1)
            readmit   = st.selectbox("Readmit within 14d (prior)", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c3:
            poly      = st.slider("Polypharmacy Count", 0, 11, 4)
            drg_wt    = st.number_input("DRG Weight", 0.5, 5.0, 1.8, step=0.1)
            spec_flag = st.selectbox("Specialist Involved", [0, 1], format_func=lambda x: "Yes" if x else "No")
            rural     = st.selectbox("Rural Patient", [0, 1], format_func=lambda x: "Yes" if x else "No")
            payer     = st.selectbox("Payer Type", [0, 1, 2], format_func=lambda x: {0:"Commercial",1:"Medicare",2:"Medicaid"}[x])

        submitted = st.form_submit_button("🔍  Run Risk Assessment", use_container_width=True)

    if submitted:
        inputs = {
            "Age": age, "Length_of_Stay": los, "Prior_Admissions_12m": prior_adm,
            "Comorbidity_Index": comorbid, "Num_Procedures": n_proc,
            "Claim_Amount_USD": claim_amt, "ED_Visits_12m": ed_visits,
            "Readmit_14d_Prior": readmit, "Polypharmacy_Count": poly,
            "DRG_Weight": drg_wt, "Specialist_Involved": spec_flag,
            "Rural_Patient": rural, "Payer_Type": payer
        }
        row = pd.DataFrame([inputs])

        prob = models["gb"].predict_proba(row)[0][1]
        risk, css_class, icon = risk_label(prob)

        # SHAP for this patient
        sv = models["explainer"].shap_values(row)
        if isinstance(sv, list):
            sv = sv[1]
        shap_vals = sv[0]
        feat_names = models["feature_names"]
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        top_drivers = [
            f"{feat_names[i]} = {row.iloc[0][feat_names[i]]:.1f}  (SHAP: {shap_vals[i]:+.3f})"
            for i in sorted_idx[:5]
        ]

        # Risk display
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{icon} Risk Level: {risk}</strong> &nbsp;|&nbsp; Score: <strong>{prob:.1%}</strong>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Risk Drivers (SHAP)**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            top_n = 6
            idx   = sorted_idx[:top_n][::-1]
            vals  = shap_vals[idx]
            names = [feat_names[i] for i in idx]
            colors = ["#DC2626" if v > 0 else "#16A34A" for v in vals]
            ax.barh(names, vals, color=colors, edgecolor="white", height=0.6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on risk score)")
            ax.set_title(f"Patient Risk Drivers — Score: {prob:.1%}", fontsize=10, fontweight="bold")
            red_patch   = mpatches.Patch(color="#DC2626", label="↑ Increases Risk")
            green_patch = mpatches.Patch(color="#16A34A", label="↓ Decreases Risk")
            ax.legend(handles=[red_patch, green_patch], fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Auto-Generated SBAR**")
            sbar = generate_sbar(inputs, prob, top_drivers)
            st.markdown(f'<div class="sbar-box">{sbar}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📊 SHAP Global Explainability</h1>
        <p>Population-level feature importance from the Gradient Boosting model — critical for regulatory compliance and auditor trust.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing SHAP values on test set..."):
        X_test_sample = models["X_test"].sample(200, random_state=42)
        sv = models["explainer"].shap_values(X_test_sample)
        if isinstance(sv, list):
            sv = sv[1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Mean |SHAP| — Feature Importance")
        mean_abs = np.abs(sv).mean(axis=0)
        feat_names = models["feature_names"]
        order = np.argsort(mean_abs)
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.barh(
            [feat_names[i] for i in order],
            mean_abs[order],
            color="#2E75B6", edgecolor="white", height=0.65
        )
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Global Feature Importance\n(Gradient Boosting — 200-patient sample)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### SHAP Summary Dot Plot")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        # Manual dot plot
        for i, fi in enumerate(order):
            vals_f = sv[:, fi]
            feat_v = X_test_sample.iloc[:, fi].values
            feat_norm = (feat_v - feat_v.min()) / (feat_v.ptp() + 1e-9)
            colors_f = plt.cm.RdYlGn_r(feat_norm)
            ax2.scatter(vals_f, [i]*len(vals_f), c=colors_f, alpha=0.5, s=12)
        ax2.set_yticks(range(len(feat_names)))
        ax2.set_yticklabels([feat_names[i] for i in order], fontsize=9)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("SHAP Value")
        ax2.set_title("SHAP Value Distribution by Feature\n(Red = high feature value, Green = low)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown("""
    #### Why SHAP Matters for Cotiviti
    - **Regulatory compliance**: Explainable AI outputs satisfy emerging FDA/ONC transparency requirements
    - **Auditor trust**: Adjusters can see exactly why a claim or patient was flagged — no black box
    - **Bias detection**: SHAP distributions reveal if protected attributes (e.g., Rural_Patient, Payer_Type) are driving flags inappropriately
    - **Provider communication**: Structured SHAP-based rationale supports denial and escalation letters
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📈 Model Performance</h1>
        <p>Evaluation metrics for Logistic Regression and Gradient Boosting classifiers on held-out test set.</p>
    </div>
    """, unsafe_allow_html=True)

    # Predictions
    lr_pred  = models["lr"].predict(models["X_test_s"])
    lr_prob  = models["lr"].predict_proba(models["X_test_s"])[:, 1]
    gb_pred  = models["gb"].predict(models["X_test"])
    gb_prob  = models["gb"].predict_proba(models["X_test"])[:, 1]
    y_test   = models["y_test"]

    # AUROC curves
    from sklearn.metrics import roc_curve
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_prob)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(lr_fpr, lr_tpr, color="#2E75B6", lw=2, label=f"Logistic Regression (AUC={models['lr_auc']:.3f})")
        ax.plot(gb_fpr, gb_tpr, color="#1F4E79", lw=2, label=f"Gradient Boosting (AUC={models['gb_auc']:.3f})")
        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Confusion Matrices")
        fig2, axes = plt.subplots(1, 2, figsize=(5.5, 4.5))
        for ax, pred, title in zip(axes, [lr_pred, gb_pred], ["Logistic Regression", "Gradient Boosting"]):
            cm = confusion_matrix(y_test, pred)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Low","High"]); ax.set_yticklabels(["Low","High"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(title, fontsize=9, fontweight="bold")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i,j], ha="center", va="center",
                            color="white" if cm[i,j] > cm.max()/2 else "black", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown("#### Classification Report — Gradient Boosting")
    cr = classification_report(y_test, gb_pred, target_names=["Low Risk","High Risk"], output_dict=True)
    cr_df = pd.DataFrame(cr).T.round(3)
    st.dataframe(cr_df, use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("GBM AUROC", f"{models['gb_auc']:.3f}")
    with c2:
        st.metric("LR AUROC", f"{models['lr_auc']:.3f}")
    with c3:
        st.metric("Test Set Size", f"{len(y_test)} patients")
