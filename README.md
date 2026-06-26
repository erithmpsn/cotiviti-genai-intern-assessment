# Cotiviti GenAI Intern Assessment — POC Demo
## Clinical Risk Classifier with SHAP Explainability

**Author:** Eric Thompson | University of Alabama at Birmingham, MSHI Candidate  
**Assessment Topic:** Topic 2 — Clinical Decision Making and Pattern Recognition in Health Care  
**Submitted to:** Cotiviti Talent Acquisition

---

## Overview

This proof-of-concept demonstrates clinical decision support and pattern recognition applied to Cotiviti's **Treatment, Payment, and Operations (TPO)** framework. It builds on the architecture from the author's TDRM (Telemetry Deterioration Risk Monitor) project, reframed for payment integrity and care management use cases.

### What It Demonstrates

| Capability | Implementation |
|---|---|
| Classification | Binary high-risk flag for escalation / audit routing |
| Prediction | Patient-level risk score (0–100%) |
| Explainability | SHAP TreeExplainer with global + patient-level views |
| SBAR Generation | Auto-generated structured clinical handoff from model output |
| Anomaly Signal | Claim amount + DRG weight flagging for payment integrity |

---

## Tech Stack

- **Frontend:** Streamlit
- **ML Models:** Logistic Regression + Gradient Boosting (scikit-learn)
- **Explainability:** SHAP TreeExplainer
- **Data:** Synthetic (1,200 patients, 13 features) — not for clinical use

---

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## App Pages

1. **Overview** — Dataset summary, architecture, TPO coverage map
2. **Patient Risk Scorer** — Enter patient attributes → get risk score + SHAP chart + SBAR
3. **SHAP Explainability** — Global feature importance and SHAP summary dot plot
4. **Model Performance** — ROC curves, confusion matrices, classification report

---

## Dataset Features (Synthetic)

| Feature | Description |
|---|---|
| Age | Patient age |
| Length_of_Stay | Inpatient days |
| Prior_Admissions_12m | Admissions in past 12 months |
| Comorbidity_Index | Charlson Comorbidity Index proxy (0–5) |
| Num_Procedures | Number of procedures on claim |
| Claim_Amount_USD | Total claim amount |
| ED_Visits_12m | Emergency department visits in past year |
| Readmit_14d_Prior | Readmission within 14 days of prior stay |
| Polypharmacy_Count | Number of concurrent medications |
| DRG_Weight | CMS Diagnosis-Related Group weight |
| Specialist_Involved | Binary flag |
| Rural_Patient | Binary flag |
| Payer_Type | 0=Commercial, 1=Medicare, 2=Medicaid |

---

## Disclaimer

This application uses **synthetic data only** and is a proof-of-concept for demonstration purposes. It is not validated for clinical use and should not be used to make real clinical or payment decisions.

---

## Related Work

This POC adapts concepts from the author's **Telemetry Deterioration Risk Monitor (TDRM)**, presented as Poster P31 at ATTIS 2026 (UAB Heersink, May 2025). IP disclosure OI2026-01273 is under review at UAB HIIE.
