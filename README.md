# Credit Risk Modeling: Can We Predict Loan Defaults Before They Happen?

## Project Overview

This project builds an end-to-end credit risk pipeline using real loan-level data. The guiding question is straightforward: **can I predict the likelihood of borrower default at approval time** using only application-time features?

I follow a practical data-science lifecycle: **Business Understanding → Analytic Approach → Data Requirements → Data Collection → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment**. The current iteration focuses on a complete case study using **Fannie Mae Single-Family Loan Performance, Q1 2022** to prove out the pipeline with free, accessible tooling.

The emphasis is twofold:

* **Career relevance**: sound decisions, defensible trade-offs, and production-minded code.
* **Educational value**: transparent steps and narrative that others can follow and reproduce.

---

## Why Q1 2022 First (the Pivot)

I initially planned to combine multiple quarters (Q1–Q4 2022 + Q1 2023). On free compute (Kaggle/WSL) that quickly hit practical limits. Rather than stall, I **pivoted to a Q1-only case study** to:

* ship a working, explainable pipeline start-to-finish;
* share progress and lessons sooner;
* keep everything reproducible on free resources.

Once this case study is solid, I’ll extend to multi-quarter modeling and add robust validation across time.

---

## Data & Labeling (Case Study)

* **Source**: Fannie Mae Single-Family Loan Performance (public).
* **Unit of analysis**: loans (the raw dataset is monthly performance; I collapse to one row per loan using origination-time features).
* **Features** (examples): credit score(s), DTI, LTV/CLTV, loan amount, rate, term, purpose, occupancy, property state, channel, first-time homebuyer flag.
* **Labeling**:

  * For this Q1 case study, a practical **proxy label** `high_risk_proxy` is used to create a learnable target on the single-quarter slice (derived from risk-relevant origination patterns).
  * The original `default_12m` exists but is mostly zeros in Q1-only context (insufficient positive cases). This is why the proxy is used for the MVP.

> Important: A single-quarter proxy makes it easy for tree models to separate classes, which can yield near-perfect in-sample metrics. This is by design for the MVP; multi-quarter expansion and temporal validation will follow.

---

## Results (What I Saw and Why)

* The best baseline (XGBoost) **perfectly recovers the proxy pattern** on Q1 data. The Streamlit app may show very high/“perfect” Precision, Recall, F1, and ROC-AUC when evaluating against `high_risk_proxy`.
* This does **not** imply production-grade generalization. It reflects:

  * strong separation in the engineered features and
  * lack of temporal shift within a single quarter.
* **Takeaway**: the pipeline works mechanically and the app is correct. To demonstrate generalization, the next step is to widen the timeframe (multi-quarter), adopt temporal splits, calibrate, and monitor.

---

## What’s in the App (Dashboard)

A Streamlit dashboard that:

* loads the trained model;
* accepts a **model-ready CSV** (encoded & scaled with the same feature schema used in training);
* outputs **probabilities and risk labels**;
* provides **interactive thresholding** and **evaluation visuals** (confusion matrix, PR/ROC curves, KPI tiles);
* includes **Schema & Data Validation** (presence/missing, NaN/Inf scan);
* includes a **Threshold Finder & Cost Panel** (Youden-J, Max-F1, Recall≥X, and cost-minimizing threshold).

> MVP scope: The app expects a **model-ready** dataset. The next iteration will bundle preprocessing in a single `Pipeline` artifact so the app can accept raw columns.

---

## Repository Structure

```
credit_risk/
│
├── data/                # Raw and processed data (gitignored)
│   ├── raw/
│   └── processed/
├── notebooks/           # Exploration and modeling notebooks
├── src/                 # Reusable scripts (preprocessing, training, scoring)
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── dashboard/           # Streamlit app
│   └── app.py
├── reports/             # Reports and figures (gitignored by default)
│   └── figures/
├── tests/               # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Reproduce (WSL-friendly)

### 1) Clone and set up environment

```bash
# WSL path; adjust if needed
cd /mnt/d/Projects
git clone https://github.com/francisblessedkim/credit_risk.git
cd credit_risk

# create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Prepare data (model-ready CSV)

If you’re following the notebooks, export the processed Q1 2022 dataset to:

```
data/processed/fannie_q1_2022_model_ready.csv
```

This file should contain the **exact feature columns used for training** plus (optionally) `high_risk_proxy` or `default_12m` for evaluation.

> Tip: If you’re not regenerating the file yourself, place your model-ready CSV in `data/processed/` and ensure its columns match the model’s expected schema.

### 3) Train (optional) and save model

If you want to retrain locally:

```bash
# example: run your training script or notebook export
python -m src.train \
  --input data/processed/fannie_q1_2022_model_ready.csv \
  --out_dir outputs
```

The repository expects a serialized model in:

```
outputs/best_model_proxy_xgboost.joblib
```

(You can change the path in the app’s sidebar.)

### 4) Run the Streamlit app

```bash
# from project root
source .venv/bin/activate
streamlit run dashboard/app.py
```

Streamlit will print a local URL (e.g., `http://localhost:8501`). Open it in your browser.

### 5) Use the app

* Upload your **model-ready CSV**.
* In the sidebar, if present, select the **ground truth** column (`high_risk_proxy` recommended for the MVP).
* Adjust the **decision threshold** and explore metrics, PR/ROC curves, and the cost panel.
* Download the scored CSV from the app.

---

## Practical Notes (What I’d Do in a Bank Setting)

If this were a production engagement (e.g., Absa/Stanbic):

* **Widen the data window** across multiple quarters/years to learn patterns that survive regime shifts.
* Use **temporal cross-validation** with strict out-of-time folds.
* Add **calibration** (Platt/Isotonic) and a **reliability diagram** in the app.
* Integrate **explainability** (global + local SHAP) for decisions and governance.
* Set up **monitoring**: stability (PSI/KS), performance drift, retrain triggers, and backtesting.
* Package preprocessing + model into a single **sklearn `Pipeline`** (or Pydantic-validated batch flow) to avoid schema drift.

---

## Version Control Hygiene (What Bit Me and How I Fixed It)

* Virtual environments and large binaries can silently bloat a repo. This project’s `.gitignore` excludes `.venv/`, `data/`, `outputs/`, and notebook checkpoints.
* If you accidentally commit heavy artifacts, use `git-filter-repo` to **rewrite history** and strip large blobs before pushing.

---

## Tech Stack

Python (Pandas, NumPy) · scikit-learn · XGBoost · imbalanced-learn · Plotly · Streamlit · Jupyter · Git/WSL

---

## Status

* [x] Business Understanding
* [x] Analytic Approach
* [x] Data Requirements
* [x] Data Collection
* [x] Data Understanding
* [x] Data Preparation
* [x] Modeling
* [x] Evaluation (Q1 proxy)
* [x] Deployment (local Streamlit MVP)
* [ ] Bundle preprocessing into a Pipeline for raw input scoring
* [ ] Multi-quarter expansion and temporal validation
* [ ] Online deployment + monitoring

---

## Author

**Francis Blessed Kim**
LinkedIn: [https://www.linkedin.com/in/francis-kim-1931681b6/](https://www.linkedin.com/in/francis-kim-1931681b6/)
GitHub: [https://github.com/francisblessedkim](https://github.com/francisblessedkim)
Medium: [https://medium.com/@kimblessedfrancis](https://medium.com/@kimblessedfrancis)

---

### Appendix: Typical “Gotchas” and Fixes

* **“Only one class in ground truth”**: you likely selected `default_12m` for Q1 (mostly zeros). Use `high_risk_proxy` for this MVP or upload a split that contains both classes.
* **Perfect metrics in the app**: expected when evaluating against the proxy on the same quarter; not indicative of out-of-time performance.
* **Duplicate prediction columns**: if you re-upload a CSV that already has `pred_…` columns, the app drops them before scoring to avoid collisions.
* **Model feature mismatch**: the app aligns uploaded columns to the model’s expected schema; missing one-hots are added as zeros, extras are dropped, and order is fixed.

