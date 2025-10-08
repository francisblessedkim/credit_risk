# dashboard/app.py
from __future__ import annotations

import io
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --- allow imports from project root when running from dashboard/ ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_model, score_df  # noqa: E402

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
SUSPECT_TARGET_NAMES = {"default_12m", "high_risk_proxy", "label", "target"}

def sanitize_cols(cols: pd.Index) -> pd.Index:
    cols = pd.Index(cols).astype(str).str.strip()
    cols = cols.str.replace(r"\s+", "_", regex=True)
    cols = cols.str.replace(r"[^0-9A-Za-z_]+", "", regex=True)
    return cols

def ensure_unique(cols: pd.Index) -> pd.Index:
    seen = {}
    out = []
    for c in cols:
        base = c
        i = 1
        while c in seen:
            i += 1
            c = f"{base}__{i}"
        seen[c] = 1
        out.append(c)
    return pd.Index(out)

def get_expected_features(model) -> Optional[List[str]]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        try:
            return list(names)
        except Exception:
            pass
    try:
        booster = model.get_booster()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    return None

def align_features_to_model(X: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    expected_set = set(expected)
    extra = [c for c in X.columns if c not in expected_set]
    missing = [c for c in expected if c not in set(X.columns)]
    if extra:
        st.warning(f"Extra columns in upload (dropped): {extra[:12]}{' ...' if len(extra) > 12 else ''}")
    if missing:
        st.warning(f"Missing columns (added as zeros): {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    for c in missing:
        X[c] = 0.0
    return X.reindex(columns=expected, fill_value=0.0)

def safe_counts(y: np.ndarray) -> pd.DataFrame:
    v = pd.Series(y).value_counts(dropna=False).rename("count")
    return v.to_frame()

def expected_cost(cm: np.ndarray, cost_fp: float, cost_fn: float) -> Tuple[float, float, int, int]:
    tn, fp, fn, tp = cm.ravel()
    total = fp * cost_fp + fn * cost_fn
    n = tn + fp + fn + tp
    return total, total / max(n, 1), fp, fn

def compute_candidates(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Return candidate thresholds and their metrics (requires ground truth)."""
    out = {}

    # ROC / Youden J
    fpr, tpr, thr_roc = roc_curve(y_true, proba)
    j = tpr - fpr
    thr_j = float(thr_roc[np.argmax(j)])
    out["youden_j"] = thr_j

    # PR / Max F1
    prec, rec, thr_pr = precision_recall_curve(y_true, proba)
    # thr_pr has length len(prec)-1; align
    f1 = (2 * prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    thr_f1 = float(thr_pr[np.argmax(f1)]) if len(thr_pr) else 0.5
    out["max_f1"] = thr_f1

    # Return also arrays for downstream visuals
    out["_roc"] = (fpr, tpr, thr_roc)
    out["_pr"] = (prec, rec, thr_pr)
    return out

def cost_min_threshold(y_true: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    # Evaluate on unique sorted thresholds (plus 0 and 1)
    thr = np.unique(np.concatenate(([0.0, 1.0], proba)))
    best_thr, best_cost = 0.5, float("inf")
    for t in thr:
        y_hat = (proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
        total, _, _, _ = expected_cost(cm, cost_fp, cost_fn)
        if total < best_cost:
            best_cost, best_thr = total, float(t)
    return best_thr

def recall_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float) -> float:
    y_hat = (proba >= thr).astype(int)
    return float(recall_score(y_true, y_hat, zero_division=0))

def min_threshold_for_recall(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> Optional[float]:
    """Smallest threshold achieving Recall >= target_recall."""
    thr = np.unique(np.concatenate(([0.0, 1.0], proba)))
    thr = np.sort(thr)[::-1]  # high to low so recall increases
    found = None
    for t in thr:
        r = recall_at_threshold(y_true, proba, t)
        if r >= target_recall:
            found = float(t)
    return found

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk Scorer (Proxy)", layout="wide")

st.title("Credit Risk Scorer — Q1 2022 (Proxy)")
st.write(
    "Upload a **model-ready** dataset (encoded & scaled; same feature columns as training). "
    "The app loads the serialized model and returns predicted probabilities and labels."
)

# Sidebar
st.sidebar.header("Configuration")
default_model_path = "/mnt/d/Projects/credit_risk/outputs/best_model_proxy_xgboost.joblib"
model_path = st.sidebar.text_input("Model path (.joblib)", value=default_model_path)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

uploaded = st.file_uploader("Upload model-ready CSV", type=["csv"])

# Load model
model = None
model_status = st.empty()
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        model_status.success(f"Loaded model: {model_path}")
    else:
        model_status.warning("Model path not found. Adjust the path in the sidebar.")
except Exception as e:
    model_status.error(f"Failed to load model: {e}")

if uploaded is not None and model is not None:
    try:
        # Read + sanitize + de-duplicate headers; drop stale prediction columns
        df = pd.read_csv(uploaded)
        df.columns = ensure_unique(sanitize_cols(df.columns))
        stale = [c for c in df.columns if c.startswith("pred_") or c.startswith("pred_label@")]
        if stale:
            df = df.drop(columns=stale, errors="ignore")

        # ---------- Schema & Data Validation ----------
        with st.expander("Schema & Data Validation", expanded=False):
            st.write("Quick checks before scoring.")
            expected = get_expected_features(model)
            if expected is not None:
                expected = list(expected)
                present = list(df.columns)
                missing = [c for c in expected if c not in present]
                extra = [c for c in present if c not in set(expected)]
                st.write("• Missing (vs. model):", missing if missing else "None")
                st.write("• Extra (vs. model):", extra[:25], "..." if len(extra) > 25 else "")
            else:
                st.info("Model did not expose training feature names; running best-effort alignment.")

            # NaN / Inf scan (top offenders)
            nans = df.isna().sum()
            infs = np.isinf(df.select_dtypes(include=[np.number])).sum()
            prob_cols = pd.Series(dtype=int)
            if not nans.empty:
                prob_cols = nans[nans > 0].sort_values(ascending=False)
            if not prob_cols.empty:
                st.write("• Columns with NaNs (top 10):")
                st.write(prob_cols.head(10).to_frame("NaNs"))
            if isinstance(infs, pd.Series) and (infs > 0).any():
                st.write("• Columns with Inf values (top 10):")
                st.write(infs[infs > 0].sort_values(ascending=False).head(10).to_frame("Infs"))

            st.caption("These checks don’t block scoring; they help catch common issues early.")

        st.write("Preview of uploaded data:")
        st.dataframe(df.head(10), width="stretch")

        # Optional ground truth for evaluation
        gt_candidates = [c for c in df.columns if c.lower() in {"default_12m", "high_risk_proxy"}]
        y_true = None
        y_true_col = None
        if gt_candidates:
            preferred = "high_risk_proxy" if "high_risk_proxy" in gt_candidates else gt_candidates[0]
            y_true_col = st.sidebar.selectbox(
                "Ground-truth column for evaluation (optional)",
                options=["(none)"] + gt_candidates,
                index=(gt_candidates.index(preferred) + 1) if preferred in gt_candidates else 0,
                help="Choose which column to compare predictions against. Leave as '(none)' to skip metrics."
            )
            if y_true_col != "(none)":
                y_true = (
                    pd.Series(df[y_true_col])
                    .astype(str).str.strip()
                    .replace({"true": "1", "false": "0", "True": "1", "False": "0"})
                    .astype(float).astype(int).values
                )
            else:
                y_true_col = None

        # Remove any target-like columns from features
        target_like = [c for c in df.columns if c.lower() in {"default_12m", "high_risk_proxy", "label", "target"}]
        X = df.drop(columns=target_like, errors="ignore")

        # Align to model schema
        expected = get_expected_features(model)
        if expected is not None:
            suspects = [c for c in expected if c.lower() in SUSPECT_TARGET_NAMES]
            if suspects:
                st.error(
                    f"Model expects target-like columns in features: {suspects}. "
                    "This suggests training included the label in X (data leakage). "
                    "Proceed for demo, but retrain excluding targets."
                )
            X = align_features_to_model(X, expected)
        else:
            st.info(
                "Model didn't expose training feature names; attempting to score as-is. "
                "If you see mismatches, retrain saving with a pandas DataFrame (to keep feature_names_in_)."
            )

        # Score
        proba, labels_default = score_df(model, X)          # model's default 0.50 cutoff
        labels_thresh = (proba >= threshold).astype(int)    # your chosen cutoff

        results = pd.concat(
            [
                df.reset_index(drop=True),
                pd.Series(proba, name="pred_prob").reset_index(drop=True),
                pd.Series(labels_default, name="pred_label@base").reset_index(drop=True),
                pd.Series(labels_thresh, name=f"pred_label@{threshold:.2f}").reset_index(drop=True),
            ],
            axis=1,
        )

        # KPIs
        total_rows = len(results)
        pos_rate = float(labels_thresh.mean())
        high_risk_count = int(labels_thresh.sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total rows", f"{total_rows:,}")
        k2.metric("Positive rate", f"{pos_rate:.2%}")
        k3.metric("Threshold", f"{threshold:.2f}")
        k4.metric("High-risk count", f"{high_risk_count:,}")

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                x=proba, nbins=30,
                labels={"x": "Predicted probability (high risk)", "y": "Count"},
                title="Risk Probability Distribution",
            )
            fig_hist.add_vline(
                x=threshold, line_width=2, line_dash="dash",
                annotation_text=f"thr={threshold:.2f}", annotation_position="top"
            )
            st.plotly_chart(fig_hist, width="stretch")
        with c2:
            counts = pd.Series(labels_thresh).value_counts().rename(index={0: "Low risk", 1: "High risk"})
            counts = counts.reindex(["Low risk", "High risk"]).fillna(0).astype(int)
            fig_bar = px.bar(
                x=counts.index, y=counts.values,
                labels={"x": "Predicted class", "y": "Count"},
                title="Predicted Class Distribution",
            )
            st.plotly_chart(fig_bar, width="stretch")

        # Feature importance (if available)
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat_names = expected if expected is not None and len(expected) == len(importances) \
                    else [f"f{i}" for i in range(len(importances))]
                fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
                top = fi_df.sort_values("importance", ascending=False).head(15)
                st.subheader("Top Features (importance)")
                fig_imp = px.bar(
                    top.sort_values("importance"),
                    x="importance", y="feature", orientation="h",
                    labels={"importance": "Importance", "feature": ""},
                    title="Top 15 Features",
                )
                st.plotly_chart(fig_imp, width="stretch")
        except Exception:
            pass

        # ---------------------- THRESHOLD FINDER & COST PANEL ----------------------
        st.subheader("Threshold Finder & Cost Panel")

        col_thr, col_cost = st.columns([2, 1])

        # Default UI values for costs and recall target
        with col_cost:
            st.markdown("**Cost settings (requires ground truth)**")
            cost_fp = st.number_input("Cost of FP (approve high-risk)", min_value=0.0, value=1_000.0, step=100.0)
            cost_fn = st.number_input("Cost of FN (reject low-risk)", min_value=0.0, value=200.0, step=50.0)

        # Finder only meaningful with ground truth
        if y_true is not None and len(np.unique(y_true)) == 2:
            # Candidates
            cand = compute_candidates(y_true, proba)
            thr_j = cand["youden_j"]
            thr_f1 = cand["max_f1"]

            # Recall target threshold
            with col_thr:
                target_recall = st.slider("Target recall (min)", 0.10, 0.99, 0.80, 0.01)
            thr_rec = min_threshold_for_recall(y_true, proba, target_recall)

            # Cost-minimizing threshold
            thr_cost = cost_min_threshold(y_true, proba, cost_fp, cost_fn)

            # Present candidates
            data_thr = {
                "Operating point": ["Current (slider)", "Youden J (ROC)", "Max F1 (PR)", f"Recall ≥ {target_recall:.2f}", "Min Cost"],
                "Threshold": [threshold, thr_j, thr_f1, thr_rec if thr_rec is not None else np.nan, thr_cost],
            }

            # Compute per-candidate metrics quickly
            def metrics_at(t: float) -> Tuple[float, float, float, float, float]:
                y_hat = (proba >= t).astype(int)
                pr = precision_score(y_true, y_hat, zero_division=0)
                rc = recall_score(y_true, y_hat, zero_division=0)
                f1 = f1_score(y_true, y_hat, zero_division=0)
                cm_ = confusion_matrix(y_true, y_hat, labels=[0, 1])
                tot_cost, avg_cost, _, _ = expected_cost(cm_, cost_fp, cost_fn)
                return pr, rc, f1, avg_cost, tot_cost

            rows = []
            for name, t in zip(data_thr["Operating point"], data_thr["Threshold"]):
                if pd.isna(t):
                    rows.append((name, np.nan, np.nan, np.nan, np.nan, np.nan))
                else:
                    pr, rc, f1, avgc, totc = metrics_at(float(t))
                    rows.append((name, pr, rc, f1, avgc, totc))

            df_thr = pd.DataFrame(
                rows, columns=["Operating point", "Precision", "Recall", "F1", "Avg Cost / loan", "Total Cost"]
            )
            df_thr.insert(1, "Threshold", [f"{t:.4f}" if pd.notna(t) else "n/a" for t in data_thr["Threshold"]])

            st.dataframe(df_thr, width="stretch")

            # Current-threshold cost box
            cm_cur = confusion_matrix(y_true, labels_thresh, labels=[0, 1])
            tot_c, avg_c, fp_c, fn_c = expected_cost(cm_cur, cost_fp, cost_fn)
            st.info(
                f"At threshold **{threshold:.2f}** → FP: {fp_c:,}, FN: {fn_c:,}, "
                f"Total cost: **{tot_c:,.0f}**, Avg/loan: **{avg_c:,.2f}**"
            )
        else:
            st.caption("Ground truth not provided (or only one class). Threshold Finder and Cost Panel need ground truth.")

        # ---------------------- Evaluation (if ground truth) ----------------------
        if y_true is not None:
            st.subheader(f"Evaluation against ground truth: {y_true_col}")

            try:
                vc = pd.Series(y_true).value_counts().rename(index={0: "0", 1: "1"}).rename("count")
                st.write("Ground-truth value counts:")
                st.write(vc.to_frame())
            except Exception:
                pass

            unique = np.unique(y_true[~pd.isna(y_true)])
            if len(unique) < 2:
                st.warning(
                    "Only one class found in ground truth. "
                    "Precision/Recall/F1 and ROC are undefined. Confusion matrix shown below."
                )
                cm = confusion_matrix(y_true, labels_thresh, labels=[0, 1])
                z = cm.astype(int)
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=z, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                        colorscale="Blues", showscale=False, text=z, texttemplate="%{text}"
                    )
                )
                fig_cm.update_layout(title=f"Confusion Matrix @ threshold {threshold:.2f}")
                st.plotly_chart(fig_cm, width="content")
            else:
                if y_true_col.lower() == "high_risk_proxy":
                    st.info(
                        "Evaluating against the **proxy label**. Since the model mirrors those rules, "
                        "near-perfect metrics are expected on this distribution. "
                        "Use a true outcome label or held-out split for non-trivial performance."
                    )

                cm = confusion_matrix(y_true, labels_thresh, labels=[0, 1])
                z = cm.astype(int)
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=z, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                        colorscale="Blues", showscale=False, text=z, texttemplate="%{text}"
                    )
                )
                fig_cm.update_layout(title=f"Confusion Matrix @ threshold {threshold:.2f}")
                st.plotly_chart(fig_cm, width="content")

                pr = precision_score(y_true, labels_thresh, zero_division=0)
                rc = recall_score(y_true, labels_thresh, zero_division=0)
                f1 = f1_score(y_true, labels_thresh, zero_division=0)
                try:
                    auc_roc = roc_auc_score(y_true, proba)
                except Exception:
                    auc_roc = float("nan")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Precision", f"{pr:.3f}")
                m2.metric("Recall", f"{rc:.3f}")
                m3.metric("F1-score", f"{f1:.3f}")
                m4.metric("ROC-AUC (prob)", f"{auc_roc:.3f}" if not np.isnan(auc_roc) else "n/a")

                # Curves
                prec, rec, _ = precision_recall_curve(y_true, proba)
                pr_auc = auc(rec, prec)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
                fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig_pr, width="stretch")

                fpr, tpr, _ = roc_curve(y_true, proba)
                roc_auc_val = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc_val:.3f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, width="stretch")

        # Results + download
        st.subheader("Results (first 50 rows)")
        st.dataframe(results.head(50), width="stretch")

        csv_buf = io.StringIO()
        results.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download predictions CSV",
            data=csv_buf.getvalue(),
            file_name="predictions_proxy.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Scoring failed: {e}")

st.caption(
    "This MVP expects a model-ready CSV (same feature columns as training). "
    "Use the Threshold Finder for Youden-J, Max-F1, Min-Cost, or Recall targets. "
    "Schema & Validation helps catch mismatches and missing data early."
)
