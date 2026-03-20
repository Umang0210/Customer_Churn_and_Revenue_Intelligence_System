"""
Model Evaluation — Customer Churn & Revenue Intelligence
=========================================================
Loads the best trained model and evaluates it on held-out test data.

Outputs:
  - Console: metrics table, threshold analysis, confusion matrix
  - reports/evaluation_report.json  — machine-readable results
  - reports/figures/09_roc_curve.png
  - reports/figures/10_confusion_matrix.png
  - reports/figures/11_threshold_analysis.png
  - reports/figures/12_feature_importance.png  (tree/XGBoost models only)

CI/CD Gate:
  Exits with code 1 if ROC-AUC < MIN_AUC_THRESHOLD (0.70).
  This is used by .github/workflows/data_pipeline.yml.

Run:
    python src/evaluate.py
"""

import os
import sys
import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [EVAL]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODELS_DIR   = BASE_DIR / "models"
MODEL_PATH   = MODELS_DIR / "churn_model.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
FEATURES_PATH= MODELS_DIR / "feature_list.json"
METADATA_PATH= MODELS_DIR / "model_metadata.json"
REPORTS_DIR  = BASE_DIR / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Gate threshold ────────────────────────────────────────────────────────────
MIN_AUC_THRESHOLD  = 0.70
EVAL_THRESHOLDS    = [0.4, 0.5, 0.6, 0.7]

sns.set_theme(style="whitegrid", font_scale=1.1)
COLOR_POS = "#e74c3c"
COLOR_NEG = "#2ecc71"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    log.info(f"Loaded model: {MODEL_PATH.name}  ({type(model).__name__})")

    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        log.info("Loaded scaler.")

    feature_names = None
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH) as f:
            feature_names = json.load(f)
        log.info(f"Loaded {len(feature_names)} feature names.")

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    return model, scaler, feature_names, metadata


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD & PREPARE TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_test_data(feature_names):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature data not found at {DATA_PATH}. Run the pipeline first."
        )

    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded data: {df.shape}")

    # Target
    target_col = None
    for c in ["churn_flag", "churn", "Churn"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise KeyError("No churn target column found.")

    y = df[target_col].copy()
    if pd.api.types.is_string_dtype(y) or y.dtype == object:
        y = y.str.strip().str.lower().map({"yes": 1, "no": 0})
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # Drop non-feature columns
    drop_cols = [target_col]
    for c in ["customer_id", "customerid", "CustomerID",
              "signup_date", "last_active_date", "ingestion_timestamp"]:
        if c in df.columns:
            drop_cols.append(c)
    X = df.drop(columns=drop_cols)

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category", "string"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(0)

    # Align to training features if known
    if feature_names:
        missing = [f for f in feature_names if f not in X.columns]
        for m in missing:
            X[m] = 0
        X = X[feature_names]

    # Reproduce the same train/test split used in train.py
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info(f"Test set: {X_test.shape}  |  Churn rate: {y_test.mean()*100:.2f}%")
    return X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def get_predictions(model, scaler, X_test):
    X = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]
    return proba


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS AT EACH THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════
def metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold":  threshold,
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":   round(f1_score(y_true, y_pred, zero_division=0), 4),
        "predicted_churn_count": int(y_pred.sum()),
        "predicted_churn_%":    round(y_pred.mean() * 100, 2),
    }


def run_threshold_analysis(y_true, y_proba):
    rows = [metrics_at_threshold(y_true, y_proba, t) for t in EVAL_THRESHOLDS]
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc          = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"{model_name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#3498db")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    out = FIGURES_DIR / "09_roc_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {out.name}")


def plot_confusion_matrix(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Retained", "Churned"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix  (threshold = {threshold})",
        fontsize=12, fontweight="bold",
    )
    tn, fp, fn, tp = cm.ravel()
    ax.text(
        0.5, -0.12,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  |  "
        f"Precision={tp/(tp+fp):.3f}  Recall={tp/(tp+fn):.3f}",
        transform=ax.transAxes, ha="center", fontsize=9, color="grey",
    )
    plt.tight_layout()
    out = FIGURES_DIR / "10_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {out.name}")


def plot_threshold_analysis(threshold_rows):
    df = pd.DataFrame(threshold_rows)
    x  = [str(t) for t in df["threshold"]]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    w = 0.25
    xs = np.arange(len(x))

    ax1.bar(xs - w,   df["precision"], width=w, label="Precision", color="#3498db",  alpha=0.85)
    ax1.bar(xs,       df["recall"],    width=w, label="Recall",    color="#e74c3c",  alpha=0.85)
    ax1.bar(xs + w,   df["f1_score"],  width=w, label="F1 Score",  color="#2ecc71",  alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"t={t}" for t in df["threshold"]])
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Precision / Recall / F1 at Different Thresholds",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(xs, df["predicted_churn_%"], "o--", color="#f39c12",
             lw=2, markersize=8, label="Predicted Churn %")
    ax2.set_ylabel("Predicted Churn %", color="#f39c12")
    ax2.tick_params(axis="y", labelcolor="#f39c12")
    ax2.legend(loc="upper left")

    for i, row in df.iterrows():
        ax1.text(i, row["f1_score"] + 0.02, f"{row['f1_score']:.3f}",
                 ha="center", fontsize=9, color="#2ecc71", fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "11_threshold_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {out.name}")


def plot_feature_importance(model, feature_names, model_name):
    """Works for Random Forest and XGBoost. Skips Logistic Regression."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        log.info("Model does not expose feature importances — skipping plot.")
        return

    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if i < 5 else "#3498db" for i in range(len(fi))]
    ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(
        f"Top 15 Feature Importances — {model_name}",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(fi.values.mean(), color="grey", linestyle="--", lw=1.2, label="Mean importance")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "12_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_evaluation_summary(model_name, auc, threshold_rows, y_true, y_proba):
    print("\n" + "═"*68)
    print(f"  EVALUATION REPORT — {model_name}")
    print("═"*68)
    print(f"  ROC-AUC Score     : {auc:.4f}")
    print(f"  Avg Precision     : {average_precision_score(y_true, y_proba):.4f}")
    print(f"  Test samples      : {len(y_true)}")
    print(f"  Actual churn rate : {y_true.mean()*100:.2f}%")
    print()
    print(f"  {'THRESHOLD':<12}  {'PRECISION':>10}  {'RECALL':>8}  {'F1':>8}  {'PRED CHURN %':>13}")
    print("  " + "-"*58)
    for row in threshold_rows:
        marker = " ←" if row["threshold"] == 0.5 else ""
        print(
            f"  {row['threshold']:<12}  {row['precision']:>10.4f}  "
            f"{row['recall']:>8.4f}  {row['f1_score']:>8.4f}  "
            f"{row['predicted_churn_%']:>12.2f}%{marker}"
        )
    print("═"*68 + "\n")

    # Classification report at 0.5
    y_pred = (y_proba >= 0.5).astype(int)
    print("  Classification Report (threshold = 0.5):")
    print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"], digits=4))


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
def save_evaluation_report(model_name, auc, threshold_rows, y_true, y_proba, passes_gate):
    report = {
        "model_name":       model_name,
        "evaluated_at":     datetime.utcnow().isoformat(),
        "test_samples":     int(len(y_true)),
        "actual_churn_rate_%": round(float(y_true.mean() * 100), 2),
        "roc_auc":          round(float(auc), 4),
        "avg_precision":    round(float(average_precision_score(y_true, y_proba)), 4),
        "min_auc_threshold": MIN_AUC_THRESHOLD,
        "passes_gate":      passes_gate,
        "threshold_analysis": threshold_rows,
        "figures": [
            "reports/figures/09_roc_curve.png",
            "reports/figures/10_confusion_matrix.png",
            "reports/figures/11_threshold_analysis.png",
            "reports/figures/12_feature_importance.png",
        ],
    }
    out = REPORTS_DIR / "evaluation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Evaluation report saved → {out}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("Evaluation pipeline started.")

    model, scaler, feature_names, metadata = load_artifacts()
    model_name = metadata.get("model_name", type(model).__name__)

    X_test, y_test = load_test_data(feature_names)
    y_proba        = get_predictions(model, scaler, X_test)

    auc             = roc_auc_score(y_test, y_proba)
    threshold_rows  = run_threshold_analysis(y_test.values, y_proba)
    passes_gate     = auc >= MIN_AUC_THRESHOLD

    # Plots
    plot_roc_curve(y_test.values, y_proba, model_name)
    plot_confusion_matrix(y_test.values, y_proba, threshold=0.5)
    plot_threshold_analysis(threshold_rows)
    plot_feature_importance(model, feature_names, model_name)

    # Console output
    print_evaluation_summary(model_name, auc, threshold_rows, y_test.values, y_proba)

    # Save report
    save_evaluation_report(model_name, auc, threshold_rows, y_test.values, y_proba, passes_gate)

    # ── CI/CD Performance Gate ──
    if not passes_gate:
        log.error(
            f"PERFORMANCE GATE FAILED: ROC-AUC = {auc:.4f} "
            f"< minimum threshold {MIN_AUC_THRESHOLD}. "
            "Do not deploy this model."
        )
        sys.exit(1)
    else:
        log.info(f"Performance gate PASSED ✓  (ROC-AUC = {auc:.4f} ≥ {MIN_AUC_THRESHOLD})")
        log.info("Evaluation pipeline complete.\n")


if __name__ == "__main__":
    main()
