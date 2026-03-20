"""
Model Training — Customer Churn & Revenue Intelligence
=======================================================
Trains three models:
    1. Logistic Regression  (baseline, linear)
    2. Random Forest        (non-linear, ensemble)
    3. XGBoost              (gradient boosting, advanced)

Auto-selects best model by ROC-AUC and saves artifacts.

Run:
    python src/train.py
"""

import os
import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.metrics            import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.pipeline           import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn(
        "XGBoost not installed. Run: pip install xgboost\n"
        "Training will proceed with Logistic Regression + Random Forest only.",
        stacklevel=2,
    )

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [TRAIN]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT    = MODELS_DIR / "churn_model.pkl"
SCALER_OUT   = MODELS_DIR / "scaler.pkl"
FEATURES_OUT = MODELS_DIR / "feature_list.json"
METADATA_OUT = MODELS_DIR / "model_metadata.json"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_features() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature data not found at {DATA_PATH}. "
            "Run ingestion.py → cleaning.py → features.py first."
        )
    df = pd.read_csv(DATA_PATH)
    log.info(f"Loaded feature data: {df.shape}")
    return df


def prepare_xy(df: pd.DataFrame):
    """
    Identify target, drop non-feature columns, encode categoricals,
    return X (DataFrame), y (Series), and feature name list.
    """
    # ── Target ──
    target_col = None
    for c in ["churn_flag", "churn", "Churn"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise KeyError("No target column found (expected: churn_flag / churn / Churn).")

    y = df[target_col].copy()
    if pd.api.types.is_string_dtype(y) or y.dtype == object:
        y = y.str.strip().str.lower().map({"yes": 1, "no": 0})
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # ── Drop non-feature columns ──
    drop_cols = [target_col]
    for c in ["customer_id", "customerid", "CustomerID", "signup_date",
              "last_active_date", "ingestion_timestamp"]:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=drop_cols)

    # ── Encode categorical columns ──
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = X[col].astype(str)
        try:
            X[col] = le.fit_transform(X[col])
        except Exception:
            X[col] = 0  # fallback for edge cases

    # ── Fill any remaining NaN ──
    X = X.fillna(0)

    feature_names = X.columns.tolist()
    log.info(f"Features: {len(feature_names)}  |  Target distribution: {y.value_counts().to_dict()}")
    return X, y, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE MODELS
# ══════════════════════════════════════════════════════════════════════════════
def get_model_candidates() -> dict:
    """
    Returns dict of model_name → (model_object, needs_scaling).
    needs_scaling=True  → scaler will be fitted and saved alongside model.
    needs_scaling=False → model handles feature scale internally.
    """
    candidates = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
            True,   # needs scaling
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            False,  # tree-based: no scaling needed
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,          # adjusted below after y is known
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
                n_jobs=-1,
            ),
            False,  # gradient boosting: no scaling needed
        )
    else:
        log.warning("XGBoost not available — training 2 models only.")

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATE ONE MODEL
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_train, X_test, y_train, y_test, needs_scaling, model_name):
    """
    Fits, predicts, and returns a metrics dict.
    Handles scaling internally — returns fitted (model, scaler or None).
    """
    scaler = None

    if needs_scaling:
        scaler   = StandardScaler()
        X_tr_fit = scaler.fit_transform(X_train)
        X_te_fit = scaler.transform(X_test)
    else:
        X_tr_fit = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_te_fit = X_test.values  if isinstance(X_test,  pd.DataFrame) else X_test

    # XGBoost: adjust scale_pos_weight for imbalance
    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        if pos > 0:
            model.set_params(scale_pos_weight=round(neg / pos, 2))

    model.fit(X_tr_fit, y_train)

    y_pred_proba = model.predict_proba(X_te_fit)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    # 5-fold CV AUC on training data
    cv_X = X_tr_fit
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, cv_X, y_train, cv=cv, scoring="roc_auc").mean()

    metrics = {
        "roc_auc":    round(auc,       4),
        "cv_auc":     round(cv_auc,    4),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1_score":   round(f1,        4),
    }

    log.info(
        f"{model_name:<22}  AUC={auc:.4f}  CV-AUC={cv_auc:.4f}  "
        f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}"
    )
    return model, scaler, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def save_artifacts(best_model, best_scaler, feature_names, metadata):
    joblib.dump(best_model, MODEL_OUT)
    log.info(f"Model saved   → {MODEL_OUT}")

    if best_scaler is not None:
        joblib.dump(best_scaler, SCALER_OUT)
        log.info(f"Scaler saved  → {SCALER_OUT}")
    else:
        # Remove stale scaler if new best doesn't need one
        if SCALER_OUT.exists():
            SCALER_OUT.unlink()
        log.info("No scaler required for this model (tree-based).")

    with open(FEATURES_OUT, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info(f"Features saved → {FEATURES_OUT}")

    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved → {METADATA_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
def print_comparison(results: dict):
    print("\n" + "="*72)
    print(f"  {'MODEL':<22}  {'AUC':>6}  {'CV-AUC':>7}  {'PREC':>6}  {'RECALL':>6}  {'F1':>6}")
    print("="*72)
    for name, m in results.items():
        marker = " ★" if m.get("selected") else ""
        print(
            f"  {name:<22}  {m['roc_auc']:>6.4f}  {m['cv_auc']:>7.4f}  "
            f"{m['precision']:>6.4f}  {m['recall']:>6.4f}  {m['f1_score']:>6.4f}{marker}"
        )
    print("="*72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("Training pipeline started.")

    df               = load_features()
    X, y, feat_names = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    candidates = get_model_candidates()
    results    = {}
    best_auc   = -1
    best_name  = None
    best_model = None
    best_scaler = None

    log.info(f"\nTraining {len(candidates)} model(s)...\n")

    for name, (model, needs_scaling) in candidates.items():
        fitted_model, fitted_scaler, metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test,
            needs_scaling=needs_scaling, model_name=name,
        )
        results[name] = metrics

        if metrics["roc_auc"] > best_auc:
            best_auc    = metrics["roc_auc"]
            best_name   = name
            best_model  = fitted_model
            best_scaler = fitted_scaler

    # Mark winner
    results[best_name]["selected"] = True

    print_comparison(results)
    log.info(f"Best model: {best_name}  (ROC-AUC = {best_auc:.4f})")

    # ── Save artifacts ──
    metadata = {
        "model_name":       best_name,
        "model_version":    "v2.0.0",
        "trained_at":       datetime.utcnow().isoformat(),
        "train_samples":    int(len(X_train)),
        "test_samples":     int(len(X_test)),
        "feature_count":    len(feat_names),
        "xgboost_available": XGBOOST_AVAILABLE,
        "metrics":          {k: v for k, v in results[best_name].items() if k != "selected"},
        "all_model_results": results,
        "min_auc_threshold": 0.70,
        "passes_gate":       best_auc >= 0.70,
    }

    save_artifacts(best_model, best_scaler, feat_names, metadata)

    if not metadata["passes_gate"]:
        log.warning(
            f"Model ROC-AUC ({best_auc:.4f}) is below minimum threshold (0.70). "
            "Review your data and features."
        )
    else:
        log.info("Performance gate passed ✓")

    log.info("Training pipeline complete.\n")
    return metadata


if __name__ == "__main__":
    main()
