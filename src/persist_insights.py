"""
Batch Predictions & Persistence — Customer Churn & Revenue Intelligence
========================================================================
Runs the trained model over the entire customer dataset, computes:
    - churn_probability     (0.0 – 1.0)
    - risk_bucket           (Low / Medium / High)
    - expected_revenue_loss (churn_probability × revenue)
    - priority_score        (expected_revenue_loss — for retention ranking)
    - clv_estimate          (simple Customer Lifetime Value proxy)

Stores results in MySQL table: customer_churn_analytics
Also exports: reports/batch_predictions.csv

Priority Score logic (from PDF spec):
    Priority Score = Risk (churn_probability) × Revenue Impact (expected_revenue_loss)
    → Identifies who to save first when resources are limited.

Run:
    python src/persist_insights.py
"""

import os
import sys
import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [PERSIST]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
FEATURES_CSV  = BASE_DIR / "data" / "processed" / "customer_features.csv"
CLEAN_CSV     = BASE_DIR / "data" / "processed" / "clean_customers.csv"
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "churn_model.pkl"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_list.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORTS_DIR   = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Risk bucket thresholds (aligned with api/app.py) ─────────────────────────
LOW_THRESHOLD    = 0.40
MEDIUM_THRESHOLD = 0.70

# ── Try .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    model = joblib.load(MODEL_PATH)
    log.info(f"Loaded model: {type(model).__name__}")

    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    if scaler:
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
# 2. LOAD FULL CUSTOMER DATASET
# ══════════════════════════════════════════════════════════════════════════════
def load_customers() -> pd.DataFrame:
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        log.info(f"Loaded feature data: {df.shape}")
    elif CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
        log.info(f"Loaded clean data (feature file not found): {df.shape}")
    else:
        raise FileNotFoundError("No processed data found. Run the pipeline first.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE FEATURES (same logic as train.py to avoid schema mismatch)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features(df: pd.DataFrame, feature_names: list) -> tuple:
    """
    Returns (X_array, customer_ids, revenue_series, raw_df_with_ids).
    Preserves customer_id and revenue columns for post-prediction enrichment.
    """
    raw_df = df.copy()

    # Extract customer ID if present
    id_col = None
    for c in ["customer_id", "customerid", "CustomerID"]:
        if c in df.columns:
            id_col = c
            break

    # Extract revenue column
    rev_col = None
    for c in ["total_spend", "TotalCharges", "totalcharges",
              "revenue", "monthly_charges", "MonthlyCharges", "monthlycharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            rev_col = c
            break

    # Drop target + non-feature cols
    drop_cols = []
    for c in ["churn_flag", "churn", "Churn",
              "customer_id", "customerid", "CustomerID",
              "signup_date", "last_active_date", "ingestion_timestamp"]:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=drop_cols)

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category", "string"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(0)

    # Align to training feature schema
    if feature_names:
        for f in feature_names:
            if f not in X.columns:
                X[f] = 0
        X = X[feature_names]

    customer_ids = raw_df[id_col] if id_col else pd.Series(
        [f"CUST_{i:05d}" for i in range(len(df))]
    )
    revenue_series = raw_df[rev_col] if rev_col else pd.Series([0.0] * len(df))

    return X.values, customer_ids, revenue_series, raw_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUN BATCH PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_batch_predictions(model, scaler, X_array) -> np.ndarray:
    X = X_array
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]
    return proba


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE BUSINESS METRICS
# ══════════════════════════════════════════════════════════════════════════════
def assign_risk_bucket(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "Low"
    elif prob < MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "High"


def compute_clv_estimate(revenue: float, churn_prob: float,
                         avg_tenure_months: float = 24.0) -> float:
    """
    Simple CLV proxy:
        CLV = monthly_revenue × expected_remaining_tenure
        expected_remaining_tenure = (1 - churn_prob) × avg_tenure_months
    """
    monthly_rev = revenue / max(avg_tenure_months, 1)
    expected_tenure = (1 - churn_prob) * avg_tenure_months
    return round(monthly_rev * expected_tenure, 2)


def build_predictions_dataframe(
    customer_ids, revenue_series, churn_probas, raw_df, metadata
) -> pd.DataFrame:
    """
    Builds the full output DataFrame with all columns the dashboard API needs.
    Priority Score = churn_probability × expected_revenue_loss
    (per PDF spec: Priority = Risk × Revenue Impact)
    """
    now = datetime.utcnow().isoformat()
    model_version = metadata.get("model_version", "v1.0.0")
    model_name    = metadata.get("model_name",    "Unknown")

    records = []
    for i, (cust_id, revenue, prob) in enumerate(
        zip(customer_ids, revenue_series, churn_probas)
    ):
        revenue       = float(revenue) if pd.notna(revenue) else 0.0
        prob          = float(np.clip(prob, 0.0, 1.0))
        risk_bucket   = assign_risk_bucket(prob)

        # Core financial outputs
        expected_revenue_loss = round(prob * revenue, 2)

        # Priority Score: combines how likely they churn AND how much it costs
        # High prob + high revenue = highest priority for retention action
        priority_score = round(prob * expected_revenue_loss, 4)

        # CLV estimate
        clv = compute_clv_estimate(revenue, prob)

        records.append({
            "customer_id":            str(cust_id),
            "churn_probability":      round(prob, 4),
            "risk_bucket":            risk_bucket,
            "revenue":                round(revenue, 2),
            "expected_revenue_loss":  expected_revenue_loss,
            "priority_score":         priority_score,
            "clv_estimate":           clv,
            "model_name":             model_name,
            "model_version":          model_version,
            "prediction_timestamp":   now,
        })

    df_out = pd.DataFrame(records)

    # Sort by priority score descending — highest risk customers first
    df_out = df_out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df_out.index += 1
    df_out.index.name = "priority_rank"

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# 6. PERSIST TO MYSQL
# ══════════════════════════════════════════════════════════════════════════════
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS customer_churn_analytics (
    id                    INT AUTO_INCREMENT PRIMARY KEY,
    customer_id           VARCHAR(64)    NOT NULL,
    churn_probability     FLOAT          NOT NULL,
    risk_bucket           VARCHAR(10)    NOT NULL,
    revenue               FLOAT          DEFAULT 0,
    expected_revenue_loss FLOAT          DEFAULT 0,
    priority_score        FLOAT          DEFAULT 0,
    clv_estimate          FLOAT          DEFAULT 0,
    model_name            VARCHAR(64),
    model_version         VARCHAR(20),
    prediction_timestamp  DATETIME,
    INDEX idx_customer     (customer_id),
    INDEX idx_risk         (risk_bucket),
    INDEX idx_priority     (priority_score DESC)
)
"""

UPSERT_SQL = """
INSERT INTO customer_churn_analytics
    (customer_id, churn_probability, risk_bucket, revenue,
     expected_revenue_loss, priority_score, clv_estimate,
     model_name, model_version, prediction_timestamp)
VALUES
    (%(customer_id)s, %(churn_probability)s, %(risk_bucket)s, %(revenue)s,
     %(expected_revenue_loss)s, %(priority_score)s, %(clv_estimate)s,
     %(model_name)s, %(model_version)s, %(prediction_timestamp)s)
ON DUPLICATE KEY UPDATE
    churn_probability     = VALUES(churn_probability),
    risk_bucket           = VALUES(risk_bucket),
    revenue               = VALUES(revenue),
    expected_revenue_loss = VALUES(expected_revenue_loss),
    priority_score        = VALUES(priority_score),
    clv_estimate          = VALUES(clv_estimate),
    model_name            = VALUES(model_name),
    model_version         = VALUES(model_version),
    prediction_timestamp  = VALUES(prediction_timestamp)
"""


def persist_to_mysql(df_out: pd.DataFrame) -> bool:
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host     = os.getenv("DB_HOST",     "localhost"),
            user     = os.getenv("DB_USER",     "churn_user"),
            password = os.getenv("DB_PASSWORD", ""),
            database = os.getenv("DB_NAME",     "churn_intelligence"),
        )
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()

        # Batch insert
        records = df_out.reset_index(drop=True).to_dict(orient="records")
        cursor.executemany(UPSERT_SQL, records)
        conn.commit()

        log.info(f"Persisted {len(records)} predictions to MySQL.")
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        log.warning(f"MySQL persistence failed: {e}")
        log.warning("Predictions saved to CSV only.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRINT BATCH SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_batch_summary(df_out: pd.DataFrame):
    total = len(df_out)
    risk_counts = df_out["risk_bucket"].value_counts()

    print("\n" + "═"*60)
    print("  BATCH PREDICTION SUMMARY")
    print("═"*60)
    print(f"  Total customers processed   : {total:,}")
    print(f"  High risk  (prob ≥ 0.70)    : {risk_counts.get('High',   0):,}  "
          f"({risk_counts.get('High',   0)/total*100:.1f}%)")
    print(f"  Medium risk (0.40–0.70)     : {risk_counts.get('Medium', 0):,}  "
          f"({risk_counts.get('Medium', 0)/total*100:.1f}%)")
    print(f"  Low risk   (prob < 0.40)    : {risk_counts.get('Low',    0):,}  "
          f"({risk_counts.get('Low',    0)/total*100:.1f}%)")
    print()
    print(f"  Avg churn probability       : {df_out['churn_probability'].mean()*100:.2f}%")
    print(f"  Total revenue at risk       : ${df_out['expected_revenue_loss'].sum():,.2f}")
    print(f"  Total CLV estimate          : ${df_out['clv_estimate'].sum():,.2f}")
    print()
    print("  TOP 10 PRIORITY CUSTOMERS:")
    print("  " + "-"*55)
    top10_cols = ["customer_id", "churn_probability", "risk_bucket",
                  "expected_revenue_loss", "priority_score"]
    top10_cols = [c for c in top10_cols if c in df_out.columns]
    top10 = df_out[top10_cols].head(10).copy()
    top10["churn_probability"] = top10["churn_probability"].map("{:.2%}".format)
    print(top10.to_string(index=False))
    print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("Batch inference pipeline started.")

    model, scaler, feature_names, metadata = load_artifacts()
    df_raw = load_customers()

    X_array, customer_ids, revenue_series, raw_df = prepare_features(
        df_raw, feature_names
    )
    log.info(f"Running predictions on {len(X_array):,} customers...")

    churn_probas = run_batch_predictions(model, scaler, X_array)

    df_out = build_predictions_dataframe(
        customer_ids, revenue_series, churn_probas, raw_df, metadata
    )

    # Save CSV (always)
    csv_path = REPORTS_DIR / "batch_predictions.csv"
    df_out.to_csv(csv_path)
    log.info(f"Batch predictions saved → {csv_path}")

    # Persist to MySQL (best-effort)
    db_success = persist_to_mysql(df_out)

    # Summary
    print_batch_summary(df_out)

    if db_success:
        log.info("Batch inference complete. Data persisted to MySQL + CSV.\n")
    else:
        log.info("Batch inference complete. Data saved to CSV only.\n")

    return df_out


if __name__ == "__main__":
    main()
