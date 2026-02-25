import pandas as pd
import joblib
import json
import mysql.connector
from pathlib import Path
from datetime import date

# ===============================
# Paths
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODEL_DIR = BASE_DIR / "models"

# ===============================
# Load assets
# ===============================
model = joblib.load(MODEL_DIR / "churn_model.pkl")

try:
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
except FileNotFoundError:
    scaler = None

with open(MODEL_DIR / "feature_list.json") as f:
    feature_list = json.load(f)

# ===============================
# DB connection
# ===============================
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="churn_user",
        password="StrongPassword123",
        database="churn_intelligence"
    )

# ===============================
# Helper functions
# ===============================
def read_processed_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    return df

def transform_features(df):
    X = df.copy()
    X = X.reindex(columns=feature_list, fill_value=0)
    return scaler.transform(X) if scaler else X

def map_risk(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"

def write_df_to_mysql(df):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 🔥 Full refresh strategy (prevents duplicate primary key error)
    cursor.execute("TRUNCATE TABLE customer_churn_analytics")

    query = """
        INSERT INTO customer_churn_analytics (
            customer_id,
            churn_probability,
            risk_bucket,
            revenue,
            expected_revenue_loss,
            priority_score,
            model_version,
            batch_run_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    data_tuples = [
        (
            row["customer_id"],
            float(row["churn_probability"]),
            row["risk_bucket"],
            float(row["revenue"]),
            float(row["expected_revenue_loss"]),
            float(row["priority_score"]),
            row["model_version"],
            row["prediction_date"]
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(query, data_tuples)

    conn.commit()
    cursor.close()
    conn.close()

# ===============================
# FINAL MASTER PIPELINE
# ===============================
def main():
    print("Loading processed data...")
    df = read_processed_data()

    print("Transforming features...")
    X = transform_features(df)

    print("Running predictions...")
    probs = model.predict_proba(X)[:, 1]

    df["churn_probability"] = probs
    df["risk_bucket"] = df["churn_probability"].apply(map_risk)

    # -----------------------------
    # Revenue Handling
    # -----------------------------
    if "monthlycharges" in df.columns:
        df["revenue"] = df["monthlycharges"]
    elif "revenue" not in df.columns:
        df["revenue"] = 0
        print("Warning: No revenue column found. Defaulting to 0.")

    # -----------------------------
    # KPI Calculations
    # -----------------------------
    df["expected_revenue_loss"] = df["churn_probability"] * df["revenue"]
    df["priority_score"] = df["churn_probability"] * df["revenue"]

    df["prediction_date"] = date.today()
    df["model_version"] = "v1.0"

    # -----------------------------
    # Ensure customer_id exists
    # -----------------------------
    if "customer_id" not in df.columns:
        if "customerid" in df.columns:
            df["customer_id"] = df["customerid"]
        else:
            df["customer_id"] = df.index + 1

    # -----------------------------
    # Final Columns (Power BI Ready)
    # -----------------------------
    final_columns = [
        "customer_id",
        "churn_probability",
        "risk_bucket",
        "revenue",
        "expected_revenue_loss",
        "priority_score",
        "prediction_date",
        "model_version"
    ]

    final_df = df[final_columns].copy()

    # -----------------------------
    # Save Final CSV
    # -----------------------------
    final_output_path = BASE_DIR / "data" / "processed" / "final_dataset.csv"
    final_df.to_csv(final_output_path, index=False)
    print(f"Final dataset saved to {final_output_path}")

    # -----------------------------
    # Write to MySQL
    # -----------------------------
    print("Writing to MySQL...")
    write_df_to_mysql(final_df)

    print("Batch scoring completed successfully.")

if __name__ == "__main__":
    main()