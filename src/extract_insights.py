"""
extract_insights.py
───────────────────
Analyzes the Telco Customer Churn dataset, model artifacts, and evaluation
metrics to produce a comprehensive dashboard_data.json for the frontend.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ==============================
# CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FEATURES_DATA_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FEATURE_LIST_PATH = BASE_DIR / "models" / "feature_list.json"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
OUTPUT_PATH = BASE_DIR / "src" / "webapp" / "static" / "dashboard_data.json"

RANDOM_STATE = 42
TARGET_COL = "churn"


def load_assets():
    """Load all project assets."""
    print("📂 Loading assets...")

    raw_df = pd.read_csv(RAW_DATA_PATH)
    features_df = pd.read_csv(FEATURES_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_LIST_PATH, "r") as f:
        feature_list = json.load(f)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return raw_df, features_df, model, feature_list, metadata


def analyze_raw_dataset(raw_df):
    """Analyze the raw dataset for overview statistics."""
    print("📊 Analyzing raw dataset...")

    # Basic shape
    rows, cols = raw_df.shape

    # Column info
    columns_info = []
    for col in raw_df.columns:
        dtype = str(raw_df[col].dtype)
        missing = int(raw_df[col].isnull().sum())
        unique = int(raw_df[col].nunique())
        columns_info.append({
            "name": col,
            "dtype": dtype,
            "missing": missing,
            "unique": unique
        })

    # Missing values summary
    total_missing = int(raw_df.isnull().sum().sum())
    total_cells = int(raw_df.size)

    return {
        "rows": rows,
        "columns": cols,
        "total_missing": total_missing,
        "total_cells": total_cells,
        "missing_pct": round(total_missing / total_cells * 100, 2),
        "columns_info": columns_info,
        "memory_mb": round(raw_df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "duplicate_rows": int(raw_df.duplicated().sum())
    }


def analyze_churn_distribution(df):
    """Analyze churn distribution."""
    print("📊 Analyzing churn distribution...")

    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(" ", "_")

    churn_col = df_copy["churn"]
    if churn_col.dtype == object:
        churn_counts = churn_col.str.strip().str.lower().value_counts().to_dict()
        churned = churn_counts.get("yes", 0)
        retained = churn_counts.get("no", 0)
    else:
        churned = int((churn_col == 1).sum())
        retained = int((churn_col == 0).sum())

    total = churned + retained
    churn_rate = round(churned / total * 100, 2) if total > 0 else 0

    return {
        "churned": churned,
        "retained": retained,
        "total": total,
        "churn_rate": churn_rate,
        "retention_rate": round(100 - churn_rate, 2)
    }


def analyze_revenue(df):
    """Analyze revenue metrics."""
    print("💰 Analyzing revenue...")

    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert numeric columns
    df_copy["totalcharges"] = pd.to_numeric(df_copy["totalcharges"], errors="coerce")
    df_copy["monthlycharges"] = pd.to_numeric(df_copy["monthlycharges"], errors="coerce")
    df_copy["tenure"] = pd.to_numeric(df_copy["tenure"], errors="coerce")

    # Map churn
    if df_copy["churn"].dtype == object:
        df_copy["churn_num"] = df_copy["churn"].str.strip().str.lower().map({"yes": 1, "no": 0})
    else:
        df_copy["churn_num"] = df_copy["churn"]

    # Revenue stats
    total_monthly = float(df_copy["monthlycharges"].sum())
    avg_monthly = float(df_copy["monthlycharges"].mean())
    total_revenue = float(df_copy["totalcharges"].sum())
    avg_revenue = float(df_copy["totalcharges"].mean())

    # Revenue by churn status
    churned_revenue = float(df_copy[df_copy["churn_num"] == 1]["totalcharges"].sum())
    retained_revenue = float(df_copy[df_copy["churn_num"] == 0]["totalcharges"].sum())

    churned_avg_monthly = float(df_copy[df_copy["churn_num"] == 1]["monthlycharges"].mean())
    retained_avg_monthly = float(df_copy[df_copy["churn_num"] == 0]["monthlycharges"].mean())

    # Monthly charges distribution
    bins = [0, 30, 50, 70, 90, 120]
    labels = ["$0-30", "$30-50", "$50-70", "$70-90", "$90-120"]
    df_copy["charge_group"] = pd.cut(df_copy["monthlycharges"], bins=bins, labels=labels)
    charge_dist = df_copy["charge_group"].value_counts().sort_index().to_dict()
    charge_dist = {str(k): int(v) for k, v in charge_dist.items()}

    return {
        "total_monthly_charges": round(total_monthly, 2),
        "avg_monthly_charges": round(avg_monthly, 2),
        "total_revenue": round(total_revenue, 2),
        "avg_revenue_per_customer": round(avg_revenue, 2),
        "churned_total_revenue": round(churned_revenue, 2),
        "retained_total_revenue": round(retained_revenue, 2),
        "churned_avg_monthly": round(churned_avg_monthly, 2),
        "retained_avg_monthly": round(retained_avg_monthly, 2),
        "charge_distribution": charge_dist
    }


def analyze_segments(df):
    """Analyze customer segments by contract, gender, senior citizen."""
    print("📊 Analyzing customer segments...")

    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(" ", "_")

    if df_copy["churn"].dtype == object:
        df_copy["churn_num"] = df_copy["churn"].str.strip().str.lower().map({"yes": 1, "no": 0})
    else:
        df_copy["churn_num"] = df_copy["churn"]

    # Contract analysis
    contract_stats = (
        df_copy.groupby("contract")
        .agg(
            count=("churn_num", "size"),
            churn_rate=("churn_num", "mean"),
            avg_monthly=("monthlycharges", "mean")
        )
        .reset_index()
    )
    contract_data = []
    for _, row in contract_stats.iterrows():
        contract_data.append({
            "contract": str(row["contract"]).strip().title(),
            "count": int(row["count"]),
            "churn_rate": round(float(row["churn_rate"]) * 100, 2),
            "avg_monthly": round(float(row["avg_monthly"]), 2)
        })

    # Gender analysis
    gender_stats = (
        df_copy.groupby("gender")
        .agg(
            count=("churn_num", "size"),
            churn_rate=("churn_num", "mean")
        )
        .reset_index()
    )
    gender_data = []
    for _, row in gender_stats.iterrows():
        gender_data.append({
            "gender": str(row["gender"]).strip().title(),
            "count": int(row["count"]),
            "churn_rate": round(float(row["churn_rate"]) * 100, 2)
        })

    # Senior citizen analysis
    senior_stats = (
        df_copy.groupby("seniorcitizen")
        .agg(
            count=("churn_num", "size"),
            churn_rate=("churn_num", "mean")
        )
        .reset_index()
    )
    senior_data = []
    for _, row in senior_stats.iterrows():
        label = "Senior" if str(row["seniorcitizen"]).strip() in ["1", "yes"] else "Non-Senior"
        senior_data.append({
            "category": label,
            "count": int(row["count"]),
            "churn_rate": round(float(row["churn_rate"]) * 100, 2)
        })

    return {
        "contract": contract_data,
        "gender": gender_data,
        "senior_citizen": senior_data
    }


def analyze_tenure(df):
    """Analyze tenure distribution and churn by tenure group."""
    print("📊 Analyzing tenure...")

    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(" ", "_")
    df_copy["tenure"] = pd.to_numeric(df_copy["tenure"], errors="coerce")

    if df_copy["churn"].dtype == object:
        df_copy["churn_num"] = df_copy["churn"].str.strip().str.lower().map({"yes": 1, "no": 0})
    else:
        df_copy["churn_num"] = df_copy["churn"]

    # Tenure groups
    bins = [-1, 6, 12, 24, 48, 100]
    labels = ["0-6 mo", "6-12 mo", "12-24 mo", "24-48 mo", "48+ mo"]
    df_copy["tenure_group"] = pd.cut(df_copy["tenure"], bins=bins, labels=labels)

    tenure_stats = (
        df_copy.groupby("tenure_group", observed=False)
        .agg(
            count=("churn_num", "size"),
            churn_rate=("churn_num", "mean"),
            avg_monthly=("monthlycharges", "mean")
        )
        .reset_index()
    )

    tenure_data = []
    for _, row in tenure_stats.iterrows():
        tenure_data.append({
            "group": str(row["tenure_group"]),
            "count": int(row["count"]),
            "churn_rate": round(float(row["churn_rate"]) * 100, 2),
            "avg_monthly": round(float(row["avg_monthly"]), 2)
        })

    # Overall tenure stats
    avg_tenure = float(df_copy["tenure"].mean())
    median_tenure = float(df_copy["tenure"].median())
    max_tenure = int(df_copy["tenure"].max())

    return {
        "groups": tenure_data,
        "avg_tenure": round(avg_tenure, 1),
        "median_tenure": round(median_tenure, 1),
        "max_tenure": max_tenure
    }


def run_model_evaluation(features_df, model, feature_list, metadata):
    """Run model evaluation and get predictions."""
    print("🤖 Running model evaluation...")

    df = features_df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Map target
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].str.strip().str.lower().map({"yes": 1, "no": 0})

    NUMERIC_COLS = ["tenure", "monthlycharges", "totalcharges"]
    CATEGORICAL_COLS = ["gender", "seniorcitizen", "contract"]

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    # Encode
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    X = X.reindex(columns=feature_list, fill_value=0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    precision = round(float(precision_score(y_test, y_pred)), 4)
    recall = round(float(recall_score(y_test, y_pred)), 4)
    f1 = round(float(f1_score(y_test, y_pred)), 4)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Threshold analysis
    thresholds = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_t = (y_proba >= t).astype(int)
        thresholds.append({
            "threshold": t,
            "precision": round(float(precision_score(y_test, y_pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_t, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_t, zero_division=0)), 4)
        })

    # Full dataset predictions for risk distribution + top customers
    y_full_proba = model.predict_proba(X)[:, 1]
    df["churn_probability"] = y_full_proba
    df["revenue"] = df["monthlycharges"] * df["tenure"]
    df["expected_revenue_loss"] = df["churn_probability"] * df["revenue"]

    # Risk distribution
    df["risk_bucket"] = pd.cut(
        df["churn_probability"],
        bins=[0.0, 0.4, 0.7, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    risk_dist = df["risk_bucket"].value_counts().to_dict()
    risk_distribution = {str(k): int(v) for k, v in risk_dist.items()}

    # Top high-risk customers
    if "customerid" in df.columns:
        id_col = "customerid"
    elif "customer_id" in df.columns:
        id_col = "customer_id"
    else:
        df["customer_id"] = [f"C{i:05d}" for i in range(len(df))]
        id_col = "customer_id"

    top_customers = (
        df.sort_values("expected_revenue_loss", ascending=False)
        [[id_col, "churn_probability", "revenue", "expected_revenue_loss", "monthlycharges", "tenure"]]
        .head(15)
    )
    top_customers_list = []
    for _, row in top_customers.iterrows():
        top_customers_list.append({
            "customer_id": str(row[id_col]),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "revenue": round(float(row["revenue"]), 2),
            "expected_loss": round(float(row["expected_revenue_loss"]), 2),
            "monthly_charges": round(float(row["monthlycharges"]), 2),
            "tenure": int(row["tenure"])
        })

    # Revenue at risk
    total_revenue = float(df["revenue"].sum())
    revenue_at_risk = float(df["expected_revenue_loss"].sum())
    high_risk_pct = round(float((df["churn_probability"] > 0.7).mean()) * 100, 2)

    return {
        "evaluation": {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        },
        "model_comparison": metadata.get("metrics", {}),
        "selected_model": metadata.get("selected_model", "random_forest"),
        "model_version": metadata.get("model_version", "v1.1.0"),
        "num_features": metadata.get("num_features", len(feature_list)),
        "feature_list": feature_list,
        "thresholds": thresholds,
        "risk_distribution": risk_distribution,
        "top_customers": top_customers_list,
        "business_kpis": {
            "total_customers": len(df),
            "high_risk_pct": high_risk_pct,
            "total_revenue": round(total_revenue, 2),
            "revenue_at_risk": round(revenue_at_risk, 2),
            "revenue_at_risk_pct": round(revenue_at_risk / total_revenue * 100, 2) if total_revenue > 0 else 0
        }
    }


def analyze_services(df):
    """Analyze internet and phone service usage patterns."""
    print("📊 Analyzing services...")

    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(" ", "_")

    if df_copy["churn"].dtype == object:
        df_copy["churn_num"] = df_copy["churn"].str.strip().str.lower().map({"yes": 1, "no": 0})
    else:
        df_copy["churn_num"] = df_copy["churn"]

    services_data = {}

    # Internet service
    if "internetservice" in df_copy.columns:
        internet_stats = (
            df_copy.groupby("internetservice")
            .agg(count=("churn_num", "size"), churn_rate=("churn_num", "mean"))
            .reset_index()
        )
        services_data["internet_service"] = []
        for _, row in internet_stats.iterrows():
            services_data["internet_service"].append({
                "service": str(row["internetservice"]).strip().title(),
                "count": int(row["count"]),
                "churn_rate": round(float(row["churn_rate"]) * 100, 2)
            })

    # Payment method
    if "paymentmethod" in df_copy.columns:
        payment_stats = (
            df_copy.groupby("paymentmethod")
            .agg(count=("churn_num", "size"), churn_rate=("churn_num", "mean"))
            .reset_index()
        )
        services_data["payment_method"] = []
        for _, row in payment_stats.iterrows():
            services_data["payment_method"].append({
                "method": str(row["paymentmethod"]).strip().title(),
                "count": int(row["count"]),
                "churn_rate": round(float(row["churn_rate"]) * 100, 2)
            })

    return services_data


def main():
    print("=" * 60)
    print("🚀 Customer Churn Intelligence — Data Extraction")
    print("=" * 60)

    raw_df, features_df, model, feature_list, metadata = load_assets()

    # 1. Dataset overview
    dataset_overview = analyze_raw_dataset(raw_df)

    # 2. Churn distribution
    churn_distribution = analyze_churn_distribution(raw_df)

    # 3. Revenue analysis
    revenue_analysis = analyze_revenue(raw_df)

    # 4. Segments
    segment_analysis = analyze_segments(raw_df)

    # 5. Tenure
    tenure_analysis = analyze_tenure(raw_df)

    # 6. Services
    services_analysis = analyze_services(raw_df)

    # 7. Model evaluation + predictions
    model_results = run_model_evaluation(features_df, model, feature_list, metadata)

    # Build final JSON
    dashboard_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": "Customer Churn & Revenue Optimization Intelligence System",
        "dataset_overview": dataset_overview,
        "churn_distribution": churn_distribution,
        "revenue_analysis": revenue_analysis,
        "segment_analysis": segment_analysis,
        "tenure_analysis": tenure_analysis,
        "services_analysis": services_analysis,
        "model": model_results
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"\n✅ Dashboard data saved to: {OUTPUT_PATH}")
    print(f"📊 Total customers analyzed: {churn_distribution['total']}")
    print(f"📈 Churn rate: {churn_distribution['churn_rate']}%")
    print(f"🤖 Model ROC-AUC: {model_results['evaluation']['roc_auc']}")
    print(f"💰 Revenue at risk: ${model_results['business_kpis']['revenue_at_risk']:,.2f}")


if __name__ == "__main__":
    main()
