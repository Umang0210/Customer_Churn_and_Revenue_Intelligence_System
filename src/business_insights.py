"""
Business Insights — Customer Churn & Revenue Intelligence
==========================================================
Computes and exports structured business KPIs from:
  - The processed feature dataset (always available)
  - The MySQL predictions table (if DB is running)

Outputs:
  - reports/business_insights.csv   — metric table
  - reports/segment_summary.csv     — churn by risk bucket
  - Console summary

This is the 'Prescriptive Intelligence' layer from the project spec:
    Priority Score = Churn Probability × Revenue Impact

Run:
    python src/business_insights.py
"""

import os
import json
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [INSIGHTS]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "processed" / "customer_features.csv"
CLEAN_CSV    = BASE_DIR / "data" / "processed" / "clean_customers.csv"
MODELS_DIR   = BASE_DIR / "models"
REPORTS_DIR  = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Try loading .env ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _revenue_col(df):
    for c in ["total_spend", "TotalCharges", "totalcharges", "revenue"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            return c
    for c in ["monthly_charges", "MonthlyCharges", "monthlycharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            return c
    return None


def _contract_col(df):
    for c in ["contract_type", "Contract", "contract"]:
        if c in df.columns:
            return c
    return None


def _tenure_col(df):
    for c in ["tenure", "Tenure"]:
        if c in df.columns and df[c].dtype != object:
            return c
    return None


def _churn_col(df):
    for c in ["churn_flag", "churn", "Churn"]:
        if c in df.columns:
            return c
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        log.info(f"Loaded feature data: {df.shape}")
    elif CLEAN_CSV.exists():
        df = pd.read_csv(CLEAN_CSV)
        log.info(f"Loaded clean data (feature file not found): {df.shape}")
    else:
        raise FileNotFoundError(
            "No processed data found. Run ingestion → cleaning → features first."
        )

    # Normalise churn column
    churn_col = _churn_col(df)
    if churn_col is None:
        raise KeyError("No churn column found in dataset.")
    df = df.rename(columns={churn_col: "churn_flag"})
    if pd.api.types.is_string_dtype(df["churn_flag"]) or df["churn_flag"].dtype == object:
        df["churn_flag"] = df["churn_flag"].str.strip().str.lower().map({"yes": 1, "no": 0})
    df["churn_flag"] = pd.to_numeric(df["churn_flag"], errors="coerce").fillna(0).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRY LOADING PREDICTIONS FROM MYSQL (optional)
# ══════════════════════════════════════════════════════════════════════════════
def load_predictions_from_db() -> pd.DataFrame | None:
    """
    Attempts to pull live prediction data from MySQL.
    Returns None gracefully if DB is not available.
    """
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host     = os.getenv("DB_HOST",     "localhost"),
            user     = os.getenv("DB_USER",     "churn_user"),
            password = os.getenv("DB_PASSWORD", ""),
            database = os.getenv("DB_NAME",     "churn_intelligence"),
        )
        query = """
            SELECT
                customer_id,
                churn_probability,
                risk_bucket,
                expected_revenue_loss,
                priority_score,
                prediction_timestamp
            FROM customer_churn_analytics
            ORDER BY prediction_timestamp DESC
            LIMIT 10000
        """
        df = pd.read_sql(query, conn)
        conn.close()
        log.info(f"Loaded {len(df)} predictions from MySQL.")
        return df
    except Exception as e:
        log.warning(f"Could not connect to MySQL ({e}). Using CSV data for insights.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE INSIGHT COMPUTATIONS
# ══════════════════════════════════════════════════════════════════════════════
def compute_kpis(df: pd.DataFrame) -> dict:
    """Returns a flat dict of KPI metric → value."""
    kpis = {}
    total = len(df)
    rev_col = _revenue_col(df)

    # ── Churn rate ──
    churn_count = df["churn_flag"].sum()
    kpis["total_customers"]       = total
    kpis["churned_customers"]     = int(churn_count)
    kpis["retained_customers"]    = int(total - churn_count)
    kpis["churn_rate_%"]          = round(churn_count / total * 100, 2)

    # ── Revenue ──
    if rev_col:
        total_rev   = df[rev_col].sum()
        churned_rev = df[df["churn_flag"] == 1][rev_col].sum()
        kpis["total_revenue_$"]           = round(float(total_rev), 2)
        kpis["revenue_at_risk_$"]         = round(float(churned_rev), 2)
        kpis["revenue_at_risk_%"]         = round(float(churned_rev / total_rev * 100), 2) if total_rev > 0 else 0
        kpis["avg_revenue_churned_$"]     = round(float(df[df["churn_flag"] == 1][rev_col].mean()), 2)
        kpis["avg_revenue_retained_$"]    = round(float(df[df["churn_flag"] == 0][rev_col].mean()), 2)
        kpis["revenue_uplift_opportunity"] = round(
            float(churned_rev * 0.15), 2
        )  # If we save 15% of at-risk customers

    # ── Contract breakdown ──
    contract_col = _contract_col(df)
    if contract_col:
        worst_contract = (
            df.groupby(contract_col)["churn_flag"].mean()
            .sort_values(ascending=False)
        )
        kpis["highest_churn_contract"]        = worst_contract.index[0]
        kpis["highest_churn_contract_rate_%"] = round(float(worst_contract.iloc[0] * 100), 2)
        kpis["lowest_churn_contract"]         = worst_contract.index[-1]
        kpis["lowest_churn_contract_rate_%"]  = round(float(worst_contract.iloc[-1] * 100), 2)

    # ── Tenure ──
    tenure_col = _tenure_col(df)
    if tenure_col:
        q25 = df[tenure_col].quantile(0.25)
        q75 = df[tenure_col].quantile(0.75)
        early  = df[df[tenure_col] <= q25]["churn_flag"].mean() * 100
        late   = df[df[tenure_col] >= q75]["churn_flag"].mean() * 100
        kpis["churn_rate_early_tenure_%"]  = round(early, 2)
        kpis["churn_rate_late_tenure_%"]   = round(late, 2)
        kpis["early_vs_late_churn_ratio"]  = round(early / late, 2) if late > 0 else "N/A"

    # ── Complaints ──
    if "complaints_count" in df.columns:
        no_complaint = df[df["complaints_count"] == 0]["churn_flag"].mean() * 100
        some_complaint = df[df["complaints_count"] >= 1]["churn_flag"].mean() * 100
        kpis["churn_rate_no_complaints_%"]  = round(no_complaint, 2)
        kpis["churn_rate_with_complaints_%"] = round(some_complaint, 2)

    # ── Payment delays ──
    if "payment_delays" in df.columns:
        no_delay   = df[df["payment_delays"] == 0]["churn_flag"].mean() * 100
        with_delay = df[df["payment_delays"] >= 1]["churn_flag"].mean() * 100
        kpis["churn_rate_no_payment_delays_%"]   = round(no_delay, 2)
        kpis["churn_rate_with_payment_delays_%"] = round(with_delay, 2)

    kpis["generated_at"] = datetime.utcnow().isoformat()
    return kpis


def compute_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    If churn_probability is available (from DB or persist_insights output),
    compute risk bucket breakdown. Otherwise fall back to actual churn_flag.
    """
    if "churn_probability" in df.columns:
        df = df.copy()
        df["risk_bucket"] = pd.cut(
            df["churn_probability"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
    elif "risk_bucket" in df.columns:
        pass
    else:
        # Synthetic bucketing based on churn flag
        df = df.copy()
        df["risk_bucket"] = df["churn_flag"].map({0: "Low", 1: "High"})

    rev_col = _revenue_col(df)

    grp = df.groupby("risk_bucket", observed=True)

    summary_parts = {"customer_count": grp.size()}

    if "churn_probability" in df.columns:
        summary_parts["avg_churn_probability_%"] = (
            grp["churn_probability"].mean().mul(100).round(2)
        )

    if "expected_revenue_loss" in df.columns:
        summary_parts["total_revenue_at_risk_$"] = grp["expected_revenue_loss"].sum().round(2)
    elif rev_col:
        churn_rate_by_bucket = grp["churn_flag"].mean()
        total_rev_by_bucket  = grp[rev_col].sum()
        summary_parts["total_revenue_at_risk_$"] = (
            (churn_rate_by_bucket * total_rev_by_bucket).round(2)
        )

    if "priority_score" in df.columns:
        summary_parts["avg_priority_score"] = grp["priority_score"].mean().round(2)

    segment_df = pd.DataFrame(summary_parts).reset_index()
    segment_df.columns = [str(c) for c in segment_df.columns]
    return segment_df


def compute_top_priority_customers(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Returns top N customers ranked by priority score:
    Priority = churn_probability × revenue (decision intelligence layer).
    """
    df = df.copy()
    rev_col = _revenue_col(df)

    if "priority_score" not in df.columns:
        if "churn_probability" in df.columns and rev_col:
            df["priority_score"] = df["churn_probability"] * df[rev_col]
        elif rev_col:
            df["priority_score"] = df["churn_flag"] * df[rev_col]
        else:
            df["priority_score"] = df["churn_flag"]

    if "expected_revenue_loss" not in df.columns and rev_col and "churn_probability" in df.columns:
        df["expected_revenue_loss"] = (df["churn_probability"] * df[rev_col]).round(2)
    elif "expected_revenue_loss" not in df.columns and rev_col:
        df["expected_revenue_loss"] = (df["churn_flag"] * df[rev_col]).round(2)

    id_col = None
    for c in ["customer_id", "customerid", "CustomerID"]:
        if c in df.columns:
            id_col = c
            break

    keep_cols = []
    if id_col:
        keep_cols.append(id_col)
    if "churn_probability" in df.columns:
        keep_cols.append("churn_probability")
    if "risk_bucket" in df.columns:
        keep_cols.append("risk_bucket")
    if rev_col:
        keep_cols.append(rev_col)
    if "expected_revenue_loss" in df.columns:
        keep_cols.append("expected_revenue_loss")
    keep_cols.append("priority_score")

    result = (
        df[keep_cols]
        .sort_values("priority_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index += 1  # 1-based rank
    result.index.name = "rank"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SAVE & PRINT
# ══════════════════════════════════════════════════════════════════════════════
def save_insights(kpis: dict, segment_df: pd.DataFrame, top_customers: pd.DataFrame):
    # KPI table
    kpi_df = pd.DataFrame(list(kpis.items()), columns=["metric", "value"])
    kpi_path = REPORTS_DIR / "business_insights.csv"
    kpi_df.to_csv(kpi_path, index=False)
    log.info(f"KPIs saved → {kpi_path}")

    # Segment summary
    seg_path = REPORTS_DIR / "segment_summary.csv"
    segment_df.to_csv(seg_path, index=False)
    log.info(f"Segment summary saved → {seg_path}")

    # Top priority customers
    top_path = REPORTS_DIR / "top_priority_customers.csv"
    top_customers.to_csv(top_path)
    log.info(f"Priority customers saved → {top_path}")


def print_kpis(kpis: dict):
    print("\n" + "═"*60)
    print("  BUSINESS INTELLIGENCE SUMMARY")
    print("═"*60)
    for k, v in kpis.items():
        if k == "generated_at":
            continue
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"  {label:<45}  {v:>12,.2f}")
        elif isinstance(v, int):
            print(f"  {label:<45}  {v:>12,}")
        else:
            print(f"  {label:<45}  {str(v):>12}")
    print("═"*60)


def print_segment_table(segment_df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  RISK SEGMENT BREAKDOWN")
    print("═"*60)
    print(segment_df.to_string(index=False))
    print("═"*60)


def print_top_customers(top_df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  TOP PRIORITY CUSTOMERS FOR RETENTION ACTION")
    print("═"*60)
    print(top_df.to_string())
    print("═"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("Business Insights pipeline started.")

    # Load feature data (always available)
    df = load_data()

    # Try to enrich with live predictions from MySQL
    pred_df = load_predictions_from_db()
    if pred_df is not None and len(pred_df) > 0:
        # Merge churn_probability and priority_score into df if customer_id matches
        for id_col in ["customer_id", "customerid", "CustomerID"]:
            if id_col in df.columns and "customer_id" in pred_df.columns:
                df = df.merge(
                    pred_df[["customer_id", "churn_probability",
                             "risk_bucket", "expected_revenue_loss", "priority_score"]],
                    left_on=id_col, right_on="customer_id", how="left",
                )
                log.info("Merged live predictions into dataset.")
                break

    # Compute insights
    kpis         = compute_kpis(df)
    segment_df   = compute_segment_summary(df)
    top_df       = compute_top_priority_customers(df, top_n=20)

    # Save & print
    save_insights(kpis, segment_df, top_df)
    print_kpis(kpis)
    print_segment_table(segment_df)
    print_top_customers(top_df)

    log.info("Business Insights pipeline complete.\n")


if __name__ == "__main__":
    main()
