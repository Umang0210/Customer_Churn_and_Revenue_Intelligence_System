# ── Standard imports (keep your existing ones) ────────────────────────────────
import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Add project root to path ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# ── Import upload router ──────────────────────────────────────────────────────
from upload_handler import router as upload_router

# Sub-routers disabled — endpoints are now handled directly in this file
HAS_SUBROUTERS = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "churn_model.pkl"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_list.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Intelligence API",
    description="Customer Churn & Revenue Optimization Intelligence System",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(upload_router)            # ← upload + pipeline trigger


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        log.warning("Model not found. Run the pipeline first.")
        return None, None, None, {}

    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)   if SCALER_PATH.exists()   else None
    features = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.exists() else None
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    log.info(f"Model loaded: {type(model).__name__}")
    return model, scaler, features, metadata


model, scaler, feature_names, model_metadata = load_model()


# ── Request schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    customer_id:      str
    revenue:          float = 0.0
    monthly_charges:  float = 0.0
    usage_frequency:  int   = 0
    complaints_count: int   = 0
    payment_delays:   int   = 0
    gender:           Optional[str] = None
    seniorcitizen:    Optional[str] = None
    contract:         Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  model is not None,
        "model_name":    model_metadata.get("model_name", "unknown"),
        "model_version": model_metadata.get("model_version", "unknown"),
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    global model, scaler, feature_names, model_metadata

    # Reload model if not loaded (e.g. after pipeline retrain)
    if model is None:
        model, scaler, feature_names, model_metadata = load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Run the pipeline first.",
        )

    # Build feature vector
    input_data = {
        "monthly_charges":  req.monthly_charges,
        "usage_frequency":  req.usage_frequency,
        "complaints_count": req.complaints_count,
        "payment_delays":   req.payment_delays,
    }

    df = pd.DataFrame([input_data])

    # Align to training features
    if feature_names:
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_names]

    X = df.values
    if scaler is not None:
        X = scaler.transform(X)

    prob = float(model.predict_proba(X)[0][1])
    prob = round(min(max(prob, 0.0), 1.0), 4)

    revenue = req.revenue or req.monthly_charges
    expected_revenue_loss = round(prob * revenue, 2)
    priority_score        = round(prob * expected_revenue_loss, 4)

    risk_bucket = "LOW" if prob < 0.4 else ("MEDIUM" if prob < 0.7 else "HIGH")

    return {
        "customer_id":            req.customer_id,
        "churn_probability":      prob,
        "risk_bucket":            risk_bucket,
        "revenue":                revenue,
        "expected_revenue_loss":  expected_revenue_loss,
        "priority_score":         priority_score,
        "model_name":             model_metadata.get("model_name", "unknown"),
        "model_version":          model_metadata.get("model_version", "unknown"),
    }


@app.get("/api/dashboard/summary")
def dashboard_summary():
    """Returns top-level KPIs for the dashboard."""
    try:
        pred_path = BASE_DIR / "reports" / "batch_predictions.csv"
        if not pred_path.exists():
            return {"error": "No predictions yet. Run the pipeline."}

        df  = pd.read_csv(pred_path)
        return {
            "total_predictions":    len(df),
            "avg_churn_probability": round(float(df["churn_probability"].mean()), 4),
            "high_risk_count":      int((df["risk_bucket"] == "High").sum()),
            "revenue_at_risk":      round(float(df["expected_revenue_loss"].sum()), 2),
            "total_revenue":        round(float(df["revenue"].sum()), 2),
            "last_updated":         model_metadata.get("trained_at", "unknown"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/priority_customers")
def priority_customers(limit: int = 20):
    """Returns top N customers by priority score."""
    try:
        pred_path = BASE_DIR / "reports" / "batch_predictions.csv"
        if not pred_path.exists():
            return []

        df = pd.read_csv(pred_path)
        top = (
            df.sort_values("priority_score", ascending=False)
            .head(limit)
            .fillna(0)
        )
        return top.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk_distribution")
def risk_distribution():
    try:
        pred_path = BASE_DIR / "reports" / "batch_predictions.csv"
        if not pred_path.exists():
            return []

        df    = pd.read_csv(pred_path)
        dist  = df["risk_bucket"].value_counts().reset_index()
        dist.columns = ["risk_bucket", "count"]
        return dist.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/reload")
def reload_model():
    """Force-reload model artifacts. Call after pipeline completes."""
    global model, scaler, feature_names, model_metadata
    model, scaler, feature_names, model_metadata = load_model()
    return {
        "status":     "reloaded",
        "model_name": model_metadata.get("model_name", "unknown"),
        "loaded":     model is not None,
    }
