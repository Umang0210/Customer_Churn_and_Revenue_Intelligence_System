"""
api/app.py â€” PATCH INSTRUCTIONS
================================
Add the following lines to your existing api/app.py.

STEP 1: Add this import near the top (after existing imports):
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from upload_handler import router as upload_router

STEP 2: Mount the router (after app = FastAPI(...)):
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.include_router(upload_router)

STEP 3: Add the /api/upload/status SSE endpoint (optional but makes
        progress polling more efficient):
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Already handled inside upload_handler.py via GET /api/upload/status
# No additional code needed.

â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FULL PATCHED app.py SHOWN BELOW â€” replace your existing file:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

# â”€â”€ Standard imports (keep your existing ones) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

security = HTTPBasic()

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    return "admin"

# â”€â”€ Add project root to path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os, shutil
_REAL_BASE = Path(__file__).resolve().parent.parent
IS_VERCEL = os.environ.get("VERCEL") == "1"
if IS_VERCEL:
    BASE_DIR = Path("/tmp/app")
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        for d in ["src", "data", "models", "reports"]:
            src_dir = _REAL_BASE / d
            dst_dir = BASE_DIR / d
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
else:
    BASE_DIR = _REAL_BASE
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# â”€â”€ Import upload router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from upload_handler import router as upload_router

# Sub-routers disabled â€” endpoints are now handled directly in this file
HAS_SUBROUTERS = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "churn_model.pkl"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_list.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# â”€â”€ App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ Mount routers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app.include_router(upload_router)            # â† upload + pipeline trigger


# â”€â”€ Load model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


# â”€â”€ Request schema â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PredictRequest(BaseModel):
    customer_id:      str
    revenue:          float = Field(default=0.0, ge=0.0)
    monthly_charges:  float = Field(default=0.0, ge=0.0)
    usage_frequency:  int   = Field(default=0, ge=0)
    complaints_count: int   = Field(default=0, ge=0)
    payment_delays:   int   = Field(default=0, ge=0)
    gender:           Optional[str] = None
    seniorcitizen:    Optional[str] = None
    contract:         Optional[str] = None


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  model is not None,
        "model_name":    model_metadata.get("model_name", "unknown"),
        "model_version": model_metadata.get("model_version", "unknown"),
        "timestamp":     datetime.utcnow().isoformat(),
        "feature_importances": model_metadata.get("feature_importances", {})
    }


@app.post("/predict")
@app.post("/api/predict")
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


@app.get("/api/dashboard/data")
def dashboard_data():
    """Returns the full dashboard data dictionary."""
    try:
        # First, try to read the recently exported processed JSON
        data_json = BASE_DIR / "data" / "processed" / "dashboard_data.json"
        if data_json.exists():
            return json.loads(data_json.read_text())
        
        # Alternatively, generate it on the fly
        import export_dashboard as exporter
        from fastapi.responses import JSONResponse
        
        data = exporter.generate_dashboard_data()
        if data:
            return JSONResponse(content=json.loads(json.dumps(data, cls=exporter.NpEncoder)))
            
        raise HTTPException(status_code=404, detail="Dashboard data not found. Run the pipeline first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/customers")
def get_customer_data(page: int = 1, page_size: int = 50):
    """Returns paginated customer data."""
    try:
        data_path = BASE_DIR / "data" / "processed" / "final_dataset.csv"
        if not data_path.exists():
            data_path = BASE_DIR / "data" / "raw" / "uploaded_dataset.csv"
            
        if not data_path.exists():
            return {"total": 0, "page": page, "page_size": page_size, "columns": [], "data": []}
            
        # Use pandas chunking or just load if it's small enough. For 7000 rows, loading in memory is fine.
        df = pd.read_csv(data_path)
        # Convert NaN to None for JSON
        df = df.replace({np.nan: None})
        
        total = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        
        page_df = df.iloc[start:end]
        
        response_data = {
            "total": total,
            "page": page,
            "page_size": page_size,
            "columns": list(df.columns),
            "data": page_df.to_dict(orient="records")
        }
        
        import export_dashboard as exporter
        from fastapi.responses import JSONResponse
        import json
        return JSONResponse(content=json.loads(json.dumps(response_data, cls=exporter.NpEncoder)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/summary")
def dashboard_summary():
    """Returns top-level KPIs for the dashboard."""
    try:
        pred_path = BASE_DIR / "data" / "processed" / "batch_predictions.csv"
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
        pred_path = BASE_DIR / "data" / "processed" / "batch_predictions.csv"
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
        from db import get_engine
        from sqlalchemy import text
        
        with get_engine().connect() as conn:
            result = conn.execute(text("SELECT risk_bucket, COUNT(*) as count FROM customer_predictions GROUP BY risk_bucket")).fetchall()
            return [{"risk_bucket": r[0], "count": r[1]} for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/feature_importances")
def feature_importances():
    try:
        from db import get_engine
        from sqlalchemy import text
        import json
        with get_engine().connect() as conn:
            # Get latest model run
            result = conn.execute(text("SELECT feature_importances FROM model_runs ORDER BY run_date DESC LIMIT 1")).fetchone()
            if not result or not result[0]:
                return {}
            return json.loads(result[0])
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

