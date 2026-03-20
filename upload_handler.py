import os
import io
import json
import shutil
import asyncio
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Upload"])

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR  = BASE_DIR / "data" / "raw"
STATUS_FILE = BASE_DIR / "data" / "pipeline_status.json"
HISTORY_FILE= BASE_DIR / "data" / "upload_history.json"
TEMPLATE_PATH = BASE_DIR / "data" / "raw" / "Sample_dataset.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Required columns (minimum viable set) ─────────────────────────────────────
REQUIRED_COLUMNS = {
    "customer_id", "monthly_charges", "churn"
}

# ── Accepted columns (used for validation hints) ──────────────────────────────
ACCEPTED_COLUMNS = {
    "customer_id", "signup_date", "last_active_date",
    "monthly_charges", "total_spend", "usage_frequency",
    "complaints_count", "payment_delays", "contract_type", "churn",
    # Also accept Telco dataset naming
    "customerid", "tenure", "monthlycharges", "totalcharges",
    "contract", "seniorcitizen", "gender",
}

ACCEPTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB    = 50
MIN_ROWS            = 10


# ══════════════════════════════════════════════════════════════════════════════
# STATUS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _write_status(status: str, message: str, step: int = 0,
                  total_steps: int = 8, details: dict = None):
    payload = {
        "status":      status,     # idle | running | success | failed
        "message":     message,
        "step":        step,
        "total_steps": total_steps,
        "progress_pct": round(step / total_steps * 100) if total_steps else 0,
        "updated_at":  datetime.utcnow().isoformat(),
        "details":     details or {},
    }
    STATUS_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def _read_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            pass
    return {"status": "idle", "message": "No pipeline has run yet.",
            "step": 0, "total_steps": 8, "progress_pct": 0}


def _append_history(entry: dict):
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    history.insert(0, entry)
    history = history[:20]  # keep last 20
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# FILE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_upload(df: pd.DataFrame, filename: str) -> dict:
    """
    Returns {"valid": True} or {"valid": False, "errors": [...], "warnings": [...]}
    Accepts both snake_case and Telco-style column names (case-insensitive).
    """
    errors   = []
    warnings = []

    # Row count
    if len(df) < MIN_ROWS:
        errors.append(f"File has only {len(df)} rows. Minimum required: {MIN_ROWS}.")

    # Normalise all column names: lowercase + strip underscores/spaces for flexible matching
    def _norm(c):
        return c.strip().lower().replace(" ", "").replace("_", "")

    cols_norm = {_norm(c) for c in df.columns}

    # Required column equivalences (normalised form → display name)
    REQUIRED_NORM = {
        "customerid":      "customer_id",
        "monthlycharges":  "monthly_charges",
        "churn":           "churn",
    }

    # Check required columns
    for norm, display in REQUIRED_NORM.items():
        if norm not in cols_norm:
            errors.append(f"Required column missing: '{display}' (or equivalent e.g. customerID, MonthlyCharges)")

    # Check target column values
    churn_col = None
    for c in df.columns:
        if _norm(c) in ("churn", "churnflag"):
            churn_col = c
            break
    if churn_col:
        unique_vals = set(df[churn_col].dropna().astype(str).str.strip().str.lower().unique())
        valid_vals  = {"0", "1", "yes", "no", "true", "false"}
        invalid     = unique_vals - valid_vals
        if invalid:
            errors.append(
                f"Column '{churn_col}' has unexpected values: {invalid}. "
                "Expected: yes/no or 0/1."
            )

    # Check numeric columns
    for num_norm in ["monthlycharges", "totalcharges", "totalspend"]:
        for c in df.columns:
            if _norm(c) == num_norm:
                non_numeric = pd.to_numeric(df[c], errors="coerce").isna().sum()
                if non_numeric > len(df) * 0.1:
                    warnings.append(
                        f"Column '{c}' has {non_numeric} non-numeric values "
                        f"({non_numeric/len(df)*100:.1f}%). These will be set to 0."
                    )
                break

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    return {"valid": True, "errors": [], "warnings": warnings}


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def _run_pipeline_background(filepath: str, filename: str):
    """
    Runs in a background thread after a successful upload.
    Updates status file at each step for the frontend to poll.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))

    _write_status("running", f"Pipeline started for: {filename}", step=0)

    try:
        # Import orchestrator
        sys.path.insert(0, str(BASE_DIR))
        import importlib
        if "run_pipeline" in sys.modules:
            importlib.reload(sys.modules["run_pipeline"])
        import run_pipeline as rp

        step_names = {
            1: "Data Ingestion",
            2: "Data Cleaning",
            3: "Feature Engineering",
            4: "Exploratory Analysis",
            5: "Model Training",
            6: "Model Evaluation",
            7: "Batch Predictions",
            8: "Business Insights",
        }

        def step_callback(step_num, step_name, status):
            _write_status(
                "running",
                f"Step {step_num}/8: {step_name} — {status}",
                step=step_num,
            )

        # Monkey-patch run_step to update status
        original_run_step = rp.run_step
        def patched_run_step(step_num, step_name, module_name):
            _write_status("running", f"Running: {step_name}", step=step_num - 1)
            result = original_run_step(step_num, step_name, module_name)
            _write_status(
                "running",
                f"Completed: {step_name}",
                step=step_num,
                details={"last_step": result},
            )
            return result
        rp.run_step = patched_run_step

        summary = rp.run_pipeline(raise_on_failure=False)

        final_status = "success" if summary["status"] == "success" else "failed"
        _write_status(
            final_status,
            "Pipeline completed successfully! Dashboard data updated."
            if final_status == "success"
            else f"Pipeline finished with {summary['fail_count']} failure(s). Check logs.",
            step=8,
            details=summary,
        )

        _append_history({
            "filename":    filename,
            "filepath":    filepath,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status":      final_status,
            "duration":    summary["total_duration"],
            "steps":       summary["steps"],
        })

    except Exception as e:
        import traceback
        _write_status(
            "failed",
            f"Pipeline error: {str(e)}",
            step=0,
            details={"error": str(e), "traceback": traceback.format_exc()},
        )
        log.error(f"Background pipeline error: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/dataset")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a CSV or Excel file. Validates schema, saves to data/raw/,
    then triggers the full ML pipeline in the background.
    """
    # ── Check current pipeline status ──
    current = _read_status()
    if current.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="A pipeline is already running. Please wait for it to complete.",
        )

    # ── Validate file extension ──
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(ACCEPTED_EXTENSIONS)}",
        )

    # ── Read file content ──
    content = await file.read()

    # ── Check file size ──
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Parse into DataFrame ──
    try:
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse file: {str(e)}",
        )

    # ── Validate schema ──
    validation = validate_upload(df, file.filename)
    if not validation["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "File validation failed.",
                "errors":  validation["errors"],
                "warnings": validation["warnings"],
            },
        )

    # ── Save file ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = f"upload_{timestamp}{suffix}"
    save_path  = UPLOAD_DIR / safe_name

    # Always save as CSV for pipeline compatibility
    csv_path = UPLOAD_DIR / f"upload_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    log.info(f"File saved: {csv_path}  ({len(df)} rows, {len(df.columns)} columns)")

    # ── Also save as the main dataset for pipeline ──
    main_csv = UPLOAD_DIR / "uploaded_dataset.csv"
    df.to_csv(main_csv, index=False)

    # ── Trigger pipeline in background ──
    _write_status("running", f"File uploaded. Starting pipeline...", step=0)
    background_tasks.add_task(
        _run_pipeline_background,
        str(main_csv),
        file.filename,
    )

    return {
        "message":   "File uploaded successfully. Pipeline started.",
        "filename":  file.filename,
        "rows":      len(df),
        "columns":   list(df.columns),
        "warnings":  validation["warnings"],
        "saved_as":  str(csv_path.name),
        "status_url": "/api/upload/status",
    }


@router.get("/status")
async def get_pipeline_status():
    """
    Returns the current pipeline run status.
    Frontend polls this every 3 seconds while pipeline is running.
    """
    return _read_status()


@router.get("/history")
async def get_upload_history():
    """
    Returns list of the last 20 uploads and their pipeline results.
    """
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return []


@router.get("/template")
async def download_template():
    """
    Returns a sample CSV template showing the expected format.
    """
    if TEMPLATE_PATH.exists():
        return FileResponse(
            path=str(TEMPLATE_PATH),
            media_type="text/csv",
            filename="churn_dataset_template.csv",
        )

    # Generate a minimal template on the fly
    template_data = {
        "customer_id":     ["C001", "C002", "C003"],
        "signup_date":     ["2022-01-15", "2021-06-20", "2023-03-10"],
        "last_active_date":["2024-01-10", "2023-08-15", "2024-02-01"],
        "monthly_charges": [75.50, 45.00, 110.25],
        "total_spend":     [1510.00, 540.00, 330.75],
        "usage_frequency": [22, 8, 30],
        "complaints_count":[0, 3, 1],
        "payment_delays":  [0, 2, 0],
        "contract_type":   ["Two year", "Month-to-month", "One year"],
        "churn":           ["No", "Yes", "No"],
    }
    template_df = pd.DataFrame(template_data)

    temp_path = UPLOAD_DIR / "_template.csv"
    template_df.to_csv(temp_path, index=False)

    return FileResponse(
        path=str(temp_path),
        media_type="text/csv",
        filename="churn_dataset_template.csv",
    )


@router.get("/column-guide")
async def column_guide():
    """
    Returns documentation on expected columns, types, and valid values.
    """
    return {
        "required_columns": [
            {
                "name": "customer_id",
                "type": "string",
                "description": "Unique customer identifier",
                "example": "C001",
            },
            {
                "name": "monthly_charges",
                "type": "float",
                "description": "Monthly billing amount in dollars",
                "example": 75.50,
            },
            {
                "name": "churn",
                "type": "string or int",
                "description": "Historical churn outcome",
                "valid_values": ["Yes", "No", "1", "0"],
            },
        ],
        "optional_columns": [
            {"name": "signup_date",      "type": "date",   "description": "Customer signup date (YYYY-MM-DD)"},
            {"name": "last_active_date", "type": "date",   "description": "Last activity date (YYYY-MM-DD)"},
            {"name": "total_spend",      "type": "float",  "description": "Total historical spend ($)"},
            {"name": "usage_frequency",  "type": "int",    "description": "Sessions/logins per month"},
            {"name": "complaints_count", "type": "int",    "description": "Number of support complaints"},
            {"name": "payment_delays",   "type": "int",    "description": "Number of late payments"},
            {"name": "contract_type",    "type": "string", "description": "Contract duration",
             "valid_values": ["Month-to-month", "One year", "Two year"]},
        ],
        "accepted_formats":  ["CSV (.csv)", "Excel (.xlsx)", "Excel 97-2003 (.xls)"],
        "max_file_size_mb":  MAX_FILE_SIZE_MB,
        "min_rows":          MIN_ROWS,
        "notes": [
            "Column names are case-insensitive.",
            "Telco dataset format (MonthlyCharges, TotalCharges, Contract) is also accepted.",
            "Extra columns are allowed and will be used as additional features.",
            "The pipeline auto-detects and handles missing values.",
        ],
    }
