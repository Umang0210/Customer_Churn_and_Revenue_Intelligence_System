"""
Pipeline Orchestrator â€” Customer Churn & Revenue Intelligence
=============================================================
Connects ALL pipeline steps end-to-end in the correct order.
Can be triggered:
  - Manually:        python run_pipeline.py
  - From API:        POST /api/upload/run-pipeline
  - From CI/CD:      GitHub Actions workflow
  - On file upload:  auto-triggered by upload_handler.py

Steps:
  1. ingestion.py
  2. cleaning.py
  3. features.py
  4. eda.py
  5. train.py
  6. evaluate.py          â† exits with code 1 if AUC < 0.70
  7. persist_insights.py
  8. business_insights.py

Usage:
    python run_pipeline.py                   # full pipeline
    python run_pipeline.py --from-step 5     # start from training
    python run_pipeline.py --skip-eda        # skip EDA (faster)
    python run_pipeline.py --skip-train      # predictions only (use existing model)
"""

import sys
import time
import uuid
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime


# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os

IS_VERCEL = os.environ.get("VERCEL") == "1"

handlers = [logging.StreamHandler(sys.stdout)]
log_file_str = "stdout"

if IS_VERCEL:
    LOG_DIR = Path("/tmp/app/logs")
else:
    LOG_DIR = Path(__file__).resolve().parent / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
run_id = [arg.split("=")[1] for arg in sys.argv if arg.startswith("--run-id=")]
run_id = run_id[0] if run_id else f"RUN-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_file = LOG_DIR / f"{run_id}.log"
handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
log_file_str = str(log_file)

import sys
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [PIPELINE]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=handlers,
)
log = logging.getLogger(__name__)

# â”€â”€ Paths & sys.path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os
import importlib
import os, shutil
_REAL_BASE = Path(__file__).resolve().parent
IS_VERCEL = os.environ.get("VERCEL") == "1"
if IS_VERCEL:
    BASE_DIR = Path("/tmp/app")
    if not BASE_DIR.exists():
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        for d in ["data", "models", "reports"]:
            src_dir = _REAL_BASE / d
            dst_dir = BASE_DIR / d
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
else:
    BASE_DIR = _REAL_BASE

# Always run from project root so relative paths (data/, models/) work
os.chdir(BASE_DIR)

# Insert src/ so pipeline modules are importable by name
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP RUNNER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_step(step_num: int, step_name: str, module_name: str) -> dict:
    """
    Imports and runs a pipeline step's main() function.
    Returns a result dict with status, duration, error.
    """
    result = {
        "step":     step_num,
        "name":     step_name,
        "module":   module_name,
        "status":   "pending",
        "duration": 0.0,
        "error":    None,
    }

    log.info(f"{'â”€'*55}")
    log.info(f"  STEP {step_num}: {step_name}")
    log.info(f"{'â”€'*55}")

    start = time.time()
    try:
        # Use importlib so we can reload on repeated runs (avoids stale cache)
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)

        if hasattr(module, "main"):
            module.main()
        else:
            raise AttributeError(f"Module '{module_name}' has no main() function.")

        result["duration"] = round(time.time() - start, 2)
        result["status"]   = "success"
        log.info(f"  âœ“ {step_name} completed in {result['duration']}s\n")

    except SystemExit as e:
        # evaluate.py calls sys.exit(1) on gate failure â€” treat as fatal
        result["duration"] = round(time.time() - start, 2)
        result["status"]   = "gate_failed"
        result["error"]    = f"Performance gate failed (exit code {e.code})"
        log.error(f"  âœ— {step_name} GATE FAILED: {result['error']}\n")

    except Exception as e:
        result["duration"] = round(time.time() - start, 2)
        result["status"]   = "failed"
        result["error"]    = str(e)
        log.error(f"  âœ— {step_name} FAILED: {e}")
        log.error(traceback.format_exc())

    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PIPELINE DEFINITION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
PIPELINE_STEPS = [
    (1, "Data Ingestion",         "ingestion"),
    (2, "Data Cleaning",          "cleaning"),
    (3, "Feature Engineering",    "features"),
    (4, "Exploratory Analysis",   "eda"),
    (5, "Model Training",         "train"),
    (6, "Model Evaluation",       "evaluate"),
    (7, "Batch Predictions",      "persist_insights"),
    (8, "Business Insights",      "business_insights"),
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_pipeline(
    from_step: int = 1,
    skip_eda: bool = False,
    skip_train: bool = False,
    raise_on_failure: bool = False,
) -> dict:
    """
    Runs the full pipeline. Returns a summary dict.
    Used by both CLI and the FastAPI upload endpoint.
    """
    pipeline_start = time.time()
    results        = []
    failed         = False

    log.info("=" * 55)
    log.info("  CHURN INTELLIGENCE PIPELINE â€” STARTING")
    log.info(f"  From step: {from_step}")
    log.info(f"  Skip EDA:  {skip_eda}")
    log.info(f"  Skip train: {skip_train}")
    log.info("=" * 55 + "\n")

    for step_num, step_name, module_name in PIPELINE_STEPS:
        # Skip steps before from_step
        if step_num < from_step:
            log.info(f"  [SKIP] Step {step_num}: {step_name}")
            results.append({
                "step": step_num, "name": step_name,
                "status": "skipped", "duration": 0.0, "error": None,
            })
            continue

        # Optional skips
        if skip_eda and step_num == 4:
            log.info(f"  [SKIP] Step 4: EDA (--skip-eda)")
            results.append({
                "step": 4, "name": step_name,
                "status": "skipped", "duration": 0.0, "error": None,
            })
            continue

        if skip_train and step_num in (5, 6):
            log.info(f"  [SKIP] Step {step_num}: {step_name} (--skip-train)")
            results.append({
                "step": step_num, "name": step_name,
                "status": "skipped", "duration": 0.0, "error": None,
            })
            continue

        result = run_step(step_num, step_name, module_name)
        results.append(result)

        if result["status"] in ("failed", "gate_failed"):
            failed = True
            if raise_on_failure or result["status"] == "gate_failed":
                break
            # For non-gate failures, continue unless it's a critical data step
            if step_num <= 3:
                log.error(
                    f"  Critical step {step_num} failed. Aborting pipeline."
                )
                break

    total_duration = round(time.time() - pipeline_start, 2)

    # â”€â”€ Summary â”€â”€
    success_count = sum(1 for r in results if r["status"] == "success")
    skip_count    = sum(1 for r in results if r["status"] == "skipped")
    fail_count    = sum(1 for r in results if r["status"] in ("failed", "gate_failed"))

    log.info("\n" + "=" * 55)
    log.info("  PIPELINE SUMMARY")
    log.info("=" * 55)
    for r in results:
        icon = "âœ“" if r["status"] == "success" else ("â†’" if r["status"] == "skipped" else "âœ—")
        log.info(f"  {icon}  Step {r['step']}: {r['name']:<28} {r['status']:<12} {r['duration']}s")
    log.info(f"\n  Total time  : {total_duration}s")
    log.info(f"  Succeeded   : {success_count}")
    log.info(f"  Skipped     : {skip_count}")
    log.info(f"  Failed      : {fail_count}")
    log.info(f"  Log file    : {log_file_str}")
    log.info("=" * 55 + "\n")

    
    if run_id:
        if IS_VERCEL:
            LOG_DIR = Path("/tmp/app/logs")
        else:
            LOG_DIR = Path(__file__).resolve().parent / "logs"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{run_id}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        log.addHandler(fh)
        # Also root logger
        logging.getLogger().addHandler(fh)

    summary = {
        "status":         "failed" if failed else "success",
        "total_duration": total_duration,
        "steps":          results,
        "success_count":  success_count,
        "skip_count":     skip_count,
        "fail_count":     fail_count,
        "log_file":       log_file_str,
        "completed_at":   datetime.utcnow().isoformat(),
    }

    if failed and raise_on_failure:
        raise RuntimeError(f"Pipeline failed. See {log_file} for details.")

    return summary


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLI
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Intelligence â€” Pipeline Orchestrator"
    )
    parser.add_argument(
        "--from-step", type=int, default=1,
        help="Start from step N (1=ingestion, 5=train, 7=predictions)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA step (faster iteration)"
    )
    parser.add_argument("--run-id", type=str, default=None, help="Unique Run ID for logging")
    parser.add_argument("--skip-train", action="store_true",
        help="Skip training + evaluation (use existing model)"
    )
    args = parser.parse_args()

    summary = run_pipeline(
        from_step=args.from_step,
        skip_eda=args.skip_eda,
        skip_train=args.skip_train,
    )

    sys.exit(0 if summary["status"] == "success" else 1)


if __name__ == "__main__":
    main()

