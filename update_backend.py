import os
import re

# 1. Update run_pipeline.py
with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    rp = f.read()

rp = rp.replace('parser.add_argument(\n        "--skip-train"', 'parser.add_argument("--run-id", type=str, default=None, help="Unique Run ID for logging")\n    parser.add_argument(\n        "--skip-train"')

rp = re.sub(
    r'if not IS_VERCEL:\n\s*LOG_DIR = Path\(__file__\).resolve\(\).parent / "logs"\n\s*LOG_DIR.mkdir\(exist_ok=True\)\n\s*log_file = LOG_DIR / f"pipeline_\{datetime.now\(\).strftime\(\'%Y%m%d_%H%M%S\'\)\}.log"\n\s*handlers.append\(logging.FileHandler\(log_file\)\)\n\s*log_file_str = str\(log_file\)',
    '''
if IS_VERCEL:
    LOG_DIR = Path("/tmp/app/logs")
else:
    LOG_DIR = Path(__file__).resolve().parent / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
import sys as _sys
_args = _sys.argv
_run_id = [a.split("=")[1] for a in _args if a.startswith("--run-id=")]
if not _run_id:
    try:
        idx = _args.index("--run-id")
        _run_id = [_args[idx+1]]
    except:
        pass
run_id = _run_id[0] if _run_id else f"RUN-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

log_file = LOG_DIR / f"{run_id}.log"
handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
log_file_str = str(log_file)''', rp)

# Ensure sys is imported correctly
rp = rp.replace('import sys\nimport time', 'import sys\nimport time\nimport uuid')

with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(rp)


# 2. Update upload_handler.py to generate run_id, pass it, and save it
with open('upload_handler.py', 'r', encoding='utf-8') as f:
    uh = f.read()

# Fix Vercel src logic globally first!
for d in ['["data", "models", "reports"]', '["data","models","reports"]']:
    uh = uh.replace(d, '["src", "data", "models", "reports"]')

# Add run_id to _run_pipeline_background
uh = uh.replace('def _run_pipeline_background(filepath: str, filename: str):', 'def _run_pipeline_background(filepath: str, filename: str, run_id: str):')

# Pass run_id to history and status
uh = uh.replace(
'''        _append_history({
            "filename":    filename,
            "filepath":    filepath,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status":      final_status,
            "duration":    round(sum(s.get("duration", 0) for s in summary["steps"]), 2),
            "steps":       summary["steps"],
        })''',
'''        _append_history({
            "run_id":      run_id,
            "filename":    filename,
            "filepath":    filepath,
            "uploaded_at": datetime.utcnow().isoformat(),
            "status":      final_status,
            "duration":    round(sum(s.get("duration", 0) for s in summary["steps"]), 2),
            "steps":       summary["steps"],
        })''')

# In upload_dataset endpoint
uh = uh.replace('background_tasks.add_task(_run_pipeline_background, str(file_path), file.filename)',
'''    import uuid
    run_id = f"RUN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:4]}"
    background_tasks.add_task(_run_pipeline_background, str(file_path), file.filename, run_id)''')

uh = uh.replace('''return {
        "message": "File uploaded successfully. Pipeline started.",
        "filename": file.filename,''',
'''return {
        "message": "File uploaded successfully. Pipeline started.",
        "run_id": run_id,
        "filename": file.filename,''')

# We need to make sure run_pipeline.py gets called with the run_id inside patched_run_step
# Actually, the background task just needs to write the run_id to STATUS_FILE
uh = uh.replace('def _write_status(status: str, message: str, step: int = 0,\n                  total_steps: int = 8, details: dict = None):',
'def _write_status(status: str, message: str, step: int = 0,\n                  total_steps: int = 8, details: dict = None, run_id: str = None):')
uh = uh.replace('"updated_at":  datetime.utcnow().isoformat(),',
'"updated_at":  datetime.utcnow().isoformat(),\n        "run_id": run_id,')
uh = uh.replace('_write_status("running", f"Running: {step_name}", step=step_num - 1)', '_write_status("running", f"Running: {step_name}", step=step_num - 1, run_id=run_id)')
uh = uh.replace('_write_status(\n                "running",\n                f"Completed: {step_name}",\n                step=step_num,\n                details={"last_step": result},\n            )',
'_write_status(\n                "running",\n                f"Completed: {step_name}",\n                step=step_num,\n                details={"last_step": result},\n                run_id=run_id\n            )')
uh = uh.replace('_write_status("failed", f"Pipeline failed: {e}")', '_write_status("failed", f"Pipeline failed: {e}", run_id=run_id)')
uh = uh.replace('_write_status("success", "Pipeline completed successfully! Dashboard data updated.", step=8, details=summary)', '_write_status("success", "Pipeline completed successfully! Dashboard data updated.", step=8, details=summary, run_id=run_id)')
uh = uh.replace('_write_status("failed", f"Pipeline finished with {fail_count} failure(s). Check logs.", step=8, details=summary)', '_write_status("failed", f"Pipeline finished with {fail_count} failure(s). Check logs.", step=8, details=summary, run_id=run_id)')


# Add logs endpoint
uh = uh.replace('@router.get("/status")',
'''@router.get("/logs/{run_id}")
async def get_logs(run_id: str):
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except:
            pass
            
    run_meta = next((r for r in history if r.get("run_id") == run_id), None)
    
    if IS_VERCEL:
        log_path = Path(f"/tmp/app/logs/{run_id}.log")
    else:
        log_path = Path(__file__).resolve().parent / "logs" / f"{run_id}.log"
        
    log_content = ""
    if log_path.exists():
        log_content = log_path.read_text(encoding="utf-8")
        
    if not run_meta and not log_path.exists():
        return {"error": "Pipeline run not found."}
        
    return {
        "run_id": run_id,
        "metadata": run_meta,
        "logs": log_content
    }

@router.get("/status")''')

with open('upload_handler.py', 'w', encoding='utf-8') as f:
    f.write(uh)

# 3. Patch src logic across other files!
for f in ["api/index.py", "src/business_insights.py", "src/eda.py", "src/evaluate.py", "src/persist_insights.py", "src/train.py"]:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        for d in ['["data", "models", "reports"]', '["data","models","reports"]']:
            content = content.replace(d, '["src", "data", "models", "reports"]')
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)

print("Python backend updated!")
