import os
import re

with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    rp = f.read()

# Make run_pipeline accept run_id
rp = rp.replace('def run_pipeline(from_step: int = 1, skip_eda: bool = False, skip_train: bool = False, raise_on_failure: bool = True) -> dict:',
'def run_pipeline(from_step: int = 1, skip_eda: bool = False, skip_train: bool = False, raise_on_failure: bool = True, run_id: str = None) -> dict:')

# We need to capture the logs that occur during run_pipeline.
# Actually, the global handlers are already set up. If we just clear and add a new one inside run_pipeline, it's fine.
patch = """
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
"""

rp = rp.replace('summary = {', patch + '\n    summary = {')

with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(rp)

# Now update upload_handler.py to pass run_id to run_pipeline
with open('upload_handler.py', 'r', encoding='utf-8') as f:
    uh = f.read()

uh = uh.replace('summary = rp.run_pipeline(raise_on_failure=False)', 'summary = rp.run_pipeline(raise_on_failure=False, run_id=run_id)')

with open('upload_handler.py', 'w', encoding='utf-8') as f:
    f.write(uh)
