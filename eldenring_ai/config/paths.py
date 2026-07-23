"""
paths.py - filesystem locations derived from the package location so the
repository is relocatable (replaces the old ~/eldenring-ai/... hardcodes).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # eldenring_ai/config/paths.py -> repo root

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

SESSION_STATS = DATA_DIR / "session_stats.json"
PTR_CACHE = DATA_DIR / "world_chr_man_ptr.cache"
WF_LOG = DATA_DIR / "wf-recorder.log"
SAVE_BACKUP = DATA_DIR / "eldenring-save-backup.sl2"

# Each training run keeps its records + event log in its own folder under data/runs/.
# These default to DATA_DIR (standalone / tests) and are repointed by use_run_dir().
RUNS_DIR = DATA_DIR / "runs"
EPISODE_RECORDS = DATA_DIR / "episode_records.jsonl"
STEP_RECORDS = DATA_DIR / "step_records.csv"
EVENT_LOG = DATA_DIR / "events.log"


def use_run_dir(run_dir):
    """Point the per-run record + event-log files at `run_dir`, called once at
    startup so each run's data lives in its own folder."""
    global EPISODE_RECORDS, STEP_RECORDS, EVENT_LOG
    EPISODE_RECORDS = run_dir / "episode_records.jsonl"
    STEP_RECORDS = run_dir / "step_records.csv"
    EVENT_LOG = run_dir / "events.log"
