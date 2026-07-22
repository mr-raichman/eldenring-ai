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

EPISODE_RECORDS = DATA_DIR / "episode_records.jsonl"
STEP_RECORDS = DATA_DIR / "step_records.csv"
