"""
paths.py - filesystem locations. Everything the project owns is derived from the
package location so the repository is relocatable (replaces the old
~/eldenring-ai/... hardcodes). SAVE_LIVE is the one exception: it points at the
game's own save, which lives wherever Steam put it.
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

# The game's live save, restored from SAVE_BACKUP after a crash or a victory so
# every episode starts from the same character state (and so a defeated Margit
# does not persist into the next run). Machine-specific: it contains the Steam
# library location and the account's Steam ID, so unlike everything above it
# cannot be derived from PROJECT_ROOT. The copy is strictly one-way - nothing
# writes SAVE_BACKUP, or a post-victory save would overwrite the canonical state.
SAVE_LIVE = Path.home() / (
    "Games/SteamLibrary/steamapps/compatdata/1245620/pfx/drive_c/users/steamuser"
    "/AppData/Roaming/EldenRing/76561198216009149/ER0000.sl2"
)

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
