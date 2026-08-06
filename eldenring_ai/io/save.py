"""
save.py - restoring the canonical Elden Ring save over the game's live save.
"""

import shutil

from eldenring_ai import config
from eldenring_ai.config import paths


def restore_save():
    """Copy the backup save over the live save, `.bak` included.

    The game must not be running: it holds the save in memory and would write it
    back out on exit, undoing the copy.

    The copy is strictly one-way. Nothing here ever writes SAVE_BACKUP, because a
    save taken after a victory would make a defeated Margit the canonical state and
    there is no other copy of it - `data/` is not in git.

    Returns None when the save was restored, or a string explaining why it was
    skipped. Callers log the reason; a failed copy must never end a training run.
    """
    backup = paths.SAVE_BACKUP
    if not backup.exists():
        return f"no backup at {backup}"
    if backup.stat().st_size < config.SAVE_MIN_BYTES:
        return f"backup at {backup} looks truncated"

    # The .bak is the game's own rollback copy; leaving a defeated-Margit one behind
    # would let a save repair undo the restore.
    shutil.copyfile(backup, paths.SAVE_LIVE)
    shutil.copyfile(backup, paths.SAVE_LIVE.parent / (paths.SAVE_LIVE.name + ".bak"))
    return None
