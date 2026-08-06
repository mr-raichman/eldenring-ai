"""
restore_save.py - manual check of the save-restore that training performs after a
crash or a victory: kill the game, copy the backup over the live save, relaunch.

Use it to prove the restore really works without waiting for a crash. Move the
character somewhere else in game (another grace, mid-run, wherever), then:

    uv run python tools/restore_save.py

The game should come back with the character exactly where the backup has it. This
runs the same eldenring_ai.io.save.restore_save() the training loop calls, so a pass
here is a pass there.
"""

import subprocess
import time

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai import config
from eldenring_ai.config import paths
from eldenring_ai.io.memory import GameMemory
from eldenring_ai.io.save import restore_save


def main():
    print(f"backup: {paths.SAVE_BACKUP}")
    print(f"live:   {paths.SAVE_LIVE}")

    memory = GameMemory()
    memory.connect()
    if memory.is_alive():
        print(f"killing the game (pid {memory.pid})...")
        memory.kill()
        print("game closed")
    else:
        print("game is not running")

    # Only safe now: a running game holds the save in memory and would write it back.
    reason = restore_save()
    if reason:
        print(f"RESTORE SKIPPED: {reason}")
        return 1
    print("save restored from backup")

    print("relaunching through Steam...")
    subprocess.Popen(
        ["steam", f"steam://rungameid/{config.ELDEN_RING_APP_ID}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    while True:
        memory.connect()
        if memory.is_alive():
            print(f"game running (pid {memory.pid}) - load the save and check the character")
            return 0
        time.sleep(config.GAME_LAUNCH_POLL_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
