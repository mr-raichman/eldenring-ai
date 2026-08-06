"""
stamina.py - live stamina monitor.

Confirms the stamina offsets in config/offsets.py are still right for this game build.
Sprint to deplete and watch the ratio fall. Reads through GameMemory.read_stamina(),
the same call the training loop uses, so a correct reading here is a correct reading
there.

    uv run python tools/stamina.py
"""

import sys
import time

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai.config.offsets import PTR4_STAMINA_CUR, PTR4_STAMINA_MAX
from eldenring_ai.io.memory import GameMemory


def main():
    memory = GameMemory()
    if not memory.connect():
        print("Elden Ring not found.")
        sys.exit(1)

    print(f"Watching stamina (ptr4 + {PTR4_STAMINA_CUR:#x} current / {PTR4_STAMINA_MAX:#x} max).")
    print("Sprint to deplete. Ctrl+C to stop.\n")

    try:
        while True:
            memory.refresh()
            # read_stamina() returns 1.0 both at full stamina and when it cannot read,
            # so say which one this is rather than printing a confident 1.000.
            if memory._last_ptr4 is None:
                print("  [player not loaded]        ", end="\r")
            else:
                print(f"  stamina: {memory.read_stamina():.3f}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nDone.")


if __name__ == "__main__":
    main()
