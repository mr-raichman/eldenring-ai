"""
area_id.py - live monitor of the player's area ID (ptr4 + 0x19A).

Cross the Margit fog gate in both directions and watch the value change:
MARGIT_AREA_ID (4) means inside the arena, 0 means loading/transition. Drives
a GameMemory so it shares the exact pointer chain and cached pointer resolver
the training loop uses.

    uv run python tools/area_id.py
"""

import sys
import time

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai.config import MARGIT_AREA_ID
from eldenring_ai.io.memory import GameMemory


def _label(area_id):
    if area_id == MARGIT_AREA_ID:
        return "Margit arena"
    if area_id == 0:
        return "loading / transition"
    return "other"


def main():
    memory = GameMemory()
    if not memory.connect():
        print("Elden Ring not found.")
        sys.exit(1)

    print("Live monitor of area ID (ptr4 + 0x19A).")
    print("Cross the fog gate in both directions and watch the value.")
    print(f"MARGIT_AREA_ID = {MARGIT_AREA_ID}. Ctrl+C to stop.\n")

    prev = None
    try:
        while True:
            memory.refresh()
            if memory._last_ptr4 is None:
                print("  [player not loaded]")
                time.sleep(0.5)
                continue
            area_id = memory._last_area_id
            if area_id != prev:
                print(f"  area_id: {prev} -> {area_id}  ({_label(area_id)})")
                prev = area_id
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
