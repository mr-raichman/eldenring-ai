"""
stamina.py - live stamina monitor.

Confirms the current/max stamina offsets (ptr4 + 0x154 / 0x158). Sprint to
deplete and watch the ratio.

    uv run python tools/stamina.py
"""

import sys
import time

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai.io.memory import GameMemory, _read_int32

STAMINA_CURRENT = 0x154
STAMINA_MAX     = 0x158


def main():
    memory = GameMemory()
    if not memory.connect():
        print("Elden Ring not found.")
        sys.exit(1)

    print("Watching stamina (ptr4 + 0x154 current / 0x158 max).")
    print("Sprint to deplete. Ctrl+C to stop.\n")
    print(f"{'Current':>10}  {'Max':>10}  {'Ratio':>10}")
    print("-" * 34)

    try:
        while True:
            memory.refresh()
            if memory._last_ptr4:
                current = _read_int32(memory.pid, memory._last_ptr4 + STAMINA_CURRENT)
                maximum = _read_int32(memory.pid, memory._last_ptr4 + STAMINA_MAX)
                ratio   = current / maximum if maximum > 0 else 0
                print(f"{current:>10}  {maximum:>10}  {ratio:>10.3f}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nDone.")


if __name__ == "__main__":
    main()
