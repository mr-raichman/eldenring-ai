"""
test_memory.py - Real-time monitor for all memory-read values.

Connects to the running Elden Ring process and prints a live dashboard
that refreshes every 0.5 seconds. Press Ctrl+C to exit.

Values displayed
----------------
  Player HP       current / max and the normalised ratio used by the agent
  Ready flag      raw int at ptr4 + 0x0A0  (0 = loading, 1 = controllable)
  Area ID         raw byte at ptr4 + 0x19A and its human-readable meaning
  Flask count     raw byte at flask_object + 0x388  (floors at 1 when empty)
  Flask flag      binary byte at flask_object + 0x3B0  (1 = has flasks, 0 = empty)

Derived booleans (computed from the above, same logic as the training loop)
  is_player_loaded   ptr4 is not null
  is_player_alive    loaded and current HP > 0
  is_player_at_full_hp  alive and current HP >= max HP
  is_player_ready    ready flag == 1 AND area ID != 0
  is_in_boss_arena   area ID == MARGIT_AREA_ID (4)

Run from the project root:
    python tests/test_memory.py
"""

import os
import sys
import time

import cv2

os.environ.setdefault("DISPLAY",         ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

from eldenring_ai.config import MARGIT_AREA_ID
from eldenring_ai.io.memory import GameMemory, _read_boss_hp_bar

# ── Area ID lookup ─────────────────────────────────────────────────────────────
# Maps the raw area ID byte to a human-readable label.
# Add entries here if you discover new area IDs during testing.

AREA_LABELS = {
    0:  "loading / transition",
    4:  "Margit's arena  ← inside fog gate",
    68: "approach corridor (outside fog gate)",
}


def area_label(area_id):
    return AREA_LABELS.get(area_id, f"unknown area ({area_id})")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def bool_str(value):
    return "YES" if value else "NO "


def divider(char="─", width=54):
    return char * width


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    print("Connecting to Elden Ring...")
    memory = GameMemory()
    if not memory.connect():
        print("Could not connect. Make sure the game is running and in-world.")
        sys.exit(1)

    print("Connected. Monitoring memory - press Ctrl+C to stop.\n")
    time.sleep(0.5)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Warning: could not open /dev/video0 - boss HP will show as '-'")
        print("Make sure wf-recorder is running and v4l2loopback is loaded.")
    try:
        while True:
            memory.refresh()

            # Capture current frame for OpenCV boss HP reading
            ret, frame = cap.read()
            boss_hp = _read_boss_hp_bar(frame) if frame is not None else "-"

            # ── Raw values ────────────────────────────────────────────────────
            ptr4_loaded  = memory._last_ptr4 is not None
            current_hp   = memory._last_current_hp  if ptr4_loaded else "-"
            max_hp       = memory._last_max_hp       if ptr4_loaded else "-"
            hp_ratio     = memory.read_player_hp()
            ready_flag   = memory._last_is_ready     if ptr4_loaded else "-"
            area_id      = memory._last_area_id      if ptr4_loaded else "-"
            flask_count  = memory._last_flasks
            flask_flag   = memory._last_flask_empty_flag

            # ── Derived booleans ──────────────────────────────────────────────
            d_loaded   = memory.is_player_loaded()
            d_alive    = memory.is_player_alive()
            d_full_hp  = memory.is_player_at_full_hp()
            d_ready    = memory.is_player_ready()
            d_arena    = memory.is_in_boss_arena()

            # ── Print dashboard ───────────────────────────────────────────────
            # \033[2J clears the terminal; \033[H moves cursor to top-left.
            print("\033[2J\033[H", end="")

            print(divider("═"))
            print("  ELDEN RING - MEMORY MONITOR")
            print(divider("═"))

            print("\n  ── Raw values ──────────────────────────────────────────")
            print(f"  Player HP        {current_hp} / {max_hp}   (ratio: {hp_ratio:.3f})")
            print(f"  Boss HP          {boss_hp}")
            print(f"  Ready flag       {ready_flag}")
            print(f"  Area ID          {area_id}  →  {area_label(area_id) if ptr4_loaded else '-'}")
            print(f"  Flask count      {flask_count}  (raw byte - floors at 1 when empty)")
            print(f"  Flask flag       {flask_flag}  (1 = has flasks,  0 = all consumed)")

            print("\n  ── Derived booleans ────────────────────────────────────")
            print(f"  is_player_loaded     {bool_str(d_loaded)}")
            print(f"  is_player_alive      {bool_str(d_alive)}")
            print(f"  is_player_at_full_hp {bool_str(d_full_hp)}")
            print(f"  is_player_ready      {bool_str(d_ready)}")
            print(f"  is_in_boss_arena     {bool_str(d_arena)}  (MARGIT_AREA_ID = {MARGIT_AREA_ID})")

            print("\n" + divider())
            print("  Refreshing every 0.5s - Ctrl+C to exit")
            print(divider())

            time.sleep(0.5)

    except KeyboardInterrupt:
        cap.release()
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()
