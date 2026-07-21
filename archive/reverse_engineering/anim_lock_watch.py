"""
confirm_anim_flag.py - Real-time visual confirmation of ptr4 + 0x0468.

Displays the current animation state as a large, glanceable label so
you can verify it matches what you see on screen while playing.

Shows:
  - Current state (FREE / PARTIAL / LOCKED) updated in place
  - A scrolling log of real transitions (debounced - ignores sub-50ms noise)
  - Raw byte value and how long the current state has been held

Run with sudo. Keep this terminal visible alongside the game.
"""

import os
import sys
import time

os.environ.setdefault("DISPLAY",         ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eldenring_ai.io.memory import GameMemory, _read_bytes

OFFSET       = 0x0468
POLL_WAIT    = 1.0 / 120          # 120 Hz - fast enough to catch real transitions
DEBOUNCE     = 0.05               # ignore state changes shorter than this (filters engine ticks)
LOG_LINES    = 12                 # how many past transitions to show

# ── Display ────────────────────────────────────────────────────────────────────

DISPLAY = {
    3: ("FREE   ", "Any action allowed"),
    1: ("PARTIAL", "Held action / jump arc - movement ok, attacks locked"),
    0: ("LOCKED ", "Animation commit - nothing allowed"),
}

def render(current_val, current_since, log):
    label, desc = DISPLAY.get(current_val, (f"???={current_val}", "unknown"))
    held = time.time() - current_since

    # Clear and redraw
    os.system("clear")
    print()
    print("  ptr4 + 0x0468  -  animation lock flag")
    print("  " + "─" * 54)
    print()
    print(f"  current value : {current_val}  ({bin(current_val)})")
    print(f"  state         : {label}")
    print(f"  meaning       : {desc}")
    print(f"  held for      : {held:.3f}s")
    print()
    print("  " + "─" * 54)
    print(f"  {'TIME':>7}   {'VAL':>4}   {'STATE':<10}  {'HELD':>8}   NOTE")
    print(f"  {'────':>7}   {'───':>4}   {'─────':<10}  {'────':>8}   ────")
    for entry in log[-LOG_LINES:]:
        t, val, lbl, dur, note = entry
        print(f"  {t:>6.2f}s   {val:>4}   {lbl:<10}  {dur:>7.3f}s   {note}")
    print()
    print("  Ctrl+C to stop.")


def main():
    print("  Connecting...")
    mem = GameMemory()
    if not mem.connect():
        print("  ERROR: game not found.")
        sys.exit(1)

    mem.refresh()
    if not mem.is_player_loaded():
        print("  ERROR: player not loaded.")
        sys.exit(1)

    print(f"  Connected. ptr4=0x{mem._last_ptr4:016X}")
    time.sleep(0.5)

    log          = []
    t0           = time.time()

    # Debounce state: we only commit a new state after it has held for DEBOUNCE seconds
    committed_val   = None
    committed_since = t0
    candidate_val   = None
    candidate_since = None

    try:
        while True:
            mem.refresh()
            if not mem.is_player_loaded():
                time.sleep(POLL_WAIT)
                continue

            raw = _read_bytes(mem.pid, mem._last_ptr4 + OFFSET, 1)[0]
            now = time.time()

            # ── Debounce logic ─────────────────────────────────────────────────
            if raw != committed_val:
                # A new raw value appeared
                if raw != candidate_val:
                    # It's a new candidate - start timing it
                    candidate_val   = raw
                    candidate_since = now
                elif now - candidate_since >= DEBOUNCE:
                    # Candidate held long enough - commit it
                    if committed_val is not None:
                        duration = now - committed_since
                        lbl      = DISPLAY.get(committed_val, (str(committed_val), ""))[0]
                        note     = "← noise?" if duration < DEBOUNCE * 2 else ""
                        log.append((committed_since - t0, committed_val, lbl, duration, note))

                    committed_val   = candidate_val
                    committed_since = candidate_since
                    candidate_val   = None
                    candidate_since = None
            else:
                # Raw matches committed - reset candidate
                candidate_val   = None
                candidate_since = None

            render(committed_val if committed_val is not None else raw,
                   committed_since, log)
            time.sleep(POLL_WAIT)

    except KeyboardInterrupt:
        if committed_val is not None:
            dur = time.time() - committed_since
            lbl = DISPLAY.get(committed_val, (str(committed_val), ""))[0]
            log.append((committed_since - t0, committed_val, lbl, dur, ""))
        render(committed_val, committed_since, log)
        print("  Stopped.")


if __name__ == "__main__":
    main()
