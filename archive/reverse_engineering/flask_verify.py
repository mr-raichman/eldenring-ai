"""
verify_flask_candidate.py - Live monitor for the promising flask pointer candidate.

Tests whether 0x6FFFFA15B880 is a stable static pointer to the flask object
by reading the flask count from it in real time and comparing against what
you see on screen.

If the count matches your in-game flask count and updates when you
drink/refill, this is the correct static pointer.

Run in a separate terminal while the game is open:
    sudo python tests/verify_flask_candidate.py
"""

import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eldenring_ai.io.memory import find_pid, _read_bytes, _read_int64

# ── Candidate to verify ────────────────────────────────────────────────────────

FLASK_STATIC_PTR  = 0x6FFFFA15B880   # the static address found at level 1
OFFSET_COUNT      = 0x388            # byte offset to flask count inside flask object
OFFSET_EMPTY_FLAG = 0x3B0            # byte offset to binary empty flag


def read_byte(pid, address):
    try:
        return _read_bytes(pid, address, 1)[0]
    except OSError:
        return None


def main():
    pid = find_pid()
    if pid is None:
        print("Elden Ring not found.")
        sys.exit(1)
    print(f"Found Elden Ring: PID {pid}")
    print(f"Monitoring flask pointer at 0x{FLASK_STATIC_PTR:X}")
    print("Drink a flask or rest at grace - values should update instantly.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Step 1: read the flask object pointer from the static address
            flask_obj = _read_int64(pid, FLASK_STATIC_PTR)

            if not flask_obj:
                print("Static pointer is null - is the game in a loading screen?")
                time.sleep(1)
                continue

            # Step 2: read flask count and empty flag from the flask object
            count      = read_byte(pid, flask_obj + OFFSET_COUNT)
            empty_flag = read_byte(pid, flask_obj + OFFSET_EMPTY_FLAG)

            count_str      = str(count)      if count      is not None else "READ ERROR"
            empty_flag_str = str(empty_flag) if empty_flag is not None else "READ ERROR"
            has_flasks     = "YES" if empty_flag == 1 else "NO"

            print(
                f"\r  flask_obj=0x{flask_obj:X}  "
                f"count={count_str}  "
                f"empty_flag={empty_flag_str}  "
                f"has_flasks={has_flasks}   ",
                end="",
                flush=True,
            )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
