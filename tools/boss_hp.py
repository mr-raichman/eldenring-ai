"""
boss_hp.py - live Margit HP monitor.

Uses the exact capture and detection pipeline as the training loop:
cv2.VideoCapture reading /dev/video0 (the wf-recorder stream) and
_read_boss_hp_bar() from io/memory.py. Prints a live-updating line.

Run with the game open and wf-recorder streaming:
    uv run python tools/boss_hp.py
"""

import sys
import time

import cv2

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai.config import V4L2_DEVICE
from eldenring_ai.io.memory import _read_boss_hp_bar


def main():
    print(f"Opening capture device {V4L2_DEVICE}...")
    cap = cv2.VideoCapture(V4L2_DEVICE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Could not open {V4L2_DEVICE}. Is wf-recorder running?")
        sys.exit(1)
    print("Capture ready.")
    print("Monitoring Margit HP - press Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\r  [ERROR] Failed to read frame - has wf-recorder stopped?",
                      end="", flush=True)
                time.sleep(0.5)
                continue

            boss_hp = _read_boss_hp_bar(frame)
            print(f"\r  Margit HP: {boss_hp * 100:6.2f}%   ", end="", flush=True)
            time.sleep(0.05)   # ~20 updates per second

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
