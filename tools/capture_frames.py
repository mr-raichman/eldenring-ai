"""
capture_frames.py - dump the AI's frame stack to PNGs for inspection.

Press Enter to grab FRAME_STACK frames (FRAME_SKIP steps apart) and save them
under tests/captures/capture_NNN/: the greyscale AI views, amplified frame
diffs, the colour frame, and the boss HP-bar crop. Needs the game open and
wf-recorder streaming.

    uv run python tools/capture_frames.py
"""

import os
import time

import cv2
import numpy as np

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai.config import BOSS_HP_REGION, FRAME_SKIP, FRAME_STACK, OBSERVATION_SHAPE, V4L2_DEVICE
from eldenring_ai.config import paths

BASE_DIR      = str(paths.PROJECT_ROOT / "tests" / "captures")
STEP_DURATION = 1 / 50.0


def _capture_once(cap, capture_count, sample_interval):
    height, width = OBSERVATION_SHAPE[0], OBSERVATION_SHAPE[1]
    out_dir = os.path.join(BASE_DIR, f"capture_{capture_count:03d}")
    os.makedirs(out_dir, exist_ok=True)

    frames_grey  = []
    colour_frame = None
    print(f"Capturing {FRAME_STACK} frames, {FRAME_SKIP} steps apart ({sample_interval*1000:.0f}ms)...")

    for i in range(FRAME_STACK):
        if i > 0:
            time.sleep(sample_interval)
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR reading frame")
            break
        if i == FRAME_STACK - 1:
            colour_frame = frame
        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grey, (width, height))
        frames_grey.append(resized)
        print(f"  Frame {i}: mean={resized.mean():.2f}  std={resized.std():.2f}")

    if colour_frame is None:
        return

    for i, fg in enumerate(frames_grey):
        label = "oldest" if i == 0 else ("newest" if i == FRAME_STACK - 1 else str(i))
        cv2.imwrite(os.path.join(out_dir, f"ai_view_{i}_{label}.png"), fg)

    for i in range(1, len(frames_grey)):
        diff      = cv2.absdiff(frames_grey[i - 1], frames_grey[i])
        amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"diff_{i-1}_to_{i}.png"), amplified)
        print(f"  Diff {i-1} to {i}: max={diff.max()}  mean={diff.mean():.3f}")

    r = BOSS_HP_REGION
    cv2.imwrite(os.path.join(out_dir, "colour.png"), colour_frame)
    cv2.imwrite(os.path.join(out_dir, "boss_hp_bar.png"),
                colour_frame[r["y1"]:r["y2"], r["x1"]:r["x2"]])
    print(f"Saved to {out_dir}")


def main():
    sample_interval = STEP_DURATION * FRAME_SKIP
    cap = cv2.VideoCapture(V4L2_DEVICE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Press Enter to capture (Ctrl+C to quit).")
    capture_count = 0
    try:
        while True:
            input("> ")
            capture_count += 1
            _capture_once(cap, capture_count, sample_interval)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
