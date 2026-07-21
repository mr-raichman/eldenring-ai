"""
vision.py - screen-capture, frame-stacking, and boss-HP vision parameters.
"""

FRAME_STACK = 12
FRAME_SKIP = 1
OBSERVATION_SHAPE = (256, 256, FRAME_STACK)  # 1280x720

WAYLAND_OUTPUT = "HDMI-A-1"
V4L2_DEVICE = "/dev/video0"

PRESS_DURATION = 0.02

BOSS_HP_REGION = {
    "x1": 466,
    "y1": 867,
    "x2": 1463,
    "y2": 871,
}
BOSS_HP_CAP_FULL = 996
