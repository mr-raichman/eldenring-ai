"""
vision.py - screen-capture, frame-stacking, and boss-HP vision parameters.
"""

FRAME_STACK = 12
FRAME_SKIP = 2
OBSERVATION_SHAPE = (256, 256, FRAME_STACK)  # 1280x720

WAYLAND_OUTPUT = "HDMI-A-1"
V4L2_DEVICE = "/dev/video0"

PRESS_DURATION = 0.02

# Capture-pipeline settle delays, in seconds. Each one waits on an external process
# or kernel module that gives no readiness signal, so the pause is the only handshake.
DEVICE_SETTLE_DELAY  = 1.0   # after modprobe v4l2loopback, before the node is checked
RECORDER_KILL_DELAY  = 0.5   # after pkill wf-recorder, before relaunching
RECORDER_START_DELAY = 2.0   # after launching wf-recorder, before reading frames

BOSS_HP_REGION = {
    "x1": 466,
    "y1": 867,
    "x2": 1463,
    "y2": 871,
}
BOSS_HP_CAP_FULL = 996
