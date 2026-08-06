"""
vision.py - screen-capture, frame-stacking, and boss-HP vision parameters.
"""

FRAME_STACK = 12
FRAME_SKIP = 2

# The policy's frame, not the screen's: captured frames are downscaled to this before
# stacking. One stack spans FRAME_STACK * FRAME_SKIP * ACTION_LOCK_DURATION seconds.
OBSERVATION_SHAPE = (256, 256, FRAME_STACK)

WAYLAND_OUTPUT = "HDMI-A-1"
V4L2_DEVICE = "/dev/video0"

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

# Detecting the right-hand cap of the boss HP bar inside BOSS_HP_REGION: the cap is
# bright and almost colourless, so a column qualifies when it is above the brightness
# floor and below the saturation ceiling. Calibrated against this build's HUD.
BOSS_HP_BRIGHTNESS_MIN = 120
BOSS_HP_SATURATION_MAX = 60

# The bar reading is a median over this many consecutive frames. A single frame can be
# spoiled by an effect drawn over the HUD, and a median discards that where a mean
# would smear it into the value.
#
# 3 and not 5: the median only settles once most of the window holds the new value, so
# every extra frame is another step of delay between the sword landing and the reward
# appearing. 5 cost two steps (0.4 s) of that, and the spike rejection it bought is now
# also covered by BOSS_HIT_MIN_DELTA, which refuses to pay for a sub-hit-sized drop.
BOSS_HP_MEDIAN_WINDOW = 3
