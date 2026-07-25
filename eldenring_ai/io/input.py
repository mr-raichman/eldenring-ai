"""
input.py - virtual Xbox 360 gamepad: the ACTIONS table, controller creation and
action execution, plus the scripted sequences that walk the character into the
Margit arena and use the grace-return item.
"""

import time
from collections import namedtuple

from evdev import AbsInfo, UInput
from evdev import ecodes as e

from eldenring_ai import config
from eldenring_ai.config import PRESS_DURATION

_VENDOR = 0x045E  # Microsoft
_PRODUCT = 0x028E  # Xbox 360 controller
_VERSION = 0x0114
_BUSTYPE = 0x03  # BUS_USB

STICK_MAX = 32767  # full stick deflection for movement


CONTROLLER_CAPABILITIES = {
    e.EV_KEY: [
        e.BTN_A,
        e.BTN_B,
        e.BTN_X,
        e.BTN_Y,
        e.BTN_TL,
        e.BTN_TR,
        e.BTN_TL2,
        e.BTN_TR2,
        e.BTN_SELECT,
        e.BTN_START,
        e.BTN_THUMBL,
        e.BTN_THUMBR,
    ],
    e.EV_ABS: [
        (
            e.ABS_X,
            AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0),
        ),  # left stick X
        (
            e.ABS_Y,
            AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0),
        ),  # left stick Y
        (e.ABS_Z, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),  # LT
        (
            e.ABS_RX,
            AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0),
        ),  # right stick X
        (
            e.ABS_RY,
            AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0),
        ),  # right stick Y
        (
            e.ABS_RZ,
            AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0),
        ),  # RT
        (
            e.ABS_HAT0X,
            AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0),
        ),  # d-pad X
        (
            e.ABS_HAT0Y,
            AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0),
        ),  # d-pad Y
    ],
}

Action = namedtuple(
    "Action", ["toggle", "key_type", "input_code", "value", "action_id"]
)

ACTIONS = {
    "No Action": Action(False, None, None, None, 0),
    "Move Forward": Action(True, e.EV_ABS, e.ABS_Y, -STICK_MAX, 1),
    "Move Backward": Action(True, e.EV_ABS, e.ABS_Y, STICK_MAX, 2),
    "Move Left": Action(True, e.EV_ABS, e.ABS_X, -STICK_MAX, 3),
    "Move Right": Action(True, e.EV_ABS, e.ABS_X, STICK_MAX, 4),
    "Hold Sprint": Action(True, e.EV_KEY, e.BTN_B, 1, 5),
    "Guard": Action(True, e.EV_ABS, e.ABS_Z, 255, 6),
    "Jump": Action(False, e.EV_KEY, e.BTN_A, 1, 7),
    "Dodge": Action(False, e.EV_KEY, e.BTN_B, 1, 8),
    "Heal": Action(False, e.EV_KEY, e.BTN_X, 1, 9),
    "Light Attack": Action(False, e.EV_KEY, e.BTN_TR, 1, 10),
    "Heavy Attack": Action(False, e.EV_ABS, e.ABS_RZ, 255, 11),
}

N_ACTIONS = len(ACTIONS)
N_TOGGLES = sum(1 for a in ACTIONS.values() if a.toggle)

# Reverse map for looking up an action name from the discrete action id.
ACTION_BY_ID = {a.action_id: name for name, a in ACTIONS.items()}

_TOGGLE_OPPOSITE = {
    "Move Forward": "Move Backward",
    "Move Backward": "Move Forward",
    "Move Left": "Move Right",
    "Move Right": "Move Left",
}


def create_controller():
    return UInput(
        CONTROLLER_CAPABILITIES,
        name="Xbox 360 Controller",
        vendor=_VENDOR,
        product=_PRODUCT,
        version=_VERSION,
        bustype=_BUSTYPE,
    )


def reset_all(gamepad, ACTIONS=ACTIONS):
    toggle_state = {action: 0 for action in ACTIONS if ACTIONS[action].toggle}
    for code in (
        e.ABS_X,
        e.ABS_Y,
        e.ABS_Z,
        e.ABS_RX,
        e.ABS_RY,
        e.ABS_RZ,
        e.ABS_HAT0X,
        e.ABS_HAT0Y,
    ):
        gamepad.write(e.EV_ABS, code, 0)
    for btn in (
        e.BTN_A,
        e.BTN_B,
        e.BTN_X,
        e.BTN_Y,
        e.BTN_TL,
        e.BTN_TR,
        e.BTN_TL2,
        e.BTN_TR2,
        e.BTN_SELECT,
        e.BTN_START,
        e.BTN_THUMBL,
        e.BTN_THUMBR,
    ):
        gamepad.write(e.EV_KEY, btn, 0)
    gamepad.syn()
    return toggle_state


def execute_action(gamepad, action, toggle_state):
    a = ACTIONS[action]

    if a.toggle:
        if toggle_state[action]:
            gamepad.write(a.key_type, a.input_code, 0)
            toggle_state[action] = 0
        else:
            gamepad.write(a.key_type, a.input_code, a.value)
            toggle_state[action] = 1
            opposite = _TOGGLE_OPPOSITE.get(action)
            if opposite:
                toggle_state[opposite] = 0

    elif action == "Dodge" and toggle_state.get("Hold Sprint", 0):
        gamepad.write(e.EV_KEY, e.BTN_B, 0)
        gamepad.syn()
        time.sleep(PRESS_DURATION)
        gamepad.write(e.EV_KEY, e.BTN_B, 1)
        gamepad.syn()
        time.sleep(PRESS_DURATION)
        gamepad.write(e.EV_KEY, e.BTN_B, 0)
        gamepad.syn()
        time.sleep(PRESS_DURATION)
        gamepad.write(e.EV_KEY, e.BTN_B, 1)
        gamepad.syn()
        # toggle_state["Hold Sprint"] remains 1

    elif action != "No Action":
        # Generic tap
        gamepad.write(a.key_type, a.input_code, a.value)
        gamepad.syn()
        time.sleep(PRESS_DURATION)
        gamepad.write(a.key_type, a.input_code, 0)
        gamepad.syn()

    gamepad.syn()
    return toggle_state


def _camera_right(gamepad, duration, speed):
    gamepad.write(e.EV_ABS, e.ABS_RX, speed)
    gamepad.syn()
    time.sleep(duration)
    gamepad.write(e.EV_ABS, e.ABS_RX, 0)
    gamepad.syn()


def walk_to_fog(gamepad):
    gamepad.write(e.EV_ABS, e.ABS_Y, -STICK_MAX)  # forward
    gamepad.write(e.EV_KEY, e.BTN_B, 1)  # sprint
    gamepad.syn()
    _camera_right(gamepad, config.FOG_TURN1_DURATION, config.FOG_TURN1_SPEED)
    time.sleep(config.FOG_SPRINT_LEG1_DURATION)
    _camera_right(gamepad, config.FOG_TURN2_DURATION, config.FOG_TURN2_SPEED)
    time.sleep(config.FOG_SPRINT_LEG2_DURATION)
    gamepad.write(e.EV_ABS, e.ABS_Y, 0)
    gamepad.write(e.EV_KEY, e.BTN_B, 0)
    # Enter fog gate
    gamepad.write(e.EV_KEY, e.BTN_Y, 1)
    gamepad.syn()
    time.sleep(PRESS_DURATION)
    gamepad.write(e.EV_KEY, e.BTN_Y, 0)
    gamepad.syn()
    time.sleep(config.FOG_GATE_LOAD_DELAY)
    # Lock on
    gamepad.write(e.EV_KEY, e.BTN_THUMBR, 1)
    gamepad.syn()
    time.sleep(PRESS_DURATION)
    gamepad.write(e.EV_KEY, e.BTN_THUMBR, 0)
    gamepad.syn()


def use_grace_return_item(gamepad):
    # Hold Y to open pouch
    gamepad.write(e.EV_KEY, e.BTN_Y, 1)
    gamepad.syn()
    time.sleep(config.POUCH_OPEN_DELAY)
    # D-pad up to highlight the slot
    gamepad.write(e.EV_ABS, e.ABS_HAT0Y, -1)
    gamepad.syn()
    time.sleep(PRESS_DURATION)
    gamepad.write(e.EV_ABS, e.ABS_HAT0Y, 0)
    gamepad.syn()
    time.sleep(config.POUCH_HIGHLIGHT_DELAY)
    # Release Y: activates the selected slot and opens confirmation
    gamepad.write(e.EV_KEY, e.BTN_Y, 0)
    gamepad.syn()
    time.sleep(config.POUCH_RELEASE_DELAY)
    # D-pad left to "Yes"
    gamepad.write(e.EV_ABS, e.ABS_HAT0X, -1)
    gamepad.syn()
    time.sleep(PRESS_DURATION)
    gamepad.write(e.EV_ABS, e.ABS_HAT0X, 0)
    gamepad.syn()
    time.sleep(config.POUCH_CONFIRM_DELAY)
    # A to confirm
    gamepad.write(e.EV_KEY, e.BTN_A, 1)
    gamepad.syn()
    time.sleep(PRESS_DURATION)
    gamepad.write(e.EV_KEY, e.BTN_A, 0)
    gamepad.syn()

def press_menu_confirm(gamepad):
    gamepad.write(e.EV_KEY, e.BTN_A, 1)
    gamepad.syn()
    time.sleep(PRESS_DURATION)
    gamepad.write(e.EV_KEY, e.BTN_A, 0)
    gamepad.syn()
