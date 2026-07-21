"""
controller_repl.py - manual virtual-gamepad REPL for testing inputs by hand.

Type a button/axis name to pulse it on the virtual controller (A, B, LT, RX+,
DU, ...), or quit/exit/q to leave. Useful for checking game bindings without
the agent. The controller is created by io/input.create_controller().

    uv run python tools/controller_repl.py
"""

import time

from evdev import ecodes as e

import _bootstrap  # noqa: F401  (sets sys.path + display env)

from eldenring_ai.io.input import create_controller

PRESS_DURATION = 0.1

BUTTON_MAP = {
    "A":     (e.EV_KEY, e.BTN_A),
    "B":     (e.EV_KEY, e.BTN_B),
    "X":     (e.EV_KEY, e.BTN_X),
    "Y":     (e.EV_KEY, e.BTN_Y),
    "LB":    (e.EV_KEY, e.BTN_TL),
    "RB":    (e.EV_KEY, e.BTN_TR),
    "RT":    (e.EV_ABS, e.ABS_RZ, 255),
    "LT":    (e.EV_ABS, e.ABS_Z,  255),
    "Back":  (e.EV_KEY, e.BTN_SELECT),
    "Start": (e.EV_KEY, e.BTN_START),
    "LP":    (e.EV_KEY, e.BTN_THUMBL),
    "RP":    (e.EV_KEY, e.BTN_THUMBR),
    "DL":    (e.EV_ABS, e.ABS_HAT0X, -1),
    "DR":    (e.EV_ABS, e.ABS_HAT0X,  1),
    "DU":    (e.EV_ABS, e.ABS_HAT0Y, -1),
    "DD":    (e.EV_ABS, e.ABS_HAT0Y,  1),
    "LX+":   (e.EV_ABS, e.ABS_X,   32767),
    "LX-":   (e.EV_ABS, e.ABS_X,  -32767),
    "LY+":   (e.EV_ABS, e.ABS_Y,   32767),
    "LY-":   (e.EV_ABS, e.ABS_Y,  -32767),
    "RX+":   (e.EV_ABS, e.ABS_RX,  32767),
    "RX-":   (e.EV_ABS, e.ABS_RX, -32767),
    "RY+":   (e.EV_ABS, e.ABS_RY,  32767),
    "RY-":   (e.EV_ABS, e.ABS_RY, -32767),
}


def press(gamepad, name):
    key = next((k for k in BUTTON_MAP if k.lower() == name.lower()), None)
    if key is None:
        print(f"Unknown input: '{name}'")
        return
    mapping = BUTTON_MAP[key]
    if mapping[0] == e.EV_KEY:
        _, code = mapping
        gamepad.write(e.EV_KEY, code, 1)
        gamepad.syn()
        time.sleep(PRESS_DURATION)
        gamepad.write(e.EV_KEY, code, 0)
        gamepad.syn()
    elif mapping[0] == e.EV_ABS:
        _, axis, value = mapping
        gamepad.write(e.EV_ABS, axis, value)
        gamepad.syn()
        time.sleep(PRESS_DURATION)
        gamepad.write(e.EV_ABS, axis, 0)
        gamepad.syn()


def main():
    gamepad = create_controller()
    print("  ".join(BUTTON_MAP.keys()))
    try:
        while True:
            cmd = input("> ").strip()
            if cmd.lower() in ("quit", "exit", "q"):
                break
            if cmd:
                press(gamepad, cmd)
    except KeyboardInterrupt:
        pass
    finally:
        gamepad.close()


if __name__ == "__main__":
    main()
