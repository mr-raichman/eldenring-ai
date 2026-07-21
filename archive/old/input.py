import evdev
from evdev import UInput, AbsInfo, ecodes as e
import time


CONTROLLER_CAPABILITIES = {
    e.EV_KEY: [
        e.BTN_SOUTH,
        e.BTN_EAST,
        e.BTN_NORTH,
        e.BTN_WEST,
        e.BTN_TL,
        e.BTN_TR,
        e.BTN_TL2,
        e.BTN_TR2,
        e.BTN_SELECT,
        e.BTN_START,
        e.BTN_THUMBL,
        e.BTN_THUMBR,
        e.BTN_DPAD_UP,
        e.BTN_DPAD_DOWN,
        e.BTN_DPAD_LEFT,
        e.BTN_DPAD_RIGHT,
    ],
    e.EV_ABS: [
        (e.ABS_X,  AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=128, resolution=0)),
        (e.ABS_Y,  AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=128, resolution=0)),
        (e.ABS_RX, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=128, resolution=0)),
        (e.ABS_RY, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=128, resolution=0)),
        (e.ABS_Z,  AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
        (e.ABS_RZ, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
    ],
}

STICK_B = 32767
STICK_F = -32768


def create_controller():
    gamepad = UInput(
        CONTROLLER_CAPABILITIES,
        name="Virtual Xbox 360 Controller",
        version=0x0114,
        vendor=0x045e,
        product=0x028e,
    )
    print(f"Virtual controller created: {gamepad.device.path}")
    return gamepad


def press(gamepad, button, duration=0.1):
    gamepad.write(e.EV_KEY, button, 1)
    gamepad.syn()
    time.sleep(duration)
    gamepad.write(e.EV_KEY, button, 0)
    gamepad.syn()


def move(gamepad, axis, direction, duration=0.1):
    gamepad.write(e.EV_ABS, axis, direction)
    gamepad.syn()
    time.sleep(duration)
    gamepad.write(e.EV_ABS, axis, 0)
    gamepad.syn()


def run(gamepad, axis, direction, duration=0.1):
    gamepad.write(e.EV_ABS, axis, direction)
    gamepad.write(e.EV_KEY, e.BTN_EAST, 1)
    gamepad.syn()
    time.sleep(duration)
    gamepad.write(e.EV_KEY, e.BTN_EAST, 0)
    gamepad.write(e.EV_ABS, axis, 0)
    gamepad.syn()

def dodge(gamepad, axis, direction):
    gamepad.write(e.EV_ABS, axis, direction)
    gamepad.syn()
    gamepad.write(e.EV_KEY, e.BTN_EAST, 1)
    gamepad.syn()
    time.sleep(0.05)
    gamepad.write(e.EV_KEY, e.BTN_EAST, 0)
    gamepad.syn()
    time.sleep(0.05)
    gamepad.write(e.EV_ABS, axis, 0)
    gamepad.syn()


def walk_to_fog(gamepad):
    """
    Hardcoded route from the grace site to Margit's fog gate.
    """
    time.sleep(1)
    move(gamepad, e.ABS_RX, STICK_B, 0.2)
    run(gamepad, e.ABS_Y, STICK_F, 3)
    move(gamepad, e.ABS_RX, STICK_B, 0.25)
    run(gamepad, e.ABS_Y, STICK_F, 1.75)
    press(gamepad, e.BTN_WEST)
    time.sleep(0.5)
    press(gamepad, e.BTN_WEST)
    time.sleep(2)
    press(gamepad, e.BTN_THUMBR)

def use_grace_return_item(gamepad):
    """
    Uses the top pouch slot item (hold Y + dpad up), then confirms
    the return-to-grace dialog with dpad left + A.
    Assumes a grace-return item is assigned to the top pouch slot.
    """
    gamepad.write(e.EV_KEY, e.BTN_WEST, 1)
    gamepad.syn()
    time.sleep(1)
    gamepad.write(e.EV_KEY, e.BTN_DPAD_UP, 1)
    gamepad.syn()
    time.sleep(0.2)
    gamepad.write(e.EV_KEY, e.BTN_DPAD_UP, 0)
    gamepad.syn()
    time.sleep(0.5)
    gamepad.write(e.EV_KEY, e.BTN_NORTH, 0)
    gamepad.syn()
    time.sleep(0.5)
    press(gamepad, e.BTN_DPAD_LEFT)
    time.sleep(0.3)
    press(gamepad, e.BTN_SOUTH)

def run_input_test():
    print("=== Input Test ===\n")
    gamepad = create_controller()
    time.sleep(1)

    print("Switch to Elden Ring")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("Jumping 3 times!\n")
    for i in range(3):
        print(f"  Jump {i + 1}...")
        press(gamepad, e.BTN_SOUTH)
        time.sleep(1)

    gamepad.close()
    print("=== Test complete ===")

def run_walk_to_fog():
    gamepad = create_controller()
    time.sleep(2)
    walk_to_fog(gamepad)
    time.sleep(1)
    gamepad.close()

def run_dodge():
    gamepad = create_controller()
    time.sleep(2)
    dodge(gamepad, e.ABS_X, STICK_B)
    time.sleep(1)
    gamepad.close()

if __name__ == "__main__":
    #run_dodge()
    #run_input_test()
    run_walk_to_fog()
