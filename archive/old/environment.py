import os
import subprocess
import threading
import time

import cv2
import gymnasium
import numpy as np
from evdev import ecodes as e
from gymnasium import spaces

from memory import get_game_state, GameMemory
from input import STICK_B, STICK_F, create_controller, dodge, move, press, walk_to_fog, use_grace_return_item

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

# ── Action space ──────────────────────────────────────────────────────────────

ACTIONS = {
    0:  "nothing",
    1:  "move_forward",
    2:  "move_back",
    3:  "move_left",
    4:  "move_right",
    5:  "jump",
    6:  "dodge",
    7:  "heal",
    8:  "light_attack",
    9:  "heavy_attack",
    10: "shield_parry",
    11: "ash_of_war",
    12: "dodge_forward",
    13: "dodge_right",
    14: "dodge_backward",
    15: "dodge_left",
}

# Per-action penalties applied every time that action is chosen,
# regardless of its outcome. High-cost actions (dodges, jumps) are
# penalised so the agent must find them genuinely useful to keep using them.
ACTION_PENALTIES = {
    "nothing":       0.000,
    "move_forward":  0.000,
    "move_back":     0.00,
    "move_left":     0.00,
    "move_right":    0.00,
    "jump":          0.0,
    "dodge":         0.0,
    "heal":          0.0005,
    "light_attack":  -0.002,
    "heavy_attack":  -0.002,
    "shield_parry":  0.00,
    "ash_of_war":    0.0,
    "dodge_forward": 0.0,
    "dodge_right":   0.0,
    "dodge_backward":0.0,
    "dodge_left":    0.0,
}

# Margit's arena area ID (ptr4 + 0x19A), confirmed on v1.16.1
MARGIT_AREA_ID = 4


class EldenRingEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        # Observation: 84x84 greyscale frame (same format used by Atari benchmarks)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        # 16 discrete actions
        self.action_space = spaces.Discrete(16)

        # Virtual Xbox controller for sending inputs to the game
        self.gamepad = create_controller()
        time.sleep(1)   # wait for the controller to register with the OS

        # Memory reader - connects to the running Elden Ring process
        self.memory = GameMemory()
        self.memory.connect()

        # Background thread captures frames from the screen continuously.
        # Training reads the latest available frame each step.
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()
        print("Waiting for first frame...")
        time.sleep(2)   # give the capture thread time to get the first frame
        print("Ready.")

        # HP tracking - we compute reward deltas from previous values
        self.prev_player_hp = 1.0
        self.prev_boss_hp   = 1.0

        self.step_count    = 0
        self.max_steps     = 3000
        self.episode_count = 0

        # Reward breakdown tracking - lets us see which signals dominate
        reward_components = [
            "boss_damaged",
            "player_damaged",
            "player_cured",
            "heal_good",
            "heal_bad",
            "heal_no_flasks",
            "survival",
            "move_forward_bonus",
            "nothing_bonus",
            "action_penalty",
        ]
        self.episode_rewards = {
            name: {"count": 0, "total": 0.0} for name in reward_components
        }
        self.total_rewards = {
            name: {"count": 0, "total": 0.0} for name in reward_components
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _add_reward(self, name, amount):
        """
        Record a named reward contribution to both episode and session totals,
        then return the amount so it can be summed inline.
        """
        self.episode_rewards[name]["count"] += 1
        self.episode_rewards[name]["total"] += amount
        self.total_rewards[name]["count"]   += 1
        self.total_rewards[name]["total"]   += amount
        return amount

    def _capture_loop(self):
        """
        Background thread: continuously capture the screen with wayshot and
        store the latest decoded frame. Runs as a daemon so it stops when
        the main process exits.
        """
        while True:
            result = subprocess.run(["wayshot", "--stdout"], capture_output=True)
            buffer = np.frombuffer(result.stdout, dtype=np.uint8)
            frame  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame

    def _capture_frame(self):
        """
        Get the latest captured frame.
        Returns (full_colour_frame, 84x84_greyscale_observation).
        The colour frame is used by OpenCV for boss HP reading.
        The observation is what the neural network receives.
        """
        with self._frame_lock:
            frame = self._latest_frame.copy()
        grey        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized     = cv2.resize(grey, (84, 84))
        observation = resized[:, :, np.newaxis]  # add channel dimension
        return frame, observation

    def _print_reward_log(self):
        """Print a breakdown of reward contributions for the finished episode."""
        ep_total  = sum(v["total"] for v in self.episode_rewards.values())
        all_total = sum(v["total"] for v in self.total_rewards.values())

        print(f"\n--- Episode {self.episode_count} | steps={self.step_count} ---")
        print(f"{'Component':<20} {'Ep count':>9} {'Ep pts':>9}  "
              f"{'All count':>9} {'All pts':>9}")
        print("-" * 62)
        for name, ep in self.episode_rewards.items():
            al = self.total_rewards[name]
            print(
                f"{name:<20} {ep['count']:>9} {ep['total']:>+9.2f}  "
                f"{al['count']:>9} {al['total']:>+9.2f}"
            )
        print("-" * 62)
        print(f"{'TOTAL':<20} {'':>9} {ep_total:>+9.2f}  {'':>9} {all_total:>+9.2f}")

    def _execute_action(self, action_name):
        """Send the controller input corresponding to the chosen action."""
        if action_name == "nothing":
            time.sleep(0.1)
        elif action_name == "move_forward":
            move(self.gamepad, e.ABS_Y, STICK_F)
        elif action_name == "move_back":
            move(self.gamepad, e.ABS_Y, STICK_B)
        elif action_name == "move_left":
            move(self.gamepad, e.ABS_X, STICK_F)
        elif action_name == "move_right":
            move(self.gamepad, e.ABS_X, STICK_B)
        elif action_name == "jump":
            press(self.gamepad, e.BTN_SOUTH)
        elif action_name == "dodge":
            press(self.gamepad, e.BTN_EAST)
        elif action_name == "heal":
            press(self.gamepad, e.BTN_NORTH)
        elif action_name == "light_attack":
            press(self.gamepad, e.BTN_TR)
        elif action_name == "heavy_attack":
            press(self.gamepad, e.BTN_TR2)
        elif action_name == "shield_parry":
            press(self.gamepad, e.BTN_TL)
        elif action_name == "ash_of_war":
            press(self.gamepad, e.BTN_TL2)
        elif action_name == "dodge_forward":
            dodge(self.gamepad, e.ABS_Y, STICK_F)
        elif action_name == "dodge_right":
            dodge(self.gamepad, e.ABS_X, STICK_B)
        elif action_name == "dodge_backward":
            dodge(self.gamepad, e.ABS_Y, STICK_B)
        elif action_name == "dodge_left":
            dodge(self.gamepad, e.ABS_X, STICK_F)

    def _calculate_reward(self, player_hp, boss_hp, action_name):
        """
        Compute the step reward from named components.
        Every contribution is recorded via _add_reward() for the episode log.
        """
        reward = 0.0

        # Boss took damage this step - positive signal for aggression
        boss_hp_delta = self.prev_boss_hp - boss_hp
        if boss_hp_delta > 0:
            reward += self._add_reward("boss_damaged", boss_hp_delta * 0.1)

        # Player took damage this step - negative signal for getting hit
        player_hp_delta = self.prev_player_hp - player_hp
        if player_hp_delta > 0:
            reward += self._add_reward("player_damaged", -0.0)
        elif player_hp_delta < 0:
            reward += self._add_reward("player_cured", 0.5)

        # Small bonus for moving toward Margit - encourages engagement
        if action_name == "move_forward":
            reward += self._add_reward("move_forward_bonus", 0.005)

        # Small bonus for doing nothing - prevents the agent being penalised
        # into random spam just to avoid the zero baseline
        if action_name == "nothing":
            reward += self._add_reward("nothing_bonus", 0.0)

        # Per-action penalty - forces actions to earn their cost
        penalty = ACTION_PENALTIES.get(action_name, 0.0)
        if penalty > 0:
            reward += self._add_reward("action_penalty", -penalty)

        # Survival bonus - small reward every step for staying alive,
        # creating a baseline incentive to keep the episode going
        reward += self._add_reward("survival", 0.001)

        return reward

    # ── Episode lifecycle ──────────────────────────────────────────────────────

    def _wait_for_respawn(self, timeout=90):
        """
        Wait until the player has fully respawned and is controllable.

        Handles three starting states:
          A) Player alive (truncated episode) - return immediately, no wait needed
          B) Already loading - skip Phase 1, go straight to Phase 2
          C) Player dead, world still loaded - wait for loading screen (Phase 1),
             then wait for respawn (Phase 2)

        Phase 2 also handles the Statue of Marika dialog by periodically
        pressing D-pad right + A to select the Site of Grace option if
        is_player_ready() hasn't fired after 20 seconds.
        """
        print("Waiting for respawn...")
        start = time.time()

        self.memory.refresh()

        # Case A: alive - nothing to wait for
        if self.memory.is_player_alive():
            print("  Player still alive (truncated episode) - skipping.")
            return

        # Case B: already loading
        if not self.memory.is_player_loaded():
            print("  Already in loading screen - skipping to Phase 2.")
        else:
            # Case C: dead but world still loaded - wait for load to begin
            print("  Phase 1: waiting for loading screen...")
            while time.time() - start < 15:
                self.memory.refresh()
                if not self.memory.is_player_loaded():
                    print("  Loading screen detected.")
                    break
                time.sleep(0.1)

        # Phase 2: wait for player to be fully ready at the grace site
        print("  Phase 2: waiting for player to be ready...")
        consecutive        = 0
        required           = 5     # stable reads required before proceeding
        last_button_press  = time.time()

        while time.time() - start < timeout:
            self.memory.refresh()
            if self.memory.is_player_ready():
                consecutive += 1
                if consecutive >= required:
                    print("  Player ready - proceeding.")
                    return
            else:
                consecutive = 0
                # Statue of Marika dialog: if stuck for 20s without becoming
                # ready, try pressing D-pad right + A to select Site of Grace
                if time.time() - last_button_press > 20:
                    print("  Pressing D-pad right + confirm (Statue of Marika dialog)...")
                    press(self.gamepad, e.BTN_DPAD_RIGHT)
                    time.sleep(0.3)
                    press(self.gamepad, e.BTN_SOUTH)
                    last_button_press = time.time()
            time.sleep(0.1)

        print("  Respawn timeout - continuing anyway.")

    def _verify_in_arena(self, timeout=5):
        """
        After walk_to_fog, confirm the player is inside Margit's arena
        using the memory area ID (ptr4 + 0x19A == 4).

        Requires 3 consecutive positive reads to filter out brief
        transition flickers. Returns True on success, False on timeout.

        5 seconds is enough because the fog gate triggers the area change
        instantly - if it hasn't happened in 5s, walk_to_fog failed.
        """
        print("  Verifying arena entry...")
        start       = time.time()
        consecutive = 0
        required    = 3

        while time.time() - start < timeout:
            self.memory.refresh()
            if self.memory.is_in_boss_arena():
                consecutive += 1
                if consecutive >= required:
                    print("  Memory confirms arena entry - starting episode.")
                    return True
            else:
                consecutive = 0
            time.sleep(0.1)

        print(f"  Arena not detected after {timeout}s - walk_to_fog likely failed.")
        return False

    def _recover_to_grace(self):
        """
        Recovery path when walk_to_fog fails and the player is outside
        the arena. Uses the pouch item to return to the last grace site,
        waits for the resulting loading screen to complete, then retries
        walk_to_fog.
        """
        print("  Recovery: using grace-return item...")
        use_grace_return_item(self.gamepad)

        # The item use triggers a loading screen after a short delay.
        # We must wait for it to start before calling _wait_for_respawn,
        # otherwise _wait_for_respawn sees "player alive" and returns
        # immediately without waiting for anything.
        print("  Recovery: waiting for loading screen to begin...")
        start = time.time()
        while time.time() - start < 15:
            self.memory.refresh()
            if not self.memory.is_player_loaded():
                print("  Recovery: loading screen detected.")
                break
            time.sleep(0.1)
        else:
            print("  Recovery: no loading screen detected - proceeding anyway.")

        # Loading screen is now active; _wait_for_respawn will work correctly
        self._wait_for_respawn()
        print("  Recovery: retrying walk_to_fog...")
        walk_to_fog(self.gamepad)

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset for a new episode:
          1. Print reward log for the episode that just ended
          2. Wait for the player to respawn at grace
          3. Walk to Margit's fog gate
          4. Verify we entered the arena (memory-based)
          5. If not, recover and try once more
          6. Read initial state for the new episode
        """
        super().reset(seed=seed)

        # Print and reset per-episode reward tracking
        if self.episode_count > 0:
            self._print_reward_log()
        self.episode_count += 1
        self.episode_rewards = {
            name: {"count": 0, "total": 0.0}
            for name in self.total_rewards
        }

        # Reset HP baselines - will be overwritten with real values below
        self.prev_player_hp = 1.0
        self.prev_boss_hp   = 1.0
        self.step_count     = 0

        # Navigate to the arena
        self._wait_for_respawn()
        walk_to_fog(self.gamepad)

        if not self._verify_in_arena(timeout=5):
            # First attempt failed - use recovery item and try once more
            self._recover_to_grace()
            if not self._verify_in_arena(timeout=5):
                # Two failures in a row - log and continue rather than loop
                print("  WARNING: Arena verification failed twice. Continuing.")

        # Read the actual starting state so reward deltas are correct
        # from the very first step. refresh() must be called before
        # get_game_state() because get_game_state() does not call it internally.
        frame, observation = self._capture_frame()
        self.memory.refresh()
        state = get_game_state(frame, self.memory)
        self.prev_player_hp = state["player_hp"]
        self.prev_boss_hp   = state["boss_hp"]

        return observation, {}

    def step(self, action):
        """
        Execute one action and return the resulting (observation, reward,
        terminated, truncated, info) tuple.

        Flow:
          1. Execute the action (sends controller input and waits)
          2. Capture the current frame and refresh memory
          3. Read game state (player HP, boss HP, arena status)
          4. Calculate reward from named components
          5. Check termination conditions
          6. Update HP baselines for the next step's delta calculation
        """
        self.step_count += 1
        action_name = ACTIONS[action]

        # Send the controller input for this action
        self._execute_action(action_name)

        # Capture frame and refresh memory together so both are current
        # for the same game moment
        frame, observation = self._capture_frame()
        self.memory.refresh()   # must be called before get_game_state()
        state = get_game_state(frame, self.memory)

        player_hp = state["player_hp"]
        boss_hp   = state["boss_hp"]

        reward = self._calculate_reward(player_hp, boss_hp, action_name)

        terminated = False

        # Player died - strong negative signal, end episode
        if player_hp <= 0:
            terminated = True
            reward -= 10.0

        # Boss defeated - strong positive signal, end episode.
        # boss_bar_visible is memory-based (area ID == 4), so this only
        # fires when we are genuinely inside the arena, preventing the
        # false positive that previously occurred when the player was
        # outside and the HP bar region appeared all-black.
        if state["boss_bar_visible"] and boss_hp <= 0.05:
            terminated = True
            reward += 100.0

        # Truncate if the episode has run too long
        truncated = self.step_count >= self.max_steps

        # Update baselines for the next step's delta calculations
        self.prev_player_hp = player_hp
        self.prev_boss_hp   = boss_hp

        info = {
            "player_hp": player_hp,
            "boss_hp":   boss_hp,
            "step":      self.step_count,
            "action":    action_name,
            "reward":    reward,
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up the virtual controller when training ends."""
        self.gamepad.close()
