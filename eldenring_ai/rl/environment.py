"""
environment.py - EldenRingEnv, the Gymnasium environment that orchestrates the
whole loop: screen capture, /proc memory reads, gamepad input, reward shaping,
episode reset/recovery, and the live dashboard for PPO training.
"""

import json
import os
import time
from collections import deque
import subprocess

import gymnasium
import numpy as np

from eldenring_ai import config
from eldenring_ai.config import paths
from eldenring_ai.io import input
from eldenring_ai.io.capture import ScreenCapture
from eldenring_ai.io.memory import GameMemory, get_game_state, find_pid, reset_boss_hp_smoothing
from eldenring_ai.rl.reward import compute_reward
from eldenring_ai.ui.dashboard import _print_dashboard

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

PERSIST_FILE = str(paths.SESSION_STATS)


class EldenRingEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = gymnasium.spaces.Dict(
            {
                "frame": gymnasium.spaces.Box(
                    low=0,
                    high=255,
                    shape=config.OBSERVATION_SHAPE,
                    dtype=np.uint8,
                ),
                "action_history": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(config.HISTORY_LENGTH * input.N_ACTIONS + input.N_TOGGLES,),
                    dtype=np.float32,
                ),
                "hp_history": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(config.HISTORY_LENGTH,),
                    dtype=np.float32,
                ),
                "boss_hp_history": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(config.HISTORY_LENGTH,),
                    dtype=np.float32,
                ),
                "st_history": gymnasium.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(config.HISTORY_LENGTH,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = gymnasium.spaces.Discrete(input.N_ACTIONS)

        self.gamepad = input.create_controller()
        time.sleep(2)

        self.memory = GameMemory()
        self.memory.connect()

        self.capture = ScreenCapture()

        self._initialize()
        self._load_persist()

    def _wait_for_condition(self, condition, timeout):
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.memory.refresh()
                if condition():
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def _initialize(self):
        # --- per-episode state (reset at the start of every episode) ---
        self._action_history = deque(
            [np.zeros(input.N_ACTIONS, dtype=np.float32)] * config.HISTORY_LENGTH,
            maxlen=config.HISTORY_LENGTH,
        )

        self._hp_history = deque([1.0] * config.HISTORY_LENGTH, maxlen=config.HISTORY_LENGTH)

        self._boss_hp_history = deque([0] * config.HISTORY_LENGTH, maxlen=config.HISTORY_LENGTH)

        self._st_history = deque([1.0] * config.HISTORY_LENGTH, maxlen=config.HISTORY_LENGTH)

        self.toggle_state = input.reset_all(self.gamepad)

        self.prev_player_hp = 1.0
        self.prev_boss_hp = 1.0
        self.prev_stamina = 1.0

        self.ep_steps = 0

        self._stop_requested = False

        # --- persisted training stats (restored from session_stats.json) ---
        self.episode_count = 0

        self.total_deaths = 0
        self.total_kills = 0
        self.total_truncations = 0
        self.total_grace_recoveries = 0

        self.training_actions = {action: 0 for action in input.ACTIONS}

        self.best_ep_reward = -1000.0
        self.best_boss_hp = 1.0
        self.best_ep_steps = 0

        self.reward_mean = deque(
            [0.0] * config.MEAN_STATS_WINDOW, maxlen=config.MEAN_STATS_WINDOW
        )
        self.boss_hp_mean = deque(
            [0.0] * config.MEAN_STATS_WINDOW, maxlen=config.MEAN_STATS_WINDOW
        )
        self.steps_mean = deque(
            [0.0] * config.MEAN_STATS_WINDOW, maxlen=config.MEAN_STATS_WINDOW
        )

        self.actions_mean = deque(
            [np.zeros(input.N_ACTIONS, dtype=np.float32)] * config.MEAN_STATS_WINDOW,
            maxlen=config.MEAN_STATS_WINDOW,
        )

        self._episode_reward_history = []

        # --- current-episode stats ---
        self.ep_reward = 0
        self.ep_boss_hp = 1.0
        self.ep_actions = {action: 0 for action in input.ACTIONS}

        self._last_obs = None

    def _zero_obs(self):
        return {
            "frame": np.zeros(config.OBSERVATION_SHAPE, dtype=np.uint8),
            "action_history": np.zeros(
                config.HISTORY_LENGTH * input.N_ACTIONS + input.N_TOGGLES,
                dtype=np.float32,
            ),
            "hp_history": np.ones(config.HISTORY_LENGTH, dtype=np.float32),
            "boss_hp_history": np.zeros(config.HISTORY_LENGTH, dtype=np.float32),
            "st_history": np.ones(config.HISTORY_LENGTH, dtype=np.float32),
        }

    def _build_obs(self, observation):
        """Assemble the Dict observation from the current frame stack and history
        buffers. Single source of truth so step() and reset() stay identical."""
        action_history = np.array(self._action_history, dtype=np.float32).flatten()
        toggle_obs = np.array(list(self.toggle_state.values()), dtype=np.float32)
        action_history = np.concatenate([action_history, toggle_obs])
        return {
            "frame": observation,
            "action_history": action_history,
            "hp_history": np.array(self._hp_history, dtype=np.float32),
            "boss_hp_history": np.array(self._boss_hp_history, dtype=np.float32),
            "st_history": np.array(self._st_history, dtype=np.float32),
        }

    def step(self, action):
        start_lock = time.time()

        if not self._is_game_running():
            return (
                self._last_obs if self._last_obs is not None else self._zero_obs(),
                0.0, True, False, {},
            )

        # --- execute the chosen action ---
        selected_action = input.ACTION_BY_ID[action]

        self.toggle_state = input.execute_action(
            self.gamepad, selected_action, self.toggle_state
        )
        self.ep_actions[selected_action] += 1
        self.training_actions[selected_action] += 1

        one_hot = np.zeros(input.N_ACTIONS, dtype=np.float32)
        one_hot[action] = 1.0
        self._action_history.append(one_hot)

        # --- capture new observation and game state ---
        try:
            frame, observation = self.capture.get_frame()
            self.memory.refresh()
            player_hp, boss_hp, stamina = get_game_state(frame, self.memory)
        except Exception:
            remaining = config.ACTION_LOCK_DURATION - (time.time() - start_lock)
            if remaining > 0:
                time.sleep(remaining)
            return (
                self._last_obs if self._last_obs is not None else self._zero_obs(),
                0.0,
                True,
                False,
                {},
            )

        # --- reward ---
        boss_reward, player_punish, reward, _ = compute_reward(
            player_hp, self.prev_player_hp,
            boss_hp,   self.prev_boss_hp,
            stamina,   self.prev_stamina,
            self._st_history, self._boss_hp_history,
            self._hp_history, self._action_history
        )

        # --- termination / truncation ---
        if player_hp <= 0:
            terminated = True
            self.total_deaths += 1
        else:
            terminated = False

        if boss_hp <= config.BOSS_DEFEAT_HP_THRESHOLD:
            terminated = True
            self.total_kills += 1
            reward += config.REWARD_BOSS_DEFEATED
            self._stop_requested = True

        if self.ep_steps >= config.MAX_STEPS_PER_EPISODE and not terminated:
            self.total_truncations += 1
            truncated = True
        else:
            truncated = False

        # --- update history buffers and previous-step values ---
        self._hp_history.append(player_hp)
        if min(max(0.0, self.prev_boss_hp - boss_hp), 1.0) > 0:
            self._boss_hp_history.append(1)
        else:
            self._boss_hp_history.append(0)

        self._st_history.append(stamina)

        self.prev_player_hp = player_hp
        self.prev_boss_hp = boss_hp
        self.prev_stamina = stamina

        self.ep_reward += reward
        self.ep_boss_hp = boss_hp
        self.ep_steps += 1

        self._episode_reward_history.append(self.ep_reward)

        obs_dict = self._build_obs(observation)

        remaining = config.ACTION_LOCK_DURATION - (time.time() - start_lock)
        if remaining > 0:
            time.sleep(remaining)

        self._last_obs = obs_dict
        return obs_dict, reward, terminated, truncated, {}

    def _is_game_running(self):
        pid = getattr(self.memory, "pid", None)
        if not pid:
            return False
        try:
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
            return state != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False

    def _launch_and_recover_game(self):
        subprocess.Popen(["steam", f"steam://rungameid/{config.ELDEN_RING_APP_ID}"])

        while True:
            self.memory.pid = find_pid()
            if self._is_game_running():
                break
            time.sleep(3)

        self.memory._world_chr_man_ptr = None
        while not self.memory.resolve_static_ptr():
            time.sleep(1)

        while not self.memory.world_loaded():
            input.press_menu_confirm(self.gamepad)
            time.sleep(config.MENU_CONFIRM_INTERVAL)

        while True:
            self.memory.refresh()
            if self.memory.is_player_ready():
                return
            time.sleep(0.2)

    def _ensure_game_running(self):
        if not self._is_game_running():
            self._launch_and_recover_game()
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.episode_count == 0:
            self.training_start_time = time.time()

        self.best_ep_reward = max(self.best_ep_reward, self.ep_reward)
        self.best_boss_hp = min(self.best_boss_hp, self.ep_boss_hp)
        self.best_ep_steps = max(self.best_ep_steps, self.ep_steps)

        if self.episode_count != 0:
            self.reward_mean.append(self.ep_reward)
            self.boss_hp_mean.append(self.ep_boss_hp)
            self.steps_mean.append(self.ep_steps)
            self.actions_mean.append(
                np.array(list(self.ep_actions.values()), dtype=np.float32)
            )

        if not config.DEBUG_MODE and self.episode_count != 0:
            _print_dashboard(self)

        self._save_persist()

        self._initialize()

        self._load_persist()

        while True:
            recovered = self._ensure_game_running()
            if not recovered:
                self._wait_for_condition(self.memory.loading_screen, 10)
                self._wait_for_condition(self.memory.is_player_ready, 15)

            time.sleep(1)

            if recovered:
                subprocess.run(["hyprctl", "dispatch", "movecursor", "9999", "9999"])

            input.walk_to_fog(self.gamepad)

            if not self._wait_for_condition(self.memory.is_in_boss_arena, 2):
                self.total_grace_recoveries += 1
                input.use_grace_return_item(self.gamepad)
            else:
                break

        reset_boss_hp_smoothing()

        while True:
            try:
                frame, observation = self.capture.get_frame()
                self.memory.refresh()
                player_hp, boss_hp, stamina = get_game_state(frame, self.memory)
                break
            except Exception:
                self.capture.close()
                self.capture = ScreenCapture()
                time.sleep(1)

        self.prev_player_hp = player_hp
        self.prev_boss_hp = boss_hp
        self.prev_stamina = stamina

        obs_dict = self._build_obs(observation)

        self.start_ep_time = time.time()
        self.episode_count += 1

        self._last_obs = obs_dict
        return obs_dict, {}

    def _save_persist(self):
        with open(PERSIST_FILE, "w") as f:
            json.dump(
                {
                    "episode_count": self.episode_count,
                    "total_deaths": self.total_deaths,
                    "total_kills": self.total_kills,
                    "total_truncations": self.total_truncations,
                    "total_grace_recoveries": self.total_grace_recoveries,
                    "training_actions": self.training_actions,
                    "best_ep_reward": self.best_ep_reward,
                    "best_boss_hp": self.best_boss_hp,
                    "best_ep_steps": self.best_ep_steps,
                    "reward_mean": list(self.reward_mean),
                    "boss_hp_mean": list(self.boss_hp_mean),
                    "steps_mean": list(self.steps_mean),
                    "actions_mean": [arr.tolist() for arr in self.actions_mean],
                    "training_start_time": self.training_start_time,
                },
                f,
            )

    def _load_persist(self):
        if os.path.exists(PERSIST_FILE):
            with open(PERSIST_FILE) as f:
                data = json.load(f)
            self.episode_count = data.get("episode_count", 0)
            self.total_deaths = data.get("total_deaths", 0)
            self.total_kills = data.get("total_kills", 0)
            self.total_truncations = data.get("total_truncations", 0)
            self.total_grace_recoveries = data.get("total_grace_recoveries", 0)
            self.training_actions = data.get(
                "training_actions", {action: 0 for action in input.ACTIONS}
            )
            self.best_ep_reward = data.get("best_ep_reward", 0.0)
            self.best_boss_hp = data.get("best_boss_hp", 1.0)
            self.best_ep_steps = data.get("best_ep_steps", 0)

            self.reward_mean = deque(
                data.get("reward_mean", [0.0] * config.MEAN_STATS_WINDOW),
                maxlen=config.MEAN_STATS_WINDOW,
            )
            self.boss_hp_mean = deque(
                data.get("boss_hp_mean", [0.0] * config.MEAN_STATS_WINDOW),
                maxlen=config.MEAN_STATS_WINDOW,
            )
            self.steps_mean = deque(
                data.get("steps_mean", [0.0] * config.MEAN_STATS_WINDOW),
                maxlen=config.MEAN_STATS_WINDOW,
            )

            self.actions_mean = deque(
                [
                    np.array(arr, dtype=np.float32)
                    for arr in data.get("actions_mean", [])
                ],
                maxlen=config.MEAN_STATS_WINDOW,
            )

            self.training_start_time = data.get("training_start_time", 0)
