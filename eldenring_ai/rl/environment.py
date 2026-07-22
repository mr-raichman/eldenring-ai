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
from eldenring_ai.ui import shared_stats
from eldenring_ai.ui.dashboard import _print_dashboard
from eldenring_ai.ui.episode_log import EpisodeRecorder
from eldenring_ai.ui.metrics import (
    COUNTERS,
    DERIVED_MEASURES,
    EPISODE_METRICS,
    EVENT_CATEGORIES,
    REWARD_COMPONENTS,
)

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")

PERSIST_FILE = str(paths.SESSION_STATS)

# Short-name -> SB3 logger tag for the PPO snapshot saved into each episode record.
_PPO_SNAPSHOT_KEYS = [
    ("explained_variance",   "train/explained_variance"),
    ("entropy_loss",         "train/entropy_loss"),
    ("value_loss",           "train/value_loss"),
    ("policy_gradient_loss", "train/policy_gradient_loss"),
    ("clip_fraction",        "train/clip_fraction"),
    ("approx_kl",            "train/approx_kl"),
    ("learning_rate",        "train/learning_rate"),
    ("ep_rew_mean",          "rollout/ep_rew_mean"),
    ("ep_len_mean",          "rollout/ep_len_mean"),
]


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

        self._recorder = EpisodeRecorder()

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
        # per-episode state (reset at the start of every episode)
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
        self._episode_outcome = None

        self._stop_requested = False

        self._recorder.reset()

        # persisted training stats (restored from session_stats.json)
        self.episode_count = 0

        self.total_deaths = 0
        self.total_fall_deaths = 0
        self.total_kills = 0
        self.total_truncations = 0
        self.total_grace_recoveries = 0

        self.training_actions = {action: 0 for action in input.ACTIONS}

        # Registry-driven episode metrics: running best and rolling-mean window per key.
        self._metric_best = {m.key: None for m in EPISODE_METRICS}
        self._metric_window = {
            m.key: deque([0.0] * config.MEAN_STATS_WINDOW, maxlen=config.MEAN_STATS_WINDOW)
            for m in EPISODE_METRICS
        }

        # current-episode stats
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

    def _episode_metric_values(self):
        """Current values for the registry metrics (the just-finished episode).
        To add a metric: add it to EPISODE_METRICS and return its value here."""
        return {
            "reward": self.ep_reward,
            "boss_hp": self.ep_boss_hp,
            "steps": self.ep_steps,
        }

    def _publish_episode_metrics(self, values):
        """Expose this episode's metrics + counters + reward records for TensorBoard,
        which StatsLoggerCallback drains into the SB3 logger."""
        payload = {m.tb_tag: values[m.key] for m in EPISODE_METRICS}
        payload.update({c.tb_tag: getattr(self, c.key) for c in COUNTERS})
        payload.update({c.tb_tag: self._recorder.composition[c.key] for c in REWARD_COMPONENTS})
        payload.update({e.tb_tag: self._recorder.event_counts[e.key] for e in EVENT_CATEGORIES})
        derived = self._recorder.derived_values()
        payload.update({d.tb_tag: derived[d.key] for d in DERIVED_MEASURES if derived[d.key] is not None})
        shared_stats.episode_metrics = payload

    def _build_episode_context(self):
        """Assemble the per-episode record context from env state and PPO stats."""
        now = time.time()
        duration = now - getattr(self, "start_ep_time", now)
        ppo = shared_stats.ppo_stats
        return {
            "episode": self.episode_count,
            "training_time_s": now - self.training_start_time,
            "total_timesteps": ppo.get("total_timesteps", 0),
            "outcome": self._episode_outcome or "timeout",
            "duration_s": duration,
            "steps_per_s": self.ep_steps / max(duration, 0.001),
            "final_boss_hp": self.ep_boss_hp,
            "actions": dict(self.ep_actions),
            "ppo": {name: ppo.get(tag) for name, tag in _PPO_SNAPSHOT_KEYS},
        }

    def step(self, action):
        start_lock = time.time()

        if not self._is_game_running():
            return (
                self._last_obs if self._last_obs is not None else self._zero_obs(),
                0.0, True, False, {},
            )

        # execute the chosen action
        selected_action = input.ACTION_BY_ID[action]

        self.toggle_state = input.execute_action(
            self.gamepad, selected_action, self.toggle_state
        )
        self.ep_actions[selected_action] += 1
        self.training_actions[selected_action] += 1

        one_hot = np.zeros(input.N_ACTIONS, dtype=np.float32)
        one_hot[action] = 1.0
        self._action_history.append(one_hot)

        # capture new observation and game state
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

        # reward (prev values captured before they are advanced below, for records)
        prev_hp_at_step = self.prev_player_hp
        prev_stamina_at_step = self.prev_stamina
        boss_reward, player_punish, reward, events = compute_reward(
            player_hp, self.prev_player_hp,
            boss_hp,   self.prev_boss_hp,
            stamina,   self.prev_stamina,
            self._st_history, self._boss_hp_history,
            self._hp_history, self._action_history
        )

        # termination / truncation
        defeat_bonus = 0.0
        if player_hp <= 0:
            terminated = True
            self.total_deaths += 1
            if any(e.startswith("FALL_DEATH") for e in events):
                self.total_fall_deaths += 1
                self._episode_outcome = "fall_death"
            else:
                self._episode_outcome = "combat_death"
        else:
            terminated = False

        if boss_hp <= config.BOSS_DEFEAT_HP_THRESHOLD:
            terminated = True
            self.total_kills += 1
            defeat_bonus = config.REWARD_BOSS_DEFEATED
            reward += defeat_bonus
            self._episode_outcome = "kill"
            self._stop_requested = True

        if self.ep_steps >= config.MAX_STEPS_PER_EPISODE and not terminated:
            self.total_truncations += 1
            truncated = True
            self._episode_outcome = "timeout"
        else:
            truncated = False

        # update history buffers and previous-step values
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

        self._recorder.record_step(
            self.ep_steps, selected_action, player_hp, boss_hp, stamina,
            prev_hp_at_step, prev_stamina_at_step,
            boss_reward, player_punish, reward, events, defeat_bonus,
        )

        self.ep_steps += 1

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

        # Process the just-finished episode only if one actually ran. ep_steps is 0 on
        # the first reset of any run - fresh OR resumed from a checkpoint (where
        # episode_count is already restored) - so this avoids corrupting best/mean
        # stats and writing a bogus zero-step record on startup.
        if self.ep_steps > 0:
            values = self._episode_metric_values()
            for m in EPISODE_METRICS:
                v = values[m.key]
                best = self._metric_best[m.key]
                if best is None:
                    self._metric_best[m.key] = v
                elif m.better == "max":
                    self._metric_best[m.key] = max(best, v)
                elif m.better == "min":
                    self._metric_best[m.key] = min(best, v)
                self._metric_window[m.key].append(v)
            self._publish_episode_metrics(values)
            self._recorder.write_episode(self._build_episode_context())
            if not config.DEBUG_MODE:
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
        data = {
            "episode_count": self.episode_count,
            "training_actions": self.training_actions,
            "training_start_time": self.training_start_time,
        }
        for c in COUNTERS:
            data[c.key] = getattr(self, c.key)
        for m in EPISODE_METRICS:
            data[m.persist_best] = self._metric_best[m.key]
            data[m.persist_mean] = list(self._metric_window[m.key])
        with open(PERSIST_FILE, "w") as f:
            json.dump(data, f)

    def _load_persist(self):
        if not os.path.exists(PERSIST_FILE):
            return
        with open(PERSIST_FILE) as f:
            data = json.load(f)

        self.episode_count = data.get("episode_count", 0)
        self.training_actions = data.get(
            "training_actions", {action: 0 for action in input.ACTIONS}
        )
        self.training_start_time = data.get("training_start_time", 0)

        for c in COUNTERS:
            setattr(self, c.key, data.get(c.key, 0))

        for m in EPISODE_METRICS:
            self._metric_best[m.key] = data.get(m.persist_best)
            self._metric_window[m.key] = deque(
                data.get(m.persist_mean, [0.0] * config.MEAN_STATS_WINDOW),
                maxlen=config.MEAN_STATS_WINDOW,
            )
