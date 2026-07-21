#!/usr/bin/env python3
"""
test_dashboard.py - Preview the dashboard without running the game.

Populates the environment with fake data and calls _print_dashboard()
directly. Edit the fake data below to test different scenarios.

Usage:
    python3 test_dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Minimal stubs so environment.py imports without the game running ──────────

import unittest.mock as mock
import types

sys.modules["evdev"]                     = mock.MagicMock()
sys.modules["evdev.ecodes"]              = mock.MagicMock()
sys.modules["eldenring_ai.io.capture"]      = mock.MagicMock()
sys.modules["eldenring_ai.io.input"]        = mock.MagicMock()
sys.modules["eldenring_ai.io.memory"]       = mock.MagicMock()
sys.modules["eldenring_ai.ui.shared_stats"] = types.SimpleNamespace(ppo_stats={
    "train/entropy_loss":         -1.8821,
    "train/explained_variance":    0.7234,
    "train/value_loss":            0.0881,
    "train/policy_gradient_loss": -0.0113,
})

from eldenring_ai import config
from eldenring_ai.rl import environment

# ── Build a fake env instance without __init__ ────────────────────────────────

env = object.__new__(environment.EldenRingEnv)

# ── Session / episode counters ────────────────────────────────────────────────

import time
import random

random.seed(42)

env.session_start_time     = time.time() - 7234
env.episode_count          = 87
env.step_count             = 1130
env.total_steps            = 74_382
env.total_deaths           = 84
env.total_kills            = 0
env.total_truncations      = 3
env.total_grace_recoveries = 2
env.best_ep_reward         = 14.89

# ── Rolling history (100-episode window) ──────────────────────────────────────
# Simulate a realistic training curve: early episodes do poorly,
# later episodes land more hits as the policy improves slightly.

env._mean_window = 100
N = 50  # number of past episodes in the window

# Boss HP: starts near 1.0, slowly improves over episodes
ep_boss_hps  = [min(1.0, 0.95 - i * 0.004 + random.uniform(-0.05, 0.05)) for i in range(N)]
ep_boss_hps  = [max(0.4, v) for v in ep_boss_hps]

# Rewards: loosely correlated with boss HP reached (lower HP = more reward)
ep_rewards   = [round((1.0 - hp) * random.uniform(8, 14), 3) for hp in ep_boss_hps]

# Steps: most episodes end in death, lengths vary 800-2000
ep_lengths   = [random.randint(800, 2000) for _ in range(N)]

env._recent_ep_rewards  = ep_rewards
env._recent_ep_lengths  = ep_lengths
env._recent_ep_boss_hp  = ep_boss_hps

# ── Current episode cumulative reward history ──────────────────────────────────
# Simulate a mid-fight episode: quiet early, reward spikes when hits land.

env._episode_reward_cumulative = 0.0
env._episode_reward_history    = []

boss_hp   = 1.0
total_dam = 0.0
for step in range(1130):
    # Occasional hit on Margit every ~80 steps
    hit_boss   = (step % 80 == 0 and step > 0)
    hit_player = (step % 120 == 0 and step > 0)

    if hit_boss:
        total_dam = min(1.0, total_dam + 0.012)
        boss_hp   = max(0.0, 1.0 - total_dam)

    good = (1 + total_dam) * config.BOSS_PARAMETER if hit_boss else 0.0
    bad  = config.PLAYER_PARAMETER if hit_player else 0.0

    attack = good**config.BOSS_HP_EXPONENT + good
    hit    = -(abs(bad)**config.PLAYER_HP_EXPONENT)
    reward = config.REWARD_SCALE * (attack + hit)

    env._episode_reward_cumulative += reward
    env._episode_reward_history.append(env._episode_reward_cumulative)

# ── HP tracking ───────────────────────────────────────────────────────────────

env.last_ep_steps        = 1130
env.max_ep_steps         = 1892
env.last_ep_min_boss_hp  = ep_boss_hps[-1]          # last completed episode
env.session_best_boss_hp = min(ep_boss_hps)          # best ever
env.min_boss_hp          = boss_hp                   # current episode so far

# ── Steps-per-second window ───────────────────────────────────────────────────

now = time.time()
env._sps_window      = [now - i * (1 / 52.0) for i in range(520)]
env._sps_window_secs = 10

# ── Reward breakdown ──────────────────────────────────────────────────────────

_COMPONENTS = environment._REWARD_COMPONENTS

env.episode_rewards = {name: {"count": 0, "total": 0.0} for name in _COMPONENTS}
env.total_rewards   = {name: {"count": 0, "total": 0.0} for name in _COMPONENTS}

env.episode_rewards["survival"]["total"]    = env._episode_reward_cumulative
env.total_rewards["survival"]["total"]      = sum(ep_rewards) + env._episode_reward_cumulative
env.total_rewards["boss_defeated"]["total"] = 0.0
env.best_ep_reward                          = max(ep_rewards)

# ── Action breakdown ──────────────────────────────────────────────────────────

actions = list(config.ACTIONS.values())

# Current episode: random policy still, but attacks slightly overrepresented
ep_total  = env.step_count
ep_counts = {name: 0 for name in actions}
for _ in range(ep_total):
    pick = random.choices(actions, weights=[
        5, 7, 5, 5, 5, 5, 5,   # toggle actions (movement less favoured early)
        5, 6, 7, 9, 7, 5, 4, 4  # press actions (attacks slightly higher)
    ])[0]
    ep_counts[pick] += 1

# All-time totals: extrapolate from current episode proportions × 87 episodes
all_counts = {name: int(ep_counts[name] / ep_total * env.total_steps) for name in actions}

env.episode_actions = ep_counts
env.total_actions   = all_counts
# Build fake per-episode action history for the last-N column
ep_action_history = []
for _ in range(N):
    ep_ac = {}
    ep_total_fake = random.randint(800, 2000)
    for a in actions:
        ep_ac[a] = int(random.uniform(0.04, 0.10) * ep_total_fake)
    ep_action_history.append(ep_ac)

env._recent_ep_actions = ep_action_history
# ── Call the real dashboard ───────────────────────────────────────────────────

env._save_persist = lambda: None
env._print_dashboard()
