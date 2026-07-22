"""
episode_log.py - EpisodeRecorder: accumulates the per-step reward decomposition and
traces during an episode, then writes a per-episode aggregate row (JSONL, appended
forever) and a rolling per-step CSV (last STEP_RECORD_EPISODES episodes). Pure - no
game I/O; iterates the ui/metrics.py registries so the tracked fields stay
single-source with the dashboard and TensorBoard.
"""

import csv
import json
import time
from collections import deque

from eldenring_ai import config
from eldenring_ai.config import paths
from eldenring_ai.ui.metrics import EVENT_CATEGORIES, REWARD_COMPONENTS

_STEP_CSV_HEADER = [
    "episode", "step", "action",
    "player_hp", "boss_hp", "stamina",
    "boss_reward", "player_punish", "reward", "events",
]


class EpisodeRecorder:
    """One instance per training run: reset() at each episode start, record_step()
    every step, write_episode() at episode end. `_recent_episodes` survives reset so
    the CSV keeps a rolling window."""

    def __init__(self):
        self._recent_episodes = deque(maxlen=config.STEP_RECORD_EPISODES)
        self.reset()

    def reset(self):
        # Per-step traces for the dashboard charts.
        self.step_rewards = []
        self.hp_trace = []
        self.boss_trace = []
        self.stamina_trace = []
        # Rows for the step CSV of the current episode (episode id prepended at write).
        self._step_rows = []
        # Reward composition, signed as displayed (boss/defeat positive, rest negative).
        self.composition = {c.key: 0.0 for c in REWARD_COMPONENTS}
        # Per-episode event counts.
        self.event_counts = {e.key: 0 for e in EVENT_CATEGORIES}
        # Derived-measure accumulators.
        self._stamina_at_boss_hit = []
        self._hp_at_hit_taken = []
        self._best_hit = 0.0
        self._worst_hit = 0.0

    def record_step(self, step, action, player_hp, boss_hp, stamina,
                    prev_player_hp, prev_stamina,
                    boss_reward, player_punish, reward, events, defeat_bonus=0.0):
        self.step_rewards.append(reward)
        self.hp_trace.append(player_hp)
        self.boss_trace.append(boss_hp)
        self.stamina_trace.append(stamina)
        self._step_rows.append([
            step, action,
            round(player_hp, 3), round(boss_hp, 3), round(stamina, 3),
            round(boss_reward, 4), round(player_punish, 4), round(reward, 4),
            "|".join(events),
        ])

        # Composition (see reward.py: exactly one branch fires per step).
        self.composition["boss_reward"] += boss_reward
        self.composition["player_punish"] -= player_punish
        if any(e.startswith("STEP_PENALTY") for e in events):
            self.composition["step_penalty"] -= config.STEP_PENALTY
        if any(e.startswith("ZERO STAMINA") for e in events):
            self.composition["stamina_penalty"] -= config.LOW_STAMINA_PUNISH
        self.composition["defeat_bonus"] += defeat_bonus
        self.composition["net"] += reward

        # Event counts by prefix.
        for cat in EVENT_CATEGORIES:
            if any(e.startswith(cat.prefix) for e in events):
                self.event_counts[cat.key] += 1

        # Derived measures use the pre-hit values (the exposure that earned/took it).
        if boss_reward > 0:
            self._stamina_at_boss_hit.append(prev_stamina)
            self._best_hit = max(self._best_hit, boss_reward)
        if player_punish > 0:
            self._hp_at_hit_taken.append(prev_player_hp)
            self._worst_hit = max(self._worst_hit, player_punish)

    def derived_values(self):
        def mean(xs):
            return round(sum(xs) / len(xs), 3) if xs else None
        boss = self.event_counts["boss_hits"]
        taken = self.event_counts["hits_taken"]
        return {
            "stamina_at_boss_hit_mean": mean(self._stamina_at_boss_hit),
            "hp_at_hit_taken_mean": mean(self._hp_at_hit_taken),
            "best_hit": round(self._best_hit, 3),
            "worst_hit": round(-self._worst_hit, 3),
            "hits_taken_per_boss_hit": round(taken / boss, 3) if boss else None,
        }

    def write_episode(self, context):
        paths.EPISODE_RECORDS.parent.mkdir(parents=True, exist_ok=True)

        row = self._aggregate_row(context)
        with open(paths.EPISODE_RECORDS, "a") as f:
            f.write(json.dumps(row) + "\n")

        episode = context["episode"]
        self._recent_episodes.append([[episode] + r for r in self._step_rows])
        with open(paths.STEP_RECORDS, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_STEP_CSV_HEADER)
            for ep_rows in self._recent_episodes:
                writer.writerows(ep_rows)

    def _aggregate_row(self, context):
        row = {
            "episode": context["episode"],
            "wall_time": round(time.time(), 1),
            "training_time_s": round(context["training_time_s"], 1),
            "total_timesteps": context["total_timesteps"],
            "outcome": context["outcome"],
            "steps": len(self.step_rewards),
            "duration_s": round(context["duration_s"], 1),
            "steps_per_s": round(context["steps_per_s"], 2),
            "net_reward": round(self.composition["net"], 3),
            "final_boss_hp": round(context["final_boss_hp"], 4),
        }
        for c in REWARD_COMPONENTS:
            row[c.key] = round(self.composition[c.key], 3)
        for e in EVENT_CATEGORIES:
            row[f"n_{e.key}"] = self.event_counts[e.key]
        row.update(self.derived_values())
        row["actions"] = context["actions"]
        row["ppo"] = context["ppo"]
        return row
