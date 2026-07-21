"""
dashboard.py - renders the per-episode training dashboard: session totals,
episode/best/rolling stats, per-action counts, live PPO metrics, and the ASCII
reward-over-time graph printed after each episode.
"""

import time

from eldenring_ai import config
from eldenring_ai.ui import shared_stats
from eldenring_ai.io.input import ACTIONS

_PPO_DISPLAY = [
    ("train/entropy_loss",         "Entropy loss"),
    ("train/explained_variance",   "Explained variance"),
    ("train/value_loss",           "Value loss"),
    ("train/policy_gradient_loss", "Policy grad loss"),
    ("train/clip_fraction", "Clip fraction"),
]

def _fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _render_reward_graph(self, n_rows=15, max_width=80):
    history = self._episode_reward_history
    if len(history) < 2:
        return []

    total_steps = len(history)
    batches = []
    for i in range(n_rows - 1):
        start = int(i       * total_steps / (n_rows - 1))
        end   = int((i + 1) * total_steps / (n_rows - 1))
        end   = min(end, total_steps)
        earned = history[end - 1] - (history[start - 1] if start > 0 else 0)
        batches.append(earned)

    max_abs = max(abs(b) for b in batches) if batches else 1.0
    if max_abs == 0:
        max_abs = 1.0

    lines = []
    for earned in batches:
        char   = "+" if earned >= 0 else "-"
        filled = int((abs(earned) / max_abs) * max_width)
        lines.append(f"  │{char * filled}")
    lines.append(f"  └{'─' * max_width}")
    return lines

def _print_dashboard(self):
    training_time = time.time() - self.training_start_time
    elapsed = time.time() - getattr(self, 'start_ep_time', time.time())
    steps_per_second = self.ep_steps / max(elapsed, 0.001)
    training_steps  = shared_stats.ppo_stats.get("total_timesteps", 0)

    boss_hp  = f"{self.ep_boss_hp * 100:.1f}%"
    best_boss_hp  = f"{self.best_boss_hp * 100:.1f}%"

    reward_mean = sum(self.reward_mean) / min(config.MEAN_STATS_WINDOW, self.episode_count)
    boss_hp_mean = sum(self.boss_hp_mean) / min(config.MEAN_STATS_WINDOW, self.episode_count)
    steps_mean = sum(self.steps_mean) / min(config.MEAN_STATS_WINDOW, self.episode_count)

    actions_mean = {}

    SEPARATOR = 72

    row = []
    row.append(f"  ELDEN RING ═ EPISODE {self.episode_count}")
    row.append("═" * SEPARATOR)
    row.append(f"  Training time  {_fmt_time(training_time)}   |   Steps/sec  {steps_per_second:.1f}   |   Steps  {training_steps:,}")
    row.append("")
    row.append(f"  Deaths  {self.total_deaths}   |   Victories  {self.total_kills}   |   Truncations  {self.total_truncations}   |   Graces  {self.total_grace_recoveries}")
    row.append("═" * SEPARATOR)
    row.append(f"  {'STATS':<15}  {'EPISODE':>12}  {'BEST':>10} {'LOCAL MEAN':>12}")
    row.append("")
    row.append(f"  {'Reward':<15}  {self.ep_reward:>+12.2f}  {self.best_ep_reward:>+10.2f} {reward_mean:>+12.2f}")
    row.append(f"  {'Margit HP':<15}  {boss_hp:>12}  {best_boss_hp:>10} {boss_hp_mean*100:>11.1f}%")
    row.append(f"  {'Steps':<15}  {self.ep_steps:>12,}  {self.best_ep_steps:>10,} {steps_mean:>12.0f}")
    row.append("═" * SEPARATOR)

    row.append(f"  {'ACTION':<15}  {'EPISODE':>12}  {'TOTAL':>10} {'LOCAL MEAN':>12}")
    row.append("")
    for name in ACTIONS:
        ep_actions    = self.ep_actions[name]
        total_actions   = self.training_actions[name]

        actions_mean[name] = 0
        for one_hot in self.actions_mean:
            actions_mean[name] += one_hot[ACTIONS[name].action_id]
        actions_mean[name] = actions_mean[name] / min(config.MEAN_STATS_WINDOW, self.episode_count)

        row.append(f"  {name:<15}  {ep_actions:>12}  {total_actions:>10} {actions_mean[name]:>12.0f}")
    row.append("═" * SEPARATOR)
    row.append(f"  {'PPO METRICS':>10} {'VALUE':>29}")
    row.append("")
    for key, label in _PPO_DISPLAY:
        val     = shared_stats.ppo_stats.get(key) if shared_stats.ppo_stats else None
        val_str = f"{val:+10.5f}" if isinstance(val, float) else str(int(val)) if val is not None else "-"
        row.append(f"  {label:<30} {val_str}")

    row.append("═" * SEPARATOR)

    graph_lines = _render_reward_graph(self, n_rows=len(row))

    max_rows = max(len(row), len(graph_lines))
    for i in range(max_rows):
        l = row[i]        if i < len(row)        else ""
        r = graph_lines[i] if i < len(graph_lines) else ""
        print(f"{l:<72}  {r}")
    print(" ")
