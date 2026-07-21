"""
dashboard.py - renders the per-episode training dashboard with Rich: a header
panel (session totals), tables for the episode metrics, per-action counts, and
live PPO metrics, plus a plotext line chart of the episode's cumulative reward.
Table rows are driven by the registry in ui/metrics.py, so adding a tracked
metric there shows up here automatically. Longer-term graphs over training live
in TensorBoard (tensorboard --logdir logs).
"""

import time

import plotext as plt
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from eldenring_ai import config
from eldenring_ai.io.input import ACTIONS
from eldenring_ai.ui import shared_stats
from eldenring_ai.ui.metrics import COUNTERS, EPISODE_METRICS

# Printed, not Live-rendered, because SB3's progress_bar already runs a Rich Live
# display and two Live displays on one stdout conflict.
_console = Console()

_PPO_DISPLAY = [
    ("train/entropy_loss",         "Entropy loss"),
    ("train/explained_variance",   "Explained variance"),
    ("train/value_loss",           "Value loss"),
    ("train/policy_gradient_loss", "Policy grad loss"),
    ("train/clip_fraction",        "Clip fraction"),
]


def _fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt(value, fmt):
    return "-" if value is None else format(value, fmt)


def _fmt_ppo(value):
    if value is None:
        return "-"
    return f"{value:+.5f}" if isinstance(value, float) else str(int(value))


def _reward_panel(self, width=64, height=12):
    """plotext line chart of the episode's cumulative reward: boss hits push it
    up, player hits pull it down. None until the episode has at least two steps."""
    history = self._episode_reward_history
    if len(history) < 2:
        return None
    plt.clf()
    plt.plot(history, marker="braille")
    plt.plotsize(width, height)
    plt.theme("clear")
    plt.xlabel("step")
    plt.ylabel("reward")
    return Panel(
        Text.from_ansi(plt.build()),
        title="Episode reward",
        title_align="left",
        box=box.HEAVY,
    )


def _print_dashboard(self):
    win = min(config.MEAN_STATS_WINDOW, max(1, self.episode_count))
    training_time = time.time() - self.training_start_time
    elapsed = time.time() - getattr(self, "start_ep_time", time.time())
    steps_per_second = self.ep_steps / max(elapsed, 0.001)
    training_steps = shared_stats.ppo_stats.get("total_timesteps", 0)

    counters = "    ".join(f"{c.label} [b]{getattr(self, c.key)}[/]" for c in COUNTERS)
    header = Panel(
        f"Training time [b]{_fmt_time(training_time)}[/]     "
        f"Steps/sec [b]{steps_per_second:.1f}[/]     "
        f"Steps [b]{training_steps:,}[/]\n{counters}",
        title=f"ELDEN RING  -  Episode {self.episode_count}",
        title_align="left",
        box=box.HEAVY,
    )

    stats = Table(box=box.SIMPLE_HEAVY, pad_edge=False)
    for col, justify in (("Stat", "left"), ("Episode", "right"), ("Best", "right"), ("Local mean", "right")):
        stats.add_column(col, justify=justify)
    values = self._episode_metric_values()
    for m in EPISODE_METRICS:
        mean = sum(self._metric_window[m.key]) / win
        stats.add_row(
            m.label,
            _fmt(values[m.key], m.fmt),
            _fmt(self._metric_best[m.key], m.fmt),
            _fmt(mean, m.fmt),
        )

    actions = Table(box=box.SIMPLE_HEAVY, pad_edge=False)
    for col, justify in (("Action", "left"), ("Episode", "right"), ("Total", "right"), ("Local mean", "right")):
        actions.add_column(col, justify=justify)
    for name in ACTIONS:
        action_id = ACTIONS[name].action_id
        mean = sum(one_hot[action_id] for one_hot in self.actions_mean) / win
        actions.add_row(
            name,
            f"{self.ep_actions[name]:,}",
            f"{self.training_actions[name]:,}",
            f"{mean:.0f}",
        )

    ppo = Table(box=box.SIMPLE_HEAVY, pad_edge=False)
    ppo.add_column("PPO metric", justify="left")
    ppo.add_column("Value", justify="right")
    for key, label in _PPO_DISPLAY:
        value = shared_stats.ppo_stats.get(key) if shared_stats.ppo_stats else None
        ppo.add_row(label, _fmt_ppo(value))

    renderables = [header, stats, actions, ppo]
    reward_panel = _reward_panel(self)
    if reward_panel is not None:
        renderables.append(reward_panel)
    _console.print(Group(*renderables))
    _console.print()
