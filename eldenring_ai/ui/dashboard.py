"""
dashboard.py - the per-episode training dashboard (Rich), printed when an episode
ends. A full-width, multi-panel layout: a header with session totals and the
episode outcome; three columns (episode result + reward composition, per-action
counts, whole-training PPO stats); a full-width events-and-quality strip; and two
plotext charts (per-step reward and the HP/boss/stamina fight trajectory). Panels
are driven by the registries in ui/metrics.py, so tracked fields appear here
automatically. Long-run trends live in TensorBoard (tensorboard --logdir logs).
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
from eldenring_ai.ui.metrics import (
    COUNTERS,
    DERIVED_MEASURES,
    EPISODE_METRICS,
    EVENT_CATEGORIES,
    REWARD_COMPONENTS,
)

# Printed, not Live-rendered, because SB3's progress_bar already runs a Rich Live
# display and two Live displays on one stdout conflict.
_console = Console()

_PPO_DISPLAY = [
    ("train/explained_variance",   "Explained var"),
    ("train/entropy_loss",         "Entropy loss"),
    ("train/value_loss",           "Value loss"),
    ("train/policy_gradient_loss", "Policy grad loss"),
    ("train/clip_fraction",        "Clip fraction"),
    ("train/approx_kl",            "Approx KL"),
    ("train/learning_rate",        "Learning rate"),
    ("rollout/ep_rew_mean",        "Ep rew mean"),
    ("rollout/ep_len_mean",        "Ep len mean"),
]

_OUTCOME = {
    "kill":         ("VICTORY",    "bold green"),
    "fall_death":   ("FALL DEATH", "bold red"),
    "combat_death": ("DEATH",      "bold yellow"),
    "timeout":      ("TIMEOUT",    "bold dim"),
}


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


def _header_panel(self, training_time, steps_per_second, training_steps):
    label, style = _OUTCOME.get(self._episode_outcome, ("-", "bold"))
    elapsed = time.time() - getattr(self, "start_ep_time", time.time())
    counters = "    ".join(f"{c.label} [b]{getattr(self, c.key)}[/]" for c in COUNTERS)
    body = (
        f"Training [b]{_fmt_time(training_time)}[/]     "
        f"Steps [b]{training_steps:,}[/]     "
        f"Steps/sec [b]{steps_per_second:.1f}[/]     "
        f"Episode [b]{elapsed:.0f}s[/] / [b]{self.ep_steps}[/] steps\n"
        f"{counters}"
    )
    title = Text.assemble(
        (f"ELDEN RING  -  Episode {self.episode_count}  -  ", "bold"),
        (label, style),
    )
    return Panel(body, title=title, title_align="left", box=box.HEAVY)


def _result_panel(self, win):
    stats = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False)
    for col, justify in (("Stat", "left"), ("Episode", "right"), ("Best", "right"), ("Mean", "right")):
        stats.add_column(col, justify=justify)
    values = self._episode_metric_values()
    for m in EPISODE_METRICS:
        mean = sum(self._metric_window[m.key]) / win
        stats.add_row(m.label, _fmt(values[m.key], m.fmt), _fmt(self._metric_best[m.key], m.fmt), _fmt(mean, m.fmt))

    comp = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False, show_header=False)
    comp.add_column("Component", justify="left")
    comp.add_column("Value", justify="right")
    composition = self._recorder.composition
    for c in REWARD_COMPONENTS:
        style = "bold" if c.key == "net" else ""
        comp.add_row(c.label, Text(f"{composition[c.key]:+.2f}", style=style))

    return Panel(
        Group(stats, Text("reward composition", style="dim"), comp),
        title="Episode result", title_align="left", box=box.HEAVY,
    )


def _actions_panel(self, win):
    actions = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False)
    for col, justify in (("Action", "left"), ("Ep", "right"), ("%", "right"), ("Total", "right"), ("Mean", "right")):
        actions.add_column(col, justify=justify)
    total_ep = max(self.ep_steps, 1)
    for name in ACTIONS:
        action_id = ACTIONS[name].action_id
        mean = sum(one_hot[action_id] for one_hot in self.actions_mean) / win
        ep_count = self.ep_actions[name]
        actions.add_row(
            name, f"{ep_count:,}", f"{100 * ep_count / total_ep:.0f}",
            f"{self.training_actions[name]:,}", f"{mean:.0f}",
        )
    return Panel(actions, title="Actions", title_align="left", box=box.HEAVY)


def _ppo_panel():
    ppo = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False)
    ppo.add_column("PPO metric", justify="left")
    ppo.add_column("Value", justify="right")
    for key, label in _PPO_DISPLAY:
        value = shared_stats.ppo_stats.get(key) if shared_stats.ppo_stats else None
        ppo.add_row(label, _fmt_ppo(value))
    return Panel(ppo, title="PPO (training)", title_align="left", box=box.HEAVY)


def _events_panel(self):
    events = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False)
    events.add_column("Event", justify="left")
    events.add_column("Count", justify="right")
    for e in EVENT_CATEGORIES:
        events.add_row(e.label, f"{self._recorder.event_counts[e.key]:,}")

    quality = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False)
    quality.add_column("Quality", justify="left")
    quality.add_column("Value", justify="right")
    derived = self._recorder.derived_values()
    for d in DERIVED_MEASURES:
        quality.add_row(d.label, _fmt(derived[d.key], d.fmt))

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(width=4)  # gutter so Count and Quality don't touch
    grid.add_column(ratio=1)
    grid.add_row(events, "", quality)
    return Panel(grid, title="Events & quality", title_align="left", box=box.HEAVY)


def _chart(series_list, title, width, height, ymin=None, ymax=None):
    plt.clf()
    for values, label in series_list:
        plt.plot(values, label=label, marker="braille")
    plt.plotsize(width, height)
    plt.theme("clear")
    if ymin is not None:
        plt.ylim(ymin, ymax)
    plt.xlabel("step")
    lines = plt.build().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return Panel(Text.from_ansi("\n".join(lines)), title=title, title_align="left", box=box.HEAVY)


def _charts_row(self):
    rec = self._recorder
    if len(rec.step_rewards) < 2:
        return None
    # Each chart sits in a HEAVY panel (2 border + 2 padding) inside a 2-column grid,
    # so the plot width must leave that chrome plus a 1-char safety margin.
    half = max(40, (_console.size.width - 10) // 2)
    reward_chart = _chart([(rec.step_rewards, "reward")], "Step reward", half, 14)
    trajectory = _chart(
        [(rec.hp_trace, "HP"), (rec.boss_trace, "Boss"), (rec.stamina_trace, "Stam")],
        "Fight trajectory", half, 14, ymin=0.0, ymax=1.0,
    )
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(reward_chart, trajectory)
    return grid


def _print_dashboard(self):
    win = min(config.MEAN_STATS_WINDOW, max(1, self.episode_count))
    training_time = time.time() - self.training_start_time
    elapsed = time.time() - getattr(self, "start_ep_time", time.time())
    steps_per_second = self.ep_steps / max(elapsed, 0.001)
    training_steps = shared_stats.ppo_stats.get("total_timesteps", 0)

    top = Table.grid(expand=True)
    top.add_column(ratio=5)
    top.add_column(ratio=4)
    top.add_column(ratio=4)
    top.add_row(_result_panel(self, win), _actions_panel(self, win), _ppo_panel())

    renderables = [
        _header_panel(self, training_time, steps_per_second, training_steps),
        top,
        _events_panel(self),
    ]
    charts = _charts_row(self)
    if charts is not None:
        renderables.append(charts)

    _console.print(Group(*renderables))
    _console.print()
