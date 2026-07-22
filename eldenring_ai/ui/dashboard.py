"""
dashboard.py - the per-episode training dashboard (Rich), printed when an episode
ends. A compact info row (episode metrics, per-action counts, events & quality,
whole-training PPO stats) sits above one full-width plotext chart that overlays the
four episode traces: reward (yellow, right axis), player HP (blue), stamina (green)
and boss HP (red) on the left 0-1 axis, layered boss at the bottom up to reward on
top. The chart height is sized to fill the rest of the terminal. Panels are driven by the registries in
ui/metrics.py. Long-run trends live in TensorBoard (tensorboard --logdir logs).
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


def _table(*columns):
    """A compact SIMPLE_HEAVY table (no blank edge rows)."""
    t = Table(box=box.SIMPLE_HEAVY, pad_edge=False, expand=True, show_edge=False)
    for name, justify in columns:
        t.add_column(name, justify=justify)
    return t


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


def _metrics_panel(self, win):
    stats = _table(("Stat", "left"), ("Ep", "right"), ("Best", "right"), ("Mean", "right"))
    values = self._episode_metric_values()
    for m in EPISODE_METRICS:
        mean = sum(self._metric_window[m.key]) / win
        stats.add_row(m.label, _fmt(values[m.key], m.fmt), _fmt(self._metric_best[m.key], m.fmt), _fmt(mean, m.fmt))
    return Panel(stats, title="Episode", title_align="left", box=box.HEAVY)


def _actions_panel(self):
    names = list(ACTIONS)
    half = (len(names) + 1) // 2
    left, right = names[:half], names[half:]
    t = _table(("Action", "left"), ("Ep", "right"), ("Action", "left"), ("Ep", "right"))
    for i in range(half):
        ln = left[i]
        row = [ln, f"{self.ep_actions[ln]}"]
        if i < len(right):
            rn = right[i]
            row += [rn, f"{self.ep_actions[rn]}"]
        else:
            row += ["", ""]
        t.add_row(*row)
    return Panel(t, title="Actions", title_align="left", box=box.HEAVY)


def _events_panel(self):
    events = _table(("Event", "left"), ("Count", "right"))
    for e in EVENT_CATEGORIES:
        events.add_row(e.label, f"{self._recorder.event_counts[e.key]:,}")

    quality = _table(("Quality", "left"), ("Value", "right"))
    derived = self._recorder.derived_values()
    for d in DERIVED_MEASURES:
        quality.add_row(d.label, _fmt(derived[d.key], d.fmt))

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(width=4)
    grid.add_column(ratio=1)
    grid.add_row(events, "", quality)
    return Panel(grid, title="Events & quality", title_align="left", box=box.HEAVY)


def _ppo_panel():
    ppo = _table(("PPO metric", "left"), ("Value", "right"))
    for key, label in _PPO_DISPLAY:
        value = shared_stats.ppo_stats.get(key) if shared_stats.ppo_stats else None
        ppo.add_row(label, _fmt_ppo(value))
    return Panel(ppo, title="PPO (training)", title_align="left", box=box.HEAVY)


def _info_row(self, win):
    grid = Table.grid(expand=True)
    grid.add_column(ratio=3)
    grid.add_column(ratio=4)
    grid.add_column(ratio=5)
    grid.add_column(ratio=4)
    grid.add_row(_metrics_panel(self, win), _actions_panel(self), _events_panel(self), _ppo_panel())
    return grid


def _stairs(values):
    """Expand a per-step series into stair-step (sample-and-hold) coordinates: each
    value is held across its step and jumps vertically to the next, so the chart
    shows exact values instead of interpolated diagonals."""
    xs, ys = [], []
    for i, v in enumerate(values):
        xs.extend((i, i + 1))
        ys.extend((v, v))
    return xs, ys


def _chart_panel(self, width, height):
    """One full-width chart overlaying the four episode traces as stair steps.
    Reward is on the right y-axis (its own scale); HP/stamina/boss share the left
    0-1 axis. Drawn bottom-to-top boss -> stamina -> HP -> reward, so reward sits on
    top. Reward is an explicit RGB yellow because plotext's named "yellow" renders
    as blue; stamina is a darker green so it doesn't read as the reward yellow."""
    rec = self._recorder
    plt.clf()
    bx, by = _stairs(rec.boss_trace)
    sx, sy = _stairs(rec.stamina_trace)
    hx, hy = _stairs(rec.hp_trace)
    rx, ry = _stairs(rec.step_rewards)
    plt.plot(bx, by, label="Boss",    color="red",          marker="braille", yside="left")
    plt.plot(sx, sy, label="Stamina", color=(34, 139, 34),  marker="braille", yside="left")
    plt.plot(hx, hy, label="HP",      color="blue",         marker="braille", yside="left")
    plt.plot(rx, ry, label="reward",  color=(255, 210, 0),  marker="braille", yside="right")
    plt.ylim(0.0, 1.0, yside="left")
    plt.plotsize(width, height)
    plt.theme("clear")
    plt.xlabel("step")
    lines = plt.build().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return Panel(Text.from_ansi("\n".join(lines)), title="Episode trace", title_align="left", box=box.HEAVY)


def _print_dashboard(self):
    win = min(config.MEAN_STATS_WINDOW, max(1, self.episode_count))
    training_time = time.time() - self.training_start_time
    elapsed = time.time() - getattr(self, "start_ep_time", time.time())
    steps_per_second = self.ep_steps / max(elapsed, 0.001)
    training_steps = shared_stats.ppo_stats.get("total_timesteps", 0)

    header = _header_panel(self, training_time, steps_per_second, training_steps)
    info = _info_row(self, win)

    if len(self._recorder.step_rewards) < 2:
        _console.print(Group(header, info))
        _console.print()
        return

    # Measure the info block and give the rest of the terminal height to the chart.
    with _console.capture() as cap:
        _console.print(Group(header, info))
    used = len(cap.get().rstrip("\n").split("\n"))
    chart_height = max(14, _console.size.height - used - 3)
    chart_width = _console.size.width - 4

    chart = _chart_panel(self, chart_width, chart_height)
    _console.print(Group(header, info, chart))
    _console.print()
