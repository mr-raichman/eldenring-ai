"""
metrics.py - the metric registry. Single source of truth for what the training
run tracks. Collection (environment.py), the terminal dashboard (dashboard.py),
TensorBoard logging (train.py), and persistence all iterate these lists, so
adding or removing a tracked metric is a one-line change here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeMetric:
    """A per-episode scalar with a best value and a rolling-window mean."""
    key: str           # logical name; also the key used in the env value dict
    label: str         # dashboard row label
    fmt: str           # format spec applied to episode / best / mean, e.g. "+.2f"
    better: str        # "max" or "min" - which direction is the running best
    tb_tag: str        # TensorBoard scalar tag, e.g. "episode/reward"
    persist_best: str  # session_stats.json key for the best value
    persist_mean: str  # session_stats.json key for the rolling window


@dataclass(frozen=True)
class Counter:
    """A monotonically increasing session total."""
    key: str           # env attribute name
    label: str         # dashboard label
    tb_tag: str        # TensorBoard scalar tag


# Per-episode metrics. `persist_*` keep the existing session_stats.json key names
# so old stats files keep loading unchanged.
EPISODE_METRICS = [
    EpisodeMetric("reward",  "Reward",    "+.2f", "max", "episode/reward",  "best_ep_reward", "reward_mean"),
    EpisodeMetric("boss_hp", "Margit HP", ".1%",  "min", "episode/boss_hp", "best_boss_hp",   "boss_hp_mean"),
    EpisodeMetric("steps",   "Steps",     ",.0f", "max", "episode/steps",   "best_ep_steps",  "steps_mean"),
]

# Session counters.
COUNTERS = [
    Counter("total_deaths",           "Deaths",      "counters/deaths"),
    Counter("total_kills",            "Victories",   "counters/victories"),
    Counter("total_truncations",      "Truncations", "counters/truncations"),
    Counter("total_grace_recoveries", "Graces",      "counters/graces"),
]
