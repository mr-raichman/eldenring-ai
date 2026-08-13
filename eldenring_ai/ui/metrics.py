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


@dataclass(frozen=True)
class RewardComponent:
    """One additive slice of an episode's net reward (see EpisodeRecorder)."""
    key: str           # recorder attribute / JSONL field
    label: str         # dashboard label
    tb_tag: str        # TensorBoard scalar tag


@dataclass(frozen=True)
class EventCategory:
    """A per-step reward event, counted per episode by matching the prefix of the
    free-form strings compute_reward emits."""
    key: str           # recorder counter / JSONL field (n_<key> convention)
    label: str         # dashboard label
    prefix: str        # event-string prefix, e.g. "BOSS_HIT"
    tb_tag: str        # TensorBoard scalar tag


@dataclass(frozen=True)
class DerivedMeasure:
    """A per-episode quality measure derived from the step traces."""
    key: str           # recorder attribute / JSONL field
    label: str         # dashboard label
    fmt: str           # format spec
    tb_tag: str        # TensorBoard scalar tag


@dataclass(frozen=True)
class PpoStat:
    """One of SB3's own training statistics, read back out of its logger."""
    key: str           # short name; the field inside each episode record's "ppo" block
    label: str         # dashboard label
    tag: str           # SB3 logger name, e.g. "train/value_loss"


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
    Counter("total_fall_deaths",      "Falls",       "counters/fall_deaths"),
    Counter("total_kills",            "Victories",   "counters/victories"),
    Counter("total_grace_recoveries", "Graces",      "counters/graces"),
    Counter("total_save_restores",    "Restores",    "counters/save_restores"),
]

# Additive slices of the episode's net reward. Signed so the dashboard can show
# them directly; boss_reward and defeat_bonus are positive, the rest negative.
REWARD_COMPONENTS = [
    RewardComponent("boss_reward",     "Boss reward",    "reward/boss_total"),
    RewardComponent("player_punish",   "Player punish",  "reward/punish_total"),
    RewardComponent("stamina_penalty", "Stamina penalty", "reward/stamina_penalty_total"),
    RewardComponent("defeat_bonus",    "Defeat bonus",   "reward/defeat_bonus"),
    RewardComponent("net",             "Net",            "reward/net"),
]

# Per-step reward events, counted per episode by prefix match on the event strings.
EVENT_CATEGORIES = [
    EventCategory("boss_hits",   "Boss hits",   "BOSS_HIT",             "events/boss_hits"),
    EventCategory("hits_taken",  "Hits taken",  "HIT_TAKEN",            "events/hits_taken"),
    EventCategory("dodge_bonus", "Dodge bonus", "DODGE_REWARD",         "events/dodge_bonus"),
    EventCategory("greedy",      "Greedy",      "GREEDY_PENALTY",       "events/greedy"),
    EventCategory("multihit",    "Multi-hit",   "MULTIPLE_HIT_PENALTY", "events/multihit"),
    EventCategory("combat_death", "Combat death", "DEATH",              "events/combat_death"),
    EventCategory("fall_death",  "Fall death",  "FALL_DEATH",           "events/fall_death"),
    EventCategory("zero_stamina", "Zero stamina", "ZERO STAMINA",       "events/zero_stamina"),
]

# Per-episode quality measures derived from the step traces.
DERIVED_MEASURES = [
    DerivedMeasure("stamina_at_boss_hit_mean", "Stamina@hit", ".2f", "derived/stamina_at_boss_hit"),
    DerivedMeasure("hp_at_hit_taken_mean",     "HP@taken",    ".2f", "derived/hp_at_hit_taken"),
    DerivedMeasure("best_hit",                 "Best hit",    "+.2f", "derived/best_hit"),
    DerivedMeasure("worst_hit",                "Worst hit",   "+.2f", "derived/worst_hit"),
    DerivedMeasure("hits_taken_per_boss_hit",  "Taken/hit",   ".2f", "derived/hits_taken_per_boss_hit"),
]

# SB3's training statistics, shown on the dashboard and snapshotted into each episode
# record. `train/*` only, deliberately: SB3 records the `rollout/*` keys (ep_rew_mean,
# ep_len_mean) inside _dump_logs, which then clears the logger dict, while
# StatsLoggerCallback drains it earlier in the iteration. They were listed here for a
# long time and were None in all 1509 records of the longest run - a permanently empty
# row reads as "no data yet" rather than "never available".
PPO_STATS = [
    PpoStat("explained_variance",   "Explained var",    "train/explained_variance"),
    PpoStat("entropy_loss",         "Entropy loss",     "train/entropy_loss"),
    PpoStat("value_loss",           "Value loss",       "train/value_loss"),
    PpoStat("policy_gradient_loss", "Policy grad loss", "train/policy_gradient_loss"),
    PpoStat("clip_fraction",        "Clip fraction",    "train/clip_fraction"),
    PpoStat("approx_kl",            "Approx KL",        "train/approx_kl"),
    PpoStat("learning_rate",        "Learning rate",    "train/learning_rate"),
]
