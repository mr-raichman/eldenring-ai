"""
shared_stats.py - process-global bridges between the env, the SB3 callbacks, and
the dashboard. Kept module-level so all three can see the same dicts.
"""

# Populated by StatsLoggerCallback in train.py after each PPO update.
# Keys match SB3's internal logger names, e.g. "rollout/ep_rew_mean".
ppo_stats: dict = {}

# Populated by EldenRingEnv at each episode end: {tensorboard_tag: value}.
# StatsLoggerCallback records these into the SB3 logger -> TensorBoard.
episode_metrics: dict = {}
