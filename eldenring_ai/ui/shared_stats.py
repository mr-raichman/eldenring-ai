"""
shared_stats.py
"""

# Populated by StatsLoggerCallback in train.py after each PPO update.
# Keys match SB3's internal logger names, e.g. "rollout/ep_rew_mean".
ppo_stats: dict = {}
