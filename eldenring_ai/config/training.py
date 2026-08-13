"""
training.py - RL hyperparameters, reward weights, and episode/checkpoint limits.
"""

MEAN_STATS_WINDOW = 20

# Reward weights
BOSS_PARAMETER = 1
DODGE_REWARD = 1.5

# Smallest boss-HP drop that counts as a hit. A real sword hit measures 0.016-0.025
# of the bar (16-25 px of BOSS_HP_CAP_FULL); anything under this is bar-reading noise.
BOSS_HIT_MIN_DELTA = 0.01

# Sets the exchange rate between landing a hit and taking one, which is the whole
# balance of the reward. At 1 the run of 2026-08-06 measured 1.57 per boss hit against
# 0.87 per non-death hit taken: the agent was paid +0.70 to trade a hit for a hit, and
# it learned exactly that (boss hits 1.15 -> 2.97 per 100 steps, hits taken flat at
# 4.3 -> 4.8). A kill needs 36 hits and four kill the player, so the real rate is 1:8.8
# the other way. 2.0 puts the reward at 2:1 against trading: measured cost 0.87 doubled
# is 1.75, against a post-change hit value of 0.877. Trades stay affordable on purpose.
PLAYER_PARAMETER = 2.0
# Taking a hit is punished by (VULNERABILITY_BASE - hp) ** VULNERABILITY_EXP, so the
# same damage costs more the lower the health it was taken at. The base sets the
# multiplier's range: at 2.0 it runs from 1x at full health to 4x at zero.
PLAYER_VULNERABILITY_BASE = 2.0
PLAYER_VULNERABILITY_EXP = 2

MIN_COMBAT_DEATH_PENALTY = 3
FALL_DEATH_PENALTY = 5
FALL_DEATH_HP_THRESHOLD = 0.75
GREEDY_PENALTY = 3

# The gate discounting a hit landed at low stamina, 1-(1-stamina)**STAMINA_EXP. At
# EXP 3 it removed 9% of the hit reward over the whole 2026-08-06 run, so it is close to
# inert; LOW_STAMINA_PUNISH is the half that taught something (zero-stamina steps 5.5 ->
# 2.3 per episode). Kept as the knob for a build where stamina management matters more.
STAMINA_EXP = 3
LOW_STAMINA_PUNISH = 0.1

# Unreachable at the horizon GAMMA sets: a kill needs ~1100 steps, so this arrives
# discounted by 0.96**1100. Kept for the day the horizon or the damage rate changes.
REWARD_BOSS_DEFEATED = 100.0

# Reward history windows (in steps; 0.2s each, max possible = HISTORY_LENGTH)
#
# DODGE_WINDOW was 10. At a 6.8% dodge share, the chance of one landing anywhere in ten
# steps is 0.51 and the bonus fired on 0.55 of boss hits: it was paying out at the rate
# of a coin flip, which is variance on the reward rather than a signal. 5 steps (1s) puts
# the chance rate near 0.30, so a real dodge-into-opening can show above it.
DODGE_WINDOW = 5
GREEDY_WINDOW = 6
MULTIHIT_WINDOW = 10

# Episode (unbounded: episodes end only on death or victory, never a step limit)
HISTORY_LENGTH = 24
BOSS_DEFEAT_HP_THRESHOLD = 0.005

# PPO
LEARNING_RATE = 2e-4
# The decay reaches LR_MIN at ~850k steps, which is 184 hours at the ~4600 steps/hour
# this setup actually achieves, so on any run so far the rate is near-constant (1.98e-4
# to 1.70e-4 over 80k steps). Kept for a run long enough to need it.
LR_DECAY_FACTOR = 0.8
LR_DECAY_STEPS  = 100_000
LR_MIN          = 3e-5
# 2048 bought 39 policy updates in a 17-hour run: at ~4600 env steps/hour the usual
# rollout sizes assume a sample budget this setup does not have. 1024 doubles the number
# of updates on the same data; BATCH_SIZE stays 512, so 2 minibatches per epoch.
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 6
GAMMA = 0.96
ENT_COEF = 0.01
TARGET_KL = 0.03

# Training run / checkpointing
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 10_000
CHECKPOINT_FREQ_MINI = 1_000
