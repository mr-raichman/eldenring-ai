"""
training.py - RL hyperparameters, reward weights, and episode/checkpoint limits.
"""

MEAN_STATS_WINDOW = 20

# Reward weights
#
# BOSS_PARAMETER was 1 for the whole 1M-step run of 2026-08-13, which unlearned the attack
# entirely (7.15% of actions to 0.03%). Measured there: a landed hit paid 0.977 and a hit
# taken cost 3.554, so landing one cost 0.91 net reward and PPO removed it correctly.
#
# 4.5 and not 5. The floor and the ceiling are both measured and they are close together:
# attacking only pays at the margin above 4.98, and a sequential trade turns PROFITABLE
# above 5.71 (a hit worth more than GREEDY_PENALTY x one hit taken). Scaling this against
# PLAYER_PARAMETER cannot widen that window - the 8% attack-to-hit conversion pins the
# ratio at 2.6:1 - so the value is chosen inside it, 21% clear of the trading edge rather
# than 12%. It sits below the 4.98 floor deliberately: that floor comes from a cost curve
# fitted on episodes where a near-uniform policy happened to swing a lot, so it carries the
# cost of being in melee range as well as the cost of swinging, and is overstated.
BOSS_PARAMETER = 4.5
DODGE_REWARD = 1.25

# Smallest boss-HP drop that counts as a hit. A real sword hit measures 0.016-0.025
# of the bar (16-25 px of BOSS_HP_CAP_FULL); anything under this is bar-reading noise.
BOSS_HIT_MIN_DELTA = 0.01

# Sets the exchange rate between landing a hit and taking one, which is the whole
# balance of the reward. At 1 the run of 2026-08-06 measured 1.57 per boss hit against
# 0.87 per non-death hit taken: the agent was paid +0.70 to trade a hit for a hit, and
# it learned exactly that (boss hits 1.15 -> 2.97 per 100 steps, hits taken flat at
# 4.3 -> 4.8). A kill needs 36 hits and four kill the player, so the real rate is 1:8.8
# the other way. 2.0 was set to put the reward at 2:1 against trading.
#
# It did not. Measured over the 1M-step run that followed: a hit taken costs 3.554 on
# average, not the 1.75 predicted, because that prediction used the mean cost of a
# NON-DEATH hit (0.87) while the penalty the agent actually meets also carries the death
# floor, the vulnerability exponent and the multihit multiplier. The realised rate was
# 3.6:1 before GREEDY_PENALTY and up to 10.9:1 after it, and the agent stopped attacking.
# Kept at 2.0 anyway: the correction is made on BOSS_PARAMETER instead, so that the death
# penalties (absolute, and not scaled by this) keep the meaning they were tuned with.
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
# DODGE_WINDOW was 10, then 5. Neither made the bonus a detector: over the 1M-step run it
# fired on 0.563 of boss hits against a chance rate of 0.710 at the observed dodge share,
# i.e. BELOW chance, so as a measure of "dodged into an opening" it carries no information.
# It is kept anyway, as Teo's call, for a different job: a positive term attached to dodging
# at all, because dodging otherwise carries only the negative reward of the hits it fails to
# avoid, and dodge-then-attack is the sequence worth teaching. 3 steps with DODGE_REWARD
# 1.25 halves its variance on the hit reward (mean multiplier 1.209 -> 1.072).
#
# The evidence against it is weak on its own terms and is worth re-measuring: 0.563 comes
# from 519 hits produced by a policy that never attacked.
DODGE_WINDOW = 3
GREEDY_WINDOW = 6
MULTIHIT_WINDOW = 10

# Attacks in the last HISTORY_LENGTH steps that are paid at the full rate. Above N, a landed
# hit pays (N/n)**EXP of it, so the TOTAL hit reward turns over instead of flattening.
#
# EXP must exceed 1 for that to happen, and this is the whole reason for the shape: the cost
# of attacking saturates (measured marginal cost falls 3.1x from a 3% to a 17% attack share),
# so per-step reward is convex in the attack rate and its optimum is always a corner - either
# never attack or attack as often as stamina allows. A concave reward is the only thing that
# puts the optimum in between. 1/(1+k*n) does not: it flattens at 1/(k*HISTORY_LENGTH) and
# leaves the corner where it was.
#
# N sets where the optimum sits (6 in 24 steps = a 25% attack share), BOSS_PARAMETER sets
# whether the agent goes there. They are separable, which is what makes this tunable.
ATTACK_SOFTCAP_N = 6
ATTACK_SOFTCAP_EXP = 2

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
