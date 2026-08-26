# Training analysis, 2026-08-27. The 1,000,000-step run

**Reviewed:** `data/runs/2026-08-13_15-41-15` in full: 7,708 episodes, 1,000,143 timesteps,
112.7 hours of wall clock, 976 policy updates, **0 kills**. Code at commit `ee31fcf`, config
snapshot `config.json` in the run directory, unchanged across the whole run (no second
snapshot was written, so no parameter moved between restarts). Sources: `episode_records.jsonl`
(7,708 records), `logs/events.out.tfevents.1786628503` (38 scalar tags, 977 points each),
`step_records.csv` (last 5 episodes only), `events.log` (354 lines), `data/session_stats.json`.

This is the first run under the Round 2 build of 2026-08-13: `PLAYER_PARAMETER` 1.0 to 2.0,
`STEP_PENALTY` deleted, `N_STEPS` 2048 to 1024, `DODGE_WINDOW` 10 to 5, `BOSS_HP_MEDIAN_WINDOW`
5 to 3. It is the longest run in the project's history by 12.5x.

**One call made in writing this:** where a claim needs the game running to settle, it is
marked *inferred* and carries the single observation that would settle it. Everything else is
measured from the files above.

---

## The finding

**Nothing failed. PPO solved the reward function exactly, and the reward function's optimum is
a policy that never attacks.**

Over 1M steps the agent's attack rate fell from 7.15% of actions to **0.03%**, monotonically,
without a single reversal. Boss hits per episode went 0.54 to 0.00. For the 100,000 steps
between 600k and 700k the agent landed **zero hits in 688 consecutive episodes** and left
Margit at 99.9% health in every one of them.

Meanwhile mean episode reward improved from -8.12 to -5.40 and every PPO diagnostic got
healthier. The optimiser did its job. The objective was wrong.

The clinching measurement: **the single best episode of the entire run scored in the 6th
percentile of reward.** Episode 117 (step 11,019) landed 6 hits and took Margit to 81.3%, the
deepest he was ever taken, and was paid **-8.295** for it, worse than 93.7% of all 7,708
episodes. Across the 10 best episodes by damage dealt, the mean reward percentile is 30.4.
The reward function ranks progress toward the goal as failure.

---

## How the run developed

50k-step blocks. `ATK%` is Light plus Heavy Attack as a share of actions; `hits` is boss hits
landed per episode; `minHP` is the lowest Margit was taken in the block.

| block | kSteps | eps | reward | steps | Margit | minHP | hits | taken | ATK% | DODGE% | HEAL% | MOVE% | entropy | EV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 49 | 525 | -6.61 | 95 | 0.981 | **0.813** | **0.54** | 3.90 | **7.15** | 21.0 | 13.60 | 38.3 | -2.185 | 0.36 |
| 1 | 99 | 455 | -6.06 | 110 | 0.995 | 0.895 | 0.18 | 3.40 | 2.69 | 22.3 | 3.66 | 50.4 | -1.920 | 0.45 |
| 2 | 149 | 442 | -5.85 | 113 | 0.997 | 0.916 | 0.11 | 2.97 | 1.31 | 23.6 | 1.77 | 42.7 | -1.781 | 0.55 |
| 3 | 199 | 430 | -5.60 | 116 | 0.998 | 0.947 | 0.05 | 2.85 | 1.06 | 18.4 | 1.06 | 53.2 | -1.875 | 0.55 |
| 4 | 249 | 453 | -5.46 | 111 | 0.999 | 0.947 | 0.02 | 2.79 | 0.25 | 15.1 | 0.23 | 44.1 | -1.753 | 0.57 |
| 5 | 299 | 429 | -5.47 | 117 | 0.999 | 0.932 | 0.02 | 2.68 | 0.21 | 16.7 | 0.16 | 58.0 | -1.559 | 0.61 |
| 6 | 349 | 396 | -5.55 | 126 | 0.999 | 0.923 | 0.03 | 2.97 | 0.36 | 21.1 | 0.13 | 65.2 | -1.510 | 0.61 |
| 7 | 399 | 397 | -5.67 | 126 | 0.999 | 0.916 | 0.04 | 3.08 | 0.46 | 20.7 | 0.22 | 55.5 | -1.641 | 0.60 |
| 8 | 449 | 392 | -5.80 | 128 | 0.999 | 0.932 | 0.04 | 2.88 | 0.34 | 27.9 | 0.07 | 57.0 | -1.400 | 0.64 |
| 9 | 499 | 369 | -5.72 | 135 | 1.000 | 0.947 | 0.01 | 2.98 | 0.11 | 25.1 | 0.03 | 62.8 | -1.327 | 0.57 |
| 10 | 549 | 362 | -5.77 | 138 | 1.000 | 0.984 | 0.00 | 2.99 | 0.07 | 28.1 | 0.04 | 64.3 | -1.350 | 0.59 |
| 11 | 599 | 363 | -5.59 | 138 | 1.000 | 0.975 | 0.01 | 2.79 | 0.04 | 30.0 | 0.02 | 56.7 | -1.334 | 0.65 |
| 12 | 649 | 345 | -5.46 | 145 | 1.000 | 0.999 | 0.00 | 2.86 | 0.02 | 24.7 | 0.01 | 61.7 | -1.312 | 0.64 |
| 13 | 699 | 343 | -5.62 | 146 | 1.000 | 0.999 | 0.00 | 3.05 | 0.02 | 23.7 | 0.03 | 61.2 | -1.329 | 0.58 |
| 14 | 749 | 335 | -5.54 | 149 | 1.000 | 1.000 | 0.00 | 3.10 | 0.02 | 21.1 | 0.01 | 64.7 | -1.341 | 0.60 |
| 15 | 799 | 337 | -5.54 | 149 | 1.000 | 0.975 | 0.01 | 3.02 | 0.02 | 24.0 | 0.01 | 66.7 | -1.270 | 0.62 |
| 16 | 849 | 330 | -5.46 | 151 | 1.000 | 0.975 | 0.00 | 2.99 | 0.03 | 22.5 | 0.00 | 66.3 | -1.268 | 0.62 |
| 17 | 899 | 340 | -5.67 | 147 | 1.000 | 0.975 | 0.01 | 3.02 | 0.06 | 24.7 | 0.01 | 63.2 | -1.348 | 0.63 |
| 18 | 949 | 323 | -5.72 | 155 | 1.000 | 0.975 | 0.01 | 3.18 | 0.08 | 24.7 | 0.02 | 58.8 | -1.344 | 0.61 |
| 19 | 1000 | 342 | -5.54 | 147 | 1.000 | 0.975 | 0.01 | 3.07 | 0.02 | 21.9 | 0.00 | 66.1 | -1.335 | 0.58 |

Three things to read off it.

**All learning happened in the first 250k steps, and it was learning to disengage.** Reward
went -6.61 to -5.46 by block 4 and then moved 0.35 in either direction for the remaining
750,000 steps. SB3's own rolling mean confirms it: best 100k-step window was 200-300k at
-5.471, the global best rolling value was -5.240 at step 600,064, and the last window was
-5.649. **The last 750,000 steps bought nothing.** That is 85 hours of wall clock.

**Margit's health is a flat line at 1.000 from block 9 onward.** The best-ever episode was
episode 117, at step 11,019, in the first 1.1% of the run. Nothing since has come close.

**Episode length grew 95 to 155 steps while damage went to zero.** The agent learned to
survive longer by refusing to engage. That is the reward's optimum, correctly found.

---

## The mechanism: attacking is measurably negative-EV

The first 100k steps are a clean natural experiment. The policy was still near-uniform, so
whether an episode contained an attack was close to random rather than a policy choice, and
the outcome can be compared across otherwise similar episodes.

| episodes with | n | net reward | hits taken | greedy penalties | multihit penalties |
|---|---|---|---|---|---|
| 0 boss hits | 711 | **-6.09** | 3.39 | 0.01 | 0.52 |
| exactly 1 hit | 199 | **-7.00** | 4.35 | 0.45 | 0.81 |
| 2 or more hits | 70 | **-7.19** | 4.60 | 1.01 | 0.74 |

**Landing a hit costs 0.91 reward. Landing two or more costs 1.10.** There is no ambiguity and
no confound from policy drift: this is the same policy, in the same 100k window, and the
episodes where it happened to connect scored worse.

The arithmetic behind it, all measured:

- A landed hit pays **0.977** on average (`boss_reward` total over counted `BOSS_HIT` events),
  capped at 1.0, or 1.5 with the dodge bonus. Best single hit reward observed in 7,708
  episodes: **1.499**.
- A hit taken costs **3.554** on average (`worst_hit` mean), and the 5th percentile is -5.0.
  Worst single step in the run: **-10.215**.
- `GREEDY_PENALTY = 3` fired on **216 of 519** counted boss hits, 0.42 per hit. So 42% of the
  time, landing a hit triples the cost of the next hit taken within 6 steps.
- The `BOSS_HIT` branch requires `player_hp_delta <= 0`. A hit landed in the same step as one
  taken pays **nothing at all**, and is not even counted. 54 of the 469 episodes that did real
  damage recorded zero `BOSS_HIT` events; **9.6% of all damage the agent ever dealt to Margit
  paid exactly zero reward.**

Episode 7707, the second-to-last of the run, is the whole lesson in one episode. 165 steps, one
Heavy Attack pressed. It connected at step 116 for **+0.699**. Two steps later a hit taken cost
**-2.995**, tripled by `GREEDY_PENALTY`. Net for the exchange: **-2.30**. The agent has seen
that exchange 519 times and drawn the correct conclusion.

### This is an over-correction from Round 2, not a new defect

Round 2 (2026-08-13) measured the pre-fix reward as paying **+0.70 to trade a hit for a hit**,
against a real-game rate of 1:8.8 the other way, and set `PLAYER_PARAMETER = 2.0` to put the
reward "at 2:1 against trading", with the stated intent that "trades stay affordable on
purpose". The comment in `config/training.py` predicted a post-change hit value of 0.877
against a doubled cost of 1.75.

**Measured outcome: hit value 0.977, mean cost of a hit taken 3.554, and 3x on top of that
whenever the hit follows an attack.** The realised rate against trading is not 2:1, it is
**3.6:1 before the greedy multiplier and up to 10.9:1 after it**. The prediction was off
because it used the mean cost of a *non-death* hit (0.87) while the penalty the agent actually
meets includes the death floor, the vulnerability exponent, and the multihit multiplier.

Round 2's diagnosis was right and its direction was right. The magnitude removed the only
positive term in the reward, and the run is what happens next.

---

## The second convergence: the ledge

**1,869 of 7,708 episodes (24.2%) ended in a fall death.** Not a bug: a second local optimum,
and the reward pays for it.

| | n | mean reward | mean steps |
|---|---|---|---|
| combat death | 5,839 | **-5.860** | 141.6 |
| fall death | 1,869 | **-5.235** | 92.6 |

**Falling off the rampart is worth 0.63 more reward than fighting, and ends the episode 35%
sooner.** `FALL_DEATH_PENALTY = 5` is a flat charge; a combat death is
`max(player_punish, 3)` on top of every non-fatal hit already taken on the way there, which is
why it averages worse than the flat 5.

This is the classic termination-seeking pathology, and `GAMMA = 0.96` sharpens it. With every
reward in the function negative and no reachable positive terminal, the discounted value of
staying alive is strictly negative, so ending the episode early is the value function's
correct answer. **12.6% of the last 100k steps' episodes scored exactly -5.000**, the clean
signature of walking off the edge without taking a hit first.

The fall rate peaked at 33.1% between 200k and 300k, exactly the window where the reward
plateaued, and settled to 17.6-25.7% for the rest of the run. It never went away.

`corr(net_reward, episode_steps)` is **-0.361** over the whole run and **-0.373** over the last
100k. Round 2 deleted `STEP_PENALTY` specifically because it made this correlation -0.475. The
deletion moved it 0.11 and left the sign intact. **Surviving longer is still scored worse**,
and the cause was never the step penalty alone: it is that hits taken are the only term with
any mass, and living longer means taking more of them.

---

## What did not fail

Worth stating plainly, because it is most of the system and it worked.

**PPO is healthy, and was never the limiter.** `approx_kl` ran 0.0013 to 0.0059 against
`TARGET_KL = 0.03`, so the early stop never fired once in 976 updates. `clip_fraction` fell
0.12 to 0.028. `explained_variance` climbed **-0.075 to 0.647**, which is the best the value
function has ever fitted in this project. `value_loss` fell 1.336 to 0.246. `train/loss` 0.412
to 0.079. Every one of these says the optimiser converged cleanly on a well-modelled objective.

**No policy collapse.** `entropy_loss` went -2.4845 (exactly ln 12, uniform at init) to -1.258,
and the empirical action entropy settled at **1.848 nats, 74% of uniform**, flat from 500k
onward. The policy is sharpened and state-dependent, not degenerate. It has simply concentrated
that entropy on seven of the twelve actions.

**The infrastructure ran 112.7 hours unattended with 354 logged interventions and no crash.**
351 grace returns (4.6% of episodes, the fog-walk retry after a bad area read) and 3 save
restores. The capture pipeline, the `/proc` reads, the AOB scan, the crash recovery and the
tiered checkpointing all held for four and a half days. That is the part of this project that
is finished.

**The in-arena step rate held at 4.70/s across the entire run**, block 0 to block 19, with no
drift. The read-after-hold ordering from `45cfc90` costs nothing measurable.

---

## Five ceilings that no amount of reward tuning removes

These are independent of the reward and would each bind on their own.

### 1. The discount horizon is 5.3 seconds; a kill is 4 minutes

`GAMMA = 0.96` at 0.213 s/step gives an effective horizon of `1/(1-0.96) = 25` steps, or
**5.3 seconds**. `REWARD_BOSS_DEFEATED = 100.0` sits roughly 1,100 steps away, arriving
discounted by `0.96^1100 = 1.5e-20`. **The victory term has never contributed a single bit of
gradient in this project's history**, which `config/training.py` already says. The agent has
literally no representation of the goal. Every signal it has ever received is negative, local,
and about the next five seconds.

### 2. Attacks connect 8% of the time, and the likely cause is the camera

7,160 attacks were pressed over 1M steps. Correcting for the damage that paid no reward, about
**575 connected: an 8.0% conversion rate.** 92% of swings hit air.

*Inferred:* the cause is lock-on. `io/input.py` `walk_to_fog()` presses `BTN_THUMBR` **once**,
3.5 seconds after entering the fog gate, and nothing ever presses it again. There is no lock-on
action in `ACTIONS` and no right-stick control at all: the policy has 12 actions and not one of
them moves the camera. In Elden Ring, lock-on breaks when the target leaves view, and Margit
leaps and repositions constantly. Once it drops mid-fight the agent cannot recover it, the
camera stops tracking, attacks lose their homing, and the left stick stops meaning "strafe" and
starts meaning "turn". The 2:1 bias toward Move Left over Move Right (202,137 against 100,584)
is consistent with a character turning in circles rather than orbiting a locked target.

**The single observation that would settle it:** watch one episode with the dashboard up and
note whether the lock-on reticle is present at step 0 and whether it survives Margit's first
leap. If it is absent at step 0, the initial press is failing on range; if it is present and
gone by step 30, it is breaking and never returning. Both are cheap to see and I cannot see
either from the files.

### 3. Movement is a toggle, and a third of movement presses cancel movement

`execute_action()` treats the four directions and Sprint and Guard as toggles: pressing
"Move Left" holds the stick fully left until "Move Left" is pressed again. The same action id
means "start moving" or "stop moving" depending on hidden state.

Reconstructing the stick from the last 5 episodes of `step_records.csv` (**608 steps only, a
small sample, so treat the exact figure as indicative**): **145 of 399 movement presses, 36.3%,
released a direction rather than engaging one.** The stick sat at neutral for 10.5% of steps,
and 9.4% of dodges were executed with a neutral stick, which in Elden Ring is a backstep rather
than a directional roll. Episode 7704 shows the shape: steps 52 to 60, stick neutral, sprint
held, seven consecutive dodges going nowhere.

The toggle state is in the observation, so this is learnable in principle, and the agent has
partly learned it. It is friction rather than a wall. But it doubles the effective action
semantics and it is the plausible proximate cause of the fall deaths: a direction engaged and
not revisited runs the character off the rampart in about 4 seconds.

### 4. Half the wall clock is spent walking back to the fog gate

| | hours | share |
|---|---|---|
| total wall clock | 112.7 | 100% |
| inside episodes | 59.1 | 52.4% |
| reset overhead | **53.6** | **47.6%** |

**25.1 seconds of reset per episode**, against a mean episode of 27.6 s. Effective throughput
**8,875 env steps/hour**. This is a large improvement on Round 2's measured 4,619 steps/hour
and 72.5% overhead, mostly because episodes are now 60% longer, but it still means **53.6 hours
of this run bought no samples at all.** Halving it is worth more than any hyperparameter in the
file.

### 5. The observation is a squashed greyscale thumbnail

1920x1080 is downscaled to 256x256 greyscale, which is a 42% horizontal squash, and stacked 12
deep. Margit's attack wind-ups, the tells the whole fight is built on, are a few pixels at that
resolution, and colour is discarded. *Inferred, and lower confidence than the other four:* this
caps how well any policy can read an incoming attack, which is what dodge timing needs. It is
listed last because the reward makes it moot for now, and because it is the most expensive to
change.

---

## Are more steps needed?

**No. More steps at this configuration make the result worse, and the arithmetic is not close.**

More steps help when the gradient points at the goal and the sample budget is short. Here the
gradient points away from the goal. It has pointed away for 750,000 steps. Running to 2M would
buy another 85 hours of the attack rate asymptoting to zero from 0.03%.

The scale of the gap, using the run's own numbers:

- Mean bar removed per landed hit: **3.29%**. **A kill needs 30.4 landed hits.**
- The player dies after **3.04** of Margit's hits.
- So a kill needs a taken-to-landed ratio of about **1:10**. The run achieved **40.8:1.**
  **The gap is a factor of 400.**
- Take the best hit rate the agent ever had, block 0 before the collapse: 0.00567 hits/step.
  Thirty landed hits at that rate needs **5,366 steps in a single episode**, against the 95-step
  episodes it was actually producing. **56x longer, without dying.**
- As a Poisson tail at that rate and that episode length: `P(>=31 hits in one episode) =
  1.1e-16`, one episode in **9.0e15**. The run produced 7,708 episodes in 113 hours.

Even freezing the policy at its most aggressive moment in history and running forever, a kill
does not happen. The problem is not the sample budget. **1M steps was more than enough to prove
the objective is wrong, and that is the result this run delivered.**

---

## Proposals

Unticked, unbuilt, ordered by measured leverage. These belong to Teo to prune, and several
contradict each other on purpose.

**A. Pay for damage as a positive terminal, not a per-hit trickle.** The only thing the agent
has never received is a reward for winning. Replace or supplement the flat per-hit payment with
a per-episode term proportional to `1 - final_boss_hp`, delivered at termination where the
discount cannot erase it, or raise `GAMMA` toward 0.99 so the horizon reaches past one exchange.
Cost: a survival-only policy stops being optimal, and the agent will start trading again, which
is what Round 2 was trying to stop. That is the point, and the correct response is to tune the
ratio rather than to remove the term.

**B. Cut `GREEDY_PENALTY` from 3 to 1.5 or delete it.** It fires on 42% of landed hits and it
is the single term that makes a successful attack net-negative. Round 2 measured it as "the only
term opposing the trade, which is why it lost". It won. Cost: the agent will eat more
post-attack punishes; that is a survivable cost while the attack rate is 0.03%.

**C. Pay for a hit landed in a trade.** The `player_hp_delta <= 0` gate silently voided 9.6% of
all damage the agent ever dealt. Whatever the trade should be priced at, zero is not a price,
it is an invisible event. Cost: none that I can see; this looks like an unintended consequence
of the branch structure rather than a decision.

**D. Make the fall strictly worse than fighting.** At -5.235 against -5.860 the ledge is
currently the better deal. `FALL_DEATH_PENALTY` needs to exceed the worst realistic combat
death, not sit under it. Cost: a flat large penalty adds variance; it also does not fix the
underlying "all rewards negative, so terminate early" structure, which is A's job.

**E. Add lock-on to the action space, or re-press it on a timer.** Cheapest version: press
`BTN_THUMBR` in `reset()` and again whenever `n` steps pass with the boss bar not visible. Full
version: a 13th action. Settle ceiling 2 first with the one observation named above, because if
lock-on is holding fine then this buys nothing.

**F. Make movement momentary instead of a toggle.** Direction persists for one step, or for a
short hold, then releases. Removes the 36% of presses that cancel movement and probably most of
the fall deaths. Cost: the agent must press a direction every step to keep moving, which spends
its action budget on locomotion; this is a real trade and may be worse.

**G. Attack the reset overhead.** 53.6 hours. If the fog walk can be replaced by a save-state
reload or the walk route shortened, it is worth more sample throughput than every other item
here combined.

**H. Do not raise `TOTAL_TIMESTEPS` until at least A or B has landed.** Nothing in this run
suggests the budget is the constraint, and everything in it suggests the objective is.

---

## Verified against inferred

**Measured from the run files, no game required:** every number in the trajectory table; the
negative-EV natural experiment; the reward-ranking inversion; the fall-death economics; all
PPO diagnostics; the wall-clock split; the 8.0% attack conversion; the 9.6% of damage paid
nothing; the toggle-press figures (from 608 steps, small sample, indicative only).

**Inferred, needs the game running:** that lock-on is breaking mid-fight and that this causes
the low attack conversion; that camera-relative movement without lock-on causes the fall
deaths; that the 256x256 greyscale observation limits reading attack wind-ups. The lock-on
question has a cheap decisive test and is named above.

**Not measurable from this run at all:** whether any policy at this observation resolution and
this step rate can beat Margit. The run never produced a policy that tried, so the question has
not been asked yet.
