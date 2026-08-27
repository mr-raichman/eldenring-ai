# Decisions

**Append only.** A decision goes in when a cold chat would default the other way. The reason
goes with it, because a decision without one gets remade. Nothing here is rewritten: where a
later decision reverses an earlier one, both stay and the later one says what it reverses.

This file is the index. The measurements behind a row live in `PLAN.md` (the round that took
it) or in the comment beside the constant; they are not repeated here.

**Created 2026-08-27**, backfilled from `PLAN.md` rounds 1-4 and the comments in
`eldenring_ai/config/`. Rows dated before that are reconstructions from those records, not
contemporaneous entries.

---

## Reward shaping

| date | decision | why |
|---|---|---|
| 2026-08-13 | `STEP_PENALTY` deleted | 24.6% of the signal doing two jobs, one wrong: the 24 steps of immunity a hit bought were worth 38% of the hit itself, so most of the attack incentive was the clock switching off. It also priced survival negatively. |
| 2026-08-13 | `PLAYER_PARAMETER` 1 -> 2.0 | The reward paid +0.70 to trade a hit for a hit, against a true rate of 1:8.8 the other way. Intended to put trading at 2:1 against. **Reversed in effect 2026-08-27:** the realised rate was 3.6:1 to 10.9:1 and the agent stopped attacking entirely. The value is kept; the correction was made on `BOSS_PARAMETER`. |
| 2026-08-13 | `DODGE_WINDOW` 10 -> 5 | At a 6.8% dodge share the bonus fired at the rate of a coin flip: variance on the one working signal. **Superseded 2026-08-27.** |
| 2026-08-13 | Death penalties are NOT scaled with `PLAYER_PARAMETER` | Changing two things at once would make the 2:1 measurement unreadable. Deferred with a condition rather than rejected. |
| 2026-08-13 | `REWARD_BOSS_DEFEATED`, `STAMINA_EXP` gate, `LR_DECAY_*` kept although inert | Teo's call. They cost nothing and become live if the horizon, the damage rate or the run length changes. Deleting them loses the knob and the record of why it is set that way. |
| 2026-08-27 | `BOSS_PARAMETER` 1 -> **4.5**, not 5 | Attacking only pays above 4.98; a sequential trade turns profitable above 5.71. 4.5 sits 21% clear of the trading edge rather than 12%, and below the 4.98 floor deliberately because that floor is fitted on untimed attacks by a near-uniform policy and is overstated. Teo's call, on the evidence that theoretically-correct hit rewards have produced spam before. |
| 2026-08-27 | `PLAYER_PARAMETER` stays 2.0; the correction goes on `BOSS_PARAMETER` | The death penalties are absolute and do not scale with it, so lowering it would silently change what a death costs relative to everything else. |
| 2026-08-27 | `GREEDY_PENALTY` stays 3 | It was read as the term that killed attacking. It is not: at `BOSS_PARAMETER` 4.5 it is the only thing keeping a trade at -1.16 while a clean hit pays +4.33. It sets the upper bound on `BOSS_PARAMETER`. |
| 2026-08-27 | **The dodge bonus is kept, retuned to W=3 / R=1.25, not removed** | Teo overruled the proposed removal. The measurement against it (fires on 0.563 of hits against a 0.710 chance rate, i.e. below chance) tests whether it *detects* dodge-into-opening, which it does not. Its job is different: a positive term attached to dodging at all, because dodging otherwise carries only the negative reward of the hits it fails to avoid, and dodge-then-attack is the sequence worth teaching. The evidence against it is also weak on its own terms - 519 hits from a policy that never attacked. Re-measure after the next run. |
| 2026-08-27 | Attack softcap added, shaped `min(1, (N/n)**EXP)` with EXP > 1 | The cost of attacking saturates, so per-step reward is convex in the attack rate and its optimum is always a corner: never attack, or attack as often as stamina allows. Only a concave reward puts it in between. `1/(1+k*n)` was rejected on arithmetic: it flattens rather than turning over and left the optimum at zero attacks. |
| 2026-08-27 | The trade-void gate (`player_hp_delta <= 0`) is left alone for now | It voids 9.6% of all damage dealt, but fixing it alone at `BOSS_PARAMETER` 4.5 moves a same-step trade from -1.83 to +2.50, straight back into the trading policy. It only works together with extending `GREEDY_PENALTY` to same-step trades. Deferred as both edits or neither. |

## RL hyperparameters

| date | decision | why |
|---|---|---|
| 2026-08-13 | `N_STEPS` 2048 -> 1024, `BATCH_SIZE` stays 512 | 2048 bought 39 policy updates in a 17-hour run. This setup does not have the sample budget the usual rollout sizes assume. Lowering the batch alongside it was rejected: two changes at once. |
| 2026-08-27 | `TOTAL_TIMESTEPS` 1M -> **2M** | The analysis document's proposal H said not to raise it until the objective was fixed, on the grounds that the 1M run plateaued at 250k and was never budget-limited. Round 4 fixed the objective, so the condition is met. **Consequence, accepted:** 225 hours of wall clock (9.4 days), and 57% of the run trains at the `LR_MIN` floor, since the schedule reaches 3e-5 at 850,300 steps. |
| 2026-08-27 | `GAMMA` stays 0.96 for this round | Its 5.3 s horizon puts a kill (~1100 steps) permanently out of reach, but raising it is a second variable in a round whose whole purpose is to read one. Recorded as open, not settled. |

## Scope and method

| date | decision | why |
|---|---|---|
| 2026-08-06 | Round 1's build scoped to bugs, sanitisation and structure, explicitly excluding parameters and tuning | Teo's call. It left P4 (reward balance) open, which round 2 then answered with measurements rather than guesses. |
| 2026-08-13 | The `Guard` action is kept | Teo confirms it is useful. A falling action share (7.9% to 2.8%) is not evidence against an action. |
| ongoing | Tier is **solo** | Nobody else runs this. An experiment left half-tuned is an experiment Teo stopped touching, not debt, and not a finding. |
| ongoing | Progress is read from damage per life, not from episode reward | The reward has been anti-correlated with progress in every run so far: the best episode of the 1M run scored in the 6th percentile. |

## Architecture and code

| date | decision | why |
|---|---|---|
| ongoing | Exactly two composition points: `rl/environment.py` and `rl/train.py` | Only those import across `io/`, `rl/` and `ui/`. Everything else is liftable. |
| 2026-08-06 | `reward.py` does NOT import `io/input`; the magic `step[8]` stays | Importing it to name the index would break the layering to remove a number that `tests/test_reward.py` already pins. The same applies to `step[10]` and `step[11]` added in 2026-08-27. |
| 2026-08-06 | No base class or protocol for the metric registries | Five dataclasses, one implementation each, consumed by iteration. |
| 2026-08-06 | `io/memory.py` is not split | Proposed on a graphify cohesion figure. A ratio pointing somewhere is not a target. |
| 2026-08-06 | `ruff` pinned to 0.16.1 with an explicit `["E4","E7","E9","F"]` rule set | An undeclared linter changes its own rules under you: `--isolated` reported 32 findings in untouched code because ruff's defaults had widened. The other 31 findings are not fixed - acting on them means a repo-wide reformat that buries every real change. |
| ongoing | `tests/` holds only pure, game-free unit tests | The recovery loop, reset sequence and capture pipeline fail loudly and need the game. Interactive diagnostics live in `tools/`. |

## Data and artifacts

| date | decision | why |
|---|---|---|
| ongoing | Nothing under `data/`, `models/` or `logs/` is deleted by anything but Teo | Gitignored, real training progress, exists in no other copy. |
| 2026-08-13 | Pre-C1 checkpoints (10k-80k) and their TensorBoard logs deleted | Their value head is fitted to an exchange rate that no longer exists, so resuming from one starts from a value function wrong by construction. Keeping the logs would put two reward functions on one TensorBoard axis. |
| 2026-08-27 | All 101 checkpoints and the 1M run's TensorBoard log deleted | Same reason as 2026-08-13, one round later: every value head is fitted to `BOSS_PARAMETER = 1`, which Round 4 replaced, so resuming from one starts from a value function wrong by construction. Nothing unique lost - the per-episode `train/*` stats live on in `episode_records.jsonl`, and `rollout/ep_rew_mean` is recomputable from `net_reward`. |
| 2026-08-13 | The orphaned 2026-07-23 records were moved into `data/runs/`, not deleted | Round 1 had them down as duplicates. They are not: they are the only copy of a run two days older than the oldest run directory. |
| ongoing | `data/eldenring-save-backup.sl2` has exactly one copy, and that is accepted | 29 MB, the canonical character state. A second copy is deferred with the condition: any near-miss. |
