# Elden Ring AI (G.A.L.E.) by Teo Raichman

I started this project with the only goal in mind of learning Linux, Python, machine learning
and all that it entails. Even though the project is fully capable of training models, I'm still
tweaking parameters and the reward function in search of a Margit defeat, so I'm open to
recommendations to improve it. It's not my main project, so I don't dedicate myself to it every
day, just when I feel like it.

## General description

This is a reinforcement learning agent called GALE with the goal of defeating **Margit, The
Fell Omen** in **Elden Ring**. It is trained using PPO, simulating inputs via a virtual
controller and taking images and some memory readings as inputs.

The idea is to teach it to play as a human would, using only vision. However, due to the
complexity of understanding what each stat bar means in-game and the inability of the model to
have memory, I decided to also implement memory readings and a rolling history of data such as
frames, boss HP, player HP, stamina, actions executed... so that it simulates what a human
would have in mind while playing, as well as making it easier for it to relate reward to action
choice once it starts detecting the patterns in its inputs.

## Philosophy and gaming strategy

### Character

I wanted the most basic possible build, so I chose Vagabond, a sword and shield class, at level
one. The character has not been leveled up, as I want it to really be able to dodge and decide
when to attack, not to win by chance because it's overpowered but to win by pure skill.

After some months of building the project and training, I opted for discarding the shield. Even
though parrying and attack blocking gave good results, it kinda took the focus away from
dodging, which is the coolest ability I wanted the AI to learn. And after training the AI on a
two-handed sword by mistake, I chose to keep it as permanent, consequently lowering the action
space.

### Micro episodes

The reward function is the most complex thing in the project and the one decision that has made
me think the most, and I'm still refining it. My current philosophy about it I call "micro
episodes", and it refers to small time periods (currently of 4.8s) that contain all the
information needed to execute the next action.

This comes from not thinking about defeating Margit at all, but just about dealing damage
without being hit, indefinitely, which is in fact the correct strategy when it comes to Souls
games. It doesn't really matter what you did or what you are going to do, because dodging now
won't directly affect a decision 10s later, as its outcome will already carry all the
information that step needs.

The reward function and the gamma parameter of 0.96, as well as the vision and stats inputs,
all refer to this time period (micro episode). This way, actions are only judged by the current
context, without adding anything about later or previous info that may contaminate the
understanding of the patterns. Choosing heal right before being hit has to be punished, but
choosing heal and being hit 10 seconds later doesn't have any relation.

## How it works

```
   wf-recorder -> v4l2loopback -> OpenCV        /proc/<pid>/mem
              (screen frames)                  (HP, stamina, area)
                     |                                |
                     +--------------+-----------------+
                                    |
                          EldenRingEnv (Gymnasium)
                                    |
                        PPO policy + custom CNN
                                    |
                          evdev UInput virtual pad
                                    |
                              Elden Ring
```

There are three channels connecting the agent to the game, and each one exists because the
game gives you nothing to work with:

- **Vision** - `wf-recorder` streams the Wayland output into a `v4l2loopback` device and
  OpenCV reads the frames from there. I downscale them to a stack of 12 grayscale 256x256
  frames. The boss HP bar I read from pixels, because I never found a reliable way to reach
  it in memory.
- **State** - player HP, stamina, readiness and the current area ID are read straight out of
  the game process through `/proc/<pid>/mem`, walking a pointer chain from a `WorldChrMan`
  base address that I find with an AOB (array-of-bytes) signature scan.
- **Control** - a virtual Xbox 360 gamepad made with evdev `UInput`. The game can't tell it
  apart from real hardware.

The observation is a Gymnasium `Dict` space: the frame stack plus 24-step histories of the
actions it took, its own HP, Margit's HP and its stamina. I give it that recent history
because a single frame can't express something like "I just committed to a heavy attack",
and without it the agent has no way of knowing it is already locked into an animation.

There are 12 discrete actions (move, sprint, guard, jump, dodge, heal, light attack, heavy
attack and doing nothing), 6 of them held toggles instead of taps.

The reward is shaped from the damage it deals and the damage it takes, plus a step penalty so
it doesn't just stand there, a stamina penalty for swinging into exhaustion, a bonus for
dodging right before landing a hit, and penalties for attacking greedily and for eating
several hits in a row. Fall deaths and combat deaths are scored separately, because falling
off the arena is a completely different mistake from losing a fight. Beating Margit is worth
a big one-off bonus, which so far is theoretical.

Episodes run from entering the fog gate until it dies or wins, with no step limit. The
environment handles the whole loop on its own: it walks the character from the grace to the
fog gate, confirms it really is inside the arena before starting, and on death it waits out
the respawn and walks back. If the game crashes it relaunches it through Steam, re-scans for
the pointer and carries on.

## Status

It works end to end and trains unattended for long runs. It's an active experiment though,
not a solved benchmark, so treat the reward shaping and the hyperparameters as things I'm
still tuning rather than a recipe that is known to converge.

## Prerequisites

This is the honest part: I built this against my own machine and it is **not portable as-is**.
Getting it running somewhere else means changing config, not just installing it.

- **OS**: Arch Linux (or similar) with a **Wayland** session. Developed on Hyprland.
- **Capture**: [`wf-recorder`](https://github.com/ammen99/wf-recorder) plus a
  **v4l2loopback** device (default `/dev/video0`).
- **Game**: _Elden Ring_ installed and launchable via **Steam** (app id `1245620`), running on
  the configured Wayland output (default `HDMI-A-1`).
- **GPU**: a CUDA-capable GPU. PyTorch is pinned to the `+cu128` build.
- **Python**: 3.12, provisioned automatically by uv (see `.python-version`).
- **[uv](https://docs.astral.sh/uv/)**: manages the environment and dependencies.
- A **save file** parked at the grace before Margit. The agent starts every episode from there.

At the very least you'll have to adjust these to match your machine, all of them in
`eldenring_ai/config/`:

| What                                | Where        |
| ----------------------------------- | ------------ |
| Wayland output name, capture device | `vision.py`  |
| Boss/player HP-bar pixel regions    | `vision.py`  |
| Memory offsets and AOB signature    | `offsets.py` |
| Walk-to-fog route timings           | `runtime.py` |

The memory offsets are the fragile part: I reverse-engineered them against one game build, so
a patch can invalidate them at any moment.

## Setup

```bash
uv sync
```

That creates `.venv/`, provisions Python 3.12 and installs everything from the lockfile,
including the CUDA build of PyTorch.

If you prefer plain pip, that works too:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Load the loopback device before capturing. The capture layer tries to do it by itself as well,
but it's one less thing to go wrong:

```bash
sudo modprobe v4l2loopback devices=1 card_label=capture exclusive_caps=1
```

## Running

Everything runs from the repository root. `uv run` executes inside `.venv` without you having
to activate it.

```bash
# Train (auto-resumes from the latest checkpoint in models/)
uv run eldenring-train
uv run python -m eldenring_ai.rl.train   # equivalent

# Manual gamepad REPL - test inputs by hand, no training
uv run python tools/controller_repl.py

# Live diagnostics (these need the game running)
uv run python tools/boss_hp.py         # Margit HP
uv run python tools/area_id.py         # area ID / fog-gate check
uv run python tools/stamina.py         # stamina
uv run python tools/capture_frames.py  # dump the AI frame stack

# Unit tests (pure logic, no game required)
uv run pytest
```

You can start the game yourself or just let the trainer launch it. It recovers from crashes on
its own and checkpoints to `models/` every so often, so you can leave it running and come back
to it later.

### What you get while it runs

There's a live dashboard that redraws in place at the end of every episode, with the reward
composition, the event rates and the PPO metrics.

Each run also writes its own folder under `data/runs/<timestamp>/`:

| File                    | Contents                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `episode_records.jsonl` | one row per episode: reward composition, event rates, derived quality measures, PPO snapshot |
| `step_records.csv`      | per-step reward and the reason for it, for the most recent episodes                          |
| `events.log`            | recovery events, aborts, arena-confirmation failures                                         |

Resuming a run continues the latest folder instead of starting a new one, so a crash doesn't
split a training run into pieces. TensorBoard logs go to `logs/`:

```bash
tensorboard --logdir logs
```

## Project layout

```
eldenring_ai/           the importable package
  config/             all tunable constants, split by concern
    training.py         hyperparameters, reward weights, episode/checkpoint limits
    vision.py           frame stack, observation shape, HP-bar regions, devices,
                        capture-pipeline settle delays
    offsets.py          WorldChrMan AOB signature + pointer-chain offsets
    runtime.py          debug toggles, game-launch parameters, recovery timeouts
                        and poll intervals, scripted-sequence calibration timings
    paths.py            all filesystem locations (relocatable, no ~ hardcodes)
  io/                 the three I/O channels
    capture.py          ScreenCapture (wf-recorder -> v4l2 -> OpenCV)
    input.py            virtual gamepad, ACTIONS, menu navigation
    memory.py           GameMemory: AOB scan, /proc reads, boss-HP vision
  rl/                 the learning loop
    environment.py      EldenRingEnv - the orchestrator
    reward.py           reward shaping (pure, unit-tested)
    features_extractor.py  CNN feature extractor
    train.py            PPO setup, callbacks, checkpointing (entry point)
  ui/                 dashboard.py (Rich), metrics.py (metric registry),
                      episode_log.py (per-episode + per-step records), shared_stats.py

tools/                live diagnostics + gamepad REPL (loose scripts, not collected by pytest)
  _bootstrap.py         shared sys.path + display-env setup, imported first by each tool
tests/                automated unit tests (pure logic, no game required)
archive/              superseded and one-off code, kept for reference (not imported)
  old/                  early flat-layout package versions
  reverse_engineering/  one-off memory offset-discovery scripts (hardcoded addresses)
  old_tools/            stale diagnostics not yet ported to the current API
data/  models/  logs/  runtime artifacts and training outputs (gitignored)
```

If you want to extend it, there are two rules I try to keep the project honest with:

- `rl/environment.py` is the only module allowed to import across `io/`, `rl/` and `ui/`. The
  `io/` modules know nothing about RL or rewards.
- Every tunable number lives in `config/`, no magic numbers in the logic. That includes the
  timeouts and the timings of the scripted movement sequences, which I measured by hand
  against the real game.

## Caveats

- **The memory offsets depend on the game version.** The `WorldChrMan` AOB signature and the
  pointer-chain offsets in `eldenring_ai/config/offsets.py` were reverse-engineered against one
  build, so a patch can break them and they'll need re-scanning.
- **The static pointer is cached** in `data/world_chr_man_ptr.cache` so it isn't re-scanned on
  every launch. Delete it after a game update.
- **Reading another process's memory needs permission.** Depending on your
  `kernel.yama.ptrace_scope` this may need elevated privileges.
- **This drives your real game, mouse and keyboard focus.** Nothing here is sandboxed, so don't
  run it on a save you care about. Back it up first.
- It plays one specific build from one specific save state. Different gear, different levels or
  a different starting grace will all need re-tuning.

## License

GNU General Public License v3.0 or later, full text in [`LICENSE`](LICENSE).

Copyright (C) 2026 Teo Raichman.
