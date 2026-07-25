# Elden Ring AI

A reinforcement-learning agent that learns to defeat **Margit, the Fell Omen** in *Elden Ring*
by playing the real, unmodified game - no game API, no emulator, no mods.

The game has no scripting interface, so the agent is wired to it the same way a person is: it
watches the screen, reads a handful of numbers out of the running process, and presses buttons
on a controller. Everything else is learned.

This is a learning project, built to get hands-on with RL, computer vision, and the messy
reality of driving software that was never meant to be automated.

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

Three channels connect the agent to the game:

- **Vision** - `wf-recorder` streams the Wayland output into a `v4l2loopback` device, OpenCV
  reads frames from it, and they are downscaled to a stack of 12 grayscale 256x256 frames.
  The boss HP bar is read from pixels, since it is not reliably reachable in memory.
- **State** - player HP, stamina, readiness, and the current area ID are read straight out of
  the game process via `/proc/<pid>/mem`, by walking a pointer chain from a `WorldChrMan`
  base address found with an AOB (array-of-bytes) signature scan.
- **Control** - a virtual Xbox 360 gamepad created with evdev `UInput`. The game cannot tell
  it from real hardware.

**Observation** is a Gymnasium `Dict` space: the frame stack plus 24-step histories of the
agent's own actions, its HP, the boss's HP, and its stamina. Giving the policy its recent
history matters because a single frame cannot express "I just committed to a heavy attack".

**Actions** are 12 discrete choices (move, sprint, guard, jump, dodge, heal, light attack,
heavy attack, no-op), 6 of which are held toggles rather than taps.

**Reward** is shaped from damage dealt and taken, with a step penalty to discourage passivity,
a stamina penalty for whiffing into exhaustion, a bonus for dodging immediately before landing
a hit, and penalties for greedy attacking and for taking multiple hits in a row. Fall deaths
and combat deaths are scored separately - falling off the arena is a different mistake from
losing a fight.

**Episodes** run from entering the fog gate to death or victory. The environment handles the
whole loop unattended: it walks the character from the grace to the fog gate, confirms it is
actually inside the arena before starting, and on death waits out the respawn and walks back.
If the game crashes it relaunches it through Steam, re-scans for the pointer, and resumes.

## Status

Working end to end and trains unattended for long runs. This is an active experiment, not a
solved benchmark - treat the reward shaping and hyperparameters as things still being tuned
rather than a recipe known to converge.

## Prerequisites

This is the honest part: the project is built against one specific Linux setup and is **not
portable as-is**. Adapting it to another machine means changing config, not just installing it.

- **OS**: Arch Linux (or similar) with a **Wayland** session. Developed on Hyprland.
- **Capture**: [`wf-recorder`](https://github.com/ammen99/wf-recorder) plus a
  **v4l2loopback** device (default `/dev/video0`).
- **Game**: *Elden Ring* installed and launchable via **Steam** (app id `1245620`), running on
  the configured Wayland output (default `HDMI-A-1`).
- **GPU**: a CUDA-capable GPU. PyTorch is pinned to the `+cu128` build.
- **Python**: 3.12, provisioned automatically by uv (see `.python-version`).
- **[uv](https://docs.astral.sh/uv/)**: manages the environment and dependencies.
- A **save file** parked at the grace before Margit. The agent starts every episode from there.

You will need to adjust at least these to match your machine, all in `eldenring_ai/config/`:

| What | Where |
| --- | --- |
| Wayland output name, capture device | `vision.py` |
| Boss/player HP-bar pixel regions | `vision.py` |
| Memory offsets and AOB signature | `offsets.py` |
| Walk-to-fog route timings | `runtime.py` |

The memory offsets in particular were reverse-engineered against one game build. A patch can
invalidate them.

## Setup

```bash
uv sync
```

That creates `.venv/`, provisions Python 3.12, and installs everything including the CUDA
build of PyTorch from the lockfile.

A plain pip workflow also works:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Load the loopback device before capturing (the capture layer will also attempt this itself):

```bash
sudo modprobe v4l2loopback devices=1 card_label=capture exclusive_caps=1
```

## Running

All commands run from the repository root. `uv run` executes inside `.venv` without activating
it.

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

Start the game first, or let the trainer launch it. Training auto-recovers from crashes and
checkpoints periodically to `models/`.

### What you get while it runs

A live dashboard redraws in place at each episode end, showing reward composition, event
rates, and PPO metrics.

Each run writes to its own folder under `data/runs/<timestamp>/`:

| File | Contents |
| --- | --- |
| `episode_records.jsonl` | one row per episode: reward composition, event rates, derived quality measures, PPO snapshot |
| `step_records.csv` | per-step reward and the reason for it, for the most recent episodes |
| `events.log` | recovery events, aborts, arena-confirmation failures |

Resuming a run continues the latest folder rather than starting a new one. TensorBoard logs go
to `logs/`:

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

Two structural rules keep this honest, if you want to extend it:

- `rl/environment.py` is the only module that imports across `io/`, `rl/`, and `ui/`. The
  `io/` modules know nothing about RL or rewards.
- Every tunable number lives in `config/`. No magic numbers in logic, including timeouts and
  the hand-measured timings of the scripted movement sequences.

## Caveats

- **Memory offsets are game-version-specific.** The `WorldChrMan` AOB signature and the
  pointer-chain offsets in `eldenring_ai/config/offsets.py` were reverse-engineered for one
  game build. A patch can invalidate them and they will need re-scanning.
- **The static pointer is cached** in `data/world_chr_man_ptr.cache` so it is not re-scanned
  on every launch. Delete it after a game update.
- **Reading another process's memory needs permission.** Depending on your
  `kernel.yama.ptrace_scope` setting this may require elevated privileges.
- **This drives your actual mouse, keyboard focus, and game.** It is not sandboxed. Do not run
  it on a save you care about - back it up first.
- The agent plays a specific character build from a specific save state. Different gear,
  levels, or a different starting grace will need re-tuning.

## License

No license file yet. Until one is added, no usage rights are granted - open an issue if you
want to use this for something.
