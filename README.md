# Elden Ring AI

A reinforcement-learning agent that learns to defeat **Margit, the Fell Omen** in
*Elden Ring* by playing the real, unmodified game - no game API, no emulator. Built from
scratch to learn RL, computer vision, and game AI.

The agent perceives the game through three channels and drives it like a human would:

- **Vision** - the screen is captured live and downscaled into a stack of grayscale frames
  that form the policy's observation.
- **State** - player HP, stamina, readiness, and the current area are read directly out of
  the game process's memory via `/proc`.
- **Control** - actions are issued through a virtual gamepad (evdev `UInput`), exactly as a
  real controller would.

A PPO policy (Stable-Baselines3) with a custom CNN feature extractor ties these together
inside a Gymnasium environment.

## Prerequisites

This project is built for a specific Linux setup:

- **OS**: Arch Linux (or similar), **Wayland** session (Hyprland).
- **Capture**: [`wf-recorder`](https://github.com/ammen99/wf-recorder) piping the game
  output into a **v4l2loopback** device (`/dev/video0`).
- **Game**: *Elden Ring* installed and launchable via **Steam** (app id `1245620`),
  running on the configured Wayland output (default `HDMI-A-1`).
- **GPU**: CUDA-capable GPU for training (PyTorch `+cu128` build).
- **Python**: 3.12 (uv provisions it automatically; see `.python-version`).
- **[uv](https://docs.astral.sh/uv/)**: manages the environment and dependencies.

Configure the Wayland output, capture device, and HP-bar pixel regions in
`eldenring_ai/config/vision.py` to match your display.

## Setup

The project is managed with uv. This creates `.venv/`, provisions Python 3.12, and
installs all dependencies (including the CUDA build of PyTorch) from the lockfile:

```bash
uv sync
```

A plain `pip` workflow is also supported via `requirements.txt`:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Load the loopback device before capturing (the capture layer will also attempt this):

```bash
sudo modprobe v4l2loopback devices=1 card_label=capture exclusive_caps=1
```

## Running

All commands run from the repository root. With uv, `uv run` executes inside `.venv`
without needing to activate it.

```bash
# Train (auto-resumes from the latest checkpoint in models/)
uv run eldenring-train                 # console entry point
uv run python -m eldenring_ai.rl.train # equivalent

# Manual gamepad REPL (test inputs by hand)
uv run python tools/controller_repl.py

# Live diagnostics (require the game running)
uv run python tools/boss_hp.py         # Margit HP
uv run python tools/area_id.py         # area ID / fog-gate check
uv run python tools/stamina.py         # stamina
uv run python tools/capture_frames.py  # dump the AI frame stack

# Unit tests
uv run pytest
```

A per-episode dashboard prints when each episode ends. Records are written under
`data/`: `episode_records.jsonl` (one row per episode - reward composition, event
rates, derived quality measures, PPO snapshot) and `step_records.csv` (per-step
reward and reason for the most recent episodes).

Training auto-recovers on crashes and checkpoints periodically to `models/`; TensorBoard
logs go to `logs/`. Watch the live graphs (reward, Margit HP, steps, PPO metrics) with:

```bash
tensorboard --logdir logs
```

## Project layout

```
eldenring_ai/            the importable package
  config/             all tunable constants, split by concern
    training.py         hyperparameters, reward weights, episode/checkpoint limits
    vision.py           frame stack, observation shape, HP-bar regions, devices
    offsets.py          WorldChrMan AOB signature + pointer-chain offsets
    runtime.py          debug toggles, game-launch parameters
    paths.py            all filesystem locations (relocatable, no ~ hardcodes)
  io/                 the three I/O channels
    capture.py          ScreenCapture (wf-recorder -> v4l2 -> OpenCV)
    input.py            virtual gamepad, ACTIONS, menu navigation
    memory.py           GameMemory: AOB scan, /proc reads, boss-HP vision
  rl/                 the learning loop
    environment.py      EldenRingEnv - the orchestrator
    reward.py           reward shaping
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
docs/                 development notes, save-backup instructions
data/  models/  logs/  runtime artifacts and training outputs (gitignored)
```

## Notes & caveats

- **Memory offsets are game-version-specific.** The WorldChrMan AOB signature and the
  pointer-chain offsets in `eldenring_ai/config/offsets.py` were reverse-engineered for a
  specific game build; a game patch can invalidate them and they will need re-scanning.
  See `docs/notes.md` for the raw measurements (HP-bar pixels, area ids, action durations).
- **The static pointer is cached** in `data/world_chr_man_ptr.cache` to avoid re-scanning on
  every launch; delete it if the game updates.
- `docs/backup.md` records where to restore the boss-fight save (`data/eldenring-save-backup.sl2`).
