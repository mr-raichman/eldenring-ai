# Running it

This is the honest part: I built this against my own machine and it is **not portable as-is**.
Getting it running somewhere else means changing config, not just installing it.

## Requirements

- **OS**: Arch Linux (or similar) with a **Wayland** session. Developed on Hyprland.
- **Capture**: [`wf-recorder`](https://github.com/ammen99/wf-recorder) plus a **v4l2loopback**
  device (default `/dev/video0`).
- **Game**: _Elden Ring_ installed and launchable through **Steam** (app id `1245620`), running
  on the configured Wayland output (default `HDMI-A-1`).
- **GPU**: a CUDA-capable GPU. PyTorch is pinned to the `+cu128` build.
- **Python**: 3.12, provisioned automatically by uv (see `.python-version`).
- **[uv](https://docs.astral.sh/uv/)**: manages the environment and dependencies.
- A **save file** parked at the grace before Margit. Every episode starts from there.

## What you have to change

All of it lives in `eldenring_ai/config/`:

| What                                | Where        |
| ----------------------------------- | ------------ |
| Wayland output name, capture device | `vision.py`  |
| Boss and player HP-bar pixel regions| `vision.py`  |
| Memory offsets and AOB signature    | `offsets.py` |
| Walk-to-fog route timings           | `runtime.py` |
| Elden Ring save-file location       | `paths.py`   |

The memory offsets are the fragile part: I reverse-engineered them against one game build, so a
patch can invalidate them at any moment.

The walk-to-fog timings are the second fragile part. They are wall-clock durations I measured by
hand against the real game (`FOG_SPRINT_LEG1_DURATION`, `FOG_TURN2_DURATION` and the rest in
`runtime.py`). Anything that makes the game run slower makes the character fall short of the fog
gate, and the environment will abort the episode rather than train outside the arena.

## Setup

```bash
uv sync
```

That creates `.venv/`, provisions Python 3.12 and installs everything from the lockfile,
including the CUDA build of PyTorch.

Load the loopback device before capturing. The capture layer tries to do it itself as well, but
it is one less thing to go wrong:

```bash
sudo modprobe v4l2loopback devices=1 card_label=capture exclusive_caps=1
```

## Training

Everything runs from the repository root. `uv run` executes inside `.venv` without you having to
activate it.

```bash
# Train. Auto-resumes from the latest checkpoint in models/
uv run eldenring-train

# Equivalent
uv run python -m eldenring_ai.rl.train
```

You can start the game yourself or let the trainer launch it. It recovers from crashes on its
own and checkpoints to `models/` periodically, so you can leave it running and come back later.

Resuming continues the latest run folder instead of starting a new one, so a crash does not split
a training run into pieces.

## Diagnostics

These need the game running and something streaming to the capture device.

```bash
# Margit's HP, through the same pixel detector training uses
uv run python tools/boss_hp.py

# Area ID, which is how the environment knows it is inside the arena
uv run python tools/area_id.py

# Stamina, read from process memory
uv run python tools/stamina.py

# Dump the AI's frame stack to data/captures/: greyscale views, amplified
# diffs, the colour frame and the boss HP-bar crop
uv run python tools/capture_frames.py

# Drive the virtual gamepad by hand, no training involved
uv run python tools/controller_repl.py

# Kill the game, restore the backup save, relaunch. The same restore training does
uv run python tools/restore_save.py
```

The tools open the capture device directly and do not start `wf-recorder` themselves, so run one
first if training is not already up:

```bash
wf-recorder -o HDMI-A-1 -f /dev/video0 --muxer=v4l2 --codec=rawvideo -x bgr24
```

## Recording footage while training

`ScreenCapture` kills the `wf-recorder` processes that write to the capture device before
launching its own, because a crash can leave one holding the node. It matches on the device, so a
recorder writing to a file is left alone and survives a recovery mid-take:

```bash
wf-recorder -o HDMI-A-1 -c h264_nvenc -f ~/fight.mkv
```

Use the hardware encoder (`h264_nvenc`) rather than the default x264, and keep the file out of
the repository. Expect roughly 1.4 GB per hour.

The consequence of matching on the device: if you change `V4L2_DEVICE` in `config/vision.py`
while a recorder is still alive on the old node, nothing will free it and you have to kill it by
hand.

## Checks

```bash
# Unit tests, pure logic, no game required
uv run pytest

# Linter. The rule set is pinned in pyproject.toml so it does not drift between ruff releases
uv run ruff check eldenring_ai tools tests
```

## What a run writes

Each run gets its own folder under `data/runs/<timestamp>/`:

| File                    | Contents                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `config.json`           | every tunable that produced this run, so its numbers can be attributed months later          |
| `episode_records.jsonl` | one row per episode: reward composition, event rates, derived quality measures, PPO snapshot |
| `step_records.csv`      | per-step reward and the reason for it, for the most recent episodes                          |
| `events.log`            | recovery events, aborts, arena-confirmation failures                                         |

`config.json` is written once per run and never read back, so `eldenring_ai/config/` stays the
only place a value is set. Resuming into a run whose config you edited in the meantime keeps the
original and writes a second, timestamped one beside it.

TensorBoard logs go to `logs/`:

```bash
tensorboard --logdir logs
```

`wf-recorder`'s own stderr goes to `data/wf-recorder.log`, which is the first place to look when
capture will not start.

## Layout

```
eldenring_ai/           the importable package
  config/             all tunable constants, split by concern
    training.py         hyperparameters, reward weights, episode/checkpoint limits
    vision.py           frame stack, observation shape, HP-bar regions and the
                        thresholds that read them, devices, capture settle delays
    offsets.py          WorldChrMan AOB signature + pointer-chain offsets
    runtime.py          debug toggles, step and button-press durations, game-launch
                        parameters, recovery timeouts and poll intervals,
                        scripted-sequence calibration timings
    paths.py            filesystem locations (relocatable except the game's save path)
  io/                 everything that touches the game
    capture.py          ScreenCapture (wf-recorder -> v4l2 -> OpenCV)
    input.py            virtual gamepad, ACTIONS, menu navigation
    memory.py           GameMemory: AOB scan, /proc reads, boss-HP vision
    save.py             restoring the backup save over the game's live save
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
docs/                 this file and the README's media
archive/              superseded and one-off code, kept for reference (not imported)
  old/                  early flat-layout package versions
  reverse_engineering/  one-off memory offset-discovery scripts (hardcoded addresses)
  old_tools/            stale diagnostics not yet ported to the current API
data/  models/  logs/  runtime artifacts and training outputs (gitignored)
```

Two rules keep the layout honest:

- Only the two composition points, `rl/environment.py` and `rl/train.py`, import across `io/`,
  `rl/` and `ui/`. Everything else stays in its own layer: `io/` knows nothing about RL or
  rewards, and `ui/` knows nothing about `io/`.
- Every tunable number lives in `config/`, no magic numbers in the logic. That includes the
  timeouts and the timings of the scripted movement sequences.

## What will bite you

- **The memory offsets depend on the game version.** The `WorldChrMan` AOB signature and the
  pointer-chain offsets in `eldenring_ai/config/offsets.py` were reverse-engineered against one
  build, so a patch can break them and they will need re-scanning. A wrong offset reads as
  plausible garbage, not as an error.
- **The resolved pointer is cached** in `data/world_chr_man_ptr.cache` so it is not re-scanned on
  every launch. Delete it after a game update, before anything else.
- **Reading another process's memory needs permission.** Depending on your
  `kernel.yama.ptrace_scope` this may need elevated privileges.
- **This drives your real game, mouse and keyboard focus.** Nothing here is sandboxed, so do not
  run it on a save you care about. Back it up first.
- **Training overwrites your live save.** After a crash or a victory it copies
  `data/eldenring-save-backup.sl2` over the save at `paths.SAVE_LIVE`, `.bak` included. The copy
  only ever goes in that direction, so the backup is the canonical state and you have to redo it
  by hand whenever you change the character.
- **Turn Steam Cloud off for Elden Ring**, or it can put the old save back underneath a running
  session.
- **Steam notifications land near the boss HP bar.** The detector reads pixels at `y` 867 to 871,
  `x` 466 to 1463. A notification over that region makes the boss-health reading plausible and
  wrong, and even when it misses, it is noise inside the 256x256 observation. Disable the Steam
  overlay before a long run.
- **It plays one specific build from one specific save state.** Different gear, different levels
  or a different starting grace will all need re-tuning.
