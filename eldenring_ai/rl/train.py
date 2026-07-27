"""
train.py - PPO setup and the training entry point: builds the environment and
policy, wires the checkpoint / stats / save-on-victory callbacks, resumes from
the latest checkpoint, and runs the learn loop.
"""

import glob
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

import torch.nn as nn

from eldenring_ai import config
from eldenring_ai.config import paths
from eldenring_ai.ui import shared_stats
from eldenring_ai.rl.features_extractor import EldenRingExtractor
from eldenring_ai.rl.environment import EldenRingEnv, PERSIST_FILE
from eldenring_ai.ui.dashboard import training_live

POLICY_KWARGS = dict(
    features_extractor_class=EldenRingExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=256),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=nn.ReLU,
)

# Filename prefix for all saved checkpoints (margit_ppo_<steps>_steps.zip, _final,
# _victory_<n> - one per win, numbered so later wins don't overwrite earlier ones).
CHECKPOINT_PREFIX = "margit_ppo_"


def _lr_schedule(model):
    """Exponential decay on the model's absolute timestep count, floored at LR_MIN.

    SB3 calls the schedule with `progress_remaining`, which it derives from the
    total_timesteps of the current learn() call - and on resume it extends that
    budget by the steps already done, which would step the rate back up every time
    training is restarted. Reading model.num_timesteps instead makes the schedule
    depend only on how far the agent has actually trained.
    """
    def schedule(_progress_remaining):
        decay = config.LR_DECAY_FACTOR ** (model.num_timesteps / config.LR_DECAY_STEPS)
        return max(config.LR_MIN, config.LEARNING_RATE * decay)

    return schedule

class StatsLoggerCallback(BaseCallback):
    def _on_step(self):
        if self.model.logger.name_to_value:
            shared_stats.ppo_stats.update(dict(self.model.logger.name_to_value))
        shared_stats.ppo_stats["total_timesteps"] = self.num_timesteps
        return True

    def _on_rollout_end(self):
        if self.model.logger.name_to_value:
            shared_stats.ppo_stats.update(dict(self.model.logger.name_to_value))
        # Push the env's per-episode metrics into TensorBoard on the timestep axis.
        for tag, value in shared_stats.episode_metrics.items():
            self.logger.record(tag, value)


class SaveOnVictoryCallback(BaseCallback):
    """Checkpoint every win and keep training - the goal is to master Margit, not to
    beat him once. Keyed on the env's kill counter rather than a flag, because the
    env has already reset (restoring the save and relaunching the game) by the time
    this callback runs again."""

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self._kills_seen = None

    def _on_step(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        kills = getattr(env, "total_kills", 0)
        if self._kills_seen is None:      # first step: adopt the resumed count
            self._kills_seen = kills
        elif kills > self._kills_seen:
            self._kills_seen = kills
            path = f"{self.save_path}_{kills}"
            print(f"\nMargit defeated ({kills}) - saving model to {path}.zip")
            self.model.save(path)
        return True

class TieredCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int = config.CHECKPOINT_FREQ_MINI, keep_freq: int = config.CHECKPOINT_FREQ, save_path: str = str(paths.MODELS_DIR), name_prefix: str = CHECKPOINT_PREFIX, **kwargs):
        assert keep_freq % save_freq == 0, "keep_freq must be a multiple of save_freq"
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, **kwargs)
        self.keep_freq = keep_freq

    def _on_step(self) -> bool:
        result = super()._on_step()

        steps = self.num_timesteps
        if steps % self.keep_freq == 0:
            for i in range(1, self.keep_freq // self.save_freq):
                stale_step = steps - (i * self.save_freq)
                pattern = os.path.join(self.save_path, f"{self.name_prefix}_{stale_step}_steps.zip")
                for f in glob.glob(pattern):
                    os.remove(f)

        return result

def find_latest_checkpoint():
    pattern     = os.path.join(str(paths.MODELS_DIR), f"{CHECKPOINT_PREFIX}*_steps.zip")
    checkpoints = glob.glob(pattern)

    if checkpoints:
        def _extract_steps(path):
            parts = os.path.basename(path).replace(".zip", "").split("_")
            try:
                return int(parts[-2])
            except (ValueError, IndexError):
                return 0
        checkpoints.sort(key=_extract_steps)
        return checkpoints[-1]

    final_path = os.path.join(str(paths.MODELS_DIR), f"{CHECKPOINT_PREFIX}final.zip")
    return final_path if os.path.exists(final_path) else None


def _latest_run_dir():
    """The most recent run directory under data/runs/, or None."""
    if not paths.RUNS_DIR.exists():
        return None
    dirs = [d for d in paths.RUNS_DIR.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def train():
    print("═══════ STARTING TRAINING ═══════\n")

    os.makedirs(str(paths.MODELS_DIR), exist_ok=True)
    os.makedirs(str(paths.LOGS_DIR),   exist_ok=True)

    latest = find_latest_checkpoint()

    if latest is None:
        # Fresh run: clean stats and start a new timestamped run directory.
        if os.path.exists(PERSIST_FILE):
            os.remove(PERSIST_FILE)
        run_dir = paths.RUNS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        # Resume: keep writing into the most recent run directory.
        run_dir = _latest_run_dir() or (paths.RUNS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    paths.use_run_dir(run_dir)

    env = EldenRingEnv()

    if latest is not None:
        print(f"Resuming from checkpoint: {latest}")
        model = PPO.load(
            latest,
            env=env,
            device="cuda",
            verbose=1,
            tensorboard_log=str(paths.LOGS_DIR),
        )
        model.learning_rate = config.LEARNING_RATE
        model.gamma         = config.GAMMA
        model.n_epochs      = config.N_EPOCHS
        model.batch_size    = config.BATCH_SIZE
        model.ent_coef      = config.ENT_COEF
        model.target_kl     = config.TARGET_KL
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            ent_coef=config.ENT_COEF,
            target_kl=config.TARGET_KL,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            tensorboard_log=str(paths.LOGS_DIR),
            device="cuda",
        )

    # Installed after construction (and after load, which restores the pickled one)
    # because the schedule closes over the model to read its timestep count.
    model.lr_schedule = _lr_schedule(model)

    logger = configure(str(paths.LOGS_DIR), format_strings=["tensorboard"])
    model.set_logger(logger)

    checkpoint_callback = TieredCheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ_MINI,
        keep_freq=config.CHECKPOINT_FREQ,
        save_path=str(paths.MODELS_DIR),
        name_prefix=CHECKPOINT_PREFIX,
    )

    # progress_bar is off: the live dashboard is our single Rich Live display, and two
    # would conflict on one stdout. Progress/ETA now live in the dashboard header.
    with training_live():
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=[
                checkpoint_callback,
                StatsLoggerCallback(),
                SaveOnVictoryCallback(
                    save_path=os.path.join(str(paths.MODELS_DIR), f"{CHECKPOINT_PREFIX}victory"),
                ),
            ],
            progress_bar=False,
            reset_num_timesteps=False,
        )

    final_path = os.path.join(str(paths.MODELS_DIR), f"{CHECKPOINT_PREFIX}final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}.zip")
    env.close()


if __name__ == "__main__":
    train()
