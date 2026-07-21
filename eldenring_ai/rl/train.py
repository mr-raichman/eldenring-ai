"""
train.py - PPO setup and the training entry point: builds the environment and
policy, wires the checkpoint / stats / stop-on-victory callbacks, resumes from
the latest checkpoint, and runs the learn loop.
"""

import glob
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

import torch.nn as nn

from eldenring_ai import config
from eldenring_ai.config import paths
from eldenring_ai.ui import shared_stats
from eldenring_ai.rl.features_extractor import EldenRingExtractor
from eldenring_ai.rl.environment import EldenRingEnv, PERSIST_FILE

POLICY_KWARGS = dict(
    features_extractor_class=EldenRingExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=256),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=nn.ReLU,
)

# Filename prefix for all saved checkpoints (margit_ppo_<steps>_steps.zip, _final, _victory).
CHECKPOINT_PREFIX = "margit_ppo_"

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


class StopOnVictoryCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        if getattr(env, "_stop_requested", False):
            print(f"\nMargit defeated - saving model to {self.save_path}.zip")
            self.model.save(self.save_path)
            return False
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


def train():
    print("═══════ STARTING TRAINING ═══════\n")

    os.makedirs(str(paths.MODELS_DIR), exist_ok=True)
    os.makedirs(str(paths.LOGS_DIR),   exist_ok=True)

    latest = find_latest_checkpoint()

    if latest is None:
        if os.path.exists(PERSIST_FILE):
            os.remove(PERSIST_FILE)

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
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            tensorboard_log=str(paths.LOGS_DIR),
            device="cuda",
        )

    logger = configure(str(paths.LOGS_DIR), format_strings=["tensorboard"])
    model.set_logger(logger)

    checkpoint_callback = TieredCheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ_MINI,
        keep_freq=config.CHECKPOINT_FREQ,
        save_path=str(paths.MODELS_DIR),
        name_prefix=CHECKPOINT_PREFIX,
    )

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[
            checkpoint_callback,
            StatsLoggerCallback(),
            StopOnVictoryCallback(
                save_path=os.path.join(str(paths.MODELS_DIR), f"{CHECKPOINT_PREFIX}victory"),
            ),
        ],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_path = os.path.join(str(paths.MODELS_DIR), f"{CHECKPOINT_PREFIX}final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}.zip")
    env.close()


if __name__ == "__main__":
    train()
