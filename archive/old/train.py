import os
import glob
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
sys.path.insert(0, os.path.dirname(__file__))

from environment import EldenRingEnv

MODELS_DIR = "models"
LOGS_DIR   = "logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


def find_latest_model():
    """
    Look for the most recently saved checkpoint in MODELS_DIR.
    Returns the path to the .zip file, or None if no models exist.

    We prefer timestamped checkpoints over 'final' saves because
    checkpoints tell us exactly how many steps were trained.
    """
    pattern = os.path.join(MODELS_DIR, "margit_ppo_*_steps.zip")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        # Fall back to the final save if no checkpoints exist
        final = os.path.join(MODELS_DIR, "margit_ppo_final.zip")
        if os.path.exists(final):
            return final
        return None

    # Sort by the step number embedded in the filename
    # e.g. margit_ppo_10000_steps.zip → 10000
    def extract_steps(path):
        name = os.path.basename(path)          # margit_ppo_10000_steps.zip
        parts = name.replace(".zip", "").split("_")
        try:
            return int(parts[-2])              # the number before "steps"
        except (ValueError, IndexError):
            return 0

    checkpoints.sort(key=extract_steps)
    return checkpoints[-1]   # return the one with the highest step count


def train():
    print("=== Training ===\n")

    env = EldenRingEnv()

    latest = find_latest_model()

    if latest is not None:
        print(f"Found existing model: {latest}")
        print("Loading and continuing training...\n")
        model = PPO.load(
            latest,
            env=env,                   # re-attach the environment
            device="cuda",
            tensorboard_log=LOGS_DIR,
        )
    else:
        print("No existing model found. Starting fresh.\n")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            device="cuda",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODELS_DIR,
        name_prefix="margit_ppo",
        verbose=1,
    )

    print("Starting training.\n")
    time.sleep(1)

    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,  # keeps the step counter cumulative
    )

    model.save(os.path.join(MODELS_DIR, "margit_ppo_final"))
    print("\nTraining complete. Model saved.")
    env.close()


if __name__ == "__main__":
    train()
