"""
test_lr_schedule.py - the learning-rate schedule decays on absolute timesteps,
never falls below the floor, and survives being pickled into a checkpoint.
"""

import ctypes
from types import SimpleNamespace

import cloudpickle

from eldenring_ai import config
from eldenring_ai.rl.train import TimestepDecaySchedule


def _rate_at(timesteps, progress_remaining=1.0):
    model = SimpleNamespace(num_timesteps=timesteps)
    return TimestepDecaySchedule(model)(progress_remaining)


def test_starts_at_the_configured_rate():
    assert _rate_at(0) == config.LEARNING_RATE


def test_decays_by_the_factor_every_decay_step():
    expected = config.LEARNING_RATE * config.LR_DECAY_FACTOR
    assert _rate_at(config.LR_DECAY_STEPS) == expected


def test_never_falls_below_the_floor():
    assert _rate_at(10_000_000) == config.LR_MIN


def test_ignores_progress_remaining():
    """SB3 derives progress_remaining from the current learn() budget, which grows
    on every resume. The rate must depend only on how far the agent has trained."""
    assert _rate_at(200_000, progress_remaining=1.0) == _rate_at(200_000, progress_remaining=0.1)


def test_decreases_monotonically():
    rates = [_rate_at(t) for t in range(0, 1_000_000, 50_000)]
    assert all(b <= a for a, b in zip(rates, rates[1:]))


def test_pickles_without_dragging_in_the_model():
    """SB3 cloudpickles lr_schedule into every checkpoint. Reaching the model from
    here reaches its env, whose evdev gamepad holds ctypes function pointers that
    cannot be pickled - which crashed every save until the schedule dropped it."""
    model = SimpleNamespace(num_timesteps=0, gamepad=ctypes.pointer(ctypes.c_int(0)))
    cloudpickle.dumps(TimestepDecaySchedule(model))
