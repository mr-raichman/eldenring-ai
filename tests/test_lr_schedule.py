"""
test_lr_schedule.py - the learning-rate schedule decays on absolute timesteps
and never falls below the floor.
"""

from types import SimpleNamespace

from eldenring_ai import config
from eldenring_ai.rl.train import _lr_schedule


def _rate_at(timesteps, progress_remaining=1.0):
    model = SimpleNamespace(num_timesteps=timesteps)
    return _lr_schedule(model)(progress_remaining)


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
