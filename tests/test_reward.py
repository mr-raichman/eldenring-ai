"""
test_reward.py - unit tests for the pure reward function.

compute_reward has no I/O, so it can be tested directly. Each test isolates
one branch of the reward logic with hand-built history buffers.
"""

import pytest

from eldenring_ai import config
from eldenring_ai.io import input
from eldenring_ai.rl.reward import compute_reward

HISTORY = config.HISTORY_LENGTH
DODGE_ID = input.ACTIONS["Dodge"].action_id  # index checked as step[8] in reward.py


def neutral(**overrides):
    """Baseline arguments for a quiet step: no damage either side, full stamina,
    empty history. Override individual values per test."""
    args = dict(
        player_hp=1.0, prev_player_hp=1.0,
        boss_hp=1.0, prev_boss_hp=1.0,
        stamina=1.0, prev_stamina=1.0,
        st_history=[1.0] * HISTORY,
        boss_hp_history=[0] * HISTORY,
        hp_history=[1.0] * HISTORY,
        action_history=[[0.0] * input.N_ACTIONS for _ in range(HISTORY)],
    )
    args.update(overrides)
    return args


def test_idle_step_applies_step_penalty():
    boss_reward, player_punish, reward, events = compute_reward(**neutral())
    assert boss_reward == 0.0
    assert player_punish == 0.0
    assert reward == -config.STEP_PENALTY
    assert "STEP_PENALTY" in events


def test_boss_hit_rewards_positively():
    boss_reward, player_punish, reward, _ = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0)
    )
    # Full stamina -> deduction factor 1.0 -> reward == BOSS_PARAMETER.
    assert boss_reward == pytest.approx(config.BOSS_PARAMETER)
    assert player_punish == 0.0
    assert reward > 0


def test_dodge_before_boss_hit_gets_bonus():
    actions = [[0.0] * input.N_ACTIONS for _ in range(HISTORY)]
    actions[-1][DODGE_ID] = 1.0
    boss_reward, _, _, events = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=actions)
    )
    assert boss_reward == pytest.approx(config.BOSS_PARAMETER * config.DODGE_REWARD)
    assert "DODGE_REWARD" in events


def test_taking_damage_punishes():
    # A single HP drop within the last 6 frames -> hit_count == 1.
    hp_hist = [1.0] * HISTORY
    hp_hist[-1] = 0.9
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.9, prev_player_hp=1.0, hp_history=hp_hist)
    )
    assert player_punish == pytest.approx(0.1)  # 0.1 * vuln(1.0) * PLAYER_PARAMETER
    assert reward < 0
    assert any(e.startswith("HIT_TAKEN") for e in events)


def test_death_floors_at_death_penalty():
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.0, prev_player_hp=1.0)
    )
    assert player_punish == pytest.approx(config.DEATH_PENALTY)
    assert reward == pytest.approx(-config.DEATH_PENALTY)
    assert "DEATH" in events


def test_zero_stamina_is_punished():
    _, _, reward, events = compute_reward(**neutral(stamina=0.0))
    assert reward == -config.LOW_STAMINA_PUNISH
    assert "ZERO STAMINA" in events
