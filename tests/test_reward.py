"""
test_reward.py - unit tests for the pure reward function.

compute_reward has no I/O, so it can be tested directly. Each test isolates
one branch of the reward logic with hand-built history buffers. The env appends
the current step's HP to hp_history AFTER calling compute_reward, so these tests
mirror that: for an isolated hit, hp_history holds no drop yet and hit_count is 1.
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


def test_isolated_hit_is_punished():
    # No prior drop in history -> the current hit is the only one -> hit_count == 1.
    # At full prev_hp the vulnerability factor is 1.0, so punish == d * PLAYER_PARAMETER.
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=1.0)
    )
    assert player_punish == pytest.approx(0.3 * config.PLAYER_PARAMETER)
    assert reward < 0
    assert "MULTIPLE_HIT_PENALTY(x1)" in events
    assert any(e.startswith("HIT_TAKEN") for e in events)


def test_combo_escalates():
    # A prior HP drop already in history + the current hit -> hit_count == 2.
    hp_hist = [1.0] * HISTORY
    hp_hist[-1] = 0.85  # one drop: 1.0 -> 0.85
    _, punish_combo, _, events = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=0.85, hp_history=hp_hist)
    )
    assert "MULTIPLE_HIT_PENALTY(x2)" in events
    # Same step as an isolated hit but doubled by hit_count.
    _, punish_isolated, _, _ = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=0.85)
    )
    assert punish_combo == pytest.approx(2 * punish_isolated)


def test_trade_is_net_negative():
    # Boss was damaged within the greedy window, then the player takes a hit.
    boss_hist = [0] * HISTORY
    boss_hist[-1] = 1
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=1.0, boss_hp_history=boss_hist)
    )
    assert "GREEDY_PENALTY" in events
    assert player_punish == pytest.approx(0.3 * config.PLAYER_PARAMETER * config.GREEDY_PENALTY)
    # A hit dealt is never worth a hit received: the trade outweighs a full boss hit.
    assert reward < -config.BOSS_PARAMETER


def test_fall_death_uses_fixed_penalty():
    # Losing more than the threshold in one step reads as a fall, not combat.
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.0, prev_player_hp=0.8)
    )
    assert player_punish == pytest.approx(config.FALL_DEATH_PENALTY)
    assert reward == pytest.approx(-config.FALL_DEATH_PENALTY)
    assert "FALL_DEATH" in events


def test_combat_death_uses_hit_logic_with_floor():
    # A killing blow from low HP: below the fall threshold -> combat death.
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.0, prev_player_hp=0.3)
    )
    vuln = (2.0 - 0.3) ** config.PLAYER_VULNERABILITY_EXP
    expected = max(0.3 * vuln * config.PLAYER_PARAMETER, config.COMBAT_DEATH_PENALTY)
    assert player_punish == pytest.approx(expected)
    assert player_punish >= config.COMBAT_DEATH_PENALTY
    assert "DEATH" in events


def test_zero_stamina_is_punished():
    _, _, reward, events = compute_reward(**neutral(stamina=0.0))
    assert reward == -config.LOW_STAMINA_PUNISH
    assert "ZERO STAMINA" in events
