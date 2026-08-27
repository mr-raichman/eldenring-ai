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
LIGHT_ID = input.ACTIONS["Light Attack"].action_id   # step[10] in reward.py
HEAVY_ID = input.ACTIONS["Heavy Attack"].action_id   # step[11] in reward.py


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


def test_idle_step_is_free():
    # The per-step penalty was removed: nothing happening costs nothing. Standing still
    # is now punished only by the opportunity cost of not landing a hit.
    boss_reward, player_punish, reward, events = compute_reward(**neutral())
    assert boss_reward == 0.0
    assert player_punish == 0.0
    assert reward == 0
    assert events == []


def test_boss_hit_rewards_positively():
    boss_reward, player_punish, reward, _ = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0)
    )
    # Full stamina -> deduction factor 1.0 -> reward == BOSS_PARAMETER.
    assert boss_reward == pytest.approx(config.BOSS_PARAMETER)
    assert player_punish == 0.0
    assert reward > 0


def test_sub_threshold_boss_drop_pays_nothing():
    # Episode 213 step 126: a 0.001 drop two steps after a real hit, the median's tail
    # settling. The reward is flat per hit, so that artifact was paid +0.971 - 12% of
    # that episode's positive reward, for nothing happening.
    boss_reward, _, reward, events = compute_reward(
        **neutral(boss_hp=0.999, prev_boss_hp=1.0)
    )
    assert boss_reward == 0.0
    assert reward == 0
    assert not any(e.startswith("BOSS_HIT") for e in events)


def test_dodge_before_boss_hit_gets_bonus():
    actions = [[0.0] * input.N_ACTIONS for _ in range(HISTORY)]
    actions[-1][DODGE_ID] = 1.0
    boss_reward, _, _, events = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=actions)
    )
    assert boss_reward == pytest.approx(config.BOSS_PARAMETER * config.DODGE_REWARD)
    assert "DODGE_REWARD" in events


def _attacking_history(n_attacks, action_id=None):
    """A history holding n_attacks presses, the rest idle."""
    actions = [[0.0] * input.N_ACTIONS for _ in range(HISTORY)]
    for i in range(n_attacks):
        actions[i][action_id if action_id is not None else LIGHT_ID] = 1.0
    return actions


def test_attack_softcap_inert_at_and_below_the_threshold():
    # The cap is a ceiling, not a per-attack tax: the Nth attack is still paid in full.
    for n in (0, config.ATTACK_SOFTCAP_N):
        boss_reward, _, _, events = compute_reward(
            **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=_attacking_history(n))
        )
        assert boss_reward == pytest.approx(config.BOSS_PARAMETER), f"n={n}"
        assert not any(e.startswith("ATTACK_SOFTCAP") for e in events), f"n={n}"


def test_attack_softcap_bites_above_the_threshold():
    n = config.ATTACK_SOFTCAP_N * 2
    boss_reward, _, _, events = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=_attacking_history(n))
    )
    expected = config.BOSS_PARAMETER * (config.ATTACK_SOFTCAP_N / n) ** config.ATTACK_SOFTCAP_EXP
    assert boss_reward == pytest.approx(expected)
    assert any(e.startswith("ATTACK_SOFTCAP") for e in events)


def test_attack_softcap_makes_total_hit_reward_turn_over():
    """The whole point of EXP > 1: total reward must FALL past the cap, not flatten.

    1/(1+k*n) saturates and leaves the optimum at a corner. This is the test that
    distinguishes the shape that works from the one that does not.
    """
    def total(n):
        boss_reward, _, _, _ = compute_reward(
            **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=_attacking_history(n))
        )
        return n * boss_reward       # hits landed scale with attacks pressed

    peak = total(config.ATTACK_SOFTCAP_N)
    assert total(config.ATTACK_SOFTCAP_N * 2) < peak
    assert total(config.ATTACK_SOFTCAP_N * 3) < total(config.ATTACK_SOFTCAP_N * 2)


def test_attack_softcap_counts_heavy_attacks_too():
    n = config.ATTACK_SOFTCAP_N * 2
    light, _, _, _ = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=_attacking_history(n, LIGHT_ID))
    )
    heavy, _, _, _ = compute_reward(
        **neutral(boss_hp=0.9, prev_boss_hp=1.0, action_history=_attacking_history(n, HEAVY_ID))
    )
    assert light == pytest.approx(heavy)


def test_isolated_hit_is_punished():
    # No prior drop in history -> the current hit is the only one -> hit_count == 1.
    # At full prev_hp the vulnerability factor is 1.0, so punish == d * PLAYER_PARAMETER.
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=1.0)
    )
    assert player_punish == pytest.approx(0.3 * config.PLAYER_PARAMETER)
    assert reward < 0
    assert not any(e.startswith("MULTIPLE_HIT_PENALTY") for e in events)
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


def test_greedy_penalty_multiplies_punish():
    # Boss was damaged within the greedy window, then the player takes a hit.
    boss_hist = [0] * HISTORY
    boss_hist[-1] = 1
    _, player_punish, reward, events = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=1.0, boss_hp_history=boss_hist)
    )
    assert "GREEDY_PENALTY" in events
    assert player_punish == pytest.approx(0.3 * config.PLAYER_PARAMETER * config.GREEDY_PENALTY)
    assert reward < 0
    # Same hit outside the greedy window costs GREEDY_PENALTY times less.
    _, punish_clean, _, events_clean = compute_reward(
        **neutral(player_hp=0.7, prev_player_hp=1.0)
    )
    assert "GREEDY_PENALTY" not in events_clean
    assert player_punish == pytest.approx(config.GREEDY_PENALTY * punish_clean)


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
    vuln = (config.PLAYER_VULNERABILITY_BASE - 0.3) ** config.PLAYER_VULNERABILITY_EXP
    expected = max(0.3 * vuln * config.PLAYER_PARAMETER, config.MIN_COMBAT_DEATH_PENALTY)
    assert player_punish == pytest.approx(expected)
    assert player_punish >= config.MIN_COMBAT_DEATH_PENALTY
    assert "DEATH" in events


def test_zero_stamina_is_punished():
    _, _, reward, events = compute_reward(**neutral(stamina=0.0))
    assert reward == -config.LOW_STAMINA_PUNISH
    assert "ZERO STAMINA" in events
