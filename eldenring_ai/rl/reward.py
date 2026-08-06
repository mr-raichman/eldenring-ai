"""
reward.py - the per-step reward, shaped from the damage dealt and taken.

Pure: no I/O, no globals beyond `config`, so every branch is unit-testable. Exactly
one of the branches below decides the returned reward, which is what lets
EpisodeRecorder reconstruct the episode's composition from the event strings.
"""

from eldenring_ai import config


def compute_reward(
    player_hp, prev_player_hp,
    boss_hp,   prev_boss_hp,
    stamina,   prev_stamina,
    st_history, boss_hp_history,
    hp_history, action_history
):
    # The boss reward is flat per hit, not proportional to the damage, so anything
    # that reads as a drop gets paid in full. Below BOSS_HIT_MIN_DELTA that is a
    # median-smoothing tail or a single spoiled pixel column, never a sword landing.
    boss_hp_delta   = prev_boss_hp - boss_hp
    if boss_hp_delta < config.BOSS_HIT_MIN_DELTA:
        boss_hp_delta = 0.0
    player_hp_delta = prev_player_hp - player_hp

    events = []

    vulnerability = (config.PLAYER_VULNERABILITY_BASE - prev_player_hp)**config.PLAYER_VULNERABILITY_EXP

    # PLAYER HIT
    if player_hp_delta > 0:
        player_punish = player_hp_delta * vulnerability * config.PLAYER_PARAMETER
        events.append(f"HIT_TAKEN(vuln={vulnerability:.2f})")

        # PLAYER HIT AFTER ATTACK
        if sum(list(boss_hp_history)[-config.GREEDY_WINDOW:]) > 0:
            player_punish *= config.GREEDY_PENALTY
            events.append("GREEDY_PENALTY")

        # PLAYER MULTIPLE HIT (current hit counts, plus recent hits in the window).
        recent_hp = list(hp_history)[-config.MULTIHIT_WINDOW:]
        hit_count = 1 + sum(
            1 for i in range(1, len(recent_hp)) if recent_hp[i - 1] - recent_hp[i] > 0
        )
        player_punish *= hit_count
        if hit_count > 1:
            events.append(f"MULTIPLE_HIT_PENALTY(x{hit_count})")

        # PLAYER DEATH: distinguish a fall (loses more HP than Margit can deal in one
        # step) from a combat death, which is left to the hit logic above.
        if player_hp <= 0:
            if player_hp_delta > config.FALL_DEATH_HP_THRESHOLD:
                player_punish = config.FALL_DEATH_PENALTY
                events.append("FALL_DEATH")
            else:
                player_punish = max(player_punish, config.MIN_COMBAT_DEATH_PENALTY)
                events.append("DEATH")
    else:
        player_punish = 0.0

    # BOSS HIT
    if player_hp_delta <= 0 and boss_hp_delta > 0:
        stamina_deduction = 1 - (1 - prev_stamina)**config.STAMINA_EXP
        boss_reward = config.BOSS_PARAMETER * stamina_deduction
        events.append(f"BOSS_HIT(deduction={stamina_deduction:.2f})")

        if any(step[8] == 1.0 for step in list(action_history)[-config.DODGE_WINDOW:]):
            boss_reward *= config.DODGE_REWARD
            events.append("DODGE_REWARD")
    else:
        boss_reward = 0.0

    # STAMINA AND INACTIVITY PUNISH
    if boss_reward == 0.0 and player_punish == 0.0:
        if stamina == 0:
            reward = -config.LOW_STAMINA_PUNISH
            events.append("ZERO STAMINA")
        elif sum(boss_hp_history) == 0:
            reward = -config.STEP_PENALTY
            events.append("STEP_PENALTY")
        else:
            reward = 0
    else:
        reward = boss_reward - player_punish


    return boss_reward, player_punish, reward, events
