"""
reward.py
"""

from eldenring_ai import config


def compute_reward(
    player_hp, prev_player_hp,
    boss_hp,   prev_boss_hp,
    stamina,   prev_stamina,
    st_history, boss_hp_history,
    hp_history, action_history
):
    boss_hp_delta   = min(max(0.0, prev_boss_hp - boss_hp), 1.0)
    player_hp_delta = prev_player_hp - player_hp

    events = []

    vulnerability = (2.0 - prev_player_hp)**config.PLAYER_VULNERABILITY_EXP

    # PLAYER HIT
    if player_hp_delta > 0:
        player_punish = player_hp_delta * vulnerability * config.PLAYER_PARAMETER
        events.append(f"HIT_TAKEN(vuln={vulnerability:.2f})")

        # PLAYER HIT AFTER ATTACK
        if sum(list(boss_hp_history)[-5:]) > 0:
            player_punish *= config.GREEDY_PENALTY
            events.append("GREEDY_PENALTY")

        # PLAYER MULTIPLE HIT
        recent_hp = list(hp_history)[-6:]
        hit_count = sum(
            1 for i in range(1, len(recent_hp)) if recent_hp[i - 1] - recent_hp[i] > 0
        )
        player_punish *= hit_count
        events.append(f"MULTIPLE_HIT_PENALTY(x{hit_count})")

        # PLAYER DEATH
        if player_hp <= 0:
            player_punish = max(player_punish * config.DEATH_PENALTY, config.DEATH_PENALTY)
            events.append("DEATH")
    else:
        player_punish = 0.0

    # BOSS HIT
    if player_hp_delta <= 0 and boss_hp_delta > 0:
        stamina_deduction = 1 - (1 - prev_stamina)**config.STAMINA_EXP
        boss_reward = config.BOSS_PARAMETER * stamina_deduction
        events.append(f"BOSS_HIT(deduction={stamina_deduction:.2f})")

        if any(step[8] == 1.0 for step in list(action_history)[-5:]):
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
