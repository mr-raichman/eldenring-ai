"""
test_episode_log.py - unit tests for EpisodeRecorder (pure, no game).

Feeds synthetic step sequences and checks that the reward composition reconstructs
the net reward, events are counted by prefix, derived measures are correct, and the
JSONL/CSV writers produce the expected shape and rolling window.
"""

import csv
import json

import pytest

from eldenring_ai import config
from eldenring_ai.config import paths
from eldenring_ai.ui.episode_log import EpisodeRecorder
from eldenring_ai.ui.metrics import EVENT_CATEGORIES, REWARD_COMPONENTS


def make_recorder(tmp_path, monkeypatch, k=3):
    monkeypatch.setattr(config, "STEP_RECORD_EPISODES", k)
    monkeypatch.setattr(paths, "EPISODE_RECORDS", tmp_path / "episodes.jsonl")
    monkeypatch.setattr(paths, "STEP_RECORDS", tmp_path / "steps.csv")
    return EpisodeRecorder()


def context(episode=1, **kw):
    ctx = dict(
        episode=episode, training_time_s=10.0, total_timesteps=100,
        outcome="timeout", duration_s=5.0, steps_per_s=2.0,
        final_boss_hp=0.5, actions={"Dodge": 1}, ppo={"value_loss": 0.3},
    )
    ctx.update(kw)
    return ctx


def _feed_mixed(rec):
    """A boss hit, a combo hit taken, an idle step-penalty, a zero-stamina step, a kill."""
    rec.record_step(0, "Light Attack", 1.0, 0.9, 0.8, 1.0, 0.8, 2.0, 0.0, 2.0, ["BOSS_HIT(deduction=1.00)"])
    rec.record_step(1, "No Action", 0.7, 0.9, 0.7, 1.0, 0.7, 0.0, 1.5, -1.5, ["HIT_TAKEN(vuln=1.00)", "MULTIPLE_HIT_PENALTY(x2)"])
    rec.record_step(2, "Move Left", 0.7, 0.9, 0.6, 0.7, 0.6, 0.0, 0.0, -config.STEP_PENALTY, ["STEP_PENALTY"])
    rec.record_step(3, "No Action", 0.7, 0.9, 0.0, 0.7, 0.0, 0.0, 0.0, -config.LOW_STAMINA_PUNISH, ["ZERO STAMINA"])
    rec.record_step(4, "Light Attack", 0.7, 0.0, 0.9, 0.7, 0.9, 2.0, 0.0, 102.0, ["BOSS_HIT(deduction=1.00)"], defeat_bonus=100.0)


def test_composition_reconstructs_net(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch)
    _feed_mixed(rec)
    comp = rec.composition
    reconstructed = sum(comp[c.key] for c in REWARD_COMPONENTS if c.key != "net")
    assert comp["net"] == pytest.approx(reconstructed)
    assert comp["net"] == pytest.approx(sum(rec.step_rewards))
    assert comp["boss_reward"] == pytest.approx(4.0)
    assert comp["player_punish"] == pytest.approx(-1.5)
    assert comp["defeat_bonus"] == pytest.approx(100.0)


def test_event_counts_by_prefix(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch)
    _feed_mixed(rec)
    assert rec.event_counts["boss_hits"] == 2
    assert rec.event_counts["hits_taken"] == 1
    assert rec.event_counts["multihit"] == 1
    assert rec.event_counts["step_penalty"] == 1
    assert rec.event_counts["zero_stamina"] == 1
    assert rec.event_counts["combat_death"] == 0
    assert rec.event_counts["fall_death"] == 0


def test_fall_vs_combat_death_not_confused(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch)
    rec.record_step(0, "Move Left", 0.0, 0.9, 0.8, 0.8, 0.8, 0.0, 5.0, -5.0, ["HIT_TAKEN(vuln=1.44)", "FALL_DEATH"])
    rec.record_step(1, "No Action", 0.0, 0.9, 0.5, 0.3, 0.5, 0.0, 4.3, -4.3, ["HIT_TAKEN(vuln=2.89)", "DEATH"])
    assert rec.event_counts["fall_death"] == 1
    assert rec.event_counts["combat_death"] == 1


def test_derived_measures(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch)
    _feed_mixed(rec)
    d = rec.derived_values()
    assert d["best_hit"] == pytest.approx(2.0)
    assert d["worst_hit"] == pytest.approx(-1.5)
    assert d["stamina_at_boss_hit_mean"] == pytest.approx((0.8 + 0.9) / 2)
    assert d["hp_at_hit_taken_mean"] == pytest.approx(1.0)
    assert d["hits_taken_per_boss_hit"] == pytest.approx(1 / 2)


def test_derived_measures_empty_are_none(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch)
    rec.record_step(0, "No Action", 1.0, 1.0, 0.5, 1.0, 0.5, 0.0, 0.0, -config.STEP_PENALTY, ["STEP_PENALTY"])
    d = rec.derived_values()
    assert d["stamina_at_boss_hit_mean"] is None
    assert d["hits_taken_per_boss_hit"] is None


def test_jsonl_row_shape(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch)
    _feed_mixed(rec)
    rec.write_episode(context(episode=7, outcome="kill"))
    lines = paths.EPISODE_RECORDS.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["episode"] == 7
    assert row["outcome"] == "kill"
    assert row["net_reward"] == pytest.approx(rec.composition["net"])
    for c in REWARD_COMPONENTS:
        assert c.key in row
    for e in EVENT_CATEGORIES:
        assert f"n_{e.key}" in row
    assert "actions" in row and "ppo" in row


def test_jsonl_handles_numpy_values(tmp_path, monkeypatch):
    # SB3 logs PPO stats as numpy float32; the record write must not choke on them.
    import numpy as np
    rec = make_recorder(tmp_path, monkeypatch)
    _feed_mixed(rec)
    ppo = {"explained_variance": np.float32(0.62), "value_loss": np.float64(0.3)}
    rec.write_episode(context(episode=1, ppo=ppo))
    row = json.loads(paths.EPISODE_RECORDS.read_text().splitlines()[0])
    assert isinstance(row["ppo"]["explained_variance"], float)
    assert row["ppo"]["explained_variance"] == pytest.approx(0.62, abs=1e-4)


def test_step_csv_keeps_rolling_window(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, monkeypatch, k=3)
    for i in range(1, 6):
        rec.reset()
        rec.record_step(0, "No Action", 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, [])
        rec.write_episode(context(episode=i))

    assert len(paths.EPISODE_RECORDS.read_text().splitlines()) == 5

    with open(paths.STEP_RECORDS) as f:
        rows = list(csv.DictReader(f))
    episodes = sorted({int(r["episode"]) for r in rows})
    assert episodes == [3, 4, 5]
