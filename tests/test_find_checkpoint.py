"""
test_find_checkpoint.py - resume must not pick a checkpoint PPO.load() cannot open.

The case that earned this test: a run killed mid-write left a 0-byte
margit_ppo__1000_steps.zip, which sorted newest and made every later resume fail with
"wasn't a zip-file".
"""

import zipfile

from eldenring_ai.config import paths
from eldenring_ai.rl import train


def _checkpoint(directory, steps, valid=True):
    path = directory / f"{train.CHECKPOINT_PREFIX}_{steps}_steps.zip"
    if valid:
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data", "{}")
    else:
        path.write_bytes(b"")
    return path


def test_picks_the_highest_step_count(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    _checkpoint(tmp_path, 1000)
    newest = _checkpoint(tmp_path, 20000)
    assert train.find_latest_checkpoint() == str(newest)


def test_skips_a_truncated_checkpoint_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    good = _checkpoint(tmp_path, 1000)
    _checkpoint(tmp_path, 2000, valid=False)
    assert train.find_latest_checkpoint() == str(good)


def test_no_readable_checkpoint_reads_as_a_fresh_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    _checkpoint(tmp_path, 1000, valid=False)
    assert train.find_latest_checkpoint() is None


def test_final_checkpoint_is_validated_too(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path)
    (tmp_path / f"{train.CHECKPOINT_PREFIX}final.zip").write_bytes(b"")
    assert train.find_latest_checkpoint() is None
