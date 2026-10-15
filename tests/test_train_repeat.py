"""Tests for cfg_train.repeat_training in rlr train / rlrom_train.

A fake RLTrainer stands in for the real one so no environment, model or
learning step is needed.
"""
import pytest

import rlrom.rlrom_train as rtr


class FakeTrainer:
    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.trained = False
        FakeTrainer.instances.append(self)

    def train(self):
        self.trained = True


@pytest.fixture(autouse=True)
def fake_trainer(monkeypatch):
    FakeTrainer.instances = []
    monkeypatch.setattr(rtr, 'RLTrainer', FakeTrainer)
    yield FakeTrainer


def test_repeat_training_default(fake_trainer):
    rtr.main_train({'model_name': 'm', 'cfg_train': {'total_timesteps': 10}})
    assert len(fake_trainer.instances) == 1
    assert fake_trainer.instances[0].trained


def test_repeat_training_5(fake_trainer):
    rtr.main_train({'model_name': 'm', 'cfg_train': {'repeat_training': 5}})
    # one fresh trainer per run (no model reuse between runs)
    assert len(fake_trainer.instances) == 5
    assert len({id(t) for t in fake_trainer.instances}) == 5
    assert all(t.trained for t in fake_trainer.instances)


def test_repeat_training_string_value(fake_trainer):
    # YAML may deliver numbers as strings
    rtr.main_train({'model_name': 'm', 'cfg_train': {'repeat_training': '3'}})
    assert len(fake_trainer.instances) == 3


def test_repeat_training_zero_raises(fake_trainer):
    with pytest.raises(ValueError):
        rtr.main_train({'model_name': 'm', 'cfg_train': {'repeat_training': 0}})
    assert fake_trainer.instances == []


def test_repeat_training_no_cfg_train(fake_trainer):
    rtr.main_train({'model_name': 'm'})
    assert len(fake_trainer.instances) == 1
