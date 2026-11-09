"""Tests for how a config names the environment it runs.

`env_id` (what gymnasium builds) and `env_name` (the name a policy was published
under) are allowed to disagree; these check the default, the override, and the
error message that teaches the difference. Nothing here needs an optional extra.
"""

import sys
import types

import pytest

import rlrom.utils as rlu


def test_env_id_defaults_to_env_name():
    env = rlu.make_env_from_cfg({'env_name': 'CartPole-v1'})
    try:
        assert env.unwrapped.spec.id == 'CartPole-v1'
    finally:
        env.close()


def test_env_id_wins_over_env_name():
    # env_name names the checkpoint, env_id the environment: they may disagree
    env = rlu.make_env_from_cfg({'env_name': 'CartPole-v1', 'env_id': 'Acrobot-v1'})
    try:
        assert env.unwrapped.spec.id == 'Acrobot-v1'
    finally:
        env.close()


def test_env_kwargs_are_forwarded():
    # FrozenLake's default map is 4x4 slippery; a 2x2 map proves the kwargs
    # reached the env constructor rather than being dropped on the way
    env = rlu.make_env_from_cfg({'env_name': 'FrozenLake-v1',
                                 'env_kwargs': {'desc': ['SF', 'FH'], 'is_slippery': False}})
    try:
        assert env.unwrapped.desc.shape == (2, 2)
    finally:
        env.close()


def test_render_mode_is_forwarded():
    env = rlu.make_env_from_cfg({'env_name': 'CartPole-v1'}, render_mode='rgb_array')
    try:
        assert env.render_mode == 'rgb_array'
    finally:
        env.close()


def test_missing_both_keys_raises():
    with pytest.raises(ValueError, match="neither 'env_id' nor 'env_name'"):
        rlu.make_env_from_cfg({})


def test_unbuildable_id_lists_the_family():
    # Hopper-v3 is still in the registry but its entry point refuses to build
    # (moved to gymnasium-robotics); the message has to name the versions that
    # do work, which is the whole point of having env_id.
    with pytest.raises(ValueError) as excinfo:
        rlu.make_env_from_cfg({'env_name': 'Hopper-v3'})
    msg = str(excinfo.value)
    assert "cannot build 'Hopper-v3'" in msg
    assert 'Hopper-v4' in msg
    assert "Set 'env_id'" in msg


def _fake_module(monkeypatch, name, with_factory):
    mod = types.ModuleType(name)
    if with_factory:
        mod.make_env_test = lambda cfg: 'built-by-the-module'
    monkeypatch.setitem(sys.modules, name, mod)
    return name


def test_factory_found_by_convention(monkeypatch):
    # no make_env_test key: the module's function of that name is used
    name = _fake_module(monkeypatch, 'rlrom_fake_env_mod_a', with_factory=True)
    factory = rlu.custom_env_factory({'import_module': name}, 'make_env_test')
    assert factory({}) == 'built-by-the-module'


def test_side_effect_only_module_falls_back_to_default(monkeypatch):
    # a module imported only to register an env class defines no factory, and
    # that must keep working instead of raising AttributeError
    name = _fake_module(monkeypatch, 'rlrom_fake_env_mod_b', with_factory=False)
    assert rlu.custom_env_factory({'import_module': name}, 'make_env_test') is None


def test_no_module_and_no_factory_uses_default():
    assert rlu.custom_env_factory({}, 'make_env_train') is None


def test_named_factory_must_exist():
    with pytest.raises(ValueError, match="has no attribute 'nope'"):
        rlu.custom_env_factory(
            {'import_module': 'rlrom.wrappers', 'make_env_test': 'nope'},
            'make_env_test')


def test_named_factory_without_import_module_raises():
    with pytest.raises(ValueError, match="no 'import_module'"):
        rlu.custom_env_factory({'make_env_test': 'make_env_test'}, 'make_env_test')


def test_action_noise_from_cfg():
    from stable_baselines3.common.noise import NormalActionNoise
    noise = rlu.action_noise_from_cfg({'type': 'normal', 'sigma': 0.2}, 3)
    assert isinstance(noise, NormalActionNoise)
    assert noise._sigma.shape == (3,)
    assert rlu.action_noise_from_cfg(None, 3) is None  # SD3 default off
    with pytest.raises(ValueError, match='unknown action_noise type'):
        rlu.action_noise_from_cfg({'type': 'cauchy'}, 3)
