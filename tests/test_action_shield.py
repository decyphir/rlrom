"""Continuous-action shield hook (cfg_specs.action_shield).

Fast and simulator-free: a tiny env with a Box action space, a monitor formula
that turns true after the first step, and a module-level shield function as the
hook target. Covers activation gating, action replacement, info reporting,
disable_actions, and config validation.
"""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rlrom.wrappers.spec_wrapper import SpecWrapper


class _RecEnv(gym.Env):
    """Records the actions it actually receives; obs = step counter."""

    observation_space = spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(self, n=3):
        self.n = n
        self.t = 0
        self.actions = []

    def reset(self, seed=None, options=None):
        self.t = 0
        self.actions = []
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        self.actions.append(np.asarray(action, dtype=float).tolist())
        done = self.t >= self.n
        return np.array([float(self.t)], dtype=np.float32), 0.0, done, False, {}


def _replace(wrapper, action):
    return np.array([0.5, -0.5], dtype=np.float32)


def _identity(wrapper, action):
    return action


def _cfg(activation='act', function='_replace', disable=False):
    specs = {
        'specs': 'signal x\nact := x[t] > 0.5\nnever := x[t] > 100',
        'obs_names': {'x': 'obs[0]'},
        'monitor_formulas': {
            'act': {'semantics': 'bool'},
            'never': {'semantics': 'bool'},
        },
        'action_shield': {'activation': activation, 'function': function},
    }
    if disable:
        specs['disable_actions'] = True
    return {'import_module': __name__, 'cfg_specs': specs}


def _run(cfg, n=3):
    env = _RecEnv(n=n)
    w = SpecWrapper(env, cfg)
    w.reset()
    infos = []
    a = np.array([0.1, 0.2], dtype=np.float32)
    for _ in range(n):
        _, _, _, _, info = w.step(a)
        infos.append(info)
    return env, w, infos


def test_shield_replaces_action_after_activation():
    env, w, infos = _run(_cfg())
    # step 1: formula not yet known at shield time -> passthrough
    assert 'shield' not in infos[0]
    assert np.allclose(env.actions[0], [0.1, 0.2])
    # step 2 (formula turned true after step 1): replaced
    assert infos[1]['shield']['machine'] == 'action_shield'
    assert infos[1]['shield']['reason'] == 'action_shield'
    assert np.allclose(env.actions[1], [0.5, -0.5])
    assert np.allclose(infos[1]['shield']['shielded'], [0.5, -0.5])
    # per-step flag recorded, off then on
    assert bool(w.episode['shield_active'][0]) is False
    assert bool(w.episode['shield_active'][1]) is True


def test_shield_inactive_when_formula_false():
    env, w, infos = _run(_cfg(activation='never'))
    assert all('shield' not in i for i in infos)
    assert np.allclose(env.actions[0], [0.1, 0.2])
    assert np.allclose(env.actions[1], [0.1, 0.2])
    assert all(bool(x) is False for x in w.episode['shield_active'])


def test_disable_actions_turns_shield_off():
    env, w, infos = _run(_cfg(disable=True))
    assert all('shield' not in i for i in infos)
    assert np.allclose(env.actions[1], [0.1, 0.2])


def test_identity_shield_reports_no_override():
    # when the shield returns the action unchanged, no override is reported
    _, _, infos = _run(_cfg(function='_identity'))
    assert all('shield' not in i for i in infos)


def test_action_shield_requires_known_activation():
    with pytest.raises(ValueError, match='not a'):
        SpecWrapper(_RecEnv(), _cfg(activation='missing'))


def test_action_shield_requires_known_function():
    with pytest.raises(ValueError, match='no attribute'):
        SpecWrapper(_RecEnv(), _cfg(function='not_a_function'))


def test_action_shield_requires_import_module():
    cfg = _cfg()
    del cfg['import_module']
    with pytest.raises(ValueError, match='import_module'):
        SpecWrapper(_RecEnv(), cfg)
