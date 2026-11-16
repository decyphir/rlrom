"""Opt-in episode metrics from the step info dict (cfg_specs.episode_info_keys).

Fast and backend-agnostic: a tiny fake env exposes outcome keys in ``info`` so
no simulator is needed. Other case studies must be unaffected when the key list
is absent.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlrom.wrappers.spec_wrapper import SpecWrapper


class _OutcomeEnv(gym.Env):
    """Steps until done; info carries a boolean flag and a growing counter."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, steps=5):
        self.steps = steps
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.steps
        info = {"arrived": done, "progress": float(self.t)}
        return (np.zeros(2, dtype=np.float32), 1.0, done, False, info)


def _run(info_keys):
    cfg = {"cfg_specs": {}}
    if info_keys is not None:
        cfg["cfg_specs"]["episode_info_keys"] = info_keys
    env = SpecWrapper(_OutcomeEnv(steps=5), cfg)
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        obs, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
    res, res_all_ep, _, _ = env.eval_specs_episode()
    env.close()
    return res, res_all_ep


def test_info_keys_aggregate_terminal_value():
    res, res_all_ep = _run(["arrived", "progress"])
    # terminal step: arrived True, progress 5 -> boolean flag averages to a rate
    assert res_all_ep["basics"]["mean_arrived"] == 1.0
    assert res_all_ep["basics"]["mean_progress"] == 5.0
    assert np.allclose(res["progress"], [5.0])


def test_no_info_keys_is_inert():
    _, res_all_ep = _run(None)
    assert "mean_arrived" not in res_all_ep["basics"]
    # the pre-existing basics are untouched
    assert set(res_all_ep["basics"]) == {"mean_ep_len", "mean_ep_rew", "completed"}
