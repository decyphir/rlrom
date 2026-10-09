"""End-to-end smoke tests: short CartPole episodes with a random policy,
with and without STL specifications (reward shaping).

Fast: no model download, no training.
"""
import gymnasium as gym
import numpy as np
import pytest

from rlrom.testers import RLTester

CARTPOLE_CFG = {
    "env_name": "CartPole-v1",
    "model_name": "random",
    "cfg_test": {"num_ep": 1, "num_steps": 50},
}

STL_CFG = {
    "env_name": "CartPole-v1",
    "model_name": "random",
    "cfg_test": {"num_ep": 1, "num_steps": 50},
    "cfg_specs": {
        # Note: avoid bare negative literals (stlrom grammar limitation,
        # e.g. `x[t] < -0.2` fails to parse).
        "specs": "signal x, theta, reward\ncart_left := x[t] < 0.5 and x[t] > 0.1",
        "obs_names": {"x": "obs[0]", "theta": "obs[2]"},
        "reward_formulas": {
            "cart_left": {"online": True, "past_horizon": 0, "weight": 5.0},
        },
        "keep_old_reward": False,
        "flatten_obs": False,
    },
}


@pytest.fixture(scope="module")
def stl_env():
    from rlrom.wrappers.stl_wrapper import STLWrapper
    env = gym.make(CARTPOLE_CFG["env_name"])
    env = STLWrapper(env, STL_CFG)
    yield env
    env.close()


def test_unwrapped_episode():
    T = RLTester(dict(CARTPOLE_CFG))
    T.run_cfg_test()
    Tres = T.test_results[-1]
    ep = Tres["episodes"][0]
    assert len(ep["observations"]) > 0
    assert len(ep["actions"]) == len(ep["rewards"])
    assert "basics" in Tres["res_all_ep"]
    assert Tres["res_all_ep"]["basics"]["mean_ep_len"] > 0


def test_stl_wrapper_step(stl_env):
    env = stl_env
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, (int, float, np.floating))
    # keep_old_reward=False: reward must come from the STL formula only
    assert -5.0 < reward < 5.0
    # episode data collected
    assert len(env.episode["stl_data"]) == 1
    assert len(env.episode["res_f"]["cart_left"]) == 1


def test_stl_wrapper_metrics(stl_env):
    env = stl_env
    env.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or env.time_step >= 50:
            break
    res, res_all_ep, res_rew_f, res_eval_f = env.eval_specs_episode()
    assert res["ep_len"][0] > 0
    assert res["ep_rew"][0] != 0  # reward shaping is active
    # per-episode per-formula metrics
    for key in ("mean", "sum", "num_sat"):
        assert key in res["cart_left"]
    assert len(res_rew_f) == 1
    # robustness of a Boolean formula is bounded by weight (BigM capped)
    assert np.max(np.abs(res_rew_f[0]["cart_left"])) <= 5.0 + 1e-6


def test_stl_wrapper_reset_reuses_driver(stl_env):
    """Reset must clear monitored data (regression: reset_monitor was broken)."""
    env = stl_env
    env.reset()
    env.step(env.action_space.sample())
    assert len(env.episode["stl_data"]) == 1
    env.reset()
    assert len(env.episode["stl_data"]) == 0
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert len(env.episode["stl_data"]) == 1
    # monitor restarted from t0: time must not accumulate across resets
    assert env.get_time()[0] == 0.0


def test_stl_reward_shaped_vs_env_reward(stl_env):
    """Wrapped reward is pure robustness (keep_old_reward=False): bounded,
    nonzero, and varying across steps."""
    env = stl_env
    env.reset()
    shaped = []
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        shaped.append(reward)
        if terminated:
            break
    shaped = np.array(shaped)
    assert np.all(np.abs(shaped) <= 5.0 + 1e-6)
    assert np.all(shaped != 0)  # Boolean semantics: robustness is +/-1, scaled
    assert len(np.unique(shaped)) > 1
