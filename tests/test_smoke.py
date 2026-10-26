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
        "specs": "signal x, theta, env_reward\ncart_left := x[t] < 0.5 and x[t] > 0.1",
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
    from rlrom.wrappers.spec_wrapper import SpecWrapper
    env = gym.make(CARTPOLE_CFG["env_name"])
    env = SpecWrapper(env, STL_CFG)
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


def test_spec_wrapper_step(stl_env):
    env = stl_env
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, (int, float, np.floating))
    # keep_old_reward=False: reward must come from the STL formula only
    assert -5.0 < reward < 5.0
    # episode data collected
    assert len(env.episode["stl_data"].get_signal(0).get_samples_list()) == 1
    assert len(env.episode["res_f"]["reward_formulas/cart_left"]) == 1


def test_spec_wrapper_metrics(stl_env):
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
    assert np.max(np.abs(res_rew_f[0]["reward_formulas/cart_left"])) <= 5.0 + 1e-6


def test_spec_wrapper_reset_reuses_driver(stl_env):
    """Reset must clear monitored data (regression: reset_monitor was broken)."""
    env = stl_env
    env.reset()
    env.step(env.action_space.sample())
    assert env.get_ep_len() == 1
    env.reset()
    assert env.get_ep_len() == 0
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.get_ep_len() == 1
    # monitor restarted from t0: time must not accumulate across resets
    assert env.get_time()[0] == 0.0


def test_per_step_record_matches_rho_without_online():
    """Per-step recorded robustness with NO 'online' option must be evaluated
    at the current time: for a state formula it equals the offline rho(f)
    curve. Regression: the default used to pin the evaluation at t0=0, so
    obs/reward-formula records (res_f) were flat and inconsistent with
    rho(f) plots."""
    from rlrom.wrappers.spec_wrapper import SpecWrapper
    cfg = dict(STL_CFG)
    cfg['cfg_specs'] = dict(STL_CFG['cfg_specs'])  # don't mutate STL_CFG
    # no 'online' key -> default anchoring (current time)
    cfg['cfg_specs']['reward_formulas'] = {
        "cart_left": {"past_horizon": 0, "weight": 5.0},
    }
    env = SpecWrapper(gym.make(CARTPOLE_CFG["env_name"]), cfg)
    env.reset()
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            break
    recorded = np.asarray(env.episode["res_f"]["reward_formulas/cart_left"], dtype=float)
    offline = np.asarray(env.get_rob("cart_left", online=False), dtype=float)
    assert len(recorded) > 0
    n = min(len(recorded), len(offline))
    np.testing.assert_allclose(recorded[:n], offline[:n], atol=1e-9)
    env.close()


def test_formula_in_two_groups_recorded_once_per_step():
    """A formula declared in both obs_formulas and reward_formulas must be
    recorded once per step per group. Regression: res_f used a shared key,
    so the value was appended twice per step and the plotted traces showed
    alternating zeros."""
    from rlrom.wrappers.spec_wrapper import SpecWrapper
    cfg = dict(STL_CFG)
    cfg['cfg_specs'] = dict(STL_CFG['cfg_specs'])  # don't mutate STL_CFG
    cfg['cfg_specs']['obs_formulas'] = {"cart_left": {"online": True, "past_horizon": 0}}
    env = SpecWrapper(gym.make(CARTPOLE_CFG["env_name"]), cfg)
    env.reset()
    n = 0
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        n += 1
        if terminated:
            break
    obs_rec = env.episode["res_f"]["obs_formulas/cart_left"]
    rew_rec = env.episode["res_f"]["reward_formulas/cart_left"]
    assert len(obs_rec) == n
    assert len(rew_rec) == n
    # same formula, same anchoring -> identical per-step values
    np.testing.assert_allclose(np.asarray(obs_rec, dtype=float),
                               np.asarray(rew_rec, dtype=float))
    env.close()


def test_formula_alias_and_formula_params():
    """'formula' selects the evaluated formula (alias of 'formula_name'),
    and 'formula_params' overrides spec-declared parameters for one config
    only (the driver's param map is global, so values are saved and
    restored around each evaluation)."""
    from rlrom.wrappers.spec_wrapper import SpecWrapper
    cfg = {
        "env_name": "CartPole-v1",
        "cfg_specs": {
            "specs": ("signal x, theta, env_reward\n"
                      "param close_dist = 1.0\n"
                      "close := x[t] < close_dist\n"),
            "obs_names": {"x": "obs[0]", "theta": "obs[2]"},
            "reward_formulas": {
                # config key != formula name: 'formula' alias in action
                "close_far": {"formula": "close", "online": True, "past_horizon": 0,
                              "formula_params": {"close_dist": 2.0}},
            },
            "flatten_obs": False,
        },
    }
    env = SpecWrapper(gym.make("CartPole-v1"), cfg)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # the first sample uses the post-action obs (x is no longer 0);
    # rob of 'x < c' at that sample is c - x
    x = float(np.asarray(obs)[0])
    rec = env.episode["res_f"]["reward_formulas/close_far"][0]
    assert abs(rec - (2.0 - x)) < 1e-6  # close_dist overridden to 2.0
    # parameter restored to the spec default after the evaluation
    assert env.stl_driver.get_param("close_dist") == 1.0
    env.close()


def test_formula_params_undeclared_raises():
    """formula_params may only reference parameters declared in the spec
    (stlrom creates params via 'param <name> = <value>' in the spec)."""
    from rlrom.wrappers.spec_wrapper import SpecWrapper
    cfg = {
        "env_name": "CartPole-v1",
        "cfg_specs": {
            "specs": "signal x, theta, env_reward\nclose := x[t] < 1.0\n",
            "obs_names": {"x": "obs[0]", "theta": "obs[2]"},
            "reward_formulas": {
                "close": {"online": True, "formula_params": {"close_dist": 2.0}},
            },
        },
    }
    env = SpecWrapper(gym.make("CartPole-v1"), cfg)
    with pytest.raises(ValueError, match="close_dist"):
        env.reset()
    env.close()


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
