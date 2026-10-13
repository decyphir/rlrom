"""Tests for reward machines integrated into STLWrapper (cfg_specs.reward_machines).

Covers: config parsing, FSM creation, per-step label evaluation (lower_rob > 0),
reward-machine stepping, episode recording, and the text display helper.
"""
import gymnasium as gym
import numpy as np
import pytest

from rlrom.wrappers.stl_wrapper import STLWrapper

RM_CFG = {
    "env_name": "CartPole-v1",
    "cfg_specs": {
        "specs": (
            "signal x, theta, reward\n"
            "on_left := x[t] < -0.2\n"
            "on_right := x[t] > 0.2\n"
        ),
        "obs_names": {"x": "obs[0]", "theta": "obs[2]"},
        "label_formulas": {
            "on_left": {"online": True, "past_horizon": 0},
            "on_right": {"online": True, "past_horizon": 0},
        },
        "reward_machines": [
            {
                "name": "left_then_right",
                "weight": 2.0,
                "reward_machine": {
                    "states": [
                        {"id": "go_left", "initial": True, "reward": 0},
                        {"id": "on_right_target", "final": True, "reward": 1},
                    ],
                    "transitions": [
                        {"from": "go_left", "to": "on_right_target",
                         "condition": "on_right", "reward": 1, "priority": 1},
                    ],
                },
            }
        ],
    },
}


@pytest.fixture(scope="module")
def rm_env():
    env = gym.make(RM_CFG["env_name"])
    env = STLWrapper(env, RM_CFG)
    yield env
    env.close()


def test_parses_reward_machines(rm_env):
    env = rm_env
    assert len(env.reward_machines) == 1
    rm = env.reward_machines[0]
    assert rm["name"] == "left_then_right"
    assert rm["weight"] == 2.0
    assert rm["machine"].state_id == "go_left"
    assert rm["machine"].is_final is False
    assert list(env.label_formulas.keys()) == ["on_left", "on_right"]


def test_reset_initializes_episode_tracking(rm_env):
    env = rm_env
    env.reset()
    assert env.episode["labels"] == []
    assert env.episode["rm_states"] == []
    assert env.episode["rm_rewards"] == []
    # label formulas appear in res_f
    assert "on_left" in env.episode["res_f"]
    assert "on_right" in env.episode["res_f"]


def test_step_records_labels_and_rm(rm_env):
    env = rm_env
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert len(env.episode["labels"]) == 1
    assert len(env.episode["rm_states"]) == 1
    assert len(env.episode["rm_rewards"]) == 1
    # labels are subsets of the declared label formulas
    for lbl in env.episode["labels"][0]:
        assert lbl in env.label_formulas
    # state index is a valid state index (0 or 1)
    assert env.episode["rm_states"][0][0] in (0, 1)


def test_rm_transitions_to_final_when_on_right_true(rm_env):
    """Inject a state where x > 0.2 so on_right is true -> RM goes to final."""
    env = rm_env
    env.reset()
    rm = env.reward_machines[0]["machine"]
    # force x to the right by monkeypatching get_sample is overkill; instead verify
    # event-driven behavior via a tiny manual machine with a canned label.
    from rlrom.wrappers.reward_machine import RewardMachine
    m = RewardMachine(RM_CFG["cfg_specs"]["reward_machines"][0]["reward_machine"])
    r = m.step(["on_right"])
    assert r == 1
    assert m.state_id == "on_right_target"
    assert m.is_final is True
    # without the label, it stays and issues the state reward (0)
    m.reset()
    r = m.step(["on_left"])
    assert r == 0
    assert m.state_id == "go_left"
    assert m.is_final is False


def test_multiple_reward_machines():
    cfg = {
        "env_name": "CartPole-v1",
        "cfg_specs": {
            "specs": "signal x, reward\non_left := x[t] < -0.2\n",
            "obs_names": {"x": "obs[0]"},
            "label_formulas": {"on_left": {"online": True, "past_horizon": 0}},
            "reward_machines": [
                {"reward_machine": {"states": [{"id": "a", "initial": True}]}},
                {"name": "second", "reward_machine": {"states": [{"id": "b", "initial": True}]}},
            ],
        },
    }
    env = gym.make("CartPole-v1")
    env = STLWrapper(env, cfg)
    assert len(env.reward_machines) == 2
    assert env.reward_machines[0]["name"] == "rm_0"  # auto-generated default name
    assert env.reward_machines[1]["name"] == "second"
    # defaults: weight 1.0, in_observation False
    assert env.reward_machines[0]["weight"] == 1.0
    assert env.reward_machines[0]["in_observation"] is False
    env.close()


def test_show_reward_machines_prints(capsys, rm_env):
    env = rm_env
    env.show_reward_machines()
    out = capsys.readouterr().out
    assert "left_then_right" in out
    assert "go_left -> on_right_target if 'on_right'" in out
    assert "Label formulas" in out


def test_no_reward_machines_is_fine():
    cfg = {
        "env_name": "CartPole-v1",
        "cfg_specs": {"specs": "signal x, reward\n", "obs_names": {"x": "obs[0]"}},
    }
    env = gym.make("CartPole-v1")
    env = STLWrapper(env, cfg)
    assert env.reward_machines == []
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert len(env.episode["rm_states"]) == 1
    assert env.episode["rm_states"][0] == []
    env.close()


# --------------------------------------------------- reward integration + termination
REWARDS_CFG = {
    "env_name": "CartPole-v1",
    "cfg_specs": {
        "specs": "signal x, reward\non_left := x[t] < -0.2\non_right := x[t] > 0.2\n",
        "obs_names": {"x": "obs[0]"},
        "env_reward_weight": 1.0,
        "label_formulas": {
            "on_left": {"online": True, "past_horizon": 0},
            "on_right": {"online": True, "past_horizon": 0},
        },
        "reward_machines": [
            {
                "name": "left_then_right",
                "weight": 3.0,
                "reward_machine": {
                    "states": [
                        {"id": "go_left", "initial": True, "reward": 0},
                        {"id": "on_right_target", "final": True, "reward": 1},
                    ],
                    "transitions": [
                        {"from": "go_left", "to": "on_right_target",
                         "condition": "on_right", "reward": 1, "priority": 1},
                    ],
                },
            }
        ],
    },
}


def _rewards_env():
    return STLWrapper(gym.make("CartPole-v1"), REWARDS_CFG)


def test_rm_states_in_dict_observation():
    env = _rewards_env()
    try:
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "rm_states" in env.observation_space.spaces
        assert "unwrapped" in env.observation_space.spaces
        obs, _ = env.reset()
        assert obs["rm_states"].shape == (1,)
        assert obs["rm_states"][0] == 0  # initial state index
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert "rm_states" in obs
    finally:
        env.close()


def test_rm_reward_integrated_and_terminates(monkeypatch):
    env = _rewards_env()
    env.reset()
    # Force on_right true so the RM takes the transition (reward 1 * weight 3).
    # We drive x > 0.2 by overriding get_sample to report a right position.
    monkeypatch.setattr(env, "get_sample", lambda obs, action, reward: np.array([0.0, 0.5, 0.0]))
    obs, reward, terminated, truncated, info = env.step(0)
    # reward = env_reward (CartPole keeps the pole up -> 1.0) + weight*rm_reward
    #        = 1.0 + 3*1 = 4.0
    assert reward == pytest.approx(4.0)
    assert env.episode["rm_rewards"][0][0] == 1
    assert env.episode["rm_states"][0][0] == 1
    assert terminated is True  # final RM state reached


def test_rm_no_transition_no_extra_reward(monkeypatch):
    env = _rewards_env()
    env.reset()
    # on_left true (not a transition from go_left) -> stays, state reward 0
    monkeypatch.setattr(env, "get_sample", lambda obs, action, reward: np.array([0.0, -0.5, 0.0]))
    obs, reward, terminated, truncated, info = env.step(0)
    # env reward 1.0 (pole up) + rm reward 0 = 1.0
    assert reward == pytest.approx(1.0)
    assert terminated is False
    assert env.episode["rm_rewards"][0][0] == 0


def test_rm_obs_is_dict_response():
    env = _rewards_env()
    try:
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, dict)
        assert "unwrapped" in obs and "rm_states" in obs
    finally:
        env.close()
