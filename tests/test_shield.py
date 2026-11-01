"""Tests for the reward-machine action shield.

Shield fields per state: ``blocked`` (list of raw action values forbidden in
the state), ``forced`` (raw action value taken unconditionally in the state)
and ``fallback`` (raw action value used when the agent's action is blocked
and the previous action is also blocked). Between machines, ``shield_priority``
(default 0) decides precedence; a tie on the max priority is an error.
"""
import gymnasium as gym
import pytest

from rlrom.wrappers.reward_machine import RewardMachine
from rlrom.wrappers.spec_wrapper import SpecWrapper


# --------------------------------------------------------------------- helpers
def _machine(name, state_fields=None, priority=0):
    return {
        "name": name,
        "shield_priority": priority,
        "reward_machine": {
            # Shield lives in the INITIAL state so no labels are needed to
            # reach it (wrapper tests do not drive the FSM transitions).
            "states": [dict({"id": "s0", "initial": True}, **(state_fields or {}))],
            "transitions": [],
        },
    }


def _make_env(machines):
    env = gym.make("CartPole-v1")  # Discrete(2): actions 0 and 1
    cfg = {"cfg_specs": {
        "specs": "signal env_reward",
        "flatten_obs": False,
        "reward_machines": machines,
    }}
    return SpecWrapper(env, cfg)


# ------------------------------------------------------- RewardMachine parsing
def test_shield_fields_parsed():
    m = RewardMachine({"states": [
        {"id": "a", "initial": True, "blocked": [1, 2], "forced": 3, "fallback": 0},
        {"id": "b", "blocked": [1]},
    ]})
    assert m.blocked_actions("a") == [1, 2]
    assert m.forced_action("a") == 3
    assert m.fallback_action("a") == 0
    assert m.blocked_actions("b") == [1]
    assert m.forced_action("b") is None
    assert m.fallback_action("b") is None
    # default to the current state
    assert m.blocked_actions() == [1, 2]
    assert m.forced_action() == 3


def test_shield_fields_default_absent():
    m = RewardMachine({"states": [{"id": "a", "initial": True}]})
    assert m.blocked_actions() == []
    assert m.forced_action() is None
    assert m.fallback_action() is None


def test_shield_field_validation():
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True, "forced": "left"}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True, "blocked": "1"}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "blocked": [1, "x"]}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "fallback": 1.5}]})


# ------------------------------------------------------------- shield behavior
def test_no_shield_passthrough():
    env = _make_env([_machine("m")])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info


def test_forced_overrides_agent_action():
    env = _make_env([_machine("m", {"forced": 1})])
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"] == {
        "machine": "m", "reason": "forced",
        "original": 0, "shielded": 1,
    }


def test_forced_same_as_agent_action_is_passthrough():
    env = _make_env([_machine("m", {"forced": 1})])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info


def test_blocked_uses_fallback_when_no_previous_action():
    env = _make_env([_machine("m", {"blocked": [1], "fallback": 0})])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["shielded"] == 0


def test_blocked_uses_previous_action_when_allowed():
    env = _make_env([_machine("m", {"blocked": [1], "fallback": 0})])
    env.reset()
    env.step(1)          # -> fallback 0 (last action becomes 0)
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->previous"
    assert info["shield"]["shielded"] == 0


def test_blocked_uses_fallback_when_previous_also_blocked():
    env = _make_env([_machine("m", {"blocked": [0, 1], "fallback": 0})])
    env.reset()
    env.step(1)          # -> fallback 0
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["shielded"] == 0


def test_blocked_no_fallback_and_no_previous_raises():
    env = _make_env([_machine("m", {"blocked": [1]})])
    env.reset()
    with pytest.raises(ValueError, match="no 'fallback'"):
        env.step(1)


def test_higher_priority_machine_wins():
    env = _make_env([
        _machine("low", {"forced": 1}, priority=0),
        _machine("high", {"forced": 0}, priority=5),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["machine"] == "high"
    assert info["shield"]["shielded"] == 0


def test_forced_takes_precedence_over_blocked_at_same_machine():
    # forced is evaluated before blocked within a machine
    env = _make_env([_machine("m", {"blocked": [1], "fallback": 0, "forced": 1})])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info  # forced == agent action: no visible change
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"]["reason"] == "forced"
    assert info["shield"]["shielded"] == 1


def test_shield_tie_on_max_priority_raises():
    env = _make_env([
        _machine("a", {"forced": 0}, priority=1),
        _machine("b", {"forced": 1}, priority=1),
    ])
    env.reset()
    with pytest.raises(ValueError, match="shield tie"):
        env.step(0)


# ------------------------------------------- cumulative blocking (multi-RM)
def test_previous_action_is_checked_against_all_machines():
    # m1 blocks 1 (fallback 0); m2 blocks 0. The previous action (0) is
    # blocked by m2, so the fallback must be used, not the previous action.
    env = _make_env([
        _machine("m1", {"blocked": [1], "fallback": 0}),
        _machine("m2", {"blocked": [0], "fallback": 1}),
    ])
    env.reset()
    env.step(1)                      # -> m1 fallback 0
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["shielded"] == 0
    # fallback is trusted even though m2 blocks 0; next step: prev=0 blocked
    # by m2, and m1's previous check fails again -> fallback
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"


def test_fallback_conflict_resolved_by_priority():
    # both block 0; m2 has higher priority and its fallback (1) is used.
    # If m1's fallback (0) had won, there would be no visible override.
    env = _make_env([
        _machine("m1", {"blocked": [0], "fallback": 0}, priority=0),
        _machine("m2", {"blocked": [0, 1], "fallback": 1}, priority=3),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"] == {
        "machine": "m2", "reason": "blocked->fallback",
        "original": 0, "shielded": 1,
    }
    # next step: previous action (1) is blocked by m2 -> fallback again
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["machine"] == "m2"


def test_fallback_tie_raises():
    env = _make_env([
        _machine("a", {"blocked": [1], "fallback": 0}, priority=2),
        _machine("b", {"blocked": [1], "fallback": 2}, priority=2),
    ])
    env.reset()
    with pytest.raises(ValueError, match="shield tie"):
        env.step(1)


def test_forced_exempt_from_lower_priority_blocker():
    # a forces 1 (priority 5); m blocks 1 (priority 0) and has no fallback:
    # the force wins, so no replacement (and no error) is needed
    env = _make_env([
        _machine("a", {"forced": 1}, priority=5),
        _machine("m", {"blocked": [1]}, priority=0),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"] == {
        "machine": "a", "reason": "forced",
        "original": 0, "shielded": 1,
    }


def test_higher_priority_blocker_overrides_forced():
    # a forces 1 (priority 5); m blocks 1 (priority 6, fallback 0):
    # the blocker wins the conflict and provides the fallback
    env = _make_env([
        _machine("a", {"forced": 1}, priority=5),
        _machine("m", {"blocked": [1], "fallback": 0}, priority=6),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"] == {
        "machine": "m", "reason": "blocked->fallback",
        "original": 1, "shielded": 0,
    }


def test_reset_clears_previous_action():
    env = _make_env([_machine("m", {"blocked": [1], "fallback": 0})])
    env.reset()
    env.step(1)          # fallback 0
    env.step(1)          # previous 0
    assert env._last_action == 0
    env.reset()
    assert env._last_action is None
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"


def test_shielded_action_recorded_in_episode():
    env = _make_env([_machine("m", {"forced": 1})])
    env.reset()
    env.step(0)
    assert env.episode["actions"] == [1]
