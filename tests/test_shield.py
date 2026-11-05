"""Tests for the reward-machine action constraints.

Per-state ``actions`` fields: ``blocked`` (list of raw action values
forbidden in the state), ``forced`` (raw action value taken unconditionally
in the state), ``fallback`` (raw action value used when the agent's action
is blocked and the previous action is also blocked) and ``enable`` (default
True; when False the state imposes no action constraints). Between sources,
``action_priority`` (machine-level, for reward machines) and
``actions.priority`` (for monitor formulas) -- both defaulting to 0 --
resolve conflicts; sources that agree (same forced action, identical
fallbacks) never conflict at equal priority, but equal priorities that
cannot be resolved (different forced actions, different fallbacks, or a
force tied by a block) raise an error.
"""
import gymnasium as gym
import pytest

from rlrom.wrappers.reward_machine import RewardMachine
from rlrom.wrappers.spec_wrapper import SpecWrapper


# --------------------------------------------------------------------- helpers
def _machine(name, state_fields=None, priority=0):
    return {
        "name": name,
        "action_priority": priority,
        "reward_machine": {
            # Shield lives in the INITIAL state so no labels are needed to
            # reach it (wrapper tests do not drive the FSM transitions).
            "states": [dict({"id": "s0", "initial": True,
                             "actions": (state_fields or {})},)],
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
        {"id": "a", "initial": True,
         "actions": {"blocked": [1, 2], "forced": 3, "fallback": 0}},
        {"id": "b", "actions": {"blocked": [1]}},
    ]})
    assert m.blocked("a") == [1, 2]
    assert m.forced("a") == 3
    assert m.fallback("a") == 0
    assert m.blocked("b") == [1]
    assert m.forced("b") is None
    assert m.fallback("b") is None
    # default to the current state
    assert m.blocked() == [1, 2]
    assert m.forced() == 3


def test_shield_fields_default_absent():
    m = RewardMachine({"states": [{"id": "a", "initial": True}]})
    assert m.blocked() == []
    assert m.forced() is None
    assert m.fallback() is None
    # actions are enabled by default
    assert m.actions_enabled() is True


def test_actions_enable_flag():
    m = RewardMachine({"states": [
        {"id": "a", "initial": True,
         "actions": {"blocked": [1], "enable": False}},
        {"id": "b", "actions": {"blocked": [1]}},
    ]})
    assert m.actions_enabled("a") is False
    assert m.actions_enabled("b") is True


def test_shield_field_validation():
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "actions": {"forced": "left"}}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "actions": {"blocked": "1"}}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "actions": {"blocked": [1, "x"]}}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "actions": {"fallback": 1.5}}]})
    with pytest.raises(ValueError):
        RewardMachine({"states": [{"id": "a", "initial": True,
                                   "actions": [1, 2]}]})


# ------------------------------------------------------------- shield behavior
def test_no_shield_passthrough():
    env = _make_env([_machine("m")])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info


def test_actions_disabled_is_passthrough():
    env = _make_env([_machine("m", {"blocked": [0, 1], "enable": False})])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info
    # and without a fallback, which would raise if the block were active
    env = _make_env([_machine("m", {"blocked": [1], "enable": False})])
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
    # same max priority, *different* forced actions -> conflict
    env = _make_env([
        _machine("a", {"forced": 0}, priority=1),
        _machine("b", {"forced": 1}, priority=1),
    ])
    env.reset()
    with pytest.raises(ValueError, match="shield conflict"):
        env.step(0)


def test_forced_tie_on_same_action_is_no_conflict():
    # same max priority, *same* forced action -> no conflict
    env = _make_env([
        _machine("a", {"forced": 1}, priority=1),
        _machine("b", {"forced": 1}, priority=1),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"]["reason"] == "forced"
    assert info["shield"]["shielded"] == 1
    assert info["shield"]["machine"] == "a, b"


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
    # same max priority, *different* fallbacks -> conflict
    env = _make_env([
        _machine("a", {"blocked": [1], "fallback": 0}, priority=2),
        _machine("b", {"blocked": [1], "fallback": 2}, priority=2),
    ])
    env.reset()
    with pytest.raises(ValueError, match="shield conflict"):
        env.step(1)


def test_fallback_tie_on_same_fallback_is_no_conflict():
    # same max priority, identical fallbacks -> no conflict
    env = _make_env([
        _machine("a", {"blocked": [1], "fallback": 0}, priority=2),
        _machine("b", {"blocked": [1], "fallback": 0}, priority=2),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["shielded"] == 0
    assert info["shield"]["machine"] == "a, b"


def test_fallback_tie_none_defers_to_set():
    # same max priority, one fallback unset: the set one is used
    env = _make_env([
        _machine("a", {"blocked": [1]}, priority=2),
        _machine("b", {"blocked": [1], "fallback": 0}, priority=2),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["shielded"] == 0


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


def test_equal_priority_blocker_on_forced_raises():
    # a forces 1 (priority 5); m blocks 1 at the *same* priority: a
    # force/block conflict that cannot be resolved by priority
    env = _make_env([
        _machine("a", {"forced": 1}, priority=5),
        _machine("m", {"blocked": [1], "fallback": 0}, priority=5),
    ])
    env.reset()
    with pytest.raises(ValueError, match="shield conflict"):
        env.step(1)


def test_equal_priority_blocker_not_blocking_forced_is_ignored():
    # a forces 1 (priority 5); m blocks 0 (not the forced action) at the
    # same priority: no conflict, the force stands
    env = _make_env([
        _machine("a", {"forced": 1}, priority=5),
        _machine("m", {"blocked": [0], "fallback": 0}, priority=5),
    ])
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"]["reason"] == "forced"
    assert info["shield"]["shielded"] == 1


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


# ------------------------------------------- monitor formula 'actions' field
def _monitor_env(formula_name, formula_opt):
    """Env with one monitor formula; truth only known after each env step, so
    its actions apply from the *next* step on (like RM states)."""
    env = gym.make("CartPole-v1")  # Discrete(2): actions 0 and 1
    cfg = {"cfg_specs": {
        "specs": f"signal x, env_reward\n{formula_name} := x[t] < 100.0",
        "obs_names": {"x": "obs[0]"},
        "monitor_formulas": {formula_name: formula_opt},
        "flatten_obs": False,
    }}
    return SpecWrapper(env, cfg)


def test_monitor_actions_parsed():
    env = _monitor_env("hit", {"online": True,
                               "actions": {"blocked": [1], "fallback": 0,
                                           "priority": 3}})
    assert env.monitor_actions == {
        "hit": {"enable": True, "blocked": [1], "forced": None,
                "fallback": 0, "action_priority": 3},
    }
    env.close()


def test_monitor_actions_validation():
    with pytest.raises(ValueError, match="actions"):
        _monitor_env("hit", {"actions": [1]})
    with pytest.raises(ValueError, match="actions.blocked"):
        _monitor_env("hit", {"actions": {"blocked": "1"}})
    with pytest.raises(ValueError, match="actions.forced"):
        _monitor_env("hit", {"actions": {"forced": "left"}})
    with pytest.raises(ValueError, match="actions.fallback"):
        _monitor_env("hit", {"actions": {"fallback": 1.5}})


def test_monitor_actions_forced_from_next_step():
    env = _monitor_env("hit", {"online": True, "actions": {"forced": 1}})
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert "shield" not in info  # truth only known after this step
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"] == {
        "machine": "hit", "reason": "forced",
        "original": 0, "shielded": 1,
    }
    env.close()


def test_monitor_actions_blocked_uses_fallback():
    # previous action (0) is blocked too -> the fallback must be used
    env = _monitor_env("hit", {"online": True,
                               "actions": {"blocked": [0, 1], "fallback": 0}})
    env.reset()
    env.step(0)
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["machine"] == "hit"
    assert info["shield"]["shielded"] == 0
    env.close()


def test_monitor_actions_blocked_uses_previous_action():
    env = _monitor_env("hit", {"online": True,
                               "actions": {"blocked": [1], "fallback": 0}})
    env.reset()
    env.step(0)          # previous action 0 is allowed
    obs, r, term, trunc, info = env.step(1)
    assert info["shield"]["reason"] == "blocked->previous"
    assert info["shield"]["shielded"] == 0
    env.close()


def test_monitor_actions_no_fallback_raises():
    # candidate and previous action are both blocked, no fallback -> error
    env = _monitor_env("hit", {"online": True, "actions": {"blocked": [0, 1]}})
    env.reset()
    env.step(0)
    with pytest.raises(ValueError, match="no 'fallback'"):
        env.step(1)
    env.close()


def test_monitor_actions_enable_false_is_inactive():
    env = _monitor_env("hit", {"online": True,
                               "actions": {"blocked": [0, 1], "enable": False}})
    env.reset()
    env.step(0)
    obs, r, term, trunc, info = env.step(1)  # would raise if active
    assert "shield" not in info
    env.close()


def test_monitor_actions_off_while_formula_false():
    # formula never true -> its actions never apply
    env = gym.make("CartPole-v1")
    cfg = {"cfg_specs": {
        "specs": "signal x, env_reward\nnever := x[t] > 100.0",
        "obs_names": {"x": "obs[0]"},
        "monitor_formulas": {"never": {"online": True,
                                       "actions": {"forced": 1}}},
        "flatten_obs": False,
    }}
    env = SpecWrapper(env, cfg)
    env.reset()
    env.step(0)
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info
    assert env._active_formula_actions == []
    env.close()


def test_monitor_actions_combine_with_machines():
    # formula blocks 0 (fallback 1); machine m blocks 1 (fallback 0).
    # agent action 1: blocked by m, previous (0) blocked by formula -> m's
    # fallback 0 is used... but 0 is blocked by the formula; fallbacks are
    # trusted even if another source blocks them.
    env = gym.make("CartPole-v1")
    cfg = {"cfg_specs": {
        "specs": "signal x, env_reward\nhit := x[t] < 100.0",
        "obs_names": {"x": "obs[0]"},
        "monitor_formulas": {"hit": {"online": True,
                                     "actions": {"blocked": [0], "fallback": 1}}},
        "reward_machines": [_machine("m", {"blocked": [1], "fallback": 0})],
        "flatten_obs": False,
    }}
    env = SpecWrapper(env, cfg)
    env.reset()
    env.step(1)
    obs, r, term, trunc, info = env.step(1)
    # candidate 1 blocked by m; previous 1 blocked by m; m's fallback 0
    assert info["shield"]["reason"] == "blocked->fallback"
    assert info["shield"]["machine"] == "m"
    assert info["shield"]["shielded"] == 0
    env.close()


def test_disable_actions_disables_all_sources():
    # machine forces 1; monitor formula forces 0 from the next step on.
    # with disable_actions, the agent's action always passes through.
    env = gym.make("CartPole-v1")
    cfg = {"cfg_specs": {
        "specs": "signal x, env_reward\nhit := x[t] < 100.0",
        "obs_names": {"x": "obs[0]"},
        "monitor_formulas": {"hit": {"online": True,
                                     "actions": {"forced": 0}}},
        "reward_machines": [_machine("m", {"forced": 1})],
        "disable_actions": True,
        "flatten_obs": False,
    }}
    env = SpecWrapper(env, cfg)
    env.reset()
    env.step(0)  # 'hit' becomes true; its forced action would be active now
    obs, r, term, trunc, info = env.step(1)
    assert "shield" not in info
    assert env.episode["actions"][-1] == 1  # agent action, unmodified
    env.close()


def test_disable_actions_absent_keeps_shield():
    # regression: without the flag, the same setup shields
    env = gym.make("CartPole-v1")
    cfg = {"cfg_specs": {
        "specs": "signal env_reward",
        "reward_machines": [_machine("m", {"forced": 1})],
        "flatten_obs": False,
    }}
    env = SpecWrapper(env, cfg)
    env.reset()
    obs, r, term, trunc, info = env.step(0)
    assert info["shield"]["reason"] == "forced"
    assert info["shield"]["shielded"] == 1
    env.close()
