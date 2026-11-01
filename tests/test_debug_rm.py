"""Tests for the debug_rm flag: per-step printing of RM states, labels and
activated transitions (and shield overrides)."""
import gymnasium as gym
import pytest

from rlrom.wrappers.spec_wrapper import SpecWrapper


def _make_env(machines, debug_rm):
    env = gym.make("CartPole-v1")
    cfg = {"cfg_specs": {
        "specs": "signal env_reward",
        "flatten_obs": False,
        "debug_rm": debug_rm,
        "reward_machines": machines,
    }}
    return SpecWrapper(env, cfg)


def _machine(name, state_fields=None, priority=0):
    return {
        "name": name,
        "shield_priority": priority,
        "reward_machine": {
            "states": [dict({"id": "s0", "initial": True}, **(state_fields or {}))],
            "transitions": [],
        },
    }


def test_debug_rm_off_prints_nothing(capsys):
    env = _make_env([_machine("m")], debug_rm=False)
    env.reset()
    env.step(0)
    env.step(1)
    assert capsys.readouterr().out == ""


def test_debug_rm_prints_state_labels_and_transitions(capsys):
    env = _make_env([_machine("m", {"forced": 1})], debug_rm=True)
    env.reset()
    env.step(0)
    out = capsys.readouterr().out
    # shield line (action 0 forced to 1)
    assert "shield" in out
    # machine line: state, labels, activated (no transitions -> stay)
    assert "rm 'm': state=s0" in out
    assert "labels=" in out
    assert "activated=[(none -> stay)]" in out
    # result line
    assert "-> state=s0 reward=" in out


def test_debug_rm_shows_activated_transition():
    # machine with one transition; drive it with an always-true label
    cfg = {"cfg_specs": {
        "specs": "signal x, env_reward\nhit := x[t] < 100.0",
        "flatten_obs": False,
        "debug_rm": True,
        "obs_names": {"x": "obs[0]"},
        "label_formulas": {"hit": None},
        "reward_machines": [{
            "name": "m",
            "reward_machine": {
                "states": [
                    {"id": "s0", "initial": True},
                    {"id": "s1"},
                ],
                "transitions": [
                    {"from": "s0", "to": "s1", "condition": "hit"},
                ],
            },
        }],
    }}
    env = SpecWrapper(gym.make("CartPole-v1"), cfg)
    env.reset()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env.step(0)  # first step: label 'hit' true -> s0 -> s1
    out = buf.getvalue()
    assert "labels=['hit']" in out
    assert "activated=[s0__to__s1__if__hit]" in out
    assert "-> state=s1" in out
