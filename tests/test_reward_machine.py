"""Tests for the decoupled RewardMachine finite-state machine.

The RewardMachine is a pure FSM (no gym, no STL): it parses ``cfg_rm``, tracks the
current state, consumes a list of *true* condition labels each step, and takes the
highest-priority enabled transition. These tests exercise that logic in isolation.
"""
import pytest

from rlrom.wrappers.reward_machine import RewardMachine, _trans_id


# --------------------------------------------------------------------------- cfg
def _minigrid_style_cfg():
    # Mirrors examples/minigrid/blocked_unlock_pickup/cfg_rm_task3.yml structure.
    return {
        "states": [
            {"id": "u5", "initial": True, "reward": 0},
            {"id": "u6", "reward": 0},
            {"id": "u8", "reward": -1},
            {"id": "u7", "final": True},
        ],
        "transitions": [
            {"from": "u5", "to": "u8", "condition": "door_is_closed", "reward": -1},
            {"from": "u6", "to": "u8", "condition": "door_is_closed", "reward": -1},
            {"from": "u8", "to": "u5", "condition": "has_key_door_open", "reward": 0},
            {"from": "u8", "to": "u6", "condition": "not_has_key_door_open_no_box", "reward": 0},
            {"from": "u6", "to": "u5", "condition": "has_key_door_open", "reward": 0},
            {"from": "u5", "to": "u6", "condition": "not_has_key_door_open_no_box", "reward": 0},
            {"from": "u6", "to": "u7", "condition": "has_box", "reward": 1},
        ],
    }


# ------------------------------------------------------------------------ basics
def test_trans_id_default():
    assert _trans_id("a", "b", "c") == "a__to__b__if__c"


def test_initial_state_and_state_helpers():
    rm = RewardMachine(_minigrid_style_cfg())
    assert rm.state_id == "u5"
    assert rm.state_idx == 0
    assert rm.is_final is False


def test_requires_exactly_one_initial_state():
    cfg = {"states": [{"id": "a"}, {"id": "b"}]}
    with pytest.raises(ValueError):
        RewardMachine(cfg)
    cfg = {"states": [{"id": "a", "initial": True}, {"id": "b", "initial": True}]}
    with pytest.raises(ValueError):
        RewardMachine(cfg)


def test_requires_states_and_required_transition_fields():
    with pytest.raises(ValueError):
        RewardMachine({"states": []})
    cfg = _minigrid_style_cfg()
    cfg["transitions"] = [{"from": "u5", "to": "u8"}]  # missing 'condition'
    with pytest.raises(ValueError):
        RewardMachine(cfg)


def test_rejects_unknown_state_in_transition():
    cfg = _minigrid_style_cfg()
    cfg["transitions"] = [{"from": "u5", "to": "nowhere", "condition": "c"}]
    with pytest.raises(ValueError):
        RewardMachine(cfg)


def test_reset_returns_to_initial():
    rm = RewardMachine(_minigrid_style_cfg())
    rm.step(["door_is_closed"])  # u5 -> u8
    assert rm.state_id == "u8"
    rm.reset()
    assert rm.state_id == "u5"


# -------------------------------------------------------------------- transitions
def test_stays_and_issues_state_reward_when_no_transition_activated():
    rm = RewardMachine(_minigrid_style_cfg())
    # no label activates u5's outgoing transitions
    reward = rm.step(["nonsense"])
    assert reward == 0
    assert rm.state_id == "u5"


def test_takes_transition_and_issues_its_reward():
    rm = RewardMachine(_minigrid_style_cfg())
    reward = rm.step(["door_is_closed"])
    assert reward == -1
    assert rm.state_id == "u8"
    # u8 has state reward -1; no activated transition -> stays, issues -1
    reward = rm.step([])
    assert reward == -1
    assert rm.state_id == "u8"


def test_priority_selects_highest():
    cfg = _minigrid_style_cfg()
    cfg["transitions"] = [
        {"from": "u5", "to": "u6", "condition": "a", "reward": 1, "priority": 1},
        {"from": "u5", "to": "u8", "condition": "a", "reward": -1, "priority": 2},
    ]
    rm = RewardMachine(cfg)
    reward = rm.step(["a"])
    assert reward == -1
    assert rm.state_id == "u8"


def test_priority_tie_raises():
    cfg = _minigrid_style_cfg()
    cfg["transitions"] = [
        {"from": "u5", "to": "u6", "condition": "a", "priority": 3},
        {"from": "u5", "to": "u8", "condition": "a", "priority": 3},
    ]
    rm = RewardMachine(cfg)
    with pytest.raises(ValueError):
        rm.step(["a"])


def test_only_one_transition_taken_per_step():
    # even if multiple labels are true, only the highest-priority one wins
    cfg = _minigrid_style_cfg()
    cfg["transitions"] = [
        {"from": "u5", "to": "u6", "condition": "a", "reward": 1, "priority": 5},
        {"from": "u5", "to": "u8", "condition": "b", "reward": -1, "priority": 1},
    ]
    rm = RewardMachine(cfg)
    reward = rm.step(["a", "b"])
    assert reward == 1
    assert rm.state_id == "u6"


def test_irrelevant_labels_ignored():
    rm = RewardMachine(_minigrid_style_cfg())
    reward = rm.step(["has_key_door_open"])  # not an outgoing transition from u5
    assert reward == 0
    assert rm.state_id == "u5"


# ---------------------------------------------------------------------------- final
def test_final_state_flag():
    cfg = _minigrid_style_cfg()
    # go u5 -> u6 -> u7 (final)
    rm = RewardMachine(cfg)
    rm.step(["not_has_key_door_open_no_box"])  # u5 -> u6
    assert rm.is_final is False
    reward = rm.step(["has_box"])  # u6 -> u7
    assert reward == 1
    assert rm.state_id == "u7"
    assert rm.is_final is True
