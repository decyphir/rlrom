"""Tests for the MuJoCo environment add-on.

Only the id resolution is exercised: it is pure logic and needs neither the
`mujoco` extra nor a download, so the suite stays runnable on a bare install.
"""

from rlrom.envs.mujoco import _env_id, resolve_env_id


def test_legacy_versions_map_to_the_gymnasium_port():
    assert resolve_env_id("Hopper-v3") == "Hopper-v4"
    assert resolve_env_id("Hopper-v2") == "Hopper-v4"
    assert resolve_env_id("HalfCheetah-v3") == "HalfCheetah-v4"
    assert resolve_env_id("InvertedDoublePendulum-v3") == "InvertedDoublePendulum-v4"


def test_current_ids_are_untouched():
    assert resolve_env_id("Hopper-v4") == "Hopper-v4"
    assert resolve_env_id("Hopper-v5") == "Hopper-v5"
    assert resolve_env_id("Walker2d-v5") == "Walker2d-v5"


def test_unported_environments_are_untouched():
    # not a ported family: rewriting it would produce a wrong id
    assert resolve_env_id("FetchReach-v3") == "FetchReach-v3"
    assert resolve_env_id("CartPole-v1") == "CartPole-v1"
    assert resolve_env_id("MiniGrid-Empty-5x5-v0") == "MiniGrid-Empty-5x5-v0"


def test_env_id_override_wins_over_env_name():
    assert _env_id({"env_name": "Hopper-v3", "env_id": "Hopper-v5"}) == "Hopper-v5"
    assert _env_id({"env_name": "Hopper-v3"}) == "Hopper-v4"


def test_missing_env_name_and_override_raise():
    import pytest

    with pytest.raises(ValueError):
        _env_id({})
