"""Tests for rlrom.utils that have no environment/model dependencies."""
import rlrom.utils as rlu


def test_parse_signal_spec_simple():
    sig, args = rlu.parse_signal_spec("torque")
    assert sig == "torque"
    assert args == []


def test_parse_signal_spec_with_args():
    sig, args = rlu.parse_signal_spec("rho(phi_goal, 2)")
    assert sig == "rho"
    assert args == ["phi_goal", "2"]


def test_get_formulas():
    specs = """
signal x, reward
phi_a := x[t] > 0
phi_b := alw_[0,5] phi_a
"""
    assert rlu.get_formulas(specs) == ["phi_a", "phi_b"]


def test_set_rec_cfg_field():
    cfg = {"a": 1, "nested": {"b": 2, "deeper": {"c": 3}}}
    out = rlu.set_rec_cfg_field(cfg, b=99, c=100)
    assert out["nested"]["b"] == 99
    assert out["nested"]["deeper"]["c"] == 100
    assert out["a"] == 1


def test_load_cfg_example_file():
    """Loading a real example cfg expands .yml/.stl fields recursively."""
    import os
    # NOTE: verbose must be >=1 or nested .yml/.stl fields are NOT expanded
    cfg = rlu.load_cfg(os.path.join("examples", "cartpole", "cfg0tr_ppo_specs.yml"))
    # load_cfg chdir's to the cfg folder; restore so other tests aren't affected
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    assert cfg["env_name"] == "CartPole-v1"
    # specs field expanded from cartpole.stl
    assert "cartpole.stl" not in cfg["cfg_specs"]["specs"]
    assert "phi_left_goal" in cfg["cfg_specs"]["specs"]
    # nested .yml expanded
    assert isinstance(cfg["cfg_train"]["algo"]["ppo"], dict)
