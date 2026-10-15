"""Shared fixtures: a synthetic training folder tree (checkpoint res files)
used by the plot and select-model tests. No real model or training needed.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

import rlrom.utils as rlu

MODEL_NAME = "plot_test_model"

# (training num, date, {steps: mean_ep_rew}); training0 has a deliberate
# tie at 32.0 (reached first at step 200) to pin the tie-breaking rule.
TRAININGS = {
    0: {"date": "2026_01_01", "rews": {100: 10.0, 200: 32.0, 300: 32.0}},
    1: {"date": "2026_01_02", "rews": {100: 11.0, 200: 21.0, 300: 22.0}},
}


def training_folder_name(train):
    return f"{MODEL_NAME}_{TRAININGS[train]['date']}__training{train}"


def _write_res(path, mean_rew, formula_ratio=0.0):
    res = {
        "res": {"ep_len": [10, 12], "ep_rew": [10.0, 12.0]},
        "res_all_ep": {
            "basics": {"mean_ep_len": 11.0, "mean_ep_rew": mean_rew},
            "eval_formulas": {"phi": {"ratio_init_sat": formula_ratio,
                                      "mean_init_rob": -0.5}},
            "reward_formulas": {"psi": {"mean_sum": 1.5 * mean_rew}},
        },
    }
    with open(path, "w") as f:
        rlu.yaml.dump(res, f)


@pytest.fixture
def training_tree(tmp_path):
    """Two synthetic trainings, three checkpoint evals each."""
    saved_cwd = os.getcwd()
    model_root = tmp_path / "models"
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()

    for train, info in TRAININGS.items():
        d = model_root / training_folder_name(train)
        d.mkdir(parents=True)
        for step, rew in info["rews"].items():
            _write_res(d / f"res_step_{step}.yml", rew, formula_ratio=step / 1000.0)

    cfg_file = cfg_dir / "cfg.yml"
    cfg_file.write_text(
        "env_name: CartPole-v1\n"
        f"model_name: {MODEL_NAME}\n"
        f"model_path: {model_root}\n"
    )
    os.chdir(cfg_dir)
    try:
        yield cfg_file
    finally:
        os.chdir(saved_cwd)


@pytest.fixture
def cfg(training_tree):
    return rlu.load_cfg(str(training_tree))
