"""Smoke tests for the training results plotting pipeline (rlr plot)."""
import pytest

import rlrom.utils as rlu
import rlrom.plots as rlp
from rlrom.rlrom_plot import main_plot, _parse_trainings, _resolve_trainings, _merge_negative_values

from conftest import training_folder_name

T0 = training_folder_name(0)
T1 = training_folder_name(1)


def test_get_training_folders(cfg):
    df = rlu.get_training_folders(cfg)
    assert len(df) == 2
    # sorted by (date, num)
    assert df["num"].tolist() == [0, 1]
    assert df["best_reward"].tolist() == [32.0, 22.0]


def test_get_training_res_last(cfg):
    df = rlu.get_training_res(cfg, -1)
    assert len(df) == 3
    assert df["steps"].tolist() == [100, 200, 300]
    assert df["mean_ep_rew"].tolist() == [11.0, 21.0, 22.0]
    assert df["training_idx"].nunique() == 1


def test_get_training_res_positions(cfg):
    df = rlu.get_training_res(cfg, [0])
    assert len(df) == 3
    assert df["mean_ep_rew"].tolist() == [10.0, 32.0, 32.0]


def test_get_df_all_trainings(cfg):
    """Regression: used to crash with a TypeError (label passed to .iloc)."""
    df = rlu.get_df_all_trainings(cfg)
    assert len(df) == 6
    assert df["training_idx"].nunique() == 2
    assert sorted(df["steps"].unique()) == [100, 200, 300]


def test_get_metric_series(cfg):
    df = rlu.get_df_all_trainings(cfg)
    assert rlu.get_metric_series(df, "mean_ep_rew").tolist() == df["mean_ep_rew"].tolist()
    s = rlu.get_metric_series(df, "phi/ratio_init_sat")
    assert len(s) == 6
    # training 0 has lower steps-wise rewards and the ratio tracks step/1000
    assert abs(s.iloc[0] - 0.1) < 1e-9
    s2 = rlu.get_metric_series(df, "psi/mean_sum")
    assert abs(s2.iloc[0] - 15.0) < 1e-9
    with pytest.raises(KeyError):
        rlu.get_metric_series(df, "does_not_exist")


def test_list_available_metrics(cfg):
    df = rlu.get_training_res(cfg, -1)
    ms = rlu.list_available_metrics(df)
    assert "mean_ep_rew" in ms
    assert "mean_ep_len" in ms
    assert "phi/ratio_init_sat" in ms
    assert "psi/mean_sum" in ms


def _run(cfg, tmp_path, name, **kargs):
    out = tmp_path / f"{name}.png"
    rc = main_plot(cfg, output=str(out), **kargs)
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_last_training(cfg, training_tree):
    _run(cfg, training_tree.parent, "last")


def test_plot_all_trainings(cfg, training_tree):
    _run(cfg, training_tree.parent, "all", trainings="all",
         metrics=["mean_ep_rew", "phi/ratio_init_sat"])


def test_plot_envelope(cfg, training_tree):
    _run(cfg, training_tree.parent, "envelope", envelope=True)


def test_plot_selected_trainings(cfg, training_tree):
    _run(cfg, training_tree.parent, "sel", trainings="0,1", envelope=True)


def test_plot_bad_index(cfg, training_tree):
    assert main_plot(cfg, output=str(training_tree.parent / "x.png"),
                     trainings="42") == 1


def test_rlrom_plot_module_e2e(training_tree, tmp_path):
    """End to end through argparse: python -m rlrom.rlrom_plot."""
    import subprocess
    import sys
    out = tmp_path / "e2e.png"
    r = subprocess.run(
        [sys.executable, "-m", "rlrom.rlrom_plot", str(training_tree),
         "--trainings", "all", "--envelope", "-o", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()


# ------------------------------------------------- --trainings parsing helpers

@pytest.mark.parametrize("line, expected", [
    ("x,theta", ["x", "theta"]),
    ("x_dot", ["x_dot"]),
    ("label_formulas/go_left", ["label_formulas/go_left"]),
    ("obs_formulas/f1, reward_formulas/f2", ["obs_formulas/f1", "reward_formulas/f2"]),
    ("rm_states/left_right", ["rm_states/left_right"]),
    ("rm_reward/left_right:weighted", ["rm_reward/left_right:weighted"]),
    ("total_reward, left_right:weighted", ["total_reward", "left_right:weighted"]),
    ("rho(phi, 50), sat(gamma)", ["rho(phi, 50)", "sat(gamma)"]),
    ("set_ep_idx(0)", ["set_ep_idx(0)"]),
])
def test_get_layout_from_string(line, expected):
    """cfg_test.plot signal names must survive the layout tokenizer intact."""
    assert rlp.get_layout_from_string(line) == [expected]


def test_get_layout_from_string_multi_line():
    lay = rlp.get_layout_from_string("x,theta\n\nx_dot")
    assert lay == [["x", "theta"], ["x_dot"]]


@pytest.mark.parametrize("spec, n, expected", [
    ('last', 5, 'last'),
    ('all', 5, 'all'),
    ('0,2', 5, [0, 2]),
    ('-1', 5, [4]),
    ('-1,-2', 5, [4, 3]),
    ('-5:-1', 8, [3, 4, 5, 6, 7]),
    ('-2:-1', 2, [0, 1]),
    ('2:', 5, [2, 3, 4]),
    (':2', 5, [0, 1, 2]),
    ('2:5', 6, [2, 3, 4, 5]),
    ('0,-2,3', 5, [0, 3]),      # duplicates removed, order kept
    ('-1,-1', 5, [4]),
])
def test_parse_and_resolve_trainings(spec, n, expected):
    assert _resolve_trainings(_parse_trainings(spec), n) == expected


@pytest.mark.parametrize("spec, n", [
    ('7', 5),        # out of range
    ('-6', 5),       # out of range (negative)
    ('3:1', 5),      # empty range
    ('-9:-5', 5),    # out of range (negative)
    ('', 5),         # empty string
    ('1:2:3:4', 5),  # too many range parts
    ('::2', 5),      # step not supported
])
def test_parse_and_resolve_trainings_errors(spec, n):
    with pytest.raises(ValueError):
        _resolve_trainings(_parse_trainings(spec), n)


@pytest.mark.parametrize("argv, expected", [
    (['--trainings', '-1,-2', '-m', 'x'], ['--trainings=-1,-2', '-m', 'x']),
    (['--trainings', '-5:-1'], ['--trainings=-5:-1']),
    (['--trainings', '-1'], ['--trainings=-1']),
    (['--trainings', 'all'], ['--trainings', 'all']),       # positive: untouched
    (['--trainings', '-m'], ['--trainings', '-m']),         # real flag: untouched
    (['--trainings', '--metric'], ['--trainings', '--metric']),  # long flag: untouched
])
def test_merge_negative_values(argv, expected):
    assert _merge_negative_values(argv) == expected
