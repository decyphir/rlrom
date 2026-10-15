"""Tests for model selection: rlrom.utils selection API and the
`rlr select-model` / `rlrom_select_model` commands.
"""
import os
import re

import pytest

import rlrom.utils as rlu
from rlrom.rlrom_select_model import main_select_model

from conftest import MODEL_NAME, training_folder_name

T0 = training_folder_name(0)
T1 = training_folder_name(1)


# ---------------------------------------------------------------- utils API

def test_get_best_models_default_last(cfg):
    # default: max mean_ep_rew in the last training
    cfg_file, mdl_file, dff = rlu.get_best_models(cfg)
    assert dff["steps"].tolist() == [300]
    assert dff["training_idx"].tolist() == [T1]
    assert mdl_file == os.path.join(T1, "model_step_300.zip") or \
        mdl_file.endswith(f"{T1}/model_step_300.zip")
    assert cfg_file.endswith(f"{T1}/cfg0.yml")


def test_get_best_models_all_tie_earliest_step(cfg):
    _, _, dff = rlu.get_best_models(cfg, 'all')
    # tie at 32.0 in training0: both rows kept, earliest step chosen
    assert dff["mean_ep_rew"].tolist() == [32.0, 32.0]
    assert dff["steps"].tolist() == [200, 300]
    assert all(t == T0 for t in dff["training_idx"])


def test_get_best_models_formula_metric(cfg):
    _, _, dff = rlu.get_best_models(cfg, 'all', metric='phi/ratio_init_sat')
    assert dff["steps"].tolist() == [300, 300]  # both trainings reach 0.3
    assert "mdl_path" in dff.columns and "cfg_path" in dff.columns


def test_get_best_models_min(cfg):
    _, _, dff = rlu.get_best_models(cfg, 'all', mode='min')
    assert dff["steps"].tolist() == [100]
    assert dff["training_idx"].tolist() == [T0]


def test_get_best_models_legacy_df_call(cfg):
    # notebook-style call: results df as first arg, train_idx ignored
    df_all = rlu.get_df_all_trainings(cfg)
    _, _, dff = rlu.get_best_models(df_all, 'all')
    assert dff["steps"].tolist() == [200, 300]


def test_get_best_models_bad_metric(cfg):
    with pytest.raises(KeyError):
        rlu.get_best_models(cfg, metric='nope')
    with pytest.raises(ValueError):
        rlu.get_best_models(cfg, mode='mean')


def test_get_model_ranking(cfg):
    df_all = rlu.get_df_all_trainings(cfg)
    top3 = rlu.get_model_ranking(df_all, top=3)
    assert rlu.get_metric_series(top3, 'mean_ep_rew').tolist() == [32.0, 32.0, 22.0]
    top2min = rlu.get_model_ranking(df_all, mode='min', top=2)
    assert rlu.get_metric_series(top2min, 'mean_ep_rew').tolist() == [10.0, 11.0]


def test_get_step_model(cfg):
    cfg_file, mdl_file = rlu.get_step_model(cfg, 150)
    assert mdl_file.endswith(f"{T1}/model_step_200.zip")
    assert cfg_file.endswith(f"{T1}/cfg0.yml")
    with pytest.raises(ValueError):
        rlu.get_step_model(cfg, 10**9)


def test_set_active_model(cfg):
    df = rlu.get_training_res(cfg, -1)
    row = df.iloc[-1]
    rlu.set_active_model(cfg, row['path'] + '/cfg0.yml', row['path'] + '/model_step_300.zip')
    mpath, cpath = rlu.get_model_fullpath(cfg)
    assert os.path.islink(mpath) and os.path.islink(cpath)
    assert os.readlink(mpath) == row['path'] + '/model_step_300.zip'
    assert os.readlink(cpath) == row['path'] + '/cfg0.yml'


# ------------------------------------------------------------- main_select_model

def _active_target(cfg):
    mpath, _ = rlu.get_model_fullpath(cfg)
    return os.readlink(mpath) if os.path.islink(mpath) else None


def test_main_select_model_default(cfg):
    assert main_select_model(cfg) == 0
    assert _active_target(cfg).endswith(f"{T1}/model_step_300.zip")


def test_main_select_model_formula_metric_all(cfg):
    assert main_select_model(cfg, trainings='all', metric='phi/ratio_init_sat') == 0
    assert _active_target(cfg).endswith('/model_step_300.zip')


def test_main_select_model_min(cfg):
    assert main_select_model(cfg, min_=True) == 0
    assert _active_target(cfg).endswith(f"{T1}/model_step_100.zip")


def test_main_select_model_step(cfg):
    assert main_select_model(cfg, step=150) == 0
    assert _active_target(cfg).endswith(f"{T1}/model_step_200.zip")


def test_main_select_model_top_no_symlink(cfg):
    assert main_select_model(cfg, top=2) == 0
    assert _active_target(cfg) is None


def test_main_select_model_errors(cfg):
    assert main_select_model(cfg, metric=['a', 'b']) == 1
    assert main_select_model(cfg, step=150, top=2) == 1
    assert main_select_model(cfg, step=150, show_selected=True) == 1
    assert main_select_model(cfg, step=10**9) == 1
    assert main_select_model(cfg, metric='nope') == 1
    assert main_select_model(cfg, trainings='42') == 1


def test_show_top_display_format(cfg, capsys):
    assert main_select_model(cfg, trainings='all', top=2) == 0
    out = capsys.readouterr().out
    # only check the table (skip the 'Scanning models in ...' progress line)
    out = out[out.index('Top 2 checkpoints'):] if 'Top 2 checkpoints' in out else ''
    # top 2 = steps 200 and 300 of training0 (tie at 32.0)
    assert f'{T0}/model_step_200.zip' in out
    assert f'{T0}/model_step_300.zip' in out
    # only short paths are shown, not the absolute (tmp) ones
    models_root = os.path.dirname(rlu.get_model_fullpath(cfg)[0])
    assert models_root not in out
    # training_idx is shown as an integer
    line = [l for l in out.splitlines() if 'model_step_200.zip' in l][0]
    assert re.search(r'200\s+0\s+32\.0', line)


def test_show_selected_no_active_model(cfg, capsys):
    assert main_select_model(cfg, show_selected=True) == 1
    assert 'No active model' in capsys.readouterr().out


def test_show_selected(cfg, capsys):
    assert main_select_model(cfg) == 0  # selects T1 step 300
    capsys.readouterr()
    assert main_select_model(cfg, show_selected=True) == 0
    out = capsys.readouterr().out
    assert 'Active model' in out
    assert f'models/{MODEL_NAME}.zip -> ' in out          # the symbolic link
    assert f'{T1}/model_step_300.zip' in out              # the file it points to
    assert 'mean_ep_rew: 22.0' in out                     # its metric value


def test_show_selected_metric(cfg, capsys):
    assert main_select_model(cfg) == 0
    capsys.readouterr()
    assert main_select_model(cfg, show_selected=True, metric='phi/ratio_init_sat') == 0
    assert 'phi/ratio_init_sat: 0.3' in capsys.readouterr().out


def test_show_selected_with_top(cfg, capsys):
    assert main_select_model(cfg) == 0
    capsys.readouterr()
    assert main_select_model(cfg, show_selected=True, top=2) == 0
    out = capsys.readouterr().out
    assert 'Active model' in out and 'Top 2 checkpoints' in out


def test_main_select_model_list_metrics(cfg, capsys):
    assert main_select_model(cfg, list_metrics=True) == 0
    out = capsys.readouterr().out
    assert 'mean_ep_rew' in out and 'phi/ratio_init_sat' in out and 'psi/mean_sum' in out


def test_rlrom_select_model_module_e2e(training_tree):
    """End to end through argparse: python -m rlrom.rlrom_select_model."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "rlrom.rlrom_select_model", str(training_tree),
         "--show-top", "3", "--trainings", "all"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert 'Top 3 checkpoints by mean_ep_rew (max):' in r.stdout


def test_rlrom_select_model_negative_trainings_e2e(training_tree):
    # space-separated negative/range value (would be rejected by raw argparse)
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "rlrom.rlrom_select_model", str(training_tree),
         "--trainings", "-2:-1", "--show-top", "6"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert 'Top 6 checkpoints' in r.stdout
