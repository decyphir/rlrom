"""Select a trained model from checkpoint results and make it active.

Selection is done on the checkpoint result DataFrames (see
:mod:`rlrom.utils`: ``get_selection_df`` / ``get_best_models``). The
selected model is made "active" by symlinking it to
``<model_path>/<model_name>.zip`` (and ``.yml``) with
``rlrom.utils.set_active_model``, which is what ``rlr test`` loads when no
explicit ``model_file`` is given.
"""
import os
import sys
import argparse

import pandas as pd

from rlrom.rlrom_plot import add_results_arguments, _parse_trainings, _resolve_trainings, _merge_negative_values


def add_select_model_options(parser, prefix=''):
    """Add the select-model specific options (min, step, show-top,
    show-selected)."""
    parser.add_argument('--min', action='store_true',
                        help=f"{prefix}Select the minimum instead of the maximum value of the metric.")
    parser.add_argument('--step', type=int, default=None,
                        help=f"{prefix}Select the first checkpoint with steps greater than this value "
                             f"(overrides --metric and --min).")
    parser.add_argument('--show-top', nargs='?', const=5, type=int, default=None,
                        help=f"{prefix}Show the top N checkpoints ranked by metric (default 5) and exit "
                             f"without selecting. Cannot be combined with --step.")
    parser.add_argument('--show-selected', action='store_true',
                        help=f"{prefix}Show the currently selected (active) model: the symbolic link, "
                             f"the file it points to, and its metric value. Does not change the "
                             f"selection. Cannot be combined with --step.")


def add_select_model_arguments(parser, prefix=''):
    """Add all the select-model options (shared results options +
    select-model options), as used by `rlrom_select_model`."""
    add_results_arguments(parser, prefix)
    add_select_model_options(parser, prefix)


def _display_df(dff, metric):
    """Build a DataFrame for display: integer training_idx and short model
    paths ('model_folder/model_name.zip')."""
    import rlrom.utils as rlu
    out = pd.DataFrame()
    out['steps'] = dff['steps']
    nums = [rlu.get_training_num(t) for t in dff['training_idx']]
    out['training_idx'] = [n if n is not None else t for n, t in zip(nums, dff['training_idx'])]
    out[metric] = rlu.get_metric_series(dff, metric)
    out['mdl_path'] = [rlu.short_model_path(p) for p in dff['mdl_path']]
    return out


def _show_top(df, metric, mode, top):
    import rlrom.utils as rlu
    try:
        ranking = rlu.get_model_ranking(df, metric=metric, mode=mode, top=top)
    except (KeyError, ValueError) as e:
        print(f'Error: {e}')
        return 1
    print(f'Top {len(ranking)} checkpoints by {metric} ({mode}):')
    print(_display_df(ranking, metric).to_string(index=False))
    return 0


def _show_selected(cfg, metric):
    import rlrom.utils as rlu
    mdl_path, cfg_path = rlu.get_model_fullpath(cfg)
    if not (os.path.islink(mdl_path) or os.path.exists(mdl_path)):
        print(f'No active model: {mdl_path} does not exist.')
        print('Use e.g. `rlr select-model cfg.yml` to select one.')
        return 1
    real = os.path.realpath(mdl_path)
    print('Active model:')
    kind = 'symlink' if os.path.islink(mdl_path) else 'file'
    print(f'  {kind:7} {rlu.short_model_path(mdl_path)} -> {rlu.short_model_path(real)}')
    if os.path.islink(cfg_path) or os.path.exists(cfg_path):
        print(f'  {"cfg":7} {rlu.short_model_path(cfg_path)} -> {rlu.short_model_path(os.path.realpath(cfg_path))}')
    df = rlu.get_selection_df(cfg, 'all')
    dff = df[df['path'] + '/' + df['model_files'] == real]
    if dff.empty:
        print('  No checkpoint results found for this model (not a checkpoint?).')
        return 0
    r = dff.sort_values('steps').iloc[0]
    val = rlu.get_metric_series(dff.sort_values('steps'), metric).iloc[0]
    print(f"  steps: {int(r['steps'])} | training_idx: {rlu.get_training_num(r['training_idx'])} | "
          f"{metric}: {val}")
    return 0


def main_select_model(cfg, trainings='last', metric=None, min_=False, step=None,
                      top=None, show_selected=False, list_metrics=False, verbose=0):
    """Select a model from the checkpoint results of cfg and make it active.

    cfg: loaded configuration (dict) with model_path/model_name as used by
    `rlr train`.
    trainings: 'last', 'all' or a comma-separated list of indices and/or
    inclusive ranges into the (date, num) sorted training folders (negative
    indices count from the end, e.g. '-1' or '-5:-1' for the last 5; the pool).
    metric: metric name (flat or 'formula/metric'), as given by --metric
    (None or a list, since the CLI collects it with action='append').
    min_: select the minimum instead of the maximum.
    step: if not None, select the first checkpoint with steps > step.
    top: if not None, print the top-N ranking and exit without selecting.
    show_selected: if True, print the currently selected (active) model and
    exit without changing the selection.
    list_metrics: print the available metric names and exit.

    Returns 0 on success, 1 on error.
    """
    import rlrom.utils as rlu

    if metric is None:
        metric = 'mean_ep_rew'
    elif isinstance(metric, list):
        if len(metric) != 1:
            print('Error: select-model takes a single --metric.')
            return 1
        metric = metric[0]

    df_trainings = rlu.get_training_folders(cfg)
    if df_trainings.empty:
        print('Error: no training folders with checkpoint results found for this configuration.')
        return 1

    if verbose >= 1:
        print(df_trainings[['date', 'num', 'best_reward', 'path']].to_string())

    try:
        sel = _parse_trainings(trainings)
        sel = _resolve_trainings(sel, len(df_trainings))
    except ValueError as e:
        print(f'Error: {e}')
        return 1

    if list_metrics:
        df = rlu.get_selection_df(df_trainings, sel)
        print('Available metrics:')
        for m in rlu.list_available_metrics(df):
            print(' -', m)
        return 0

    if show_selected:
        if step is not None:
            print('Error: --show-selected cannot be combined with --step.')
            return 1
        rc = _show_selected(cfg, metric)
        if top is not None:
            df = rlu.get_selection_df(df_trainings, sel)
            rc = _show_top(df, metric, 'min' if min_ else 'max', top) or rc
        return rc

    df = rlu.get_selection_df(df_trainings, sel)
    mode = 'min' if min_ else 'max'

    if step is not None:
        if top is not None:
            print('Error: --show-top cannot be combined with --step.')
            return 1
        dff = rlu.add_model_paths(df[df['steps'] > step]).sort_values('steps')
        if dff.empty:
            print(f'Error: no checkpoint with steps > {step}.')
            return 1
        print(f'Selected first checkpoint with steps > {step}:')
        print(_display_df(dff, metric).to_string(index=False))
        cfg_file, mdl_file = dff['cfg_path'].iloc[0], dff['mdl_path'].iloc[0]
    elif top is not None:
        return _show_top(df, metric, mode, top)
    else:
        try:
            cfg_file, mdl_file, dff = rlu.get_best_models(df, metric=metric, mode=mode)
        except (KeyError, ValueError) as e:
            print(f'Error: {e}')
            return 1
        print(f'Selected checkpoint with best {metric} ({mode}):')
        print(_display_df(dff, metric).to_string(index=False))

    rlu.set_active_model(cfg, cfg_file, mdl_file)
    return 0


def main():
    # rlrom_select_model cfg_main.yml [--trainings last|all|0,2,5] [-m METRIC] [--min] [--step N]
    #                        [--show-top N] [--show-selected]

    parser = argparse.ArgumentParser(description='Select a trained model from checkpoint results '
                                                 'and make it active (symlink for rlr test).')
    parser.add_argument('main_cfg', type=str, help='Path to the (training) configuration file in YAML format.')
    add_select_model_arguments(parser)
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level (1: list the training folders).')
    args = parser.parse_args(_merge_negative_values(sys.argv[1:]))

    from rlrom.cfg_utils import load_cfg
    if os.path.exists(args.main_cfg):
        cfg = load_cfg(args.main_cfg)
    else:
        print(f"Error: Config file {args.main_cfg} was not found.")
        sys.exit(1)

    sys.exit(main_select_model(cfg, trainings=args.trainings, metric=args.metric,
                               min_=args.min, step=args.step, top=args.show_top,
                               show_selected=args.show_selected,
                               list_metrics=args.list_metrics, verbose=args.verbose))


if __name__ == "__main__":
    main()
