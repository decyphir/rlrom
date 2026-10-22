"""Plot training results from checkpoint res files.

Produces line plots and min/max envelopes of metrics (episode reward,
episode length, STL formula metrics, ...) collected at each checkpoint
evaluation during `rlr train`.

The res files (``res_step_*.yml``) are produced by the training callbacks
and collected into DataFrames by :mod:`rlrom.utils`
(``get_training_folders`` / ``get_training_res`` / ``get_df_all_trainings``).
"""
import os
import sys
import re
import argparse


def _parse_trainings(s):
    """Parse the --trainings option: 'last', 'all', or a comma-separated
    list of indices and/or inclusive ranges (e.g. '0,2' or '-5:-1' for the
    last 5 trainings). Returns 'last'/'all' or a list of ints and/or slice
    objects; see _resolve_trainings()."""
    if s in ('last', 'all'):
        return s
    out = []
    for p in str(s).split(','):
        p = p.strip()
        if p == '':
            continue
        parts = p.split(':')
        if len(parts) > 3:
            raise ValueError(f"invalid --trainings range: '{p}'")
        if len(parts) == 1:
            out.append(int(p))
        else:
            vals = []
            for part in parts:
                vals.append(int(part) if part.strip() != '' else None)
            while len(vals) < 3:
                vals.append(None)
            if vals[2] is not None:
                raise ValueError(f"range step is not supported: '{p}'")
            out.append(slice(vals[0], vals[1]))
    if not out:
        raise ValueError(f"invalid --trainings value: '{s}'")
    return out


def _resolve_trainings(sel, n):
    """Resolve a parsed --trainings selection against n trainings.

    sel: the output of _parse_trainings(). Returns 'last'/'all' unchanged,
    or a list of unique non-negative positions (0..n-1). Negative indices
    count from the end; ranges are inclusive on both ends (e.g. '-5:-1' is
    the last 5). Raises ValueError on out-of-range indices or empty ranges.
    """
    if sel in ('last', 'all'):
        return sel

    def fix(v, default):
        v = default if v is None else v
        orig = v
        if v < 0:
            v += n
        if not (0 <= v < n):
            raise ValueError(f'training index {orig} out of range (0..{n - 1}, or negative from the end)')
        return v

    out = []
    for part in sel:
        if isinstance(part, slice):
            a = fix(part.start, 0)
            b = fix(part.stop, n - 1)
            if a > b:
                raise ValueError(f'empty range (start {a} > stop {b})')
            out.extend(range(a, b + 1))
        else:
            out.append(fix(part, part))
    out = list(dict.fromkeys(out))  # unique, order preserved
    if not out:
        raise ValueError('no training selected')
    return out


_NEG_TRAININGS_VALUE = re.compile(r'^-\d[\d,:-]*$')


def _merge_negative_values(argv, opt_names=('--trainings',)):
    """argparse rejects option values that look like options (e.g.
    `--trainings -1,-2` or `--trainings -5:-1`); rewrite them in the
    unambiguous `--opt=value` form before parsing. Only tokens that look
    like negative training selections are merged, so real flags (e.g. -m)
    are left alone."""
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in opt_names and i + 1 < len(argv) and _NEG_TRAININGS_VALUE.match(argv[i + 1]):
            out.append(f'{a}={argv[i + 1]}')
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def add_results_arguments(parser, prefix=''):
    """Add the results options shared by the plot and select-model actions
    (`rlr plot`, `rlr select-model`, `rlrom_plot`, `rlrom_select_model`).

    prefix: e.g. '[plot] ' to tag the options when they are mixed with
    options of other actions in a flat parser.
    """
    parser.add_argument('--trainings', type=str, default='last',
                        help=f"{prefix}Trainings to consider: 'last' (default, most recent), 'all', or a "
                             f"comma-separated list of indices (negative count from the end) and/or "
                             f"inclusive ranges, e.g. '0,2' or '-5:-1' for the last 5; indices refer to "
                             f"the (date, num) sorted list of training folders (see --verbose 1).")
    parser.add_argument('--metric', '-m', action='append', default=None,
                        help=f"{prefix}Metric name: a flat name such as mean_ep_rew or mean_ep_len, or a "
                             f"formula/metric pair such as phi_left_goal/ratio_init_sat "
                             f"(default: mean_ep_rew). Repeatable for plot, single for select-model. "
                             f"Use --list-metrics to see available names.")
    parser.add_argument('--list-metrics', action='store_true',
                        help=f"{prefix}List the metrics available in the results and exit.")


def add_plot_options(parser, prefix=''):
    """Add the plot-specific options (envelope, output, ...)."""
    parser.add_argument('--envelope', action='store_true',
                        help=f"{prefix}Plot the mean/min-max envelope band across trainings. With "
                             f"--trainings last, all trainings are used; with --trainings all, the "
                             f"per-training lines are shown as well.")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help=f"{prefix}Save the figure to this file (png/pdf/svg) instead of showing it.")
    parser.add_argument('--title', type=str, default=None,
                        help=f"{prefix}Figure title.")
    parser.add_argument('--log-y', action='store_true',
                        help=f"{prefix}Logarithmic y scale.")


def add_plot_arguments(parser, prefix=''):
    """Add all the plotting options (shared results options + plot options),
    as used by `rlrom_plot`."""
    add_results_arguments(parser, prefix)
    add_plot_options(parser, prefix)


def main_plot(cfg, trainings='last', metrics=None, envelope=False,
              output=None, title=None, log_y=False, list_metrics=False, verbose=0):
    """Plot training results from the checkpoint res files of cfg.

    cfg: loaded configuration (dict) with model_path/model_name as used by
    `rlr train`.
    trainings: 'last', 'all' or a comma-separated list of indices and/or
    inclusive ranges into the (date, num) sorted training folders (negative
    indices count from the end, e.g. '-1' or '-5:-1' for the last 5).
    metrics: metric name(s) as understood by rlrom.utils.get_metric_series.
    envelope: plot the mean/min-max band across the selected trainings
    (implies 'all' when trainings is 'last').
    output: if given, save the figure there instead of showing it.

    Returns 0 on success, 1 on error.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import rlrom.utils as rlu
    import rlrom.plots as rlp

    if metrics is None:
        metrics = ['mean_ep_rew']

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
    pool = 'all' if (sel == 'all' or (envelope and sel == 'last')) else sel
    df = rlu.get_selection_df(df_trainings, pool)

    if list_metrics:
        print('Available metrics:')
        for m in rlu.list_available_metrics(df):
            print(' -', m)
        return 0

    style = 'line'
    if envelope:
        style = 'both' if sel == 'all' else 'envelope'

    fig = rlp.plot_training_results(df, metrics=metrics, style=style,
                                    title=title, log_y=log_y)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f'saved plot to {output}')
    elif matplotlib.get_backend().lower() == 'agg':
        default_out = 'rlr_plot.png'
        fig.savefig(default_out, dpi=150, bbox_inches='tight')
        print(f'no interactive display available, saved plot to {default_out}')
    else:
        plt.show()
        
    return 0


def main():
    # rlrom_plot cfg_main.yml [--trainings last|all|0,2,5] [-m METRIC]... [--envelope] [-o out.png]

    parser = argparse.ArgumentParser(description='Plot training results from checkpoint res files.')
    parser.add_argument('main_cfg', type=str, help='Path to the (training) configuration file in YAML format.')
    add_plot_arguments(parser)
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level (1: list the training folders).')
    args = parser.parse_args(_merge_negative_values(sys.argv[1:]))

    from rlrom.cfg_utils import load_cfg
    if os.path.exists(args.main_cfg):
        cfg = load_cfg(args.main_cfg)
    else:
        print(f"Error: Config file {args.main_cfg} was not found.")
        sys.exit(1)

    sys.exit(main_plot(cfg, trainings=args.trainings, metrics=args.metric,
                       envelope=args.envelope, output=args.output, title=args.title,
                       log_y=args.log_y, list_metrics=args.list_metrics, verbose=args.verbose))


if __name__ == "__main__":
    main()
