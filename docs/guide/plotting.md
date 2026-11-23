# Plotting and results

Two different things are plotted, from two different commands:

- **episode plots** — signals and formula values over the steps of a test
  episode, configured in `cfg_test.plot` and drawn by `rlr test`;
- **training curves** — metrics against training steps, from the checkpoints'
  result files, drawn by `rlr plot`.

## Episode plots

```yaml
cfg_test:
  num_ep: 1
  num_steps: 500
  plot:
    - title: Observations
      out: plots/obs.png          # optional: also save the figure
      ax:                         # one axis per list item, stacked, shared x
        - x, theta                # comma-separated signals on one axis
        - x_dot
    - title: Formulas
      ax:
        - monitor_formulas/go_left
        - [monitor_formulas/cart_in_A, monitor_formulas/stay_A]
```

Figures are drawn for the first episode after all episodes have run. Signal names
come from the grammar below.

### Signal names

| Name | What it plots |
|---|---|
| a declared signal | `action_names` / `obs_names` / `aux_sig_names` entries, and `env_reward` |
| `reward` | the shaped reward the agent saw |
| `monitor_formulas/<f>` | the per-step value recorded for that formula |
| `obs_formulas/<f>`, `reward_formulas/<f>` | same, for the deprecated sections |
| `eval_formulas/<f>` | offline robustness of the formula over the trace |
| `rm_states/<machine>` | reward-machine state index per step |
| `rm_reward/<machine>` | per-step reward the machine issued |
| `rho(<f>[,past_horizon])` | robustness, offline (sees the whole trace) |
| `rob(<f>[,past_horizon])` | alias of `rho` |
| `sat(<f>)` | satisfaction over time |
| `<name>:weighted` | the value multiplied by its configured weight |

Discrete signals — rewards, state indices, satisfaction, labels — are drawn as
steps, continuous ones with linear interpolation.

### Live plots

```yaml
cfg_test:
  plot:
    - title: Live observations
      live: True
      live_window: 100       # optional trailing window, in monitor time units
      out: plots/live.png
      ax:
        - x, theta
        - reward
```

A `live: true` item opens a window that updates every environment step, per
episode. During a live window: <kbd>space</kbd> pauses and resumes the updates
(the episode keeps running), <kbd>Esc</kbd> closes the windows and ends the
episode. On a headless backend the window is skipped and only `out:` is written.
Live items are not redrawn at the end of the run.

## Training curves

`rlr train` writes `res_step_*.yml` at every evaluation. `rlr plot` reads them.

```bash
rlr plot cfg.yml                                   # last training, mean_ep_rew
rlr plot cfg.yml --trainings all                   # every training, one line each
rlr plot cfg.yml --trainings all --envelope        # + mean/min-max band
rlr plot cfg.yml -m mean_ep_rew -m mean_ep_len     # one subplot per metric
rlr plot cfg.yml -m phi_left_goal/ratio_init_sat -o plot.png
```

- `--trainings last|all|0,2,-1|-5:-1` — indices follow the (date, number) order
  of the training folders; negative indices count from the end; ranges are
  inclusive.
- `-m NAME` — flat names (`mean_ep_rew`, `mean_ep_len`), or `formula/metric`
  pairs. `--list-metrics` prints what is available in the results at hand.
- `--envelope`, `-o FILE`, `--title`, `--log-y` as described in the
  [command line reference](cli.md).

The same plots are available programmatically through
`rlrom.plots.plot_training_results`, which takes the polars dataframe built from
the result files by `rlrom.utils.get_df_all_trainings`.

## Reading the metrics

Per-episode results live in `res` (arrays), aggregates in `res_all_ep`:

| Group | Contents |
|---|---|
| `basics` | `mean_ep_len`, `mean_ep_rew`, `completed`, plus `mean_<key>` for each `cfg_specs.episode_info_keys` |
| `reward_formulas` | per formula: `mean`, `sum`, `num_sat` |
| `eval_formulas` | per formula: `mean_rob`, `mean_sum`, `mean_num_sat`, `ratio_init_sat`, `mean_init_rob` |

`completed` is the fraction of episodes that reached `cfg_test.num_steps`
instead of terminating early — with a `terminal: true` formula it doubles as a
violation rate.
