# Command line

One entry point, one parser, five actions:

```text
rlr <action> <cfg.yml> [options]
```

```bash
rlr test         examples/cartpole/cfg0_hug.yml
rlr train        examples/cartpole/cfg0tr_ppo_specs.yml
rlr show         examples/highway_env/cfg0tr.yml --expand
rlr plot         examples/cartpole/cfg0tr_ppo_specs.yml --trainings all --envelope
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml -m mean_ep_rew
```

Every action accepts every option; the `[ … ]` tag in the table says which action
an option affects. `rlr --help` prints the same list.

| Option | Actions | Meaning |
|---|---|---|
| `action` | all | `test`, `train`, `show`, `plot` or `select-model` |
| `main_cfg` | all | path to the main configuration file (default `cfg_main.yml`) |
| `--cfg-train FILE` | test, train | override the `cfg_train` section |
| `--cfg-test FILE` | test, train | override the `cfg_test` section |
| `--cfg-specs FILE` | test, train | override the `cfg_specs` section |
| `--set-params [k=v …]` | test, train | override keys that already exist |
| `--verbose N` | all | verbosity level |
| `--expand` | show | resolve nested `.yml`/`.stl`/`import_module` references |
| `--subcfg KEY` | show | print one sub-configuration (dot path, or first match at any depth) |
| `--trainings SEL` | plot, select-model | `last` (default), `all`, or indices/ranges: `0,2`, `-5:-1` |
| `--metric M`, `-m M` | plot, select-model | metric name; repeatable for `plot` |
| `--list-metrics` | plot, select-model | list the metric names found in the results |
| `--envelope` | plot | add the mean/min-max band across trainings |
| `-o FILE`, `--output FILE` | plot | save to `png`/`pdf`/`svg` instead of showing |
| `--title TEXT` | plot | figure title |
| `--log-y` | plot | logarithmic y axis |
| `--min` | select-model | select the minimum instead of the maximum |
| `--step N` | select-model | first checkpoint after N steps (overrides the metric) |
| `--show-top [N]` | select-model | print the top N ranked checkpoints and exit |
| `--show-selected` | select-model | print the active model and its metric |

## `test`

Runs `cfg_test.num_ep` episodes of `cfg_test.num_steps` steps from
`cfg_test.init_seed`, loading `<model_path>/<model_name>.zip`, and prints one
summary line per episode plus the aggregate. With `cfg_test.plot`, figures are
produced at the end; with `live: true`, windows update during the episodes.

```bash
rlr test examples/cartpole/cfg0_hug.yml
rlr test examples/cartpole/cfg0tr_ppo_specs.yml --set-params num_ep=20 render_mode=None
```

A policy can also be `random` or keyboard-driven, depending on the config's
`model_name` / `cfg_env.keyboard_control`.

## `train`

Runs `cfg_train.total_timesteps` steps, evaluating every `cfg_train.eval_freq`
steps. Each evaluation writes `model_step_*.zip` and `res_step_*.yml` into
`models/<model_name>_<date>__training<N>/`, and the last one is published as
`models/<model_name>.zip`.

```bash
rlr train examples/cartpole/cfg0tr_ppo_specs.yml
rlr train examples/metadrive/cfg0tr_ppo.yml --cfg-train cfg_smoke_ppo_train.yml
```

## `show`

Prints a configuration. Without `--expand`, the raw top-level YAML, with nested
references left as paths. With `--expand`, the resolved tree. `--subcfg` narrows
it to one section.

```bash
rlr show examples/highway_env/cfg0tr.yml --expand --subcfg cfg_specs
```

`show` deliberately does not import the training stack, so it works in an
environment where only the light dependencies are installed.

## `plot`

Plots the `res_step_*.yml` files of past trainings. See
[Plotting and results](plotting.md).

## `select-model`

Chooses a checkpoint and makes it the active model, by updating the
`<model_name>.zip` / `.yml` pair (a symlink to the chosen checkpoint, plus its
config). Ties on the best value are broken by the earliest step.

```bash
rlr select-model cfg.yml                              # best mean_ep_rew, last training
rlr select-model cfg.yml -m phi_left_goal/ratio_init_sat --trainings all
rlr select-model cfg.yml --step 100000                # first checkpoint after 100k
rlr select-model cfg.yml --min                        # worst instead of best
rlr select-model cfg.yml --show-top 5 --trainings all # rank, do not select
rlr select-model cfg.yml --show-selected              # what is active now
```

## Standalone entry points

The same code is also installed under separate names, for scripts that prefer
one command per job: `rlrom_test`, `rlrom_train`, `rlrom_plot`,
`rlrom_select_model`. They take the same arguments as the corresponding `rlr`
action, without the action word.
