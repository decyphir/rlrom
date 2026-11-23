# Troubleshooting

## The agent behaves randomly after loading a checkpoint

`model_use_specs` disagrees with how the policy was trained. A policy trained on
wrapped observations expects the formula robustnesses (and possibly reward-machine
states) appended to the observation; loaded against the raw environment it sees a
different vector. Set it to `true` for policies trained with `cfg_specs`, `false`
for raw ones.

The same symptom follows from a checkpoint trained on a different observation
layout — see the next item.

## Observation fields are missing

`cfg_specs.flatten_obs` defaults to `true`: `dict` observations are flattened into
one vector, which drops the named fields such as `rm_states`. Set
`flatten_obs: false` if you need them, and keep in mind that the two layouts are
not interchangeable for a trained policy.

## `init_seeds` has no effect

The tester reads `init_seed` (singular). `init_seeds` in an example config is
ignored, so every run starts at seed 0. Use `init_seed`.

## Results are written somewhere unexpected

Loading a config changes the process's working directory to the config file's
folder. Model folders, `.stl` paths and outputs are therefore relative to the
config, not to where you typed the command.

## `--set-params` values are strings

`--set-params num_ep=20` passes `"20"`. It works where the code casts to `int`,
but a comparison against a number can silently misbehave. Only
`render_mode=None` is converted back to `None`.

## A config mentions a model that does not exist

Model binaries are **not tracked in git**. Several examples point at
`models/<name>.zip` that you must train or fetch first; some are symlinks into
untracked `__training*` folders that may not exist on your machine. Check with
`ls -l examples/<env>/models/`, then either train the config or run
`rlr select-model`.

## Reward-metric aggregates are empty

`res_all_ep['reward_formulas']` is only filled when the config also declares
`eval_formulas`. If you shape the reward with formulas and want their aggregate
metrics, declare them under `eval_formulas` as well.

## `multi_objective: true` gives an inconsistent reward vector

The vector can differ in length from the policy's `reward_space`, and
reward-machine rewards are not included in it. Treat the multi-objective path as
experimental; no example depends on it.

## `semantics: bool` does not make rewards ±1

The monitor runs in its default robustness semantics: rewards are continuous
robustness values, not satisfaction flags. `bool` narrows the reported value but
does not turn the reward into ±1.

## Training is slower than expected

The default torch thread count (one per core) can make small-MLP updates much
slower on many-core machines. `cfg_train.torch_num_threads: 4` is usually a
better starting point than leaving it unset.

## MetaDrive: only one engine per process

Training with in-process evaluation must use `n_envs > 1` so the environments
live in subprocesses; otherwise the tester cannot create its own engine. With
`n_envs: 1` and `cfg_train.eval` set, the run fails at the first evaluation.

## Plots look flat or shifted

A formula evaluated at a fixed `t0` reports that instant's value for the whole
episode; a per-step formula records its value when the step happened. If a curve
does not look like the `rho(...)` curve of the same formula, compare the
anchoring options in [STL specifications](specifications.md) before suspecting
the monitor.

## Where to look next

- `rlr show cfg.yml --expand` — what the run actually received, after all
  includes and overrides.
- `cfg_specs.debug_signals`, `debug_formulas`, `debug_rm`, `debug_actions` — per
  step tracing of signals, formulas, machines and overridden actions.
- `episode['stl_data']` and `episode['res_f']` — the raw samples and per-step
  formula values of a test episode, from a Python session.
