# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Dependencies: `wandb` and `ir-sim` were removed from the base install (nothing
  referenced them), and the notebook-only stack (`jupyter`, `ipympl`, `plotly`,
  `kaleido`) moved to a `notebooks` extra. `pip install rlrom` now pulls 16
  packages instead of 21.
- Example notebooks carry no stored outputs any more (they are reproducible by
  running them): the three heaviest shrink by ~2.5 MB in total.

### Fixed

- MiniGrid: the reward machine is declared under `cfg_specs.reward_machines`
  with its label formulas in `monitor_formulas`; it was wired through the
  retired `cfg_specs.cfg_rm` key and never ran.
- `load_cfg(verbose=0)` expanded no `.yml`/`.stl` reference at all — the
  verbosity flag doubled as the switch that turns expansion off, so a
  `reward_machine: fsm.yml` reference stayed a string. `verbose` now only
  controls what is printed.
- `rlrom.wrappers` is a regular package (it had no `__init__.py`), so static
  analysis and `mkdocstrings` can collect it.
- Notebooks no longer import `bokeh`.

### Removed

- `examples/minigrid/blocked_unlock_pickup/` — an older task family whose configs
  used the retired `cfg_specs.cfg_rm` key, whose entry point imported `bokeh`,
  and whose machine conditions were not labels of the specification it loaded.
- `examples/cartpole/rm_stuck_left.yml` — an inert machine (`weight: 0`, and its
  condition `stuck_left` was never declared as a monitor formula).
- `examples/simglucose/` — needs the `simglucose` package, which is not a
  dependency; the notebook was not even valid JSON.
- `examples/sandbox.ipynb`, `examples/sandbox_analysis.ipynb` — scratch.

### Added

- `tests/test_example_configs.py`: guards that no example uses the retired
  `cfg_specs.cfg_rm` key and that every reward-machine transition condition is
  declared as a monitor formula.

## [0.3.0] - 2026-11-23

Everything below the 0.2.0 entry: this release replaces the STL wrapper with a
unified `SpecWrapper`, adds reward machines and action constraints as
first-class features, and completes the CLI. It is **not backward compatible**
with 0.2.0 — see *Removed* and *Changed*.

### Added

- **`SpecWrapper`** (`rlrom.wrappers.spec_wrapper`): one wrapper for STL
  monitoring and reward machines, built by `wrap_env_specs`.
- **A single `monitor_formulas` section.** Each formula is evaluated once per
  step and yields a label for reward machines; `obs_weight` also appends it to
  the observation, `reward_weight` adds its robustness to the reward,
  `terminal: true` ends the episode when it becomes true, and `actions`
  constrains the action.
- **Reward machines** (`rlrom.wrappers.reward_machine`): a pure FSM driven by
  STL labels, wired through `cfg_specs.reward_machines`, with weights, optional
  `in_observation`, state rewards, final states and per-state action
  constraints.
- **Shielding / action constraints**: `forced`, `blocked`, `fallback` and
  `priority` from reward-machine states and from monitor formulas, with
  deterministic conflict resolution, plus `disable_actions` to compare shielded
  and unshielded runs.
- **Continuous-action shield hook** (`cfg_specs.action_shield`) for `Box` action
  spaces: a Python callback gated by an activation formula, with per-step
  `episode['shield_active']` recording and `info['shield']` reporting.
- **`cfg_specs.episode_info_keys`**: opt-in aggregation of environment `info`
  keys into `mean_<key>` episode metrics.
- **CLI**: `rlr test|train|show|plot|select-model`, with `--cfg-train`,
  `--cfg-test`, `--cfg-specs` overrides, `--set-params k=v`, `--verbose`,
  and `show --expand`/`--subcfg`.
- **`rlr plot`** and the `rlrom_plot` entry point: training curves from
  checkpoint result files, several trainings, metrics and envelopes, saved to
  png/pdf/svg.
- **`rlr select-model`** and the `rlrom_select_model` entry point: pick a
  checkpoint by metric (or by step) and make it active for `rlr test`.
- **`cfg_train.repeat_training`**: run the same training N times from scratch.
- **Test-time plotting**: `cfg_test.plot` figures built with `stlrom.Signal`,
  including live-updating windows (`live: true`).
- **Algorithm coverage**: `sb3-contrib` algorithms, SBX (JAX) algorithms,
  multi-objective algorithms through `mo-gymnasium`/`morl-baselines`, and a
  tabular Q-learning implementation for discrete environments.
- **Examples**: MuJoCo Hopper (fetched policy, specifications, alternative
  tasks), MetaDrive (curved highway with a shield), MiniGrid
  `blocked_unlock_pickup`, Acrobot, MountainCar, and a rewritten highway-env
  case study; notebooks for CartPole, highway-env, Acrobot, MiniGrid and
  simglucose.
- **Test suite**: 188 tests over config handling, the wrapper, reward machines,
  shielding, episode figures, model selection, environment construction and
  training repeats.

### Changed

- `requires-python` from `>=3.10` to `>=3.12`.
- `stlrom` dependency from `>=0.2.3,<0.3.0` to `>=0.4.0`.
- Environment construction separates `env_id` (the id `gym.make` builds) from
  `env_name` (the name the policy was published under), adds `env_kwargs`, and
  finds custom `make_env_*` factories through `import_module`.
- `flatten_obs` is configurable on `cfg_specs`; dict observations are kept when
  it is off.
- Training output is organised as `models/<model_name>_<date>__training<N>/`,
  with the active model selected through a symlink and its config.
- `cfg_train.eval`, when present, is used for the periodic in-training
  evaluation instead of `cfg_test`.
- Config loading (`load_cfg`, `find_subcfg`, `set_rec_cfg_field`) moved to the
  light `rlrom.cfg_utils`, so `rlr show` and `rlr plot` run without the training
  stack installed.
- Episode plots are built from the recorded monitor data instead of re-running
  the environment.

### Removed

- `rlrom.wrappers.stl_wrapper.StlWrapper` — replaced by `SpecWrapper`.
- `rlrom.envs` and its `supported_envs`, `supported_models`, `cfg_envs` module
  level API — environment metadata now comes from the config and
  `cfg_utils`/`utils`.
- The `bokeh` dependency; plotting uses matplotlib and `stlrom.Signal`.

### Fixed

- Singleton NumPy arrays are parsed as signals instead of failing on
  dimensionality.
- `reset` is handled correctly inside `step`, and the next observation reaches
  the wrapper.
- A missing `model_name` in the config raises instead of failing later.
- Declaring the same formula in two deprecated sections with a non-zero weight
  raises `ValueError` instead of double counting.
- Failures when saving a model or dumping the config are caught and reported.

## [0.2.0] - 2025-10-25

STL wrapper release on PyPI: monitoring of STL formulas around a Gymnasium
environment, robustness-based reward shaping and observation augmentation,
testing through `rlr test` and training through `rlr train`. Earlier versions
on PyPI: 0.1.0, 0.1.1, 0.1.2.

[Unreleased]: https://github.com/decyphir/rlrom/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/decyphir/rlrom/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/decyphir/rlrom/releases/tag/v0.2.0
