# RLRom — Code Base Overview

*Generated: 2026-07 (initial analysis). Last updated: 2026-10. Update this file incrementally as the project evolves.*

## What is RLRom

A Python research library for **Reinforcement Learning + Robust Online Monitoring**.
It uses **Signal Temporal Logic (STL)** formulas (via the external `stlrom` package) to:

1. **Test** RL agents: monitor learned behaviors against interpretable STL specs and
   compute robustness-based satisfaction metrics.
2. **Train** RL agents: wrap the environment so that STL robustness values enter the
   observation and/or reward (robustness-based reward shaping), and/or through a
   **reward machine** (a finite-state machine whose transition conditions are STL
   formulas — an extension of the reward-machine notion).

Language: Python ≥ 3.12, managed with `uv` (`.venv` in repo root). Package in
`src/rlrom`, installed as `rlrom`.

## Top-level layout

```
rlrom/
├── pyproject.toml        # deps, console scripts: rlr, rlrom_test, rlrom_train
├── src/rlrom/            # THE PACKAGE (all core logic lives here)
├── examples/             # per-environment case studies (configs, STL files, notebooks,
│   │                     #   trained model checkpoints, tensorboard logs)
│   ├── cartpole/         #   classic CartPole case study
│   ├── highway_env/      #   highway-env driving case (STL reward shaping + reward machine)
│   ├── minigrid/         #   MiniGrid (reward machines on grid worlds, blocked_unlock_pickup)
│   ├── mountaincar/      #   DDPG/SAC/TD3 configs
│   ├── acrobot/          #   Acrobot (a2c/dqn + STL specs configs)
│   └── simglucose/       #   simglucose experiment thread (separate package, not a dep)
├── tests/                # maintained pytest suite (see "Test suite")
└── docs/                 # this folder — reference summaries
```

**Ignore when exploring:** `examples/*/models/*__training*`, `examples/*/tb_logs`,
`.venv`, `uv.lock` (huge). Trained checkpoints are experiment artifacts, not code.

**Separate sub-projects:**
- `stlclash/` – a standalone benchmark repository (its own `pyproject.toml`, not a dependency of rlrom) for multi‑objective RL with STL specifications.
- `stlrom/` – the external C++/Python library that provides STL parsing, monitoring and robustness evaluation. It is bundled in this repository and built as a Python wheel; `rlrom` relies on it via `stlrom.STLDriver`.


## Package structure (`src/rlrom`)

| File | Lines | Role |
|---|---|---|
| `rlrom_run.py` | 106 | Main CLI entry (`rlr`): `rlr {test,train,show} cfg.yml [--cfg-train ..] [--cfg-test ..] [--cfg-specs ..] [--set-params k=v ...] [--verbose N] [--expand] [--subcfg KEY]`. Loads YAML cfg, dispatches to `main_train`/`main_test`/`main_show`. All imports of the heavy stack (`rlrom.utils`/`testers`/`trainers`) are deferred until after cfg parsing, so `rlr show` works without SB3/jax/torch. |
| `cfg_utils.py` | 142 | Lightweight config helpers (stdlib + numpy + ruamel.yaml only): `load_cfg` (recursive `.yml`/`.stl`/`import_module` expansion, `this_cfg_pathdir` chdir), `set_rec_cfg_field`, `find_subcfg` (dot-path or first-occurrence sub-config for `rlr show`). Extracted from `utils.py` in 2026-10 so config-only CLI ops don't import the heavy dependency stack. |
| `rlrom_train.py` | 53 | Thin wrapper: CLI → `RLTrainer(cfg).train()`. |
| `rlrom_test.py` | 58 | Thin wrapper: CLI → `RLTester(cfg).run_cfg_test()`. |
| `trainers.py` | 304 | `RLTrainer`: env creation, algorithm instantiation, SB3 `model.learn()`, checkpoints, periodic eval via `RlromCallback` (saves `model_step_*.zip` + `res_step_*.yml` per eval, logs metrics to tensorboard). |
| `testers.py` | 440 | `RLTester`: runs episodes with a trained model (or `random`/keyboard), collects episodes, computes metrics (`eval_episode`), matplotlib plotting of episode signals via stlrom `Signal.plot` (`get_fig`), static and live `cfg_test.plot` figures (see below), re-testing checkpoints (`retest_checkpoints_models`). |
| `wrappers/spec_wrapper.py` | 766 | **Core of the project.** `SpecWrapper(gym.Wrapper)` — see below. Also `wrap_env_specs(env, cfg)`: composes the wrapper stack (SpecWrapper → optional FlattenObservation). |
| `wrappers/reward_machine.py` | 153 | `RewardMachine` (pure FSM, no gym/STL, drives RM dynamics). The legacy `RewardMachineWrapper` has been removed (2026-10). |
| `utils.py` | 586 | Algorithm registry `ALGO_NAMES_CLASSES` (SB3, sbx/jax, sb3-contrib, morl-baselines, tabular Q), HuggingFace model loading, training-results dataframes (`get_training_folders`, `get_training_res`, `get_best_models`, ...), misc. Config helpers moved to `cfg_utils.py` but re-exported (`utils.load_cfg`, `utils.set_rec_cfg_field` still work). |
| `plots.py` | 325 | matplotlib live-plotting helpers: `LivePlot` (driven by `cfg_test.plot` items with `live: True`), legacy `RLFig`/`live_line` used in notebooks, `get_layout_from_string` signal-layout tokenizer. |
| `extra_algos/tabular_q_learning.py` | 102 | `TabularQLearning`, SB3-compatible-ish tabular Q for discrete envs (MiniGrid). |
| `cfgs/*.yml` | — | Per-gym-env default config templates (CartPole, MountainCar, highway-env, ...). |

## The wrapper pipeline (central concept)

Built by `wrap_env_specs` in `wrappers/spec_wrapper.py`:

```
gym env
  → SpecWrapper(env, cfg)                # if cfg has 'cfg_specs' (reward machines included)
      → FlattenObservation(env)         # if cfg_specs.flatten_obs (default True)
```

`model_use_specs` (top-level cfg): whether the *agent* was trained with the wrapped
observations/rewards (`true`) or with the raw env (then tests use `last_obs['unwrapped']`).

### SpecWrapper (`wrappers/spec_wrapper.py`) — read this file first

- Driven by `cfg['cfg_specs']`:
  - `specs` / `stl_specs`: STL signal declarations + formula definitions. Either an
    inline string or a path to a `.stl` file (loaded by `utils.load_cfg`).
  - `action_names`, `obs_names`, `aux_sig_names`: map signal name → Python expression
    evaluated per step in `get_sample` (e.g. `x: "obs[0]"`, `ego_vx: "80*obs[0][3]"`).
    Note: these expressions are `eval()`'d — by design (user-provided configs).
  - `reward_formulas`: STL formulas added to the reward
    (`new_reward = env_reward*old_reward_weight + Σ w_f · ρ_f(t)`).
  - `label_formulas`: STL formulas monitored each step; a label is *true* when its
    monitored robustness lower bound is > 0. The set of true labels is fed to every
    reward machine at each step.
  - `reward_machines`: list of reward-machine configs (each: optional `name`, `weight`,
    `in_observation`, and a `reward_machine:` sub-dict = the FSM `cfg_rm` parsed by the
    `RewardMachine` class). In single-objective mode RM rewards are weighted (`weight`×)
    and **summed into the overall agent reward**; reaching a `final` RM state sets
    `terminated=True`. (Not yet wired into the multi-objective vector reward.)
  - Reward-machine state indices are exposed in the observation as a dict field
    `rm_states` (a Box, one per machine) alongside `unwrapped` and optional
    `obs_formulas` — this supports any underlying obs space. `reset()` also resets all
    machines; `show_reward_machines()` prints a text summary of states/transitions/labels.
  - The episode dict additionally records `labels` (true label names per step),
    `rm_states` (state indices per machine per step) and `rm_rewards` (per-machine
    rewards per step), and `res_f` holds per-step robustness for `label_formulas` too.
  - `obs_formulas`: formulas appended to the observation (obs space becomes
    DictSpace `{unwrapped, obs_formulas}`).
  - `eval_formulas`: formulas evaluated after/during episodes for metrics only.
  - `end_formulas`: formulas that can force `terminated=True` (if `lower_rob > 0`).
  - `real_time_step` (default 1), `BigM`, `multi_objective` (mo-gymnasium vector reward
    for MORL), `keep_old_reward`, `debug_signals`, `debug_formulas`, `flatten_obs`.
- Key methods: `step` (sample → stlrom driver → robustness → shaped reward/obs),
  `eval_formula_cfg` (online/offline robustness with `t0`, `past_horizon`, semantics
  `rob`/`bool`/`lower_rob`/`upper_rob`, bounds), `eval_specs_episode` (per-episode
  metrics: ep_len, ep_rew, per-formula mean/sum/num_sat, init_sat/init_rob),
  `set_episode_data` + `get_time`/`get_sig`/`get_rob`/`get_values_from_str`
  (replay & plotting).
- Maintains `self.episode` dict: observations, actions, rewards, rewards_wrapped,
  dones, `stl_data` (raw signal samples), `res_f` (per-formula robustness traces).
- Uses `stlrom.STLDriver` with Boolean semantics for rewards (see TODO in code).

### RewardMachine (`wrappers/reward_machine.py`)

- **Decoupled pure FSM** (no gym, no STL): parses `cfg_rm`, tracks current state, and on
  each `step(labels)` (a list of *true* condition names) takes the highest-`priority`
  enabled transition or, if none, stays and issues the state reward (default 0).
  Guard: two enabled transitions sharing the max priority raise `ValueError`.
  Exposes `reset()`, `state_id`, `state_idx`, `is_final`.
- Config: `states` (`id`, optional `initial`/`final`/`reward=0`; exactly one initial),
  `transitions` (`from`/`to`/`condition` strings required; `reward=0`, `priority=0`,
  `id` default `{from}__to__{to}__if__{condition}`).
- The `conditions:`/`debug_rm`/`in_observation`/`rm_id` keys in `cfg_rm` are *not* part of
  the FSM — they belong to the wrapping/STL layer.

### RewardMachineWrapper — removed (2026-10)

The legacy `RewardMachineWrapper` (inline transition logic coupled to
`SpecWrapper.eval_formula_cfg`) has been **removed** from
`wrappers/reward_machine.py`; only the pure `RewardMachine` FSM class remains.
SpecWrapper now drives reward machines directly via `cfg_specs.reward_machines`.

## Configuration system (how everything is wired)

- One main YAML cfg per run; values ending in `.yml` are **recursively replaced by
  the file's content** (`utils.load_cfg`); `.stl` fields load file *text*.
- `this_cfg_pathdir` is auto-set to the cfg file's dir and the process **chdir's
  there** — relative paths (models, STL files, `import_module`) resolve from the
  cfg's folder.
- Top-level keys: `env_name`, `import_module` (python module for custom env code),
  `make_env_train`/`make_env_test` (optional custom env factory functions),
  `model_name`, `model_path`, `model_use_specs`, `cfg_env`, `cfg_specs`,
  `cfg_train` (algo block, n_envs, total_timesteps, eval_freq, `eval` sub-test-cfg),
  `cfg_test` (num_ep, num_steps, init_seeds, render_mode, model_file, res_file).
- Algorithms selected in `cfg_train.algo` by key name from `ALGO_NAMES_CLASSES`
  (`ppo`, `sac`, ..., `sbx_*`, `moppo`, `gpipd`, `qlearning`, ...). MORL algos
  (morl-baselines) use `.train(total_timesteps, eval_env, **train_kwargs)`
  instead of SB3 `.learn()`.
- Training output: `models/<model_name>_<date>__training<N>/` containing `cfg0.yml`,
  `model_step_*.zip`, `res_step_*.yml`; final model symlinked/copied to
  `models/<model_name>.zip` (+ `.yml`).

## Test-time plotting (`cfg_test.plot`)

`RLTester.run_cfg_test` (via `rlr test`) can display one matplotlib figure per item of a
`cfg_test.plot` list (episode 0), after all episodes are run. Figures are built with
`stlrom.Signal.plot` (discrete signals are encoded as `Signal` with derivative 0 and
rendered as step functions):

```yaml
cfg_test:
  ...
  plot:
    - title: Observations            # optional figure title
      out: plots/obs.png             # optional: also save the figure to file
                                     # (relative paths resolve from the cfg's dir)
      ax:                            # each item = one axis (stacked, shared x)
        - x,theta                    # comma-separated signal names on the same axis
        - x_dot
    - title: Formulas
      ax:
        - label_formulas/go_left
        - [label_formulas/cart_in_A, label_formulas/stay_A]   # list form also works
```

Signal names (resolved to `stlrom.Signal` by `SpecWrapper.get_signal_from_str`,
values by `SpecWrapper.get_values_from_str`):

- bare names from `action_names` / `obs_names` / `aux_sig_names` (e.g. `push`, `x`)
- `reward` — the raw environment reward
- `obs_formulas/<f>`, `reward_formulas/<f>`, `label_formulas/<f>` — the per-step
  values recorded in the episode (robustness for obs/reward formulas, lower-bound
  robustness for labels)
- `eval_formulas/<f>` — offline robustness over the trace
- `rm_states/<machine>` — per-step state index of a reward machine
- `<machine>` (bare name) or `rm_reward/<machine>` — per-step (unweighted) reward
  issued by that machine
- `:weighted` suffix on formulas, RM rewards or bare machine names — value × weight
- `total_reward` — the final wrapped reward (put it on the same axis line as
  `:weighted` sub-rewards to compare contributions)
- `rho(<f>[,past_horizon])`, `rob(<f>[,past_horizon])`, `sat(<f>)`, bare `<f>` —
  offline robustness / satisfaction
- `set_ep_idx(<n>)` — switch episode mid-layout (as before)

Example: `examples/cartpole/cfg0tr_rm_stay_left_right.yml` (two figures:
"Observations" and "Formulas").

### Live-updating plots (`live: True`)

A `plot` item can instead of a static end-of-run figure open a live window that
updates every environment step, per episode (`rlr test` only; headless/Agg
backends skip it silently):

```yaml
cfg_test:
  plot:
    - title: Live observations
      live: True                  # open a live window during the episodes
      live_window: 20             # optional: trailing x window in time units
                                  # (default: show the whole episode so far)
      out: plots/live.png         # still supported: saved from the last episode
      ax:
        - x, theta
        - reward
```

Implemented by `rlrom.plots.LivePlot` (figures are created in
`RLTester.run_episode`, one `step()` per env step, closed when the episode ends):

- **space** pauses/resumes the live updates (the episode itself keeps running;
  the window stays responsive while paused)
- **Escape** closes the windows and stops the current episode early
- discrete signals (derivative 0, e.g. `reward`, RM states) are drawn as steps,
  matching the static `Signal.plot` rendering
- `live: True` items are skipped in the static post-run display (only their
  `out:` file is written); non-live items are displayed as before


## Canonical example configs (start here)

- `examples/cartpole/cfg0tr_ppo_specs.yml` + `cartpole.stl` — simplest full
  train-with-STL-reward example (CartPole, PPO, one reward formula, one eval formula).
  Also the smoke-test config used by `.pi/AGENTS.md` (`uv run rlr test ...`).
- `examples/cartpole/cfg0tr_ppo_rm.yml` — minimal **label-formula + reward-machine**
  example (2-state FSM on CartPole left/right, drives the new
  `cfg_specs.reward_machines` path, covered by `tests/test_reward_machine_spec.py`).
  Related direction variants: `cfg0tr_ppo_go_left.yml` / `cfg0tr_ppo_go_right.yml`.
- `examples/highway_env/cfg0tr.yml` (+ `cfg_specs.yml`, `hw-env_specs.stl`,
  `highway.py` with custom `make_env_*`) — richer: obs_names expressions, eval +
  reward formulas, custom module.
- `examples/highway_env/cfg0tr_rmf.yml` + `cfg_specs_rmf.yml` — legacy reward-machine
  example (old `cfg_rm` path). **Currently non-functional:** the
  `RewardMachineWrapper` it relied on was removed in 2026-10; use
  `cfg_specs.reward_machines` instead.
- `examples/minigrid/` and `minigrid/blocked_unlock_pickup/` — FSM/RM case study
  with multi-state RM, negative rewards, `TabularQLearning`.
- `examples/simglucose/` — moved from `tests/` (2026-07): simglucose experiment
  thread (not a dependency; requires the `simglucose` package installed separately).

## Test suite status

- `tests/` is now a maintained pytest suite — see "Test suite" section below.
- There is **no CI**. Verify changes with `uv run pytest tests/` and/or targeted
  example commands, e.g. `uv run rlr test examples/cartpole/cfg0tr_ppo_specs.yml`.
  Note that `load_cfg` chdir's to the cfg file's folder before running.

## Known issues / caveats (noted during initial read)

0. **`wrap_env_specs` flattens dict obs when reward machines / obs formulas are
   present.** The `cfg_specs.reward_machines` (and `obs_formulas`) paths make
   `SpecWrapper` return a dict obs (`unwrapped` + `rm_states`/`obs_formulas`), but
   `wrap_env_specs` adds `FlattenObservation` by default (when `cfg_specs.cfg_rm` is
   absent), flattening that dict away. Follow-up needed: in `wrap_env_specs`, don't
   flatten when `reward_machines` (or `obs_formulas`) are present and the user wants
   to keep those fields.
   **`multi_objective` ignores RM rewards** as well: the vector-reward branch of
   `SpecWrapper.step` does not append reward-machine rewards.
0b. **`rlr show` without `--expand` bypasses `load_cfg`**, so the raw top-level YAML
   is printed (nested `.yml`/`.stl` fields shown as paths, `import_module` not
   imported, no chdir). With `--expand` it goes through `load_cfg`. `--subcfg KEY`
   selects a dot-path or first-occurrence sub-configuration.
1. ~~`SpecWrapper.step` used `self.old_reward_weight` while `__init__` only sets
   `env_reward_weight`~~ — **fixed** (now uses `env_reward_weight` consistently).
2. ~~`SpecWrapper.reset_monitor` was broken (`stl_driver.data.reset_signal_data`),
   so `SpecWrapper` crashed on the first `reset()`~~ — **fixed**: `reset_monitor`
   recreates the `stlrom.STLDriver` (stlrom has no reset API; re-parsing the same
   driver accumulates parse state). `get_time` now reads `episode['stl_data']`
   instead of iterating `stl_driver.data` (not iterable in stlrom).
3. `RewardMachine` standalone class: was dead/broken code, now a **decoupled pure FSM**
   (`cfg_rm` parsing, priority-based transitions, `step(labels)` → reward). The legacy
   `RewardMachineWrapper` has been **removed** (2026-10); SpecWrapper drives the FSM directly.
4. ~~`RewardMachineWrapper.step` multiplied RM reward by `self.unwrapped._reward()`~~
   — moot, the class has been removed.
5. `SpecWrapper.eval_formula_cfg` online-mode `t0 = max(t0, tend - f_hor)` is marked
   FIXME in code (start-of-episode interpretation issues for past formulas).
   Also: in `step`, `end_formulas` appends the *last* `obs_formulas` robustness value
   (`robs_f`) to their `res_f` entry instead of their own — `label_formulas` and
   `reward_formulas` entries record their own values correctly.
6. `load_cfg` mutates cwd (chdir to cfg dir) — global side effect for the process.
   Also: with `verbose=0` it silently does **not** expand nested `.yml`/`.stl`
   fields (uses default `verbose=1`).
7. `testers.py:make_env_test` does `sys.modules[cfg['import_module']]` — assumes the
   module was already imported (training path imports it; pure test runs may not).

## Test suite

`tests/` is now a maintained pytest suite (run: `uv run pytest tests/`):

- `tests/test_utils.py` — `parse_signal_spec`, `get_formulas`, `set_rec_cfg_field`,
  `load_cfg` expansion on a real example cfg.
- `tests/test_smoke.py` — fast end-to-end on CartPole with a random policy: unwrapped
  `RLTester` run, `SpecWrapper` step/metrics, reset/data-reuse, reward-shaping bounds.
- `tests/test_reward_machine.py` — unit tests of the pure `RewardMachine` FSM
  (parsing, state validation, priority transitions, tie guard, state rewards).
- `tests/test_reward_machine_spec.py` — `cfg_specs.reward_machines` integration through
  `SpecWrapper`: label evaluation, RM stepping, episode recording, obs `rm_states` field,
  `show_reward_machines()`.

Old scratch material (gradio app tests, stale-API `RLModelTester` tests, simglucose
experiment) was removed or moved to `examples/simglucose/` (2026-07 cleanup).

## Conventions

- Config-driven by default: prefer changing YAML over code for experiment changes.
- Signal/formula names in STL are the contract between `.stl` files and cfg
  `*_names`/`*_formulas` blocks.
- Metric naming: `res` (per-episode, arrays) → `res_all_ep` (aggregates: `basics`,
  `reward_formulas`, `eval_formulas`).
