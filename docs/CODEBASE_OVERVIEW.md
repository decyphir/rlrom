# RLRom — Code Base Overview

*Generated: 2026-07 (initial analysis). Update this file incrementally as the project evolves.*

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
│   └── mountaincar/      #   DDPG/SAC/TD3 configs
├── tests/                # mixed: some stale tests (old API), notebooks, simglucose experiments
└── docs/                 # this folder — reference summaries
```

**Ignore when exploring:** `examples/*/models/*__training*`, `examples/*/tb_logs`,
`.venv`, `uv.lock` (huge). Trained checkpoints are experiment artifacts, not code.

## Package structure (`src/rlrom`)

| File | Lines | Role |
|---|---|---|
| `rlrom_run.py` | 83 | Main CLI entry (`rlr`): `rlr {test,train,show} cfg.yml [--cfg-train ..] [--cfg-test ..] [--cfg-specs ..] [--set-params k=v ...]`. Loads YAML cfg, dispatches to `main_train`/`main_test`. |
| `rlrom_train.py` | 53 | Thin wrapper: CLI → `RLTrainer(cfg).train()`. |
| `rlrom_test.py` | 58 | Thin wrapper: CLI → `RLTester(cfg).run_cfg_test()`. |
| `trainers.py` | 304 | `RLTrainer`: env creation, algorithm instantiation, SB3 `model.learn()`, checkpoints, periodic eval via `RlromCallback` (saves `model_step_*.zip` + `res_step_*.yml` per eval, logs metrics to tensorboard). |
| `testers.py` | 385 | `RLTester`: runs episodes with a trained model (or `random`/keyboard), collects episodes, computes metrics (`eval_episode`), bokeh plotting of signals (`get_fig`), re-testing checkpoints (`retest_checkpoints_models`). |
| `wrappers/stl_wrapper.py` | 510 | **Core of the project.** `STLWrapper(gym.Wrapper)` — see below. |
| `wrappers/reward_machine.py` | 125 | `RewardMachineWrapper(gym.Wrapper)` — YAML-defined FSM, STL-conditions on transitions. |
| `wrappers/specs_wrapper.py` | 18 | `wrap_env_specs(env, cfg)`: composes the wrapper stack (STLWrapper → RewardMachineWrapper or FlattenObservation). |
| `utils.py` | 660 | Config loading (`load_cfg`: recursive YAML `.yml`/`.stl` field substitution, `this_cfg_pathdir` chdir), algorithm registry `ALGO_NAMES_CLASSES` (SB3, sbx/jax, sb3-contrib, morl-baselines, tabular Q), HuggingFace model loading, training-results dataframes (`get_training_folders`, `get_training_res`, `get_best_models`, ...), misc. |
| `plots.py` | 166 | matplotlib live-plotting helpers used in notebooks. |
| `extra_algos/tabular_q_learning.py` | 102 | `TabularQLearning`, SB3-compatible-ish tabular Q for discrete envs (MiniGrid). |
| `cfgs/*.yml` | — | Per-gym-env default config templates (CartPole, MountainCar, highway-env, ...). |

## The wrapper pipeline (central concept)

Built by `wrap_env_specs` in `wrappers/specs_wrapper.py`:

```
gym env
  → STLWrapper(env, cfg)               # if cfg has 'cfg_specs'
      → RewardMachineWrapper(env, cfg_rm)   # if cfg_specs has 'cfg_rm'
      → FlattenObservation(env)              # else (if cfg_specs.flatten_obs, default True)
```

`model_use_specs` (top-level cfg): whether the *agent* was trained with the wrapped
observations/rewards (`true`) or with the raw env (then tests use `last_obs['unwrapped']`).

### STLWrapper (`wrappers/stl_wrapper.py`) — read this file first

- Driven by `cfg['cfg_specs']`:
  - `specs` / `stl_specs`: STL signal declarations + formula definitions. Either an
    inline string or a path to a `.stl` file (loaded by `utils.load_cfg`).
  - `action_names`, `obs_names`, `aux_sig_names`: map signal name → Python expression
    evaluated per step in `get_sample` (e.g. `x: "obs[0]"`, `ego_vx: "80*obs[0][3]"`).
    Note: these expressions are `eval()`'d — by design (user-provided configs).
  - `reward_formulas`: STL formulas added to the reward
    (`new_reward = env_reward*old_reward_weight + Σ w_f · ρ_f(t)`).
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

### RewardMachineWrapper (`wrappers/reward_machine.py`)

- YAML `cfg_rm` dict: `states` (id, initial, final, reward), `transitions`
  (from, to, condition, reward), `conditions` (STL formula name + eval options),
  `in_observation` (append RM state index to obs), `debug_rm`.
- **Assumes it wraps an STLWrapper** (uses `get_wrapper_attr('eval_formula_cfg')`);
  transition conditions are checked with `lower_rob > 0` (safe/conservative).
- Reaches the `final` state → episode terminated. Reward passed to agent is
  `rm_reward * env._reward()` (odd — see known issues).
- Note: a standalone `RewardMachine` class exists (nearly empty/buggy) — the
  `RewardMachineWrapper` is the real implementation.

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

## Canonical example configs (start here)

- `examples/cartpole/cfg0tr_ppo_specs.yml` + `cartpole.stl` — simplest full
  train-with-STL-reward example (CartPole, PPO, one reward formula, one eval formula).
- `examples/highway_env/cfg0tr.yml` (+ `cfg_specs.yml`, `hw-env_specs.stl`,
  `highway.py` with custom `make_env_*`) — richer: obs_names expressions, eval +
  reward formulas, custom module.
- `examples/highway_env/cfg0tr_rmf.yml` + `cfg_specs_rmf.yml` — reward-machine example.
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

1. ~~`STLWrapper.step` used `self.old_reward_weight` while `__init__` only sets
   `env_reward_weight`~~ — **fixed** (now uses `env_reward_weight` consistently).
2. ~~`STLWrapper.reset_monitor` was broken (`stl_driver.data.reset_signal_data`),
   so `STLWrapper` crashed on the first `reset()`~~ — **fixed**: `reset_monitor`
   recreates the `stlrom.STLDriver` (stlrom has no reset API; re-parsing the same
   driver accumulates parse state). `get_time` now reads `episode['stl_data']`
   instead of iterating `stl_driver.data` (not iterable in stlrom).
3. `RewardMachine` standalone class: `self_current_state = 0` (typo, no `self.`) and
   `reset` returns undefined `self.u0` — dead/broken code, superseded by
   `RewardMachineWrapper`.
4. `RewardMachineWrapper.step` multiplies RM reward by `self.unwrapped._reward()` —
   suspicious scaling; verify intended semantics before relying on it.
5. `STLWrapper.eval_formula_cfg` online-mode `t0 = max(t0, tend - f_hor)` is marked
   FIXME in code (start-of-episode interpretation issues for past formulas).
6. `load_cfg` mutates cwd (chdir to cfg dir) — global side effect for the process.
   Also: with `verbose=0` it silently does **not** expand nested `.yml`/`.stl`
   fields (uses default `verbose=1`).
7. `testers.py:make_env_test` does `sys.modules[cfg['import_module']]` — assumes the
   module was already imported (training path imports it; pure test runs may not).
8. stlrom grammar quirk: a bare negative literal on the right of a comparison can
   fail to parse (e.g. `x[t] < -0.2`); `x[t] < 0.2` / `x[t] > -0.7` are fine.

## Test suite

`tests/` is now a maintained pytest suite (run: `uv run pytest tests/`):

- `tests/test_utils.py` — `parse_signal_spec`, `get_formulas`, `set_rec_cfg_field`,
  `load_cfg` expansion on a real example cfg.
- `tests/test_smoke.py` — fast end-to-end on CartPole with a random policy: unwrapped
  `RLTester` run, `STLWrapper` step/metrics, reset/data-reuse, reward-shaping bounds.

Old scratch material (gradio app tests, stale-API `RLModelTester` tests, simglucose
experiment) was removed or moved to `examples/simglucose/` (2026-07 cleanup).

## Conventions

- Config-driven by default: prefer changing YAML over code for experiment changes.
- Signal/formula names in STL are the contract between `.stl` files and cfg
  `*_names`/`*_formulas` blocks.
- Metric naming: `res` (per-episode, arrays) → `res_all_ep` (aggregates: `basics`,
  `reward_formulas`, `eval_formulas`).
