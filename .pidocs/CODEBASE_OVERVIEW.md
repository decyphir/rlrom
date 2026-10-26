# RLRom — Code Base Overview

*Generated: 2026-07 (initial analysis). Last updated: 2026-10-26 (res_f
per-group namespacing fix, `formula`/`formula_name` alias + `formula_params`
support, end_formulas recording fix, vestigial `obs_name` option removed).
Previous: 2026-10-25 (STLData / stlrom 0.4.0 data-API migration + plotting
fixes review). Update this file incrementally as the project evolves. The
stlrom library itself is documented
in [`STLROM_OVERVIEW.md`](STLROM_OVERVIEW.md) — read both.*

## What is RLRom

A Python research library for **Reinforcement Learning + Robust Online
Monitoring**. It uses **Signal Temporal Logic (STL)** formulas (via the
external `stlrom` package, C++ core) to:

1. **Test** RL agents: monitor learned behaviors against interpretable STL
   specs and compute robustness-based satisfaction metrics.
2. **Train** RL agents: wrap the environment so that STL robustness values
   enter the observation and/or reward (robustness-based reward shaping),
   and/or through a **reward machine** (a finite-state machine whose
   transition conditions are STL label formulas).

Language: Python ≥ 3.12, managed with `uv` (`.venv` in repo root). Package in
`src/rlrom`, installed as `rlrom`.

## Top-level layout

```
rlrom/
├── pyproject.toml        # deps (incl. stlrom >= 0.4.0 from PyPI), console scripts
├── src/rlrom/            # THE PACKAGE (all core logic lives here)
├── examples/             # per-environment case studies (configs, STL files, notebooks,
│   │                     #   trained model checkpoints, tensorboard logs)
│   ├── cartpole/         #   classic CartPole case study (primary dev target)
│   ├── highway_env/      #   highway-env driving case (STL reward shaping + reward machine)
│   ├── minigrid/         #   MiniGrid (reward machines on grid worlds, blocked_unlock_pickup)
│   ├── mountaincar/      #   DDPG/SAC/TD3 configs
│   ├── acrobot/          #   Acrobot (a2c/dqn + STL specs configs)
│   └── simglucose/       #   simglucose experiment thread (separate package, not a dep)
├── tests/                # maintained pytest suite (see "Test suite")
├── stlrom/               # SEPARATE checkout of the stlrom library (own git repo,
│                         #   own .venv/build) — ahead of the PyPI 0.4.0 wheel
├── stlclash/             # standalone benchmark repo (own pyproject, not a dep)
└── .pidocs/              # this folder — reference docs
```

**Ignore when exploring:** `examples/*/models/*__training*`, `examples/*/tb_logs`,
`.venv`, `uv.lock`, `stlrom/.venv`, `stlrom/build` (huge). Trained checkpoints
are experiment artifacts, not code.

**Sub-projects:**
- `stlclash/` — standalone benchmark repository (multi-objective RL + STL specs),
  its own `pyproject.toml`, not a dependency of rlrom.
- `stlrom/` — the STL monitoring library (C++ + pybind11). rlrom uses the
  **PyPI 0.4.0 wheel** (via `uv.lock`), while `stlrom/` is a dev checkout of
  the same repo that is **ahead** of that release. See
  [`STLROM_OVERVIEW.md`](STLROM_OVERVIEW.md).

## Package structure (`src/rlrom`)

| File | Lines | Role |
|---|---|---|
| `rlrom_run.py` | 128 | Main CLI entry (`rlr`): `rlr {test,train,show,plot,select-model} cfg.yml [action options]` plus `--cfg-train/--cfg-test/--cfg-specs` overrides, `--set-params k=v ...`, `--verbose N`; `show` adds `--expand`, `--subcfg KEY`. Heavy imports deferred until after cfg parsing so `rlr show`/`plot` work without the full stack. |
| `cfg_utils.py` | 180 | Lightweight config helpers (stdlib + numpy + ruamel.yaml only): `load_cfg` (recursive `.yml`/`.stl`/`import_module` expansion, `this_cfg_pathdir` chdir), `set_rec_cfg_field`, `find_subcfg` (dot-path or first-occurrence sub-config). No heavy imports. |
| `rlrom_train.py` | 68 | Thin wrapper: CLI → `RLTrainer(cfg).train()`. |
| `rlrom_test.py` | 58 | Thin wrapper: CLI → `RLTester(cfg).run_cfg_test()`. |
| `rlrom_plot.py` | 236 | `rlr plot` / `rlrom_plot`: plot training metrics from checkpoint `res_step_*.yml` files (`--trainings last|all|0,2,-5:-1`, repeatable `--metric/-m`, `--envelope`, `-o out`, `--log-y`, `--list-metrics`). Headless backends save `rlr_plot.png`. |
| `rlrom_select_model.py` | 209 | `rlr select-model` / `rlrom_select_model`: pick a checkpoint by metric (`--min`, `--step`, `--show-top`, `--show-selected`), used to set the active model. |
| `trainers.py` | 304 | `RLTrainer`: env creation (`make_env_train`, `make_vec_envs`), algorithm instantiation, SB3 `model.learn()`, checkpoints, periodic eval via `RlromCallback` (saves `model_step_*.zip` + `res_step_*.yml` per eval, tensorboard logging). |
| `testers.py` | 440 | `RLTester`: runs episodes with a trained model (or `random`/keyboard), collects episodes, computes metrics (`eval_episode`), matplotlib episode figures via `stlrom.Signal.plot` (`get_fig`), static + live `cfg_test.plot` figures, `retest_checkpoints_models`. |
| `wrappers/spec_wrapper.py` | 799 | **Core of the project.** `SpecWrapper(gym.Wrapper)` + `wrap_env_specs(env, cfg)` — see below. |
| `wrappers/reward_machine.py` | 153 | `RewardMachine` (pure FSM, no gym/STL). The legacy `RewardMachineWrapper` was removed (2026-10). |
| `utils.py` | 723 | Algorithm registry `ALGO_NAMES_CLASSES` (SB3, sbx/jax, sb3-contrib, morl-baselines, tabular Q), HuggingFace model loading, training-results polars dataframes (`get_training_folders`, `get_training_res`, `get_df_all_trainings`, `get_metric_series`, `get_selection_df`, ...), `append_to_field_array`, misc. Re-exports config helpers from `cfg_utils`. |
| `plots.py` | 333 | matplotlib helpers: `LivePlot` (live-updating figures for `cfg_test.plot` items with `live: True`), legacy `RLFig`/`live_line`, `get_layout_from_string` signal-layout tokenizer, training-results plotting (`plot_training_results`, envelopes, `plot_df_*`). |
| `extra_algos/tabular_q_learning.py` | 102 | `TabularQLearning`, tabular Q for discrete envs (MiniGrid). |
| `cfgs/*.yml` | — | Per-gym-env default config templates (CartPole, MountainCar, highway-env, ...). |

## The wrapper pipeline (central concept)

Built by `wrap_env_specs` in `wrappers/spec_wrapper.py`:

```
gym env
  → SpecWrapper(env, cfg)                # if cfg has 'cfg_specs' (reward machines included)
      → FlattenObservation(env)          # if cfg_specs.flatten_obs (default True)
```

`model_use_specs` (top-level cfg): whether the *agent* was trained with the
wrapped observations/rewards (`true`) or with the raw env (then tests use
`last_obs['unwrapped']`).

### SpecWrapper (`wrappers/spec_wrapper.py`) — read this file first

Driven by `cfg['cfg_specs']`:

- `specs` / `stl_specs`: STL signal declarations + formula definitions, inline
  string or path to a `.stl` file. If absent, a default declaration is built
  from `action_names` + `obs_names` + `env_reward`.
- `action_names`, `obs_names`, `aux_sig_names`: signal name → Python
  expression, `eval()`'d per step in `get_sample` (e.g. `x: "obs[0]"`,
  `ego_vx: "80*obs[0][3]"`). By design user-provided.
- **`env_reward`**: the raw environment reward is sampled into the STL driver
  under the signal name `env_reward` (expression `'reward'`). The
  agent-visible/shaped reward is exposed separately as `reward` in the
  plotting namespace only. (Renamed from `reward` in 2026-10 to disambiguate.)
- `reward_formulas`: STL formulas added to the reward
  (`new_reward = env_reward*env_reward_weight + Σ w_f·ρ_f(t)`).
- `label_formulas`: STL formulas monitored each step; a label is *true* when
  its monitored robustness lower bound > 0. True labels drive reward machines.
- `reward_machines`: list of RM configs (`name`, `weight`, `in_observation`,
  and `reward_machine:` sub-dict = the FSM `cfg_rm` parsed by `RewardMachine`).
  In single-objective mode RM rewards are weighted and **summed into the
  overall agent reward**; reaching a `final` RM state sets `terminated=True`.
- `obs_formulas`: formulas appended to the observation (obs becomes
  DictSpace `{unwrapped, obs_formulas, rm_states?}`).
- `eval_formulas`: formulas evaluated (re)after episodes for metrics only.
- `end_formulas`: formulas that can force `terminated=True` (`lower_rob > 0`).
- `real_time_step` (default 1), `BigM` (default `stlrom.Signal.get_BigM()`),
  `multi_objective` (mo-gymnasium vector reward for MORL), `env_reward_weight`
  (default 1.0), `debug_signals`, `debug_formulas`, `flatten_obs`.

**STLData integration (stlrom 0.4.0, 2026-10 migration — see
`STLROM_OVERVIEW.md`):**

- One `stlrom.STLDriver` per wrapper; `reset()` **recreates the driver**
  (stlrom has no reset API; `reset_monitor`) and sets
  `episode['stl_data'] = stl_driver.data` — a **live `STLData` handle**
  (pybind returns the same Python object for the driver's internal `STLData`;
  it is shared with the driver's monitors). Episodes therefore keep their data
  alive after the next reset, enabling replay/plotting.
- `episode` dict: `observations`, `actions`, `rewards` (shaped),
  `env_rewards` (raw), `dones`, `last_obs`, `stl_data` (STLData handle),
  `labels` (true label names per step), `rm_states` (state idx per machine per
  step), `rm_rewards` (per-machine rewards per step), `res_f` (per-step
  robustness for obs/reward/label/end formulas; **namespaced keys** since
  2026-10-26 — `'obs_formulas/<f>'`, `'reward_formulas/<f>'`, ... — so a
  formula used in several groups is recorded once per step per group).
- `get_ep_len()` / `get_time()` / `get_sig(name)` read `episode['stl_data']`
  via `get_signal(idx_or_name)` / `get_samples_list()`.
- `eval_formula_cfg(f_name, f_opt)` — per-step evaluation via
  `stl_driver.get_online_rob(f, t0)` → `[estimate, lower, upper]`.
  **Time anchoring** (changed 2026-10-25): by default the formula is
  evaluated at the *current time* (minus `past_horizon`) — the formula's
  value "now" — so the recorded per-step values (`res_f`: reward shaping,
  labels, obs formulas) are consistent with `rho(f)`/`rob(f)` plots (exact
  for state formulas; for temporal formulas the per-step value is online
  (past-only) while `rho(f)` is offline (sees the future)). `online: True`
  / `eval_all_steps: True` state that anchoring explicitly; an explicit
  `t0` (without `online`) or `online: False` pins the evaluation to a fixed
  time (default 0) — e.g. whole-episode formulas tracked from the start
  (`init_sat`/`init_rob` metrics). Other options: `formula` (legacy alias
  `formula_name`) — the actual STL formula to evaluate, defaulting to the
  config key, so the same formula can be configured several times with
  different options (e.g. `past_horizon` 0 vs 1 = value now vs one step
  earlier); `formula_params: {<param>: <value>}` — overrides stlrom
  parameters declared in the spec (`param <name> = <default>`) for this
  config only (applied and restored around each evaluation, since the
  driver's param map is global); `past_horizon`, `tend`, `semantics`
  (`rob`/`lower_rob`/`upper_rob`/`bool`), `lower_bound`/`upper_bound`.
  (Before the change the default was
  fixed `t0=0`, which made recorded values flat for state formulas and
  inconsistent with `rho(f)` — e.g. `danger` in the highway specs.)
  Note `self.semantics = 'Boolean'` in `__init__` is a **dead string** —
  `stlrom.Signal.set_semantics` is never called, so stlrom runs in its
  default `SPACE` semantics (TODO in code).
- `eval_specs_episode(episode, res, ...)` — per-episode metrics: `ep_len`,
  `ep_rew`, per-formula `mean`/`sum`/`num_sat` (reward formulas from `res_f`),
  eval formulas re-played step by step (offline or `eval_all_steps`),
  aggregates in `res_all_ep` (`basics`, `reward_formulas`, `eval_formulas`
  with `mean_rob`/`mean_sum`/`mean_num_sat`/`ratio_init_sat`/`mean_init_rob`).

**Plotting / signal access (rewritten 2026-10, STLData-compatible since
2026-10-25):**

- `get_signal_from_str(name)` → `stlrom.Signal` for a plot token. Raw mapped
  signals (obs/actions/aux, `env_reward`) are returned **directly as the
  driver's `stlrom.Signal`** (its own plotting handles interpolation; driver
  samples are appended with `interp=False`, i.e. zero-order-hold). Other
  tokens are re-encoded into a fresh `Signal`: discrete (zero-order-hold)
  signals — rewards, actions, label/states, `sat(...)` — are encoded with
  `append_constant_sample` (derivative 0 → rendered as steps); continuous
  signals use `append_linear_sample`.
- `get_values_from_str(name)` → `(values, sig_type)` — same resolver, values
  only (used by live plots).
- Both delegate to `_resolve_signal_str(name)` which returns
  `(values, sig_type, discrete, sig)` — `sig` is a `stlrom.Signal` when the
  token maps directly to one, else `None`. Supported tokens:
  - bare mapped signal names (`action_names`/`obs_names`/`aux_sig_names`,
    plus `env_reward`) — via `get_sig`
  - `reward` — the shaped (agent-visible) reward
  - `obs_formulas/<f>`, `reward_formulas/<f>`, `label_formulas/<f>` — per-step
    values from `res_f` (labels render as discrete steps)
  - `eval_formulas/<f>` — offline robustness over the trace (`get_rob`)
  - `rm_states/<machine>` — per-step state index (discrete)
  - `rm_reward/<machine>` or bare `<machine>` — per-step RM reward
  - `rho(<f>[,past_horizon])`, `rob(<f>[,past_horizon])`, `sat(<f>)` — via
    `get_rob` (offline) / satisfaction
  - bare `<f>` — recorded `res_f` values if present, else offline robustness
  - `:weighted` suffix — multiply by the configured formula or RM weight
    (or `env_reward_weight` for the `env_reward` raw signal, which then
    returns a re-encoded scaled copy instead of the driver's Signal)
  - (`set_ep_idx(<n>)` is handled in `RLTester.get_fig`, not in the resolver.)

Key methods: `step` (sample → driver → robustness → shaped reward/obs → RM
step), `set_episode_data(episode)` (replay: fresh driver, copy the episode's
`STLData` in, restore `time_step`/`current_time`), `show_reward_machines()`
(text summary of states/transitions/labels), and `get_rob(formula, horizon,
online)` — robustness at every time of the trace: **offline** (default in the
plot path) evaluates `stl_driver.get_rob(formula, t0)` per step over the full
trace; **online** replays the trace sample-by-sample into a *fresh* driver so
each step is evaluated without the future (stlrom has no reset API). Unknown
formula names raise `ValueError` (stlrom itself only prints and returns
0./empty).

### RewardMachine (`wrappers/reward_machine.py`)

- **Decoupled pure FSM** (no gym, no STL): parses `cfg_rm`, tracks current
  state; on `step(labels)` (list of *true* condition names) takes the
  highest-`priority` enabled transition or stays and issues the state reward
  (default 0). Two enabled transitions sharing the max priority raise
  `ValueError`. Exposes `reset()`, `state_id`, `state_idx`, `is_final`,
  `states`, `states_idx_map`, `state_rewards`, `transitions`, `_is_initial`.
- Config: `states` (`id`, optional `initial`/`final`/`reward`; exactly one
  initial), `transitions` (`from`/`to`/`condition` required; `reward=0`,
  `priority=0`, `id` default `{from}__to__{to}__if__{condition}`).
- `conditions:`/`debug_rm`/`in_observation`/`rm_id` keys in `cfg_rm` are *not*
  part of the FSM — they belong to the wrapping/STL layer.

## Configuration system (how everything is wired)

- One main YAML cfg per run; values ending in `.yml` are **recursively
  replaced by the file's content** (`cfg_utils.load_cfg`); `.stl` fields load
  file *text*.
- `this_cfg_pathdir` is auto-set to the cfg file's dir and the process
  **chdir's there** — relative paths (models, STL files, `import_module`)
  resolve from the cfg's folder.
- Top-level keys: `env_name`, `import_module`, `make_env_train`/
  `make_env_test` (custom env factories), `model_name`, `model_path`,
  `model_use_specs`, `cfg_env`, `cfg_specs`, `cfg_train` (algo block, n_envs,
  total_timesteps, eval_freq, `eval` sub-test-cfg), `cfg_test` (num_ep,
  num_steps, init_seeds, render_mode, model_file, res_file, plot).
- Algorithms selected in `cfg_train.algo` by key name from
  `ALGO_NAMES_CLASSES` (`ppo`, `sac`, ..., `sbx_*`, `moppo`, `gpipd`,
  `qlearning`, ...). MORL algos (morl-baselines) use
  `.train(total_timesteps, eval_env, **train_kwargs)` instead of SB3 `.learn()`.
- Training output: `models/<model_name>_<date>__training<N>/` with `cfg0.yml`,
  `model_step_*.zip`, `res_step_*.yml`; final model symlinked/copied to
  `models/<model_name>.zip` (+ `.yml`).

## Test-time plotting (`cfg_test.plot`)

`RLTester.run_cfg_test` (via `rlr test`) can build one matplotlib figure per
item of a `cfg_test.plot` list (episode 0), after all episodes run. Figures
are built with `stlrom.Signal.plot` (discrete signals encoded as constant
samples → step rendering):

```yaml
cfg_test:
  ...
  plot:
    - title: Observations            # optional figure title
      out: plots/obs.png             # optional: save the figure
      ax:                            # each item = one axis (stacked, shared x)
        - x,theta                    # comma-separated signal names on the axis
        - x_dot
    - title: Formulas
      ax:
        - label_formulas/go_left
        - [label_formulas/cart_in_A, label_formulas/stay_A]   # list form works
```

Signal names follow the `get_signal_from_str` grammar above (resolved by
`SpecWrapper.get_signal_from_str`, values by `get_values_from_str`).

### Live-updating plots (`live: True`)

A `plot` item with `live: True` opens a live window that updates every env
step, per episode (`rlr test` only; headless/Agg backends skip the window
silently but still honor `out:`):

```yaml
cfg_test:
  plot:
    - title: Live observations
      live: True                  # open a live window during the episodes
      live_window: 20             # optional trailing x window (time units)
      out: plots/live.png         # saved from the last episode
      ax:
        - x, theta
        - reward
```

Implemented by `rlrom.plots.LivePlot` (created in `RLTester.run_episode`,
one `step()` per env step, closed at episode end):

- **space** pauses/resumes updates (episode keeps running; window stays
  responsive); **Escape** closes the windows and ends the episode early.
- discrete signals drawn with `drawstyle='steps-post'` (detected via
  `_is_discrete_signal`: all samples have derivative 0).
- `live: True` items are skipped in the static post-run display (only their
  `out:` file is written); non-live items display as before.
- Legacy notebook helper: `rlrom.plots.RLFig`/`live_line` (driven by
  `RLTester.fig_layout`), `plot_signal`/`plot_rob` convenience wrappers.

## Canonical example configs (start here)

- `examples/cartpole/cfg0tr_ppo_specs.yml` + `cartpole.stl` — simplest full
  train-with-STL-reward example (CartPole, PPO, one reward formula, one eval
  formula). Also the smoke-test config for `.pi/AGENTS.md`
  (`uv run rlr test ...`).
- `examples/cartpole/cfg0tr_rm_AB.yml` + `rm_AB.yml` — **label-formula +
  reward-machine** example (2-state FSM on CartPole left/right, drives
  `cfg_specs.reward_machines`; covered by
  `tests/test_reward_machine_spec.py`). Related: `cfg0tr_rm_left_right*.yml`,
  `cfg0tr_rm_stay_left_right.yml` (two-figure plotting example),
  `cfg0tr_stl_go_left.yml`, `rm_stuck_left.yml`.
- `examples/highway_env/cfg0tr.yml` (+ `cfg_specs.yml`, `hw-env_specs.stl`,
  `highway.py` with custom `make_env_*`) — richer: obs_names expressions,
  eval + reward formulas, custom module. `cfg_specs_multi_objective.yml`
  exercises the (currently broken) multi-objective branch.
- `examples/highway_env/cfg0tr_rmf.yml` + `cfg_specs_rmf.yml` — legacy
  reward-machine example (old `cfg_rm` path). **Currently non-functional:**
  the `RewardMachineWrapper` it relied on was removed in 2026-10; use
  `cfg_specs.reward_machines` instead.
- `examples/minigrid/` and `minigrid/blocked_unlock_pickup/` — FSM/RM case
  study with multi-state RM, negative rewards, `TabularQLearning`.
- `examples/simglucose/` — simglucose experiment thread (not a dependency;
  needs the `simglucose` package).

## Test suite

Run: `uv run pytest tests/` (fast, no models/downloads). **Status
2026-10-25: all 116 tests pass** (the 3 `test_episode_fig.py` failures from
the mid-migration state were fixed the same day — see Known issues A).

- `tests/test_utils.py` — `parse_signal_spec`, `get_formulas`,
  `set_rec_cfg_field`, `load_cfg` expansion on a real example cfg.
- `tests/test_smoke.py` — fast end-to-end on CartPole with a random policy:
  unwrapped `RLTester` run, `SpecWrapper` step/metrics (asserts
  `episode["stl_data"].get_signal(0)` sample counts — the STLData API),
  reset/data-reuse, reward-shaping bounds.
- `tests/test_reward_machine.py` — unit tests of the pure `RewardMachine` FSM
  (parsing, state validation, priority transitions, tie guard, state rewards).
- `tests/test_reward_machine_spec.py` — `cfg_specs.reward_machines`
  integration through `SpecWrapper`: label evaluation, RM stepping, episode
  recording, obs `rm_states` field, `show_reward_machines()`.
- `tests/test_episode_fig.py` — the `cfg_test.plot` pipeline:
  `get_signal_from_str` returns `stlrom.Signal`s, `get_fig` builds
  multi-axis figures with correct legends, unknown formulas produce a
  warning, `plot` items save `out:` files, headless `live: True` no-op
  behavior, `LivePlot` line data/drawstyle/xlim.
- `tests/test_plot.py` — `get_layout_from_string` tokenizer,
  `plot_training_results` figures.
- `tests/test_select.py` — `rlr select-model` selection logic
  (`_parse_trainings`/`_resolve_trainings`).
- `tests/test_train_repeat.py` — repeated short trainings sanity.

No CI. `load_cfg` chdir's to the cfg file's folder before running.

## Known issues / caveats (2026-10-25 review)

**A. STLData (stlrom 0.4.0) migration — completed 2026-10-25:**

1. ~~`get_sig` returned a `stlrom.Signal` while the resolver expected values~~
   — **fixed**: `_resolve_signal_str` now returns a 4-tuple
   `(values, sig_type, discrete, sig)`; raw mapped signals pass the driver's
   `Signal` through to `get_signal_from_str` (values are extracted from the
   samples for value-based consumers such as live plots).
2. ~~`get_rob` used the pre-0.4.0 array API~~ — **fixed**: offline mode
   evaluates `stl_driver.get_rob(formula, t0)` per step; online mode replays
   the trace into a fresh `STLDriver` (one `add_sample` + `get_online_rob`
   per step); unknown formulas raise `ValueError` (checked via
   `get_monitor(f).formula is None`).
3. ~~`data == []` / `stl_data == []` empty-checks never fired~~ — **fixed**:
   `get_ep_len`/`get_time` now use `isinstance(stl_data, stlrom.STLData)`.
4. ~~`res_f` shared one key per formula name across groups~~ — **fixed
   2026-10-26**: `reset()`/`step()` now use namespaced keys
   (`'obs_formulas/<f>'`, `'reward_formulas/<f>'`, ...). Before, a formula
   present in both `obs_formulas` and `reward_formulas` (e.g. `danger` in
   the highway specs) was appended **twice per step** to one shared list
   — the plot of `obs_formulas/<f>` / `reward_formulas/<f>` alternated
   between the two evaluations (with a `lower_bound` on the reward side
   this looked like "0 every second step"). Bare plot tokens resolve via
   `_res_f_trace` (reward group first, then obs/label/end); prefer the
   explicit `prefix/<f>` token. `eval_specs_episode` reward metrics read
   the namespaced key.
5. ~~`end_formulas` recorded the last `obs_formulas` value in `res_f`~~
   — **fixed 2026-10-26**: the end-formula loop now appends its own
   `eval_res['lower_rob']` (the value driving the termination check),
   like the label loop.
6. ~~Vestigial `obs_name` option (in `obs_formulas` configs)~~ — **removed
   2026-10-26** (code + example cfgs). It only wrote the config dict into
   `signals_map`, which (a) hijacked bare plot tokens of obs-formula names
   into the raw-signal branch (empty signal instead of the recorded
   `res_f` trace) and (b) was a crash landmine: a config key/`obs_name`
   colliding with a declared signal made `get_sample` `eval(<dict>)` →
   `TypeError`. Obs formulas still reach the agent as the positional
   `obs['obs_formulas']` vector; `cfg_specs.obs_names` (the signal→env
   expression map) is unchanged.

**B. Latent bugs (not exercised by the suite):**

4. **`multi_objective: True` crashes in `SpecWrapper.__init__`**: it reads
   `self.keep_old_reward`, which is never set (the cfg key `keep_old_reward`
   is not read anywhere in the wrapper). Also, `keep_old_reward` is a
   *documented* cfg key (used in `tests/test_smoke.py` and
   `examples/highway_env/cfg_specs_multi_objective.yml`) but the wrapper
   ignores it entirely — `env_reward_weight` (default 1.0) always applies.
   The vector-reward branch also does **not** include reward-machine rewards.
5. **Semantics are not actually Boolean**: `self.semantics = 'Boolean'` is a
   dead string; `stlrom.Signal.set_semantics` is never called, so stlrom runs
   in default `SPACE` semantics (test comments claiming "Boolean semantics"
   are stale; rewards are continuous robustness values, not ±1).
6. **Replay "online" robustness can see the future**: `eval_specs_episode`
   re-plays eval formulas through `eval_formula_cfg` →
   `stl_driver.get_online_rob(f, t0)`, but the driver already holds the
   *whole* trace, so `online`/`eval_all_steps` values at step t are computed
   with data from after t. (The old `get_rob(online=True)` replayed samples
   one-by-one into a fresh monitor; that path is currently broken — issue 2.)
7. ~~**`end_formulas` records the wrong value in `res_f`**~~ — **fixed
   2026-10-26**, see A.5 (the step loop appended the last `obs_formulas`
   robustness `robs_f`; it now appends the end-formula's own
   `lower_rob`).
8. **`RLTester.eval_all_episodes` is dead/broken**: it calls
   `self.env.get_wrapper_attr('eval_episode')`, but no wrapper exposes
   `eval_episode` (SpecWrapper has `eval_specs_episode`).
9. **`wrap_env_specs` flattens dict obs when RMs/obs formulas are present**
   (default `flatten_obs: True`), flattening the `rm_states`/`obs_formulas`
   dict fields away unless the user sets `flatten_obs: False`.
10. **`episode['stl_data']` liveness relies on an undocumented pybind11
    object-registry behavior** (same Python object returned for the driver's
    internal STLData; shared with monitors). Works today, but a stlrom
    rebuild/change could silently turn it into a stale snapshot.

**C. Staleness / environment:**

11. **Installed stlrom (PyPI 0.4.0, 2026-09-18) lags the `stlrom/` checkout**
    (tubes refactor, `draw_tube` plotting, `STLData.inflate_signal`,
    `samples_to_points`, new tests). Rebuild/reinstall from `stlrom/` to
    pick those up; note the checkout's pybind also still does not expose
    `STLData.reset_signal_data`/`is_empty`/`get_size`/`set_data_vector`.
12. **`load_cfg` mutates cwd** (chdir to cfg dir) — global side effect.
    `rlr show` without `--expand` bypasses `load_cfg` (raw top-level YAML,
    nested refs shown as paths); `--subcfg KEY` selects a sub-configuration.
13. **`testers.make_env_test`** does `sys.modules[cfg['import_module']]` —
    assumes the module was already imported (training path imports it; pure
    test runs may not).
14. `SpecWrapper.seed()` is legacy gym v0 style (calls `reset(seed=...)`).

## Conventions

- Config-driven by default: prefer changing YAML over code for experiment
  changes.
- Signal/formula names in STL are the contract between `.stl` files and cfg
  `*_names`/`*_formulas` blocks.
- Metric naming: `res` (per-episode, arrays) → `res_all_ep` (aggregates:
  `basics`, `reward_formulas`, `eval_formulas`).
- Keep `.pidocs/` current: after any change that alters structure, APIs,
  config keys, or the known-issues list, update the corresponding section in
  the same change (see `.pi/AGENTS.md`).
