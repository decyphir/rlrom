# RLRom — Code Base Overview

*Reference map of the repository as it currently stands: package structure,
wrapper pipeline, config system, canonical examples, known issues. Keep it
current when a change alters structure, APIs or config keys — it describes
the present state, not the history (that is what git is for). The stlrom
library is documented in [`STLROM_OVERVIEW.md`](STLROM_OVERVIEW.md).*

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
3. **Constrain** agents: reward-machine states and monitor formulas can
   force/block raw actions (a shield applied inside the wrapper).

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
│                         #   own .venv/build); untracked here, not a submodule
├── latex/                # paper chapter (LaTeX sources + build artifacts)
└── .pidocs/              # this folder — reference docs
```

**Ignore when exploring:** `examples/*/models/*__training*`, `examples/*/tb_logs`,
`.venv`, `uv.lock`, `stlrom/.venv`, `stlrom/build` (huge). Trained checkpoints
are experiment artifacts, not code.

**Sub-projects:**
- `stlrom/` — the STL monitoring library (C++ + pybind11). rlrom uses the
  **PyPI 0.4.0 wheel** (via `uv.lock`); the checkout is a local dev tree whose
  development branches are **not** used by rlrom. See
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
| `trainers.py` | 311 | `RLTrainer`: env creation (`make_env_train`, `make_vec_envs`), algorithm instantiation, SB3 `model.learn()`, checkpoints, periodic eval via `RlromCallback` (saves `model_step_*.zip` + `res_step_*.yml` per eval, tensorboard logging). |
| `testers.py` | 444 | `RLTester`: runs episodes with a trained model (or `random`/keyboard), collects episodes, computes metrics (`eval_episode`), matplotlib episode figures via `stlrom.Signal.plot` (`get_fig`), static + live `cfg_test.plot` figures, `retest_checkoints_models` (sic). |
| `wrappers/spec_wrapper.py` | 1203 | **Core of the project.** `SpecWrapper(gym.Wrapper)` + `wrap_env_specs(env, cfg)` — see below. |
| `wrappers/reward_machine.py` | 222 | `RewardMachine` (pure FSM, no gym/STL): transitions + state `actions` (shielding accessors). |
| `wrappers/stl_wrapper.py` | 751 | **Dead code** (untracked, imported by nothing): an unused `STLWrapper` duplicating `SpecWrapper`. Candidate for deletion before it ships in the wheel. |
| `utils.py` | 743 | Algorithm registry `ALGO_NAMES_CLASSES` (SB3, sbx/jax, sb3-contrib, morl-baselines, tabular Q), HuggingFace model loading, training-results polars dataframes (`get_training_folders`, `get_training_res`, `get_df_all_trainings`, `get_metric_series`, `get_selection_df`, ...), `append_to_field_array`, misc. Re-exports config helpers from `cfg_utils`. |
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
  plotting namespace only — `reward` is the token for the shaped value.
- **`monitor_formulas`** — the single formula section. Each entry is evaluated
  **once per step** and always produces a *label* for the reward machines
  (true when its monitored lower robustness bound > 0). Extra fields turn it
  into more than a label:
  - `obs_weight: w` — also appended to the observation, scaled by `w`
  - `reward_weight: w` — also added to the reward (`w·ρ`); a bare `weight`
    is honoured too
  - `terminal: true` — terminate the episode when the formula becomes true
  - `actions: {enable, blocked, forced, fallback, priority}` — shielding (see
    below)
  - evaluation options: `online`, `t0`, `past_horizon`, `tend`, `semantics`,
    `lower_bound`/`upper_bound`, `formula` (alias `formula_name`),
    `formula_params`
  Reward: `new_reward = env_reward*env_reward_weight + Σ w_f·ρ_f(t)`
  (+ weighted RM rewards). `obs_formulas`/`reward_formulas` are still accepted
  (deprecated, `DeprecationWarning`); declaring the same
  formula both ways with a non-zero weight raises `ValueError`.
  `label_formulas` is accepted as an alias of `monitor_formulas` (error if
  both are present).
- **`reward_machines`**: list of RM configs (`name`, `weight`,
  `in_observation`, `action_priority`, and a `reward_machine:` — or `fsm:` —
  sub-dict = the FSM `cfg_rm` parsed by `RewardMachine`). In single-objective
  mode RM rewards are weighted and **summed into the overall agent reward**;
  reaching a `final` RM state sets `terminated=True`.
- `eval_formulas`: formulas evaluated (re)after episodes for metrics only.
- `real_time_step` (default 1), `BigM` (default `stlrom.Signal.get_BigM()`),
  `multi_objective` (mo-gymnasium vector reward for MORL), `keep_old_reward`,
  `env_reward_weight` (default 1.0), `debug_signals`, `debug_formulas`,
  `debug_rm`, `debug_actions`, `disable_actions` (turns all shielding off),
  `flatten_obs` (default True).

### Shielding / action constraints

Two config sites can constrain the action before it reaches the env, resolved
by `SpecWrapper._apply_shield(action) -> (action, info)`:

- a reward-machine **state**: `actions: {enable, blocked, forced, fallback}`,
  active while the machine is in that state, with the machine-level
  `action_priority` as its priority;
- a **monitor formula**: the same `actions` dict, active from the **next**
  step after the formula became true (its truth only known post-step), with
  `actions.priority` (default 0).

`forced` wins; `blocked` is cumulative across sources; a blocked action is
replaced by the previous action (if unblocked) else by the highest-priority
blocker's `fallback`. Priority (default 0) is only consulted on conflict, and
*sources that agree never conflict*:

- forcing the same action at a tie is fine; forcing different actions at the
  max priority raises `ValueError`;
- a strictly higher-priority blocker beats a forcer (and supplies the
  replacement); a lower-priority blocker is ignored; a blocker **tying** the
  forcer on the blocked action is an error;
- equal-max-priority blockers must agree on `fallback` (identical, or `None`
  deferring to the one that is set).

`cfg_specs.disable_actions: True` turns every constraint off (agent action
always passes through) — handy when comparing shielded vs. unshielded runs.
The override is reported in `info['shield']` (`machine`/`reason`/`original`/
`shielded`) and printed with `debug_actions`. Raw action values only (ints).
Tests: `tests/test_shield.py` (38 cases), `tests/test_debug_rm.py`.

**STLData integration (stlrom 0.4.0 — see `STLROM_OVERVIEW.md`):**

- One `stlrom.STLDriver` per wrapper; `reset()` **recreates the driver**
  (stlrom has no reset API; `reset_monitor`) and sets
  `episode['stl_data'] = stl_driver.data` — a **live `STLData` handle**
  (pybind returns the same Python object for the driver's internal `STLData`;
  it is shared with the driver's monitors). Episodes therefore keep their data
  alive after the next reset, enabling replay/plotting.
- `episode` dict: `observations`, `actions`, `rewards` (shaped),
  `env_rewards` (raw), `dones`, `last_obs`, `stl_data` (STLData handle),
  `labels` (true label names per step), `rm_states` (state idx per machine per
  step), `rm_rewards` (per-machine rewards per step), `res_f` (per-step values
  of every monitored formula, **namespaced by the section that declared it**
  (`'monitor_formulas/<f>'`, `'obs_formulas/<f>'`, `'reward_formulas/<f>'`),
  so a formula declared in several of the deprecated sections is recorded
  once per group.
- `get_ep_len()` / `get_time()` / `get_sig(name)` read `episode['stl_data']`
  via `get_signal(idx_or_name)` / `get_samples_list()`.
- `eval_formula_cfg(f_name, f_opt)` — per-step evaluation via
  `stl_driver.get_online_rob(f, t0)` → `[estimate, lower, upper]`.
  **Time anchoring**: by default the formula is
  evaluated at the *current time* (minus `past_horizon`) — the formula's
  value "now" — so the recorded per-step values (`res_f`: reward shaping,
  labels, obs formulas) are consistent with `rho(f)`/`rob(f)` plots (exact
  for state formulas; for temporal formulas the per-step value is online
  (past-only) while `rho(f)` is offline (sees the future)). `online: True`
  / `eval_all_steps: True` state that anchoring explicitly; an explicit
  `t0` (without `online`) or `online: False` pins the evaluation to a fixed
  time (default 0) — e.g. whole-episode formulas tracked from the start
  (`init_sat`/`init_rob` metrics). Other options: `formula` (alias
  `formula_name`) — the actual STL formula to evaluate, defaulting to the
  config key, so the same formula can be configured several times with
  different options (e.g. `past_horizon` 0 vs 1 = value now vs one step
  earlier); `formula_params: {<param>: <value>}` — overrides stlrom
  parameters declared in the spec (`param <name> = <default>`) for this
  config only (applied and restored around each evaluation, since the
  driver's param map is global); `past_horizon`, `tend`, `semantics`
  (`rob`/`lower_rob`/`upper_rob`/`bool`), `lower_bound`/`upper_bound`.
  Note `self.semantics = 'Boolean'` in `__init__` is a **dead string** —
  `stlrom.Signal.set_semantics` is never called, so stlrom runs in its
  default `SPACE` semantics (TODO in code).
- `eval_specs_episode(episode, res, ...)` — per-episode metrics: `ep_len`,
  `ep_rew`, per-formula `mean`/`sum`/`num_sat` (reward formulas from `res_f`),
  eval formulas re-played step by step (offline or `eval_all_steps`),
  aggregates in `res_all_ep` (`basics` — including `completed`, the fraction
  of episodes that reached `cfg_test.num_steps` instead of terminating early;
  `reward_formulas`; `eval_formulas` with
  `mean_rob`/`mean_sum`/`mean_num_sat`/`ratio_init_sat`/`mean_init_rob`).
  Caveat: the reward-formula aggregates are computed inside the
  `if self.eval_formulas` branch, so they stay empty when no `eval_formulas`
  is configured.

**Plotting / signal access:**

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
  - `monitor_formulas/<f>`, `obs_formulas/<f>`, `reward_formulas/<f>`,
    `label_formulas/<f>` (alias of `monitor_formulas`) — per-step values from `res_f`
    (labels render as discrete steps)
  - `eval_formulas/<f>` — offline robustness over the trace (`get_rob`)
  - `rm_states/<machine>` — per-step state index (discrete)
  - `rm_reward/<machine>` or bare `<machine>` — per-step RM reward
  - `rho(<f>[,past_horizon])`, `rob(<f>[,past_horizon])`, `sat(<f>)` — via
    `get_rob` (offline) / satisfaction
  - bare `<f>` — recorded `res_f` values if present, else offline robustness
  - `:weighted` suffix — multiply by the configured formula or RM weight
    (`reward_weight` → `obs_weight` → `weight` for monitor formulas; or
    `env_reward_weight` for the `env_reward` raw signal, which then
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
- Config: `states` (`id`, optional `initial`/`final`/`reward`, and an
  optional `actions` dict for shielding — `enable` (default True), `blocked`
  (list of raw action values), `forced`, `fallback`; exactly one initial),
  `transitions` (`from`/`to`/`condition` required; `reward=0`, `priority=0`,
  `id` default `{from}__to__{to}__if__{condition}`).
- Action accessors (used by the wrapper's shield): `actions_enabled()`,
  `blocked()`, `forced()`, `fallback()` (each with an optional `state_id`).
  Values are validated as ints by `_validate_action_value`.
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
        - monitor_formulas/go_left
        - [monitor_formulas/cart_in_A, monitor_formulas/stay_A]   # list form works
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
  formula). Also the smoke-test config for `AGENTS.md` (`uv run rlr test ...`).
- `examples/cartpole/cfg0tr_rm_AB.yml` + `rm_AB.yml` — **label-formula +
  reward-machine** example (2-state FSM on CartPole left/right, drives
  `cfg_specs.reward_machines`; covered by
  `tests/test_reward_machine_spec.py`). Related: `cfg0tr_rm_left_right*.yml`,
  `cfg0tr_rm_stay_left_right.yml` (two-figure plotting example),
  `cfg0tr_stl_go_left.yml`, `rm_stuck_left.yml`.
- `examples/highway_env/cfg0tr.yml` + `cfg_specs.yml` (+ `cfg_env.yml`,
  `hw-env_specs.stl`, `highway.py` with custom `make_env_*`) — the single
  maintained highway config. It is the
  reference for a full `monitor_formulas` setup: three reward-shaped formulas
  (`reward_weight`), `danger` with `obs_weight` + shielding, and four
  lane/danger-conditioned constraints (three of them at `actions.priority: 1`,
  the rest defaulting to 0). `cfg0_manual.yml` + `cfg_env_manual.yml` stay for
  keyboard control.
- `examples/minigrid/` and `minigrid/blocked_unlock_pickup/` — FSM/RM case
  study with multi-state RM, negative rewards, `TabularQLearning`.
- `examples/simglucose/` — simglucose experiment thread (not a dependency;
  needs the `simglucose` package).

## Test suite

Run: `uv run pytest tests/` (fast, no models/downloads). **Current status:
all 167 tests pass (~15 s).**

- `tests/test_utils.py` — `parse_signal_spec`, `get_formulas`,
  `set_rec_cfg_field`, `load_cfg` expansion on a real example cfg.
- `tests/test_smoke.py` — fast end-to-end on CartPole with a random policy:
  unwrapped `RLTester` run, `SpecWrapper` step/metrics (asserts
  `episode["stl_data"].get_signal(0)` sample counts and
  `episode["res_f"]["monitor_formulas/<f>"]` traces), reset/data-reuse,
  reward-shaping bounds, deprecation/conflict errors on the deprecated
  formula sections.
- `tests/test_shield.py` — action constraints: RM-state and monitor-formula
  `actions`, forced/blocked/fallback, priority conflicts and ties,
  `info['shield']`.
- `tests/test_debug_rm.py` — `debug_rm`/`debug_actions` per-step prints.
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

## Known issues / caveats

None of these are covered by the test suite, so they can regress silently.

1. **`multi_objective` is inconsistent**: `reward_dim` is sized from
   `keep_old_reward` while the returned vector includes the raw env reward iff
   `env_reward_weight` is truthy, so the vector can differ in length from
   `reward_space`. The vector branch also does not include reward-machine
   rewards. No example config exercises the branch (`multi_objective` only
   appears set to `False` in `examples/highway_env/cfg_specs.yml`).
2. **Semantics are not actually Boolean**: `self.semantics = 'Boolean'` is a
   dead string; `stlrom.Signal.set_semantics` is never called, so stlrom runs
   in its default `SPACE` semantics. Rewards are continuous robustness values,
   not ±1, and some test comments claiming otherwise are misleading.
3. **Replayed "online" robustness can see the future**:
   `eval_specs_episode` re-plays eval formulas through `eval_formula_cfg` →
   `stl_driver.get_online_rob(f, t0)`, but the driver already holds the whole
   trace, so `online`/`eval_all_steps` values at step t use data from after t.
   (`SpecWrapper.get_rob(online=True)` is the correct path: it replays
   sample-by-sample into a fresh driver.)
4. **`RLTester.eval_all_episodes` is dead/broken**: it calls
   `self.env.get_wrapper_attr('eval_episode')`, but no wrapper exposes
   `eval_episode` (SpecWrapper has `eval_specs_episode`).
5. **`wrap_env_specs` flattens dict observations** (`flatten_obs: True` by
   default), which removes the `rm_states`/`obs_formulas` fields unless the
   user sets `flatten_obs: False`.
6. **`episode['stl_data']` liveness relies on an undocumented pybind11
   object-registry behavior** (same Python object returned for the driver's
   internal STLData, shared with monitors). A stlrom rebuild could silently
   turn it into a stale snapshot.
7. **The `stlrom/` checkout is not what rlrom runs** (that is the PyPI 0.4.0
   wheel). Its branches carry extra APIs (tubes, `plot_tube`, ...) that rlrom
   does not use. The pybind layer does not expose
   `STLData.reset_signal_data`/`is_empty`/`get_size`/`set_data_vector`.
8. **`load_cfg` mutates cwd** (chdir to the cfg's directory) — a global side
   effect. `rlr show` without `--expand` bypasses `load_cfg` (raw top-level
   YAML, nested refs shown as paths); `--subcfg KEY` selects a
   sub-configuration.
9. **`testers.make_env_test`** does `sys.modules[cfg['import_module']]` —
   assumes the module was already imported (the training path imports it, a
   pure test run may not).
10. `SpecWrapper.seed()` is gym v0 style (calls `reset(seed=...)`).
11. `--set-params k=v` stores **strings** (`ParseKwargs`), so numeric overrides
    arrive as text; only `render_mode` has an explicit `'None'` string
    special-case in `make_env_test`.
12. `eval_specs_episode` computes the `res_all_ep['reward_formulas']`
    aggregates *inside* the `if self.eval_formulas` branch — with monitor
    formulas as rewards and no `eval_formulas`, reward aggregates are empty.

## Conventions

- Config-driven by default: prefer changing YAML over code for experiment
  changes.
- Signal/formula names in STL are the contract between `.stl` files and cfg
  `*_names`/`*_formulas` blocks.
- Metric naming: `res` (per-episode, arrays) → `res_all_ep` (aggregates:
  `basics`, `reward_formulas`, `eval_formulas`).
- Keep `.pidocs/` describing the current state: after any change that alters
  structure, APIs, config keys or the known-issues list, update the
  corresponding section in the same change (see `AGENTS.md`).
