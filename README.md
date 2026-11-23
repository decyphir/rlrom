# RLRom

[![PyPI](https://img.shields.io/pypi/v/rlrom)](https://pypi.org/project/rlrom/)
[![Python versions](https://img.shields.io/pypi/pyversions/rlrom)](https://pypi.org/project/rlrom/)
[![License](https://img.shields.io/pypi/l/rlrom)](LICENSE)

Test and train reinforcement learning agents with **robust online monitoring of
Signal Temporal Logic (STL) specifications**.

RLRom puts a monitor around a Gymnasium environment. A specification is a set of
STL formulas over signals you name yourself (observations, actions, auxiliary
features). From the robustness values the monitor produces, RLRom can:

1. **Test** an agent — satisfaction ratios and robustness statistics over
   episodes, with per-episode plots of signals and formulas.
2. **Train** an agent — robustness values enter the observation and/or the
   reward (reward shaping), and/or drive a **reward machine** whose transitions
   are STL-labelled.
3. **Constrain** an agent — monitor formulas and reward-machine states force or
   block actions, from simple `blocked`/`forced` lists up to a custom shield
   callback for continuous action spaces.

Specifications live in YAML configuration files, so experiments are config
changes rather than code changes.

## Install

Python 3.12 or newer is required.

```bash
pip install rlrom
```

`stlrom`, the C++ monitoring engine underneath, is distributed as source, so
pip builds it while installing: you need a C++ toolchain (on Debian/Ubuntu,
`sudo apt install build-essential`). CMake comes with the build backend, so
there is nothing else to install — but the first install takes a few minutes
longer than a pure-Python package.

Some case studies need extra environments:

| Extra | Brings | Case study |
|---|---|---|
| `highway-env` | `highway-env` | `examples/highway_env` |
| `mujoco` | MuJoCo + Gymnasium MuJoCo envs | `examples/mujoco` |
| `metadrive` | MetaDrive driving simulator | `examples/metadrive` |
| `jax` | SBX (JAX) algorithms | `examples/*` |
| `morl` | multi-objective algorithms | — |
| `old_gym` | legacy `gym` 0.26 | — |

```bash
pip install "rlrom[mujoco]"
```

From a source checkout, use `uv`:

```bash
uv sync --extra mujoco       # or without extras
uv run rlr --help
uv run pytest tests/
```

## Getting started

### Command line

`rlr` runs a YAML configuration file. Examples are in the `examples/` folder.

```bash
# Test a policy fetched from the Hugging Face Hub, monitoring STL formulas
rlr test examples/cartpole/cfg0_hug.yml

# Train with STL specifications
rlr train examples/cartpole/cfg0tr_ppo_specs.yml
```

`rlr show cfg.yml` prints a configuration (optionally expanded with `--expand`,
restricted with `--subcfg KEY`), and `rlr --help` lists every option. The other
actions are `plot` and `select-model`, described below.

To repeat the same training several times (each run starts from scratch and gets
its own `<model>_<date>__trainingN` folder), set `repeat_training: N` under
`cfg_train`:

```yaml
cfg_train:
  repeat_training: 5
  # ...other training options
```

### Plotting training results

Training writes checkpoint results (`res_step_*.yml`) as it goes. `rlr plot`
turns them into figures — one training, several, or an envelope over repeats:

```bash
# Last training, mean episode reward
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml

# All trainings as lines plus the mean/min-max envelope, saved to a PNG
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml --trainings all --envelope -o training.png

# Several metrics, one subplot each
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml -m mean_ep_rew -m phi_left_goal/ratio_init_sat
```

Options: `--trainings last|all|0,2,-1|-5:-1` (negative indices count from the
end, ranges are inclusive; folders are ordered by date and number, `--verbose 1`
lists them), `-m METRIC` (repeatable), `--envelope`, `-o FILE`
(png/pdf/svg), `--title`, `--log-y`, and `--list-metrics` to discover the
available metric names. The standalone `rlrom_plot` command takes the same
options.

### Selecting a model

`rlr test` loads `<model_path>/<model_name>.zip` (plus its `.yml`). A trained
model becomes active through that symlink, which `rlr select-model` creates from
the training results:

```bash
# Best mean episode reward in the last training (default)
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml

# Best formula satisfaction ratio across all trainings
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml -m phi_left_goal/ratio_init_sat --trainings all

# First checkpoint after 100k steps / worst (min)
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --step 100000
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --min

# Inspect without selecting: top-5 ranking, or the currently active model
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --show-top 5 --trainings all
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --show-selected
```

Options: `--trainings` (pool, default `last`), `-m METRIC` (default
`mean_ep_rew`), `--min`, `--step N` (overrides the metric), `--show-top N`,
`--show-selected`, `--list-metrics`. Ties on the best value are broken by the
earliest step. The standalone `rlrom_select_model` command takes the same
options. Typical workflow: `rlr train` (with `repeat_training: N`) →
`rlr select-model` → `rlr test`.

### Notebooks

Programmatic use is demonstrated in the notebooks:

- [`examples/cartpole/cartpole_notebook.ipynb`](examples/cartpole/cartpole_notebook.ipynb) —
  the classic CartPole case study, with STL reward shaping and a reward machine.
- [`examples/highway_env/highway_notebook.ipynb`](examples/highway_env/highway_notebook.ipynb) —
  autonomous driving with [highway-env](https://github.com/Farama-Foundation/HighwayEnv),
  formula-driven rewards and action constraints.
- [`examples/minigrid/minigrid_notebook.ipynb`](examples/minigrid/minigrid_notebook.ipynb) —
  reward machines on grid-world tasks.
- [`examples/acrobot/acrobot_notebook.ipynb`](examples/acrobot/acrobot_notebook.ipynb) —
  a second classic-control environment.

## Documentation

Configuration keys, the wrapper pipeline, STL specification files, reward
machines, shielding, the CLI reference and per-example walkthroughs are being
consolidated into a browsable documentation site; this README is the current
entry point. In the meantime, each example folder carries the configs and STL
files it is described by, and `rlr show --expand cfg.yml` prints a configuration
with every nested reference resolved.

## Citation

If you use RLRom in academic work, please cite the OVERLAY 2025 paper (also in
[`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{gomez-ruizRLRomMonitoringTraining2025,
  title     = {RLRom: Monitoring and Training Reinforcement Learning Agents using Signal Temporal Logic},
  author    = {G{\'o}mez-Ruiz, Ana Mar{\'i}a and Donz{\'e}, Alexandre and Dang, Thao},
  booktitle = {Short Paper Proceedings of the 7th International Workshop on Artificial Intelligence and Formal Verification, Logic, Automata, and Synthesis (OVERLAY 2025)},
  series    = {CEUR Workshop Proceedings},
  volume    = {4142},
  pages     = {167--172},
  publisher = {CEUR-WS.org},
  year      = {2025},
  url       = {https://ceur-ws.org/Vol-4142/paper18.pdf}
}
```

To reference the software itself, use the version you ran, e.g. `RLRom 0.3.0`
with the repository URL above.

## License

BSD 3-Clause, © Université Grenoble Alpes. See [`LICENSE`](LICENSE).
