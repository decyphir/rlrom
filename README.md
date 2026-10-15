# RLRom

This module integrates Robust Online Monitoring methods with Reinforcement Learning stuff. The motivation is first to test RL agents using interpretable monitors, then use these monitors to train models to perform complex tasks, and/or converge toward behaviors that reliably satisfy certain requirements. 

## Install

Make sure build tools are installed, e.g., with `apt`:
```
$ sudo apt install build-essential
```
Then install with pip: 
```
pip install rlrom 
``` 
## Getting Started

### Command Line Interface

RLRom reads configuration files in the YAML format as inputs. Examples are provided in the examples folder. A command line interface is provided through command `rlr` which can be called with various arguments. For instance, the `rlr test` command reads a configuration file and runs tests:
```bash
$ rlr test examples/cartpole/cfg0_hug.yml
```
will run a few episode of the cartpole classic environment, fetching a model on huggingface and monitor a formula on these episodes. 
`rlr show cfg.yml` prints a configuration (optionally expanded with `--expand` or restricted with `--subcfg KEY`); `rlr --help` lists all options.

For training with or without STL specifications, use the `rlr train` command, e.g.:

```bash
$ rlr train examples/cartpole/cfg0tr_ppo_specs.yml
```

To repeat the same training several times (each run starts from scratch and gets its own `<model>_<date>__trainingN` checkpoint folder), set `repeat_training: N` under `cfg_train` (defaults to 1):

```yaml
cfg_train:
  repeat_training: 5
  # ...other training options
```

Training progress can be plotted from the checkpoint result files (``res_step_*.yml``) produced by `rlr train`. A typical workflow is to train several times (e.g. with `repeat_training: 5` in `cfg_train`) and then plot the envelope:

```bash
# Last training, mean episode reward
$ rlr plot examples/cartpole/cfg0tr_ppo_specs.yml

# All trainings as lines, plus the mean/min-max envelope, saved to a PNG
$ rlr plot examples/cartpole/cfg0tr_ppo_specs.yml --trainings all --envelope -o training.png

# Several metrics (formula/metric pairs supported), one subplot each
$ rlr plot examples/cartpole/cfg0tr_ppo_specs.yml -m mean_ep_rew -m phi_left_goal/ratio_init_sat
```

Useful `plot` options: `--trainings last|all|0,2,-1|-5:-1` (negative indices count from the end, ranges are inclusive; training folders are listed by date and number, `--verbose 1` shows them), `-m METRIC` (repeatable), `--envelope`, `-o FILE` (png/pdf/svg), `--title`, `--log-y` and `--list-metrics` (discovers the available metric names). A standalone `rlrom_plot cfg.yml` command with the same options is also provided.

### Selecting a model

A trained model is "active" through a symlink: `rlr test` loads `<model_path>/<model_name>.zip` (plus `.yml`) from the configuration. `rlr select-model` selects a checkpoint from the training results and (re)creates these symlinks, e.g.:

```bash
# Best mean episode reward in the last training (default)
$ rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml

# Best formula satisfaction ratio across all trainings
$ rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml -m phi_left_goal/ratio_init_sat --trainings all

# First checkpoint after 100k steps / worst (min)
$ rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --step 100000
$ rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --min

# Inspect without selecting: top-5 ranking across all trainings, or the currently selected model
$ rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --show-top 5 --trainings all
$ rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --show-selected
```

Options: `--trainings last|all|0,2,-1|-5:-1` (pool; negative indices count from the end, ranges are inclusive; default last), `-m METRIC` (default `mean_ep_rew`), `--min` (select the minimum), `--step N` (first checkpoint after N steps, overrides the metric), `--show-top N` (ranked table, default 5, no selection), `--show-selected` (show the active model: symbolic link, target file and metric value), `--list-metrics`. Ties on the best value are broken by the earliest step. A standalone `rlrom_select_model cfg.yml` command is also provided. Typical workflow: `rlr train` (with `repeat_training: N`) → `rlr select-model` → `rlr test`.

More details are provided in the notebooks (see below.)

### Notebook Examples

More programmatic features are demonstrated in notebooks, in particular

- [The Cart Pole notebook](examples/cartpole/cartpole_notebook.ipynb) presents a case study around the classic Cart Pole Gymnasium environment.

- [The Highway-env notebook](examples/highway_env/highway_notebook.ipynb) presents a slightly more involved autonomous driving example based on the [highway-env](https://github.com/Farama-Foundation/HighwayEnv) environment.