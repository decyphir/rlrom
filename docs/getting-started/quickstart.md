# Quickstart

All commands run from the repository root, so the example paths resolve. If you
installed from PyPI and have no checkout, copy the `examples/` folder from the
[repository](https://gricad-gitlab.univ-grenoble-alpes.fr/donzeal/rlrom).

## 1. Test a policy

`rlr test` runs episodes and monitors the specifications found in the config. A
config that fetches its policy from the Hugging Face Hub needs no training:

```bash
rlr test examples/cartpole/cfg0_hug.yml
```

The run prints a line per episode, then a summary: `mean_ep_len`,
`mean_ep_rew`, `completed` (the fraction of episodes that ran to the horizon
instead of ending early), and for every `eval_formulas` entry either
`sum`/`mean`/`num_sat` for per-step formulas or `ratio_sat` for whole-episode
ones.

## 2. Train with a specification

The CartPole example adds the robustness of one STL formula to the reward with
weight 10, so the agent is paid for moving the cart left (`x_dot < 0` and
`x < -1`, from `cartpole.stl`):

```bash
rlr train examples/cartpole/cfg0tr_ppo_specs.yml
```

Training writes a folder per run:

```text
examples/cartpole/models/ppo_specs_<date>__training0/
├── cfg0.yml              # the resolved configuration of the run
├── model_step_*.zip      # checkpoints saved at every evaluation
└── res_step_*.yml        # metrics measured at every evaluation
```

The last checkpoint is also published as `models/<model_name>.zip`, which is
what `rlr test` loads.

!!! tip "Repeating a run"

    Set `repeat_training: N` under `cfg_train` to run the same training N times
    from scratch, each with its own `__trainingN` folder.

## 3. Plot the results

```bash
# Mean episode reward of the most recent training
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml

# All trainings, with the mean/min-max envelope, saved to a file
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml --trainings all --envelope -o training.png

# One subplot per metric
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml -m mean_ep_rew -m phi_left_goal/ratio_init_sat
```

`--list-metrics` prints the metric names available in the results.

## 4. Pick a checkpoint

Every checkpoint is a candidate; `rlr select-model` picks one by metric and
makes it the active model:

```bash
# Best mean episode reward in the most recent training
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml

# Rank the top 5 over all trainings, without selecting
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --show-top 5 --trainings all

# What is active right now
rlr select-model examples/cartpole/cfg0tr_ppo_specs.yml --show-selected
```

Then `rlr test examples/cartpole/cfg0tr_ppo_specs.yml` evaluates that model.

## 5. Inspect a configuration

Configs are assembled from other files (`.yml` references are replaced by the
file's content, `.stl` references by the file's text). One command shows the
result:

```bash
rlr show examples/highway_env/cfg0tr.yml --expand
rlr show examples/highway_env/cfg0tr.yml --expand --subcfg cfg_specs
```

## Where to go next

- [Concepts](concepts.md) — signals, formulas, robustness, and the wrapper
  pipeline every config describes.
- [STL specifications](../guide/specifications.md) — how a `.stl` file and the
  config's signal names fit together.
- [Examples](../examples/index.md) — one page per environment, with the exact
  commands.
