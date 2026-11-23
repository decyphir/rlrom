# Configuration reference

A run is defined by one YAML file. Values ending in `.yml` are replaced by the
content of that file, and values ending in `.stl` by its text — this is how a
configuration is assembled from `cfg_env.yml`, `cfg_specs.yml`,
`cfg_train/…yml` and a `.stl` specification. `rlr show cfg.yml --expand` prints
the result.

!!! warning "The process changes directory"

    Loading a config makes the process `chdir()` into the config file's
    directory, so every relative path in it (model folder, `.stl` paths,
    `import_module`) resolves from there. That is why `rlr test
    examples/cartpole/cfg0_hug.yml` works from the repository root but does not
    write its results where you might expect — they land next to the config.
    The side effect is global: a script that loads a config and then opens a
    relative path will look in the wrong place.

## Top level

| Key | Meaning |
|---|---|
| `env_id` | the id passed to `gym.make`; must be registered. Defaults to `env_name` |
| `env_name` | the name the policy was published under; used for checkpoint lookup. May name a version gymnasium no longer registers |
| `env_kwargs` | arguments forwarded to `gym.make` |
| `model_checkpoint` | Hub filename, when it differs from `<algo>-<env_name>.zip` |
| `import_module` | Python module of the example, for `make_env_*` factories and shield functions |
| `make_env_train` / `make_env_test` | custom env factories; by convention named after `import_module` when omitted |
| `model_name` | checkpoint name, relative to `model_path` |
| `model_path` | folder holding the models (default `./models`) |
| `model_use_specs` | whether the policy was trained with the wrapped observations |
| `cfg_env` | environment section |
| `cfg_specs` | signals, formulas, reward machines, constraints |
| `cfg_train` | training section |
| `cfg_test` | testing section |

`env_id` and `env_name` are separate on purpose: the Hugging Face policies of
MuJoCo are named after Gym-era `-v3` ids while gymnasium registers `-v4`, so the
checkpoint lookup uses `env_name: Hopper-v3` and the environment is built from
`env_id: Hopper-v4`.

## `cfg_env`

| Key | Meaning |
|---|---|
| `env_id` / `env_name` / `env_kwargs` | as above, when the section overrides them |
| `render_mode` | `human`, `rgb_array` or `None`; overrides the one passed to `make` |
| `keyboard_control` | drive the environment by hand (see `cfg0_manual.yml`) |

## `cfg_train`

| Key | Meaning |
|---|---|
| `algo` | either `{<algo name>: <path to algo yml>}` or the algorithm's own settings |
| `n_envs` | number of vectorised environments |
| `total_timesteps` | training budget |
| `eval_freq` | steps between evaluations; each writes a checkpoint and its metrics |
| `eval` | the test configuration used by those evaluations (defaults to `cfg_test`) |
| `repeat_training` | run the whole training N times (default 1) |
| `torch_num_threads` | cap on torch threads — see [Troubleshooting](troubleshooting.md) |
| `progress_bar` | show a progress bar |
| `train_kwargs` | passed to the algorithm's `.learn()` / `.train()` |

The algorithm block holds stable-baselines3 constructor arguments, so any keyword
the algorithm accepts can be set there. `action_noise` is the one that needs
translating:

```yaml
action_noise:
  type: normal          # or ornstein_uhlenbeck
  mean: 0.0
  sigma: 0.1
```

## `cfg_test`

| Key | Meaning |
|---|---|
| `num_ep` | number of episodes |
| `num_steps` | horizon of each episode |
| `init_seed` | first seed; episode *k* uses `init_seed + k` (see the warning below) |
| `render_mode` | `human`, `rgb_array` or `None` |
| `model_file` | explicit checkpoint file, bypassing `model_path`/`model_name` |
| `res_file` | where to write the results |
| `reload_model_every_ep` | reload the policy before each episode |
| `plot` | episode plots — see [Plotting](plotting.md) |

!!! warning "`init_seed`, not `init_seeds`"

    The tester reads `init_seed` (singular). Several example configs still write
    `init_seeds`, which is silently ignored and leaves the default of `0`. When
    you need a specific seed range, use the singular key.

## `cfg_specs`

Documented on its own pages: [STL specifications](specifications.md),
[Reward shaping](reward-shaping.md), [Reward machines](reward-machines.md),
[Shielding](shielding.md). The keys that belong to none of them:

| Key | Meaning |
|---|---|
| `real_time_step` | monitor time step (default 1) |
| `multi_objective` | return a mo-gymnasium reward vector (experimental) |
| `keep_old_reward` | keep the environment reward in the vector reward |
| `debug_signals` / `debug_formulas` / `debug_rm` / `debug_actions` | per-step printing |
| `episode_info_keys` | environment `info` keys to aggregate as `mean_<key>` |

## Overrides on the command line

```bash
rlr test cfg.yml --cfg-specs other_specs.yml --set-params num_ep=50 render_mode=None
```

- `--cfg-specs`, `--cfg-train`, `--cfg-test` replace that section with the
  content of another file (that file is expanded too).
- `--set-params k=v` sets a key that already exists in the config. Values arrive
  as **strings**; only `render_mode: None` is special-cased back to `None`.
- `--verbose N` raises the log level.
