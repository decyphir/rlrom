# Configuration and utilities

`rlrom.cfg_utils` is deliberately light: standard library, numpy and
`ruamel.yaml` only. That is why `rlr show` and `rlr plot` work without the
training stack installed.

::: rlrom.cfg_utils.load_cfg

::: rlrom.cfg_utils.find_subcfg

::: rlrom.cfg_utils.set_rec_cfg_field

## Environments

`rlrom.utils.make_env_from_cfg` builds the environment from the configuration:
it separates `env_id` from `env_name`, forwards `env_kwargs`, applies
`render_mode`, and reports the family's registered ids when an id does not
exist. `custom_env_factory` finds an example's `make_env_*` function through
`import_module`.

## Algorithms

The `cfg_train.algo` key selects from `rlrom.utils.ALGO_NAMES_CLASSES`, which
maps names to stable-baselines3, `sb3-contrib`, SBX and `morl-baselines` classes,
plus `qlearning` for discrete environments. Anything the chosen class accepts as
a constructor keyword can be written in the algorithm block.

## Results as dataframes

Training results are also available as polars dataframes, which is what the
plotting code consumes: `get_training_folders`, `get_training_res`,
`get_df_all_trainings`, `get_metric_series`, `get_selection_df` in
`rlrom.utils`.
