# Training and testing

`RLTrainer` and `RLTester` are what `rlr train` and `rlr test` call. Both take the
resolved configuration dict and are usable directly from Python.

::: rlrom.trainers.RLTrainer
    options:
      members:
        - __init__
        - train
        - make_env_train
        - make_vec_envs

::: rlrom.testers.RLTester
    options:
      members:
        - __init__
        - run_cfg_test
        - run_episode
        - eval_episode
        - print_res_all_ep
        - get_fig

## Results

Both classes accumulate results in two layers:

| Layer | Contents |
|---|---|
| `res` | per-episode arrays: lengths, returns, per-step formula values |
| `res_all_ep` | aggregates: `basics` (`mean_ep_len`, `mean_ep_rew`, `completed`), `reward_formulas`, `eval_formulas` |

`rlr plot` and `rlr select-model` read the same data from the
`res_step_*.yml` files that training writes at every evaluation.
