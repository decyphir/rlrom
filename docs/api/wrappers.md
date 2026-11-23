# Wrappers

`SpecWrapper` is the core of the project: it samples signals, drives the monitor,
shapes rewards and observations, steps the reward machines and applies the
constraints. Everything in [the guide](../getting-started/concepts.md) describes
what it does with its configuration.

## Building the stack

::: rlrom.wrappers.spec_wrapper.wrap_env_specs
    options:
      show_root_heading: true

## `SpecWrapper`

::: rlrom.wrappers.spec_wrapper.SpecWrapper
    options:
      members:
        - __init__
        - step
        - reset
        - get_sample
        - eval_formula_cfg
        - eval_specs_episode
        - get_rob
        - get_ep_len
        - get_time
        - get_sig
        - get_signal_from_str
        - get_values_from_str
        - set_episode_data
        - show_reward_machines

## Reward machines

::: rlrom.wrappers.reward_machine.RewardMachine
    options:
      members:
        - __init__
        - reset
        - step
        - state_id
        - state_idx
        - is_final
        - actions_enabled
        - blocked
        - forced
        - fallback
