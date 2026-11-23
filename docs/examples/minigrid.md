# MiniGrid

Discrete grid worlds, a custom environment, and a reward machine on a visible
key.

Both configs are current: the machine is declared under
`cfg_specs.reward_machines` and its labels are declared in `monitor_formulas`, so
it actually runs.

## Files

| File | What it is |
|---|---|
| `cfg0tr_ppo.yml` | PPO on the custom unlock environment, no specification |
| `cfg0tr_ppo_specs.yml` | the same with the specification, the label formulas and the machine |
| `cfg_rm.yml` | the reward machine: find the key, then reach the door |
| `minigrid.stl` | the formulas: `has_key`, `not_has_key`, `door_in_front`, `key_is_visible`, `drop_key` |
| `unlock_custom.py` | the custom environment, registered through `import_module` |
| `minigrid_notebook.ipynb`, `plot_*.ipynb` | narrative notebook and training plots |

## Running it

```bash
rlr train examples/minigrid/cfg0tr_ppo_specs.yml
rlr test  examples/minigrid/cfg0tr_ppo_specs.yml
```

The environment is not a stock gym id: `unlock_custom.py` registers
`MiniGrid-CustomUnlock-v1` and the config selects it through `import_module` +
`env_name`. That mechanism — a module next to the config that provides the
custom `make_env_*` and the environment class — is the pattern to copy for any
environment gymnasium does not register.

## What it is useful for

- **Labels from a grid observation.** `obs_names` reads two cells of the 7×7×3
  observation and derives a third signal with an inline expression:

    ```yaml
    obs_names:
      ego: "obs[81]"
      obj_front: "obs[78]"
      key_visible: "1 if any(i % 3 == 0 for i, val in enumerate(obs) if val == 5) else 0"
    ```

- **Whole-episode formulas.** `find_key` and `find_key_door` are evaluated in
  `eval_formulas` with `eval_all_steps: false`, i.e. once over the episode, which
  is the right shape for "did it eventually do X".
- **Sequenced reward.** `cfg_rm.yml` is a three-state machine: get the key
  (`u0 → u1`), then face the door (`u1 → u2`, final, reward 1), and fall back to
  `u0` if the key is dropped. The reward depends on the stage reached, not on a
  weighted sum of formula robustnesses — that is what a reward machine buys.

The wiring is worth reading once, because both halves are required:

```yaml
cfg_specs:
  monitor_formulas:          # the labels the machine's transitions test
    has_key:          {online: True, past_horizon: 0}
    not_has_key:      {online: True, past_horizon: 0}
    door_in_front:    {online: True, past_horizon: 0}
  reward_machines:
    - name: unlock
      weight: 1.0
      in_observation: True
      reward_machine: cfg_rm.yml
```

A machine whose conditions are not declared as monitor formulas never leaves its
initial state, and nothing complains: the labels simply are never true.
