# MiniGrid

Discrete grid worlds, custom environments, and a reward-machine task family.
This example is the least current of the set — read the status note before
relying on it.

!!! warning "Partly stale"

    The MiniGrid configs in this folder still wire their reward machine through
    `cfg_specs.cfg_rm`, which the current wrapper no longer reads: machines are
    declared under `cfg_specs.reward_machines` (see
    [Reward machines](../guide/reward-machines.md)). The specifications and the
    custom environment parts still apply; the machine does not. The maintenance
    entry point `blocked_unlock_pickup/main.py` also imports `bokeh`, which is no
    longer a dependency of `rlrom`.

## Files

| File | What it is |
|---|---|
| `cfg0tr_ppo.yml` | PPO on the custom unlock environment |
| `cfg0tr_ppo_specs.yml` | the same with specifications and the (legacy) `cfg_rm` reference |
| `cfg_rm.yml`, `minigrid.stl` | machine and label definitions |
| `unlock_custom.py` | the custom environment, registered through `import_module` |
| `blocked_unlock_pickup/` | the task family: `task3.stl`, `cfg_rm_task1/2/3.yml`, `env_task2.py`, `env_task3.py`, `main.py` |
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
- **Sequenced tasks.** The `blocked_unlock_pickup` machines encode stages
  (find key → open door → pick up → deliver) with negative rewards for reverting
  — the task-3 machine has `reward: -1` states and transitions. That is the
  reward-machine concept without a weighted-sum approximation.

To use the machines here today, move the machine file under `reward_machines`:

```yaml
cfg_specs:
  reward_machines:
    - name: unlock
      weight: 1.0
      in_observation: True
      reward_machine:
        # content of cfg_rm.yml
```
