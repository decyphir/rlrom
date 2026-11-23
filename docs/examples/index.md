# Examples

Each example is a self-contained folder under `examples/`: configurations, the
`.stl` specification, sometimes a Python module for a custom environment or a
shield. Commands run from the repository root.

| Example | Shows | Extra | Model needed |
|---|---|---|---|
| [CartPole](cartpole.md) | STL reward shaping, reward machines, the minimal config | — | fetched or trained |
| [Highway-env](highway-env.md) | reward shaping + observation term + action constraints | `highway-env` | trained |
| [MiniGrid](minigrid.md) | reward machines on discrete tasks, tabular Q-learning | — | trained |
| [MuJoCo](mujoco.md) | specifications as the task, alternative tasks on one plant | `mujoco` | fetched or trained |
| [MetaDrive](metadrive.md) | a continuous-action shield on a driving task | `metadrive` | trained |
| [Acrobot and MountainCar](classic-control.md) | algorithm coverage on classic control | — | fetched or trained |

Two habits apply to all of them:

- Model binaries are **not tracked in git**. Where a config expects one, either
  train it or use the fetched-policy variant.
- `rlr show cfg.yml --expand` is the fastest way to see what a config becomes
  after its includes are resolved.
