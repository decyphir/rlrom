# Acrobot and MountainCar

Two smaller classic-control folders. They are the shortest examples of *using the
right algorithm* and *monitoring without custom code*.

## Acrobot

| File | What it is |
|---|---|
| `acrobot.stl` | the specification |
| `cfg0.yml` | no specification, default algorithm |
| `cfg0_hug.yml` | a published policy, fetched |
| `cfg0_specs.yml`, `cfg_specs_acrobot.yml` | monitored runs |
| `cfg0tr_a2c.yml`, `cfg0tr_dqn.yml` | algorithm variants |
| `acrobot_notebook.ipynb` | narrative notebook |

```bash
rlr test examples/acrobot/cfg0_hug.yml
rlr train examples/acrobot/cfg0tr_a2c.yml
```

The two training configs differ only in `cfg_train.algo`, which is the point:
the algorithm is a configuration choice, and `A2C`/`DQN`/`PPO`/`SAC`/`TD3` and
the `sb3-contrib` and SBX variants are all reachable the same way.

## MountainCar

| File | What it is |
|---|---|
| `cfg0tr_ddpg.yml` | DDPG on `MountainCarContinuous-v0` |
| `cfg0tr_sac.yml` | SAC on the same plant |
| `cfg0tr_td3.yml` | TD3 on the same plant |

```bash
rlr train examples/mountaincar/cfg0tr_sac.yml
rlr plot  examples/mountaincar/cfg0tr_sac.yml --trainings all --envelope
```

Three algorithms, one plant, no specifications: a good place to compare
algorithm behaviour on a cheap task before adding monitors to the mix, and to see
what `--envelope` over repeated trainings does to a noisy curve.
