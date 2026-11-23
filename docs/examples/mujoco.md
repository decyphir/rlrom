# MuJoCo

Locomotion, with no custom environment module at all: the optional extra supplies
the environments and the configs use plain `gym.make`. This example is about
treating a specification as the task.

Requires the extra: `pip install "rlrom[mujoco]"`.

## Files

| File | What it is |
|---|---|
| `cfg0_min.yml` | random policy — a smoke test with no model |
| `cfg0_hug.yml` | a published TD3 policy fetched from the Hugging Face Hub |
| `cfg0_hug_specs.yml`, `hopper.stl` | the same policy, monitored (alive, upright, forward progress, torque saturation) |
| `cfg0tr_td3.yml`, `td3_hopper.yml` | TD3 with the environment's own reward — the baseline |
| `cfg0tr_td3_backward.yml` | the same plant, walking backwards |
| `cfg0tr_td3_jump.yml`, `hopper_jump.stl` | jumping in place, where the specification *is* the task |

## Fetch a policy and monitor it

```bash
rlr test examples/mujoco/cfg0_hug.yml
rlr test examples/mujoco/cfg0_hug_specs.yml
```

The Hub policies are named after Gym-era ids, so these configs demonstrate the
`env_id` / `env_name` split:

```yaml
env_name: Hopper-v3     # what the checkpoint is called
env_id: Hopper-v4       # what gymnasium registers
```

The observation and action spaces are identical between the two, so the policy
transfers.

## Three tasks on one plant

| Config | What changes |
|---|---|
| `cfg0tr_td3.yml` | the environment's own reward |
| `cfg0tr_td3_backward.yml` | `env_kwargs: {forward_reward_weight: -1.0}` and nothing else |
| `cfg0tr_td3_jump.yml` | the environment's reward is switched off and replaced by two formulas |

The jump task has no reward knob to turn — nothing in the environment pays for
height, and `forward_reward_weight` can be zeroed or negated but there is no
height term — so the specification is the objective. The environment reward is
switched off and three formulas carry the whole reward:

```yaml
cfg_specs:
  env_reward_weight: 0
  monitor_formulas:
    staying:                       # above standing height, not travelling
      semantics: bool
      past_horizon: 10
      reward_weight: .2
    alive:
      semantics: bool
      reward_weight: .1
    jumping:
      semantics: bool
      reward_weight: .7
```

Jumping means `z > 1.5` **and** `|v_x| < 0.2` within a short window: the baseline
policy already spends half its time above 1.5 while travelling at 2.7 m/s, so
height alone would not describe a different task. The weights are the
experiment — the header of `cfg0tr_td3_jump.yml` records that an earlier,
unnormalised 20/10 split worked while 5/10 learned to fall over and end the
episode in six steps, because an unmet liveness formula is a per-step penalty and
ending the episode is one way to stop paying it.

Compare these runs on task metrics — mean `z`, mean `|v_x|`, time spent above the
jump height without travelling — not on return: the rewards are in robustness
units and are not comparable with the baseline's.
