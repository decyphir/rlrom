# CartPole

The reference example: small enough to train in a minute, and it covers every
feature once.

## Files

| File | What it is |
|---|---|
| `cartpole.stl` | the specifications: pole upright, cart left/right goals, region predicates |
| `cfg0_min.yml` | random policy, no model, no specification — the smoke test |
| `cfg0_hug.yml` | a policy downloaded from the Hugging Face Hub, monitored |
| `cfg0tr_ppo_specs.yml` | PPO with one `reward_weight` formula — the simplest training with a specification |
| `cfg0tr_ppo.yml` | the same plant, plain environment reward |
| `cfg0tr_rm_AB.yml`, `rm_AB.yml` | reward-machine setup |
| `cfg0tr_rm_stay_left_right.yml` | two-figure plotting example |
| `cartpole_ppo.yml` | PPO hyper-parameters for this plant |

## Test a monitored policy

```bash
rlr test examples/cartpole/cfg0_hug.yml
```

No training and no local model: the policy is fetched from the Hub on first use.

## Train with a reward formula

```bash
rlr train examples/cartpole/cfg0tr_ppo_specs.yml
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml --trainings all --envelope
```

In that config the environment reward is kept and one formula is added:

```yaml
cfg_specs:
  monitor_formulas:
    cart_on_left:          # x < -1, from cartpole.stl
      online: True
      past_horizon: 0
      reward_weight: 10
```

so the agent is paid both for surviving and for moving the cart left. Comparing
`cfg0tr_ppo.yml` (no specification) with `cfg0tr_ppo_specs.yml` (one formula) is
the shortest illustration of what reward shaping changes.

## Reward machines

```bash
rlr train examples/cartpole/cfg0tr_rm_AB.yml
rlr test  examples/cartpole/cfg0tr_rm_AB.yml
```

`cfg_specs` declares the eight label formulas and references the machine:

```yaml
  reward_machines:
    - rm_AB.yml
```

`rm_AB.yml` is the two-goal machine — go left, stay in A, go right, stay in B,
with `u_fail` as a final state — and it also sets `env_reward_weight: 0.25`, so
most of the return comes from the machines. See [Reward machines](../guide/reward-machines.md)
for the config reference and [Shielding](../guide/shielding.md) for the
constraint mechanism the same states can carry.
