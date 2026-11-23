# Reward shaping and observations

`cfg_specs.monitor_formulas` is the section that turns monitors into learning
signals. Every entry is evaluated once per step and always produces a label;
extra fields make it do more.

```yaml
cfg_specs:
  monitor_formulas:
    cart_on_left:
      online: true
      past_horizon: 0
      reward_weight: 10      # robustness enters the reward
    danger:
      obs_weight: 1.0        # robustness enters the observation
      actions: {blocked: [1], fallback: 0}   # and constrains the action
```

## Per-formula fields

| Field | Effect |
|---|---|
| `obs_weight: w` | append `w·ρ` to the observation |
| `reward_weight: w` | add `w·ρ` to the reward (alias `weight`) |
| `terminal: true` | end the episode while the formula holds |
| `actions` | constrain the action — see [Shielding](shielding.md) |
| evaluation options | `online`, `t0`, `past_horizon`, `semantics`, `formula`, `formula_params`, … (see [STL specifications](specifications.md)) |

The agent's reward is:

```text
new_reward = env_reward · env_reward_weight  +  Σ_f  w_f · ρ_f(t)  +  Σ_m  RM_m
```

with `env_reward_weight` defaulting to `1.0`, so a formula can *replace* the
environment's reward (`env_reward_weight: 0`) or merely steer it.

!!! tip "Weighting is load-bearing"

    An unsatisfied `ev` formula is a per-step penalty, while a safety formula's
    robustness is capped by how far inside the safe set the state is. A weighted
    mixture of liveness and safety can therefore make suicide — ending the
    episode early — optimal. The MuJoCo
    [jump task](../examples/mujoco.md) is the worked example: `20·ρ(alive) +
    10·ρ(jump)`, not the two terms with equal weight.

## Observations

With `obs_weight`, each configured formula appends one number to the
observation, in config order. The whole wrapper can also add reward-machine
states to the observation (`in_observation: true` on a machine).

!!! warning "Flattening hides names"

    `cfg_specs.flatten_obs` defaults to `true`, which flattens dict observations
    into a single vector — convenient for standard algorithms, but it removes the
    named fields (`rm_states`, …). Set `flatten_obs: false` when you need to read
    them, and remember that a policy trained with one layout cannot be used with
    the other.

## Old sections

`obs_formulas` and `reward_formulas` are deprecated but still accepted. Declaring
the same formula in two sections with a non-zero weight raises `ValueError`
instead of counting it twice — move it to `monitor_formulas` and use
`obs_weight` / `reward_weight`. `label_formulas` is an accepted alias of
`monitor_formulas`; declaring both is an error.

## Terminal conditions

`terminal: true` ends the episode as soon as the formula becomes true. This is the
mechanism for turning a violation into an episode end, and it is usually paired
with a negative reward or with `env_reward_weight: 0` so that the episode's return
reflects the specification rather than the environment's own reward.
