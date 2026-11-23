# Reward machines

A reward machine is a finite-state machine whose transitions fire when an STL
formula holds. It gives structure that a flat weighted sum of robustnesses
cannot: the reward can depend on *where the agent is in the task*, not only on
how well it satisfies each formula right now.

A machine is configured in its own file and referenced from `cfg_specs`:

```yaml
# cfg_specs of the run
reward_machines:
  - rm_stuck_left.yml
  - rm_AB.yml
```

```yaml
# rm_AB.yml
name: rm_AB
weight: 0.75
reward_machine:
  in_observation: True
  states:
    - id: u_go_left
      initial: True
    - id: u_stay_A
      reward: 1
    - id: u_fail
      final: True
  transitions:
    - from: u_go_left
      to: u_stay_A
      condition: cart_in_A
      priority: 1
      reward: 1
    - from: u_stay_A
      to: u_fail
      condition: not_A
      priority: 1
      reward: 0
```

## Conditions are labels

A transition's `condition` is the name of a formula declared in
`cfg_specs.monitor_formulas`. The label is true at step *t* when that formula's
monitored lower robustness bound is positive at *t*, so every formula you use
here must also be declared there — with `online: true` and the `past_horizon`
you mean. The example above uses `past_horizon: 0` for instantaneous conditions
and `past_horizon: 50` for "has been true for a while" ones.

## Reference

| Key (inside `reward_machine`) | Meaning |
|---|---|
| `states[].id` | state name, used by transitions |
| `states[].initial` | exactly one state must be initial |
| `states[].final` | reaching it ends the episode (`terminated=True`) |
| `states[].reward` | reward issued per step while in the state (default 0) |
| `states[].actions` | action constraints — see [Shielding](shielding.md) |
| `transitions[].from` / `.to` | required |
| `transitions[].condition` | required; a label name |
| `transitions[].reward` | reward issued when the transition fires (default 0) |
| `transitions[].priority` | highest enabled priority wins (default 0); two enabled transitions sharing the maximum raise `ValueError` |
| `transitions[].id` | defaults to `{from}__to__{to}__if__{condition}` |

On the wrapper side (the keys next to `reward_machine:`):

| Key | Meaning |
|---|---|
| `name` | machine name, used in metrics (`rm_states/<name>`, `rm_reward/<name>`) |
| `weight` | multiplier applied to the machine's per-step reward before it is added to the agent's reward |
| `in_observation` | append the machine's current state to the observation |

In single-objective mode the machines' weighted rewards are **summed** into the
agent's reward, together with the environment reward and the shaped formula
terms. Several machines can run at once, each on its own labels.

## Inspecting a machine

The wrapper can print the FSM it parsed, with the labels currently true:

```python
env.get_wrapper_attr('show_reward_machines')()
```

and `cfg_specs.debug_rm: true` prints each step's labels and state transition.

Per-step state indices are recorded in `episode['rm_states']` and can be plotted
as `rm_states/<machine>` in `cfg_test.plot`.

## State actions are computed, not declared

`RewardMachine` is a plain FSM with no dependency on gym or STL: it parses the
config, tracks state, and on each step of labels takes the highest-priority
enabled transition or stays put, issuing the state's reward. Action constraints
attached to states are exposed through its accessors and applied by the wrapper,
one step later, exactly like formula-derived constraints — see
[Shielding](shielding.md).
