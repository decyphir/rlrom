# Shielding

A **shield** replaces the agent's action before the environment sees it.
RLRom offers two mechanisms: scalar `forced`/`blocked` constraints declared in
the configuration, and a Python callback for continuous action spaces.

## Scalar constraints

The same `actions` dictionary can be attached to a monitor formula and to a
reward-machine state:

```yaml
cfg_specs:
  monitor_formulas:
    danger:
      past_horizon: 2
      semantics: bool
      actions:
        priority: 1
        enable: True
        blocked: [1, 3]     # action indices never allowed here
        fallback: 4         # used when the agent picked a blocked action
    car_left:
      semantics: bool
      actions:
        blocked: [0]
        fallback: 1
```

| Key | Meaning |
|---|---|
| `enable` | `false` switches this constraint off without removing it (default `true`) |
| `blocked` | list of raw action values that may not be taken |
| `forced` | action value that must be taken |
| `fallback` | replacement action when the agent's choice is blocked |
| `priority` | tie-breaker between constraints (default 0) |

A formula-derived constraint takes effect from the **next** step: the formula's
truth is only known once the step has been taken. A reward-machine state's
constraints are active while the machine is in that state, with the machine-level
`action_priority`.

### Resolution

1. `forced` wins.
2. `blocked` is cumulative across all active sources.
3. A blocked action is replaced by the previous action if that one is not
   blocked, else by the highest-priority blocker's `fallback`.
4. Priority breaks conflicts only. Sources that agree never conflict.

Two enabled sources that force different actions at the same maximum priority
raise `ValueError`, as does a blocker tying a forcer on the action it blocks.
Equal-priority blockers must agree on `fallback` (identical, or one `None`
deferring to the other).

The override is reported in `info['shield']` (`machine`, `reason`, `original`,
`shielded`) and printed with `cfg_specs.debug_actions: true`.

## Turning constraints off

```yaml
cfg_specs:
  disable_actions: True
```

Every constraint — including the callback below — is bypassed and the agent's
action always passes through. This is how shielded and unshielded results are
compared on the *same* checkpoint.

## Continuous actions

`blocked`/`forced` are scalar: they make sense for discrete action sets. For a
`Box` action space, name a Python function instead:

```yaml
cfg_specs:
  action_shield:
    activation: shield_activate   # optional monitor formula
    function: metadrive_shield    # attribute of cfg_specs.import_module
```

```python
def metadrive_shield(spec_wrapper, action):
    """Called with the action about to be sent; return a replacement or None."""
    ...
```

- It runs **after** the scalar logic, so it sees the action that will really be
  executed.
- With `activation`, the shield acts while that formula was true at the previous
  step (the same one-step delay as formula-derived constraints); without it, the
  shield acts on every step.
- Every other key under `action_shield` is handed to the function through
  `spec_wrapper.action_shield['cfg']`.
- The override is reported in `info['shield']` with
  `machine: <action_shield name>` and recorded per step in
  `episode['shield_active']`, which `rlr test` reports as a mean.

The function lives in the example's module, not in YAML — there is no table DSL.
Note that the STL formulas keep their own `param` declarations, so a shield's
thresholds and the specification's thresholds must be kept in sync by hand.

!!! warning "A shield is not a proof"

    The constraint acts on the action; whether that keeps the state safe is the
    shield's design problem, not RLRom's. A passive shield applied at test time
    changes the closed loop — measure the result with `rlr test` on many seeds,
    and see the [MetaDrive example](../examples/metadrive.md) for a case where
    two plausible shield laws gave opposite outcomes.

The worked configuration is `examples/highway_env/cfg_specs.yml`: three
reward-shaped formulas, `danger` with an observation term plus a constraint, and
four lane/neighbour-conditioned constraints. See the
[Highway-env example](../examples/highway-env.md).
