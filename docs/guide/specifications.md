# STL specifications

A specification is an `.stl` file: signal declarations, optional parameters, and
formula definitions. RLRom gives the file to the monitor and refers to the
formulas by name from the configuration.

## Anatomy of a `.stl` file

```text
signal push,x,x_dot,theta,theta_dot,env_reward

param epsi = 0.03

pole_up := abs(theta[t]) < epsi
keep_up := alw_[0,100] pole_up
phi_goal_keep_up := ev_[0,400] keep_up
```

- `signal` declares the names the monitor will receive samples for. They must
  match the names RLRom produces — see the next section.
- `param` declares a constant. It can be overridden per formula from the config
  (`formula_params`), which is how one `.stl` file serves several experiments.
- `name := expression` defines a formula. `[t]` is the current sample; temporal
  operators take a time window (`alw_[0,100]`, `ev_[0,400]`).

The full grammar — every operator, its arity and its semantics — is in the
[stlrom grammar reference](https://github.com/decyphir/stlrom/blob/main/docs/stl_grammar.md).
RLRom uses [stlrom](https://github.com/decyphir/stlrom) as its engine; anything
that parses there is available here.

## Which signals the monitor sees

The `signal` line must name what the wrapper samples. Defaults, if you do not
declare anything, are built from `action_names`, `obs_names` and `env_reward`.

```yaml
cfg_specs:
  action_names:
    push: "action"          # the raw action
  obs_names:
    x:     "obs[0]"
    theta: "obs[2]"
  aux_sig_names:
    speed:  "obs[3] * 20"   # any expression you like
  specs: cartpole.stl       # path relative to the config file
```

The right-hand side is a Python expression evaluated every step with `obs`,
`action` (and the reward) in scope. `env_reward` is always sampled, as the raw
environment reward.

!!! note "`specs` vs `stl_specs`"

    Both keys are accepted and mean the same thing: the declarations to load,
    either a path to a `.stl` file or the text itself.

## Monitor formulas vs eval formulas

| | `monitor_formulas` | `eval_formulas` |
|---|---|---|
| Evaluated | every step, online | over the playback of the episode |
| Can feed | observations, rewards, terminal flags, labels, constraints | metrics only |
| Suits | state and bounded-past formulas (`x[t] < 0`, `alw_[0,100] p`) | whole-episode formulas (`ev_[0,400] p`) |

A formula declared in `monitor_formulas` *also* produces a label for the
[reward machines](reward-machines.md), and can be evaluated over the episode
afterwards for metrics by declaring it in `eval_formulas` too.

## Evaluation options

Both sections accept the same per-formula options:

| Option | Meaning |
|---|---|
| `online: true` | evaluate at the current time as the trace grows |
| `t0: <t>` | pin the evaluation to a fixed time (default `0` when not online) |
| `past_horizon: <n>` | shift the evaluation `n` steps into the past |
| `tend: <t>` | end of the window for the evaluation |
| `semantics` | `rob`, `lower_rob`, `upper_rob` or `bool` |
| `lower_bound` / `upper_bound` | clamp the robustness before use |
| `formula` (alias `formula_name`) | the formula to evaluate, when the key is not the formula name |
| `formula_params` | `{<param>: <value>}` overriding `param` declarations for this entry only |
| `eval_all_steps: true` | evaluate at every step during playback (eval formulas) |

Using `formula` lets one formula appear several times with different options —
for example `past_horizon: 0` for its current value and `past_horizon: 1` for its
value one step earlier.

!!! warning "Where the value is read"

    A per-step value is computed when the step happens. A formula evaluated at
    `t0: 0` without `online` reports its value at time 0 for the whole run (that
    is how initial-satisfaction metrics are defined), not its value at the
    current step. Check a formula's curve with `cfg_test.plot` before using it as
    a reward.
