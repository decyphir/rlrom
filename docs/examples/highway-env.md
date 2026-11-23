# Highway-env

The most complete specification setup in the repository: three reward-shaped
formulas, one observation term, and four action constraints, on a three-lane
highway with traffic.

Requires the extra: `pip install "rlrom[highway-env]"`.

## Files

| File | What it is |
|---|---|
| `cfg0tr.yml` | the run: train + test, points at the three sections below |
| `cfg_env.yml` | the environment (gym id, lanes, traffic) |
| `cfg_specs.yml` | signals, formulas, constraints |
| `hw-env_specs.stl` | the formulas |
| `highway.py` | custom `make_env_*` factory used by the config |
| `cfg0_manual.yml`, `cfg_env_manual.yml` | drive the environment by keyboard |
| `highway_notebook.ipynb` | the same material as a narrative notebook |

## Run it

```bash
rlr train examples/highway_env/cfg0tr.yml
rlr test  examples/highway_env/cfg0tr.yml
```

The example expects `models/right_lane.zip`, which is not tracked — train the
config first, or point `model_name` at a checkpoint you have.

## What the specification does

Signals are read out of a dict observation, with unit conversions done in the
expressions themselves:

```yaml
obs_names:
  ego_presence: "obs[0][0]"
  ego_vx: "80*obs[0][3]"
```

then:

| Formula | Role in the config |
|---|---|
| `ego_right_lane` | `reward_weight: .25` |
| `ego_fast` | `reward_weight: .5` |
| `ego_moves` | `reward_weight: .25` |
| `danger` | `obs_weight: 1.0` **and** blocks `{1, 3}` (idle, faster), fallback `4` (slower), `priority: 1` |
| `danger_and_right_lane` | blocks lane-right, `priority: 1` |
| `danger_and_left_lane` | blocks lane-left, `priority: 1` |
| `car_left` | blocks lane-left, fallback idle |
| `car_right` | blocks lane-right, fallback idle |

The last two illustrate the intended pattern for contextual constraints: a lower
priority than the danger rules, so that when both apply the danger rule's
`fallback` is what gets executed. The resolution rules are in
[Shielding](../guide/shielding.md).

`cfg_specs.yml` also ends with the two switches worth knowing when comparing
runs: `debug_actions` and `disable_actions`.
