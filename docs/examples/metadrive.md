# MetaDrive

A driving case study, and the only example of a **continuous-action shield**. It
is also the most machine-hungry: the simulator plus its assets.

Requires the extra: `pip install "rlrom[metadrive]"`.

!!! warning "Heavier setup"

    - The simulator resolves from **git**, not PyPI: the 0.4.3 release pins
      `panda3d==1.10.13` and `requires-python <3.12`, while `main` accepts newer
      Python. `pyproject.toml` pins the commit.
    - ~170 MB of assets are a separate download, pulled into the installed
      package on first use (`python -m metadrive.pull_asset`).
    - **One engine per process.** Training with in-process evaluation must use
      `n_envs > 1` so environments run in subprocesses; `n_envs: 1` with
      `cfg_train.eval` set fails when the tester asks for its own engine.

## Files

| File | What it is |
|---|---|
| `metadrive_env.py` | `MetaDriveRLEnv` (gymnasium/seed-compatible wrapper), the shield function, the env factories |
| `metadrive.stl` | lane signals and the region formulas `phi_center`, `phi_lane`, `phi_notcenter`, `phi_danger`, plus activation formulas |
| `cfg_env.yml` | 100 scenarios on the SCS map (straight–curve–straight), static objects, no traffic |
| `cfg_env_clean.yml` | the same road with no static objects at all |
| `cfg_specs.yml` | regions, activation, `action_shield`, `episode_info_keys` |
| `metadrive_ppo.yml`, `cfg0tr_ppo.yml` | PPO baseline on the road |
| `cfg0tr_ppo_clean.yml` | the same on the empty road |
| `cfg0tr_ppo_shield.yml` | the baseline plus the specifications and the shield |
| `cfg_smoke_ppo_train.yml`, `cfg_smoke_ppo_test.yml` | ~15 s overrides for pipeline checks |
| `learn_shield.py` | label safe/unsafe actions from baseline traces into `shield_table.json` |

## Running it

A smoke test first — it needs no trained model:

```bash
rlr train examples/metadrive/cfg0tr_ppo.yml \
  --cfg-train cfg_smoke_ppo_train.yml --cfg-test cfg_smoke_ppo_test.yml
```

Then the real thing:

```bash
rlr train examples/metadrive/cfg0tr_ppo.yml        # vanilla PPO baseline
rlr train examples/metadrive/cfg0tr_ppo_shield.yml  # baseline + shield
rlr test  examples/metadrive/cfg0tr_ppo_shield.yml  # evaluate with the shield
rlr test  examples/metadrive/cfg0tr_ppo.yml         # evaluate without it
```

Testing a checkpoint that already exists, with the shield applied passively:

```bash
rlr test examples/metadrive/cfg0tr_ppo_shield.yml --set-params model_name=<checkpoint>
```

`rlr test` reports the case study's own outcome metrics through
`cfg_specs.episode_info_keys`:

```yaml
  episode_info_keys: [arrive_dest, route_completion, crash, out_of_road]
```

which is what makes `mean_arrive_dest`, `mean_route_completion`,
`mean_crash`, `mean_out_of_road` and `mean_shield_active` appear in the summary
next to the formula metrics.

## The shield

The action is a 2-D `Box` (steering, acceleration), so this example uses the
[`action_shield` hook](../guide/shielding.md) rather than scalar constraints:

```yaml
cfg_specs:
  action_shield:
    activation: shield_activate_position   # act unless centred and aligned
    function: metadrive_shield
```

`metadrive_shield` is a feedback law: a lane-keeping term on the lateral
deviation and heading error, plus a speed target that falls as the car drifts.
Its thresholds are read from the STL `param` values so that spec and shield can
be kept in agreement.

Two findings from building it, both worth knowing before writing your own:

- **Activation decides everything.** A formula that reads "not centred" over a
  window is satisfied almost always for a fast policy, so it shields almost every
  step; positional activations (`shield_activate_position`,
  `shield_activate_drift`) were the ones that behaved.
- **A constant action is not a controller.** `learn_shield.py` learns a
  region → action table from recorded traces. As a piecewise-constant law it has
  no error feedback, so the car drifts to the edge of the lane and only then
  changes cell — worse than the hand-written feedback law in the case study's
  measurements. The learned path stays opt-in (`action_shield.use_table`).

This folder is a research case study: it carries the paper's reproduction
material, and its numbers live with the runs, not in this documentation.
