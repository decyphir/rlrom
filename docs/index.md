# RLRom

Test and train reinforcement learning agents with **robust online monitoring of
Signal Temporal Logic (STL) specifications**.

RLRom wraps a [Gymnasium](https://gymnasium.farama.org/) environment with a
monitor. You describe what the agent should and should not do as STL formulas
over signals you name yourself — observations, actions, or any feature you can
compute from the environment state. From the robustness values the monitor
produces at every step, RLRom can:

<div class="grid cards" markdown>

-   **Test**

    ---

    Run a trained policy, monitor the specifications over whole episodes, and
    report satisfaction ratios and robustness statistics.

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   **Train**

    ---

    Put robustness into the observation and/or the reward (reward shaping), or
    drive a reward machine whose transitions are labelled by formulas.

    [:octicons-arrow-right-24: Reward shaping](guide/reward-shaping.md)

-   **Constrain**

    ---

    Force or block actions from monitor formulas and reward-machine states, up
    to a custom shield for continuous action spaces.

    [:octicons-arrow-right-24: Shielding](guide/shielding.md)

</div>

Specifications and experiments live in YAML configuration files, so a new
experiment is a config change rather than a code change.

## Install

```bash
pip install rlrom
```

Python 3.12 or newer. The STL engine (`stlrom`) is distributed as source and is
compiled during the install, so a C++ toolchain is required — see
[Install](getting-started/install.md) for the details and the optional extras.

## Quickstart

```bash
# Monitor a policy downloaded from the Hugging Face Hub
rlr test examples/cartpole/cfg0_hug.yml

# Train CartPole with an STL reward
rlr train examples/cartpole/cfg0tr_ppo_specs.yml

# Plot the training curves
rlr plot examples/cartpole/cfg0tr_ppo_specs.yml
```

More in [Quickstart](getting-started/quickstart.md).

## Where things are

| | |
|---|---|
| Source | [gricad-gitlab.univ-grenoble-alpes.fr/donzeal/rlrom](https://gricad-gitlab.univ-grenoble-alpes.fr/donzeal/rlrom) (mirror: [github.com/decyphir/rlrom](https://github.com/decyphir/rlrom)) |
| Package | [pypi.org/project/rlrom](https://pypi.org/project/rlrom/) |
| Issues | [GitLab issues](https://gricad-gitlab.univ-grenoble-alpes.fr/donzeal/rlrom/-/issues) |
| Citation | [OVERLAY 2025 paper](about/citation.md) |
