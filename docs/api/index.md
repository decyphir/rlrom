# API overview

The command line covers the usual workflow. This section is for using RLRom from
Python, which is how the notebooks work.

| Module | What lives there |
|---|---|
| `rlrom.wrappers.spec_wrapper` | `SpecWrapper` and `wrap_env_specs` — the whole monitoring layer |
| `rlrom.wrappers.reward_machine` | `RewardMachine`, the pure FSM |
| `rlrom.trainers` | `RLTrainer` — environment construction, algorithm instantiation, `learn()`, checkpoints |
| `rlrom.testers` | `RLTester` — episodes, metrics, figures |
| `rlrom.cfg_utils` | config loading and lookup, with no heavy imports |
| `rlrom.utils` | algorithm registry, environment factories, result dataframes |
| `rlrom.plots` | training curves, live plots, signal layout |

The minimal programmatic shape is:

```python
from rlrom.cfg_utils import load_cfg
from rlrom.trainers import RLTrainer
from rlrom.testers import RLTester

cfg = load_cfg('examples/cartpole/cfg0tr_ppo_specs.yml')  # chdir's to the cfg's folder
trained = RLTrainer(cfg).train()
results = RLTester(cfg).run_cfg_test()
```

For a single episode without the trainer:

```python
from rlrom.wrappers.spec_wrapper import wrap_env_specs
import gymnasium as gym

env = wrap_env_specs(gym.make('CartPole-v1'), cfg)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# the wrapper records the trace as it goes: signals, labels, formula values
stl_data = env.get_wrapper_attr('stl_driver').data   # the recorded signals
```

!!! note "How these pages are generated"

    The signature blocks below are extracted from the source by `mkdocstrings`
    in static mode — the documentation build does not import `rlrom`, so it does
    not need torch or any simulator installed. Docstrings are the contract;
    where a docstring is thin, the guide pages are the reference.
