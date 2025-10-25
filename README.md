# RLRom

This module integrates Robust Online Monitoring methods with Reinforcement Learning stuff. The motivation is first to test RL agents using interpretable monitors, then use these monitors to train models to perform complex tasks, and/or converge toward behaviors that reliably satisfy certain requirements. 

## Install

Make sure build tools are installed, e.g., with `apt`:
```
$ sudo apt install build-essential
```
Then install with pip: 
```
pip install rlrom 
``` 
## Getting Started

### Command Line Interface
RLRom reads configuration files in the YAML format as inputs. Examples are provided in the examples folder. A command line interface is provided through command `rlr` which can be called with various arguments. For instance, the `rlr test` command reads a configuration file and runs tests:
```bash
$ rlr test examples/cartpole/cfg0_hug.cfg
```
will run a few episode of the cartpole classic environment, fetching a model on huggingface and monitor a formula on these episodes. 

For training with or without STL specifications, use the `rlr train` command, e.g.:

```bash
$ rlr train examples/cartpole/cfg0tr_ppo_specs.cfg
```
More details are provided in the notebooks (see below.)

### Notebook Examples

More programmatic features are demonstrated in notebooks, in particular

- [The Cart Pole notebook](examples/cartpole/cartpole_notebook.ipynb) presents a case study around the classic Cart Pole Gymnasium environment.

- [The Highway-env notebook](examples/highway-env/highway_env_notebook.ipynb) presents a slightly more involved autonomous driving example based on the [highway-env](https://github.com/Farama-Foundation/HighwayEnv) environment.