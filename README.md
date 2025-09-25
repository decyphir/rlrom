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

RLRom reads configuration files in the YAML format as inputs. Examples are provided in the examples folder. A command line interface is provided through the script `rlrom_run`. For instance, 
```
$ rlrom_run test examples/cartpole/cfg_cartpole.cfg
```
will run a few episode of the cartpole classic environment, fetching a model on huggingface and monitor a formula on these episodes. 

More programmatic features are demonstrated in notebooks, in particular [this notebook](examples/highway_env/highway_notebook.ipynb) which presents a case study around [highway-env](https://github.com/Farama-Foundation/HighwayEnv) environment. 