# RLRom

This module integrates Robust Online Monitoring methods with Reinforcement Learning stuff. The motivation is first to test RL agents using interpretable monitors, then use these monitors to train models to perform complex tasks, and/or converge toward behaviors that reliably satisfy certain requirements. 

## Install

Those are needed for building some of the required python modules, in particular [stlrom](https://github.com/decyphir/stlrom) for STL monitoring.
- CMake
- swig 

Then installing should be as simple as
```
pip install rlrom 
``` 

Note that some environments still require an older version of Gym. It can be installed with 
```
pip install rlrom[old_gym]
``` 


## Getting Started

RLRom reads configuration files in the YAML format as inputs. Examples are provided in the examples folder. A command line interface is provided through the script `rlrom_run`. For instance, 
```
$ rlrom_run test examples/cartpole/cfg_cartpole.cfg
```
will run a few episode of the cartpole classic environment, fetching a model on huggingface and monitor a formula on these episodes. 

More programmatic features are demonstrated in notebooks, in particular [this notebook](examples/highway_env/highway_notebook.ipynb) which presents a case study around [highway-env](https://github.com/Farama-Foundation/HighwayEnv) environment. 