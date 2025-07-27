# RLRom

This module integrates Robust Online Monitoring methods with Reinforcement Learning stuff.
The motivation is first to test RL agents using interpretable monitors, then use these monitors to train models to perform complex tasks, and/or converge toward behaviors that reliably satisfy certain requirements. 

## Install

Those are needed for building some of the required python modules: 
- CMake
- swig 

Then run the following:
```
pip install --upgrade -r requirements.txt 
``` 

## Testing

In the current version, features are mostly demonstrated in [this notebook](examples/highway_env/highway_notebook.ipynb) in the example folder, which present a case study around [highway-env](https://github.com/Farama-Foundation/HighwayEnv) environment. 