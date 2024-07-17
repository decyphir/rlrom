# RLRom 

This module integrates Robust Online Monitoring methods with Reinforcement Learning stuff.
The first motivation is testing/monitoring RL models. 

## Install

Those are needed for building some of the required python modules: 
- CMake
- swig 

Then run the following:
```
pip install -r requirements.txt
``` 

## Running 

Run `python run_app.py`, then open browser.

## Features

- Select an environment among a list of supported ones.
- To load a model, choose between 
  - **Random**: random actions
  - **Local**: Upload  model zip files created with stable-baselines-3, then choose one
  - **Hugging Face**: Fetch the list of models available on Hugging Face, then choose one
- Choose between running with or without human render
- Runs from a list of seeds and store traces
- Compute total rewards
- Plots observation, reward, actions, individually or together of any trace, with flexible layout
- Evaluate (monitor) and plot quantitative and Boolean satisfaction of any Signal Temporal Logic formula (STL)
- Sort runs against STL formula robustness
