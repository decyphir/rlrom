import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import highway_env  # noqa: F401

import rlrom.utils as utils
from rlrom.trainers import RLTrainer, STLWrapperCallback
from rlrom.testers import stl_wrap_env
from pprint import pprint
import numpy as np

# Instantiate environment
def make_train_env(cfg):
    
    # env configuration    
    cfg_env = cfg['cfg_env']            
    if cfg_env.get('manual_control', False):
        print("WARNING: manual_control was set to True. I'm setting it back to False")
        cfg_env['manual_control'] = False    
    
    env = gym.make("highway-fast-v0")    
    env.unwrapped.configure(cfg_env)
        
    cfg_specs = cfg.get('cfg_specs', None)    
    if cfg_specs is not None:
        env = stl_wrap_env(env, cfg_specs)                    
            
    return env
    

if __name__ == "__main__":
    # train_highway_env cfg_main.yml [--cfg-train cfg_train.yml]    
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train or test PPO on highway environment')
    parser.add_argument('main_cfg', type=str, help='Path to main configuration file')
    parser.add_argument('--cfg-train', type=str, help='Path to training configuration file')
    parser.add_argument('--num-trainings',type=int, help='Number of repeats of training')
    args = parser.parse_args()
    
    # Start with default configuration
    custom_cfg = dict()        
    
    # Load main config file
    if os.path.exists(args.main_cfg):
        custom_cfg = utils.load_cfg(args.main_cfg)
    else:
        print(f"Warning: Config file {args.main_cfg} not found. Using default settings.")
    
    # Override with train config if specified
    if args.cfg_train:
        if os.path.exists(args.cfg_train):
            custom_cfg['cfg_train'] = args.cfg_train
            print(f"Using training config from {args.cfg_train}")
        else:
            print(f"Warning: Training config file {args.cfg_train} not found.")
    pprint(custom_cfg)
        
    make_env= lambda: make_train_env(custom_cfg)


    if args.num_trainings:
        num_trainings= args.num_trainings
    else:
        num_trainings = 1
        
    trainer = RLTrainer(custom_cfg)
    for _ in range(0, num_trainings):
        trainer.train(make_env)