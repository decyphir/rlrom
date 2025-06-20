import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import stlrom
from rlrom.wrappers.stl_wrapper import STLWrapper

from rlrom.envs import *
import rlrom.utils as utils
import time
import matplotlib.pyplot as plt
import highway_env  # noqa: F401
import rlrom.trainers as trainers

from pprint import pprint
import numpy as np

cfg=dict()
cfg['exp_type'] = 'vanilla'; # type of experiment 'vanilla' for normal reward and environment 
cfg['cfg_env'] = 'cfg_env.yml'
def init_cfg_train():
        model_path = './models'
        model_name = 'hw_vanilla_ppo'
        n_envs = 12
        batch_size = 64
        neurons = 128
        total_timesteps = 200_000
        return locals()
cfg['cfg_train'] = init_cfg_train()


def stl_wrap_env(env, cfg_specs):
    driver= stlrom.STLDriver()
    stl_specs_str = cfg_specs['specs']
    driver.parse_string(stl_specs_str)
    obs_formulas = cfg_specs.get('obs_formulas',[])
    end_formulas = cfg_specs.get('end_formulas',[])
    env = STLWrapper(env,driver,signals_map=cfg_specs, obs_formulas = obs_formulas,end_formulas=end_formulas)
    return env
        
# Instantiate environment
def make_train_env(cfg):
    # env configuration
    
    cfg_env = cfg['cfg_env']            
    if cfg_env.get('manual_control', False):
        print("WARNING: manual_control was set to True. I'm setting it back to False")
        cfg_env['manual_control'] = False    
    env = gym.make("highway-fast-v0")    
    env.unwrapped.configure(cfg_env)
    
    # STL wrapper TODO

    if cfg['exp_type']=='stl':
        cfg_specs = cfg.get('cfg_specs')
        env = stl_wrap_env(env, cfg_specs)
            
    return env
    
class HwEnvTestCallback(BaseCallback):
    def __init__(self, verbose=0, cfg=cfg):
        super().__init__(verbose)
        self.cfg = cfg
        self.custom_rollout_metric=0
        self.is_vanilla =  cfg['exp_type']=='vanilla'

    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        # Access the rollout buffer
        buffer = self.model.rollout_buffer
        episodes = utils.get_episodes_from_rollout(buffer)
        num_ep = len(episodes)

        print('Number of episodes:', num_ep )
        
        v_mean = []
        v_sum = []
        for i in range(0,num_ep):
            print('----------------------------------------------------')
            print('EPISODE ', i)            
            obs = episodes[i]['observations']
            v =[]            

            if self.is_vanilla:                        
                for o in obs:        
                    v.append(80*o[0,3])
            else:
                for o in obs:        
                    v.append(80*o[3])

            print(v)
            print('mean v:',np.mean(v) )
            v_mean.append(np.mean(v))
            v_sum.append(np.sum(v))

        print('mean ego_vx:', v_mean)
        print('sum ego_vx:', v_sum)           
    
        self.logger.record("eval/mean_velocity", np.mean(v_mean))
        self.logger.record("eval/mean_final_dist", np.mean(v_sum))
        return True

class HwEnvTester: 
    def __init__(self,cfg):
        cfg_env = cfg['cfg_env']
        self.manual_control = True
        if cfg_env.get('manual_control', False):
            model = 'manual'
            print("INFO: manual_control set to True, stay alert.")            
        else:                                       
            model_name, _ = utils.get_model_fullpath(cfg)
            print("INFO: Loading model ", model_name)
            model= PPO.load(model_name)
            self.manual_control = False
        
        env = gym.make("highway-v0", render_mode='human')
        env.unwrapped.configure(cfg_env)

        # wrap env with stl_wrapper. We'll have to check if not done already        
        cfg_specs = cfg.get('cfg_specs')
        if cfg['exp_type']=='stl':
            env = stl_wrap_env(env, cfg_specs)
        
        self.cfg = cfg
        self.model=model
        self.env= env        
        self.agent_is_vanilla = cfg['exp_type']=='vanilla'

    def get_action(self, obs):        
        if self.agent_is_vanilla:   # agent was trained without stl_wrapper, so we need to use wrapped_obs to predict action
            obs = self.env.wrapped_obs
        
        if self.manual_control is True:
            action = self.env.action_space.sample()
        else:
            action, _ = self.model.predict(obs)
        return action

    def run_seed(self, seed=None, num_steps=100):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()        
        for _ in range(num_steps):    
            action = self.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)    
            if terminated:
                print('Crash')
                break    
        self.env.close()
        return 

if __name__ == "__main__":
    # train_highway_env cfg_main.yml [--cfg-train cfg_train.yml]    
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train or test PPO on highway environment')
    parser.add_argument('main_cfg', type=str, help='Path to main configuration file')
    parser.add_argument('--cfg-train', type=str, help='Path to training configuration file')
    
    args = parser.parse_args()
    
    # Start with default configuration
    custom_cfg = dict()
    custom_cfg['exp_type'] = 'vanilla'
    
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

    # Callback(s)
    callbacks = CallbackList([
        HwEnvTestCallback(verbose=1, cfg=custom_cfg),        
    ])

    trainers.train_ppo(custom_cfg,make_env, callbacks)