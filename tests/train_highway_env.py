from context import *
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import rlrom.wrappers.stl_wrapper
import stlrom
from rlrom.envs import *
import rlrom.utils
import time
import matplotlib.pyplot as plt
import highway_env  # noqa: F401



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


def train_ppo(cfg):
    cfg = utils.load_cfg(cfg)    
    
    # env configuration
    cfg_env = cfg['cfg_env']        
    
    if cfg_env.get('manual_control', False):
        print("WARNING: manual_control was set to True. I'm setting it back to False")
        cfg_env['manual_control'] = False

    # hyperparams, training configuration
    cfg_train = cfg['cfg_train']        
    n_envs = cfg_train['n_envs']
    batch_size = cfg_train['batch_size']
    neurons = cfg_train['neurons']
    learning_rate = cfg_train.get('learning_rate', 5e-4)
    total_timesteps = cfg_train['total_timesteps']    
    model_name = cfg_train['model_name']
    tb_dir= cfg_train.get('model_path/tb_logs/','./tb_logs/')        
    tb_prefix =  f"{model_name}_{int(time.time())}"

    policy_kwargs = dict(
      #activation_fn=th.nn.ReLU,
      net_arch=dict(pi=[neurons, neurons], qf=[neurons, neurons])
    )

    # Instantiate environment
    def make_env():
        env = gym.make("highway-fast-v0")
        env.unwrapped.configure(cfg_env)
        #if env_mode==EnvMode.TERM_SLOW:
    #    cfg = cfg_envs['highway-env']
    #    driver= stlrom.STLDriver()
    #    driver.parse_string(cfg['specs'])        
    #    env = rlrom.wrappers.stl_wrapper.STLWrapper(env,driver,signals_map=cfg, terminal_formulas={'ego_slow_too_long'}        
        return env
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)    

    # Instantiate model
    model = PPO("MlpPolicy",env,
     device='cpu',
     policy_kwargs=policy_kwargs,
     n_steps=batch_size * 12 // n_envs,
     batch_size=batch_size,
     n_epochs=10,
     learning_rate=learning_rate,
     gamma=0.9,
     verbose=1,
     tensorboard_log=tb_dir
    )

    # Train the agent
    model.learn(
      total_timesteps=total_timesteps,
      progress_bar=True,
      tb_log_name=tb_prefix
    )

    # Saving the agent
    model_name, cfg_name = utils.get_model_fullpath(cfg)
    model.save(model_name) #TODO try except 
    with open(cfg_name,'w') as f:
         yaml.safe_dump(cfg, f)
    
    return model

class Tester: 
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
        
        # todo switch exp_type
        env = gym.make("highway-v0", render_mode='human')
        env.unwrapped.configure(cfg_env)

        # wrap env with stl_wrapper. We'll have to check if not done already
        cfg_specs = cfg.get('cfg_specs')
        driver= stlrom.STLDriver()
        driver.parse_string(cfg_specs['specs'])        
        env = rlrom.wrappers.stl_wrapper.STLWrapper(env,driver,signals_map=cfg_specs)
        
        self.cfg = cfg
        self.model=model
        self.env= env
        
    def get_action(self):
        wobs = self.env.wrapped_obs
        if self.manual_control is True:
            action = self.env.action_space.sample()
        else:
            action, _ = self.model.predict(wobs)
        return action

    def run_seed(self, seed=None, num_steps=100):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()        
        for _ in range(num_steps):    
            action = self.get_action()
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
    train_ppo(custom_cfg)