from stable_baselines3 import PPO #,A2C,SAC,TD3,DQN,DDPG
import rlrom.utils as utils
from rlrom.testers import RLTester
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback,BaseCallback, CallbackList
import numpy as np
import yaml
import time, datetime
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from rlrom.wrappers.stl_wrapper import stl_wrap_env
import os
import sys
import importlib
import torch as th

def make_env_train(cfg):
        
    if 'make_env_train' in cfg:
      # recover and execute the make_env_train custom function      
      to_import = cfg.get('import_module')                
      sys.path.append('')
      if to_import is not None: #TODO better exception handling ?
        imported = importlib.import_module(to_import)
        print(f'Imported module {to_import}')      
        custom_make_env = getattr(imported, cfg['make_env_train'])        
        env = custom_make_env(cfg)      
    else:  
      # default
      env_name = cfg.get('env_name','')                           
      env = gym.make(env_name, render_mode=None)
      
    cfg_specs = cfg.get('cfg_specs', None)            
    if cfg_specs is not None:
        model_use_spec = cfg.get('model_use_specs', False)
        if model_use_spec:          
          env = stl_wrap_env(env, cfg_specs)
          env = gym.wrappers.FlattenObservation(env)
            
    return env


class RlromCallback(BaseCallback):
  def __init__(self, verbose=0, cfg_main=dict(),model_name_now=''):
    super().__init__(verbose)
    self.cfg = utils.set_rec_cfg_field(cfg_main,render_mode=None)
    
    cfg_train = cfg_main.get('cfg_train')
    n_envs = cfg_train.get('n_envs',1)
    self.eval_freq = cfg_train.get('eval_freq', 1000)//n_envs
    
    self.checkpoints_folder = ''
    
    
    print(f'n_envs = {n_envs}, Eval freq: {self.eval_freq}')
  

  def _on_step(self):
    #print("step:", self.n_calls)
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      self.eval_policy()
          
    return True
    

  def eval_policy(self):
        
    self.tester = RLTester(self.cfg)
    self.tester.model = self.model
    Tres = self.tester.run_cfg_test(reload_model=False)                
          
    res_all_ep = Tres['res_all_ep']
    for metric_name, metric_value in res_all_ep['basics'].items():
      log_name = 'basics/'+metric_name    
      self.logger.record(log_name, metric_value)
      
    for f_name, f_value in res_all_ep['eval_formulas'].items():    
      for metric_name, metric_value in f_value.items():
        log_name = 'eval_f/'+f_name+'/'+metric_name 
        self.logger.record(log_name, metric_value)
    for f_name, f_value in res_all_ep['reward_formulas'].items():    
      for metric_name, metric_value in f_value.items():
        log_name = 'rew_f/'+f_name+'/'+metric_name 
        self.logger.record(log_name, metric_value)
    
    self.tester.print_res_all_ep(Tres)

    return True



class RLTrainer:
  def __init__(self, cfg):    
    self.cfg = utils.load_cfg(cfg)    
    self.cfg_train = self.cfg.get('cfg_train', {})
    self.model_use_specs = self.cfg.get('model_use_specs', False)
    self.env_name = self.cfg.get('env_name')
    self.model_name = self.cfg.get('model_name')
    self.make_env= lambda: make_env_train(self.cfg)
    self.model = None

  def train(self):
        
    cfg_algo = self.cfg_train.get('algo')
    model_name = self.cfg.get('model_name')
    
    if cfg_algo is not None:       
      has_cfg_specs = 'cfg_specs' in self.cfg
      if has_cfg_specs:
        s = self.cfg.get('cfg_specs')        
        has_cfg_specs = s != None

    if has_cfg_specs:   
      callbacks = CallbackList([        
        RlromCallback(verbose=1, cfg_main=self.cfg, model_name_now=self.get_model_name_now())        
          ])
    else:
      callbacks = [] 
       
    if self.model is None:   
      if 'ppo' in cfg_algo:                           
        model = self.init_PPO()
    else:
      model= self.model

    # Training          
    total_timesteps = self.cfg_train.get('total_timesteps',1000)    
    progress_bar = self.cfg_train.get('progress_bar',True)    
    tb_prefix =  self.get_model_name_now()    

    # Train the agent
    model.learn(
      total_timesteps = total_timesteps,
      callback = callbacks,
      tb_log_name = tb_prefix,
      progress_bar= progress_bar,
    )
    
    # Saving the agent
    self.save_model()

    return model

  def save_model(self,path=None):
    model_name, cfg_name = utils.get_model_fullpath(self.cfg)
    self.model.save(model_name) #TODO try except 
    with open(cfg_name,'w') as f:
         yaml.safe_dump(self.cfg, f)


  def init_PPO(self):
    cfg_ppo = {}

    # Get options for ppo    
    cfg_algo = self.cfg_train.get('algo')    
    if cfg_algo.get('ppo') is not None:                     
      cfg_ppo = cfg_algo.get('ppo')

    # Policy     
    if 'policy' not in cfg_ppo:
      cfg_ppo['policy']= 'MlpPolicy'

    if 'policy_kwargs' in cfg_ppo:
      print('Reading policy_kwargs')
      cfg_ppo['policy_kwargs']= policy_cfg2kargs(cfg_ppo['policy_kwargs'])
    
    # Environments
    n_envs = self.cfg_train.get('n_envs',1)
    if n_envs>1:
       env = make_vec_env(self.make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)    
    else:
       env = self.make_env()

    print(cfg_ppo)
    self.model= PPO(env=env, **cfg_ppo )

    return self.model
  
  def get_model_name_now(self):
    s = self.model_name
    dd = datetime.datetime.now()
    s_dd= dd.strftime("_%Y_%m_%d")
    return s+s_dd

def policy_cfg2kargs(cfg_policy):
  act_fn =  {
    "ReLU": th.nn.ReLU,
    "Tanh": th.nn.Tanh,
    "ELU": th.nn.ELU,
  }
  if 'activation_fn' in cfg_policy:
    if isinstance(cfg_policy['activation_fn'], str):
      cfg_policy['activation_fn']= act_fn[cfg_policy['activation_fn']]
  
  return cfg_policy

