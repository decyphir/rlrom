from stable_baselines3 import PPO #,A2C,SAC,TD3,DQN,DDPG
import rlrom.utils as utils
from rlrom.testers import RLTester
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback,BaseCallback, CallbackList
import numpy as np
import yaml
import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from rlrom.wrappers.stl_wrapper import stl_wrap_env
import os
import sys
import importlib
import torch as th

def make_env_train(cfg):
    print(f'currend folder: {os.getcwd()}')
        
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
  def __init__(self, verbose=0, cfg_main=dict()):
    super().__init__(verbose)
    self.cfg = utils.set_rec_cfg_field(cfg_main,render_mode=None)
    
    cfg_train = cfg_main.get('cfg_train')
    n_envs = cfg_train.get('n_envs',1)
    self.eval_freq = cfg_train.get('eval_freq', 1000)//n_envs
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
    self.model_use_specs = cfg.get('model_use_specs', False)
    

  def train(self,make_en_train=make_env_train):
    
    make_env= lambda: make_env_train(self.cfg)
    cfg_algo = self.cfg_train.get('algo')
    model_name = self.cfg.get('model_name')
    
    if cfg_algo is not None:       
      has_cfg_specs = 'cfg_specs' in self.cfg
      if has_cfg_specs:
        s = self.cfg.get('cfg_specs')        
        has_cfg_specs = s != None

    if has_cfg_specs:   
      callbacks = CallbackList([        
        RlromCallback(verbose=1, cfg_main=self.cfg)        
          ])
    else:
      callbacks = [] 
       
    if cfg_algo.get('ppo') is not None:                     
      cfg_ppo = cfg_algo.get('ppo')
      print('Training  with PPO...',cfg_ppo )          
      model = self.train_ppo(cfg_ppo, make_env, model_name,callbacks)
    
    # Saving the agent
    model_name, cfg_name = utils.get_model_fullpath(self.cfg)
    model.save(model_name) #TODO try except 
    with open(cfg_name,'w') as f:
         yaml.safe_dump(self.cfg, f)

    return model

  def train_ppo(self,cfg_algo, make_env,model_name, callbacks):
      
    # hyperparams, training configuration      
    n_envs = self.cfg_train.get('n_envs',1)
    batch_size = cfg_algo.get('batch_size',128)
    neurons = cfg_algo.get('neurons',128)
    learning_rate = float(cfg_algo.get('learning_rate', '5e-4'))
    n_epoch = cfg_algo.get('n_epoch', 10)
    gamma = cfg_algo.get('gamma', .99)
    gae_lambda = cfg_algo.get('gae_gamma', .8)
    clip_range = cfg_algo.get('clip_range', .2)
    ent_coef = cfg_algo.get('ent_coef', 0.0)
    vf_coef = cfg_algo.get('vf_coef', .5)
    normalize_advantage = cfg_algo.get('normalize_advantage', False)
    total_timesteps = cfg_algo.get('total_timesteps',1000)
            
    cfg_tb = cfg_algo.get('tensorboard',dict()) 
    tb_dir, tb_prefix= self.get_tb_dir(cfg_tb,model_name)
    
    policy_kwargs = dict(
      activation_fn=th.nn.Tanh,
      net_arch=dict(pi=[neurons, neurons], qf=[neurons, neurons])
    )
    
    if n_envs>1:
       env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)    
    else:
       env = make_env()

    # Instantiate model
    model = PPO("MlpPolicy",env,
      device='cpu',
      policy_kwargs=policy_kwargs,
      n_steps=batch_size * 12 // n_envs,
      batch_size=batch_size,
      n_epochs=n_epoch,
      ent_coef=ent_coef,
      learning_rate=learning_rate,
      gamma=gamma,
      gae_lambda=gae_lambda,
      clip_range=clip_range,
      vf_coef=vf_coef,
      normalize_advantage=normalize_advantage,
      verbose=1,
      tensorboard_log=tb_dir
    )
 
    # Train the agent
    model.learn(
      total_timesteps=total_timesteps,
      callback = callbacks,
      tb_log_name=tb_prefix,
      progress_bar=True
    )
    
    return model

  def get_tb_dir(self, cfg, model_name):
    tb_dir = cfg.get('tb_path','./tb_logs')
    tb_prefix =  f"{model_name}_{int(time.time())}"
    return tb_dir, tb_prefix