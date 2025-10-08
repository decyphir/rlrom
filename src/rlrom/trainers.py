from stable_baselines3 import PPO #,A2C,SAC,TD3,DQN,DDPG
import rlrom.utils as utils
from rlrom.utils import policy_cfg2kargs, add_now_suffix
from rlrom.testers import RLTester
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from rlrom.wrappers.stl_wrapper import stl_wrap_env
from rlrom.wrappers.reward_machine import RewardMachine
import os
import sys
import importlib
from rlrom.utils import yaml
import copy #for deep copy of dicts (*not* default !)

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
          env = FlattenObservation(env)

        cfg_rm = cfg_specs.get('cfg_rm', None)            
        if cfg_rm is not None:
            env = RewardMachine(env, cfg_rm)  
            print("env", env)
            
    return env


class RlromCallback(BaseCallback):
  def __init__(self, verbose=0, cfg_main=dict(), chkpt_dir='', cfg_name=''):
    super().__init__(verbose)
    self.cfg = utils.set_rec_cfg_field(cfg_main,render_mode=None)
    
    cfg_train = cfg_main.get('cfg_train')
    self.n_envs = cfg_train.get('n_envs',1)
    self.eval_freq = cfg_train.get('eval_freq', 1000)//self.n_envs        
    self.chkpt_dir = chkpt_dir
    self.chkpt_model_root_name = os.path.join(chkpt_dir, 'model_step_')
    self.chkpt_res_root_name = os.path.join(chkpt_dir, 'res_step_')
    
    cfg_filename= os.path.join(self.chkpt_dir,'cfg0.yml')
    
    print(f'n_envs: {self.n_envs}, Eval freq: {self.eval_freq}, Checkpoints folder: {self.chkpt_dir}')
    print(f'Saving configuration file to {cfg_filename}')
    
    with open(cfg_filename,'w') as f:
         yaml.dump(self.cfg, f)


  def _on_step(self):
    #print("step:", self.n_calls)
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      self.eval_and_save_model()           
    return True
    
  def  eval_and_save_model(self):
    Tres = self.eval_policy()
    model_filename = self.chkpt_model_root_name+str(self.n_calls*self.n_envs)
    print(f'saving model to {model_filename}...')
    self.model.save(model_filename)      
    res_filename = self.chkpt_res_root_name+str(self.n_calls*self.n_envs)+'.yml'            
    Tres.pop('episodes',[]) # TODO make a more generic save result thing, with options to keep episodes maybe
    print(f'saving test results to {res_filename}...')
    with open(res_filename,'w') as f:
       yaml.dump(Tres, f)

  def _on_training_end(self):
    self.eval_and_save_model()
    return super()._on_training_end()

  def eval_policy(self):
        
    self.tester = RLTester(self.cfg)
    self.tester.model = self.model
    Tres = self.tester.run_cfg_test(reload_model=False)                
          
    res_all_ep = Tres['res_all_ep']
    for metric_name, metric_value in res_all_ep['basics'].items():
      log_name = 'basics/'+metric_name    
      self.logger.record(log_name, metric_value)

    if 'eval_formula' in res_all_ep:
      for f_name, f_value in res_all_ep['eval_formulas'].items():    
        for metric_name, metric_value in f_value.items():
          log_name = 'eval_f/'+f_name+'/'+metric_name 
          self.logger.record(log_name, metric_value)
    if 'reward_formulas' in res_all_ep:
      for f_name, f_value in res_all_ep['reward_formulas'].items():    
        for metric_name, metric_value in f_value.items():
          log_name = 'rew_f/'+f_name+'/'+metric_name 
          self.logger.record(log_name, metric_value)

    self.tester.print_res_all_ep(Tres)

    return Tres

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
    
    # setup folder for checkpoints and saving the cfg
    chkpt_dir, cfg_name = self.set_checkpoint_dir()

    callbacks = CallbackList([        
        RlromCallback(verbose=1, cfg_main=self.cfg, chkpt_dir=chkpt_dir, cfg_name=cfg_name)        
          ])
       
    if self.model is None:   
      if 'ppo' in cfg_algo:                           
        model = self.init_PPO()
    else:
      model= self.model

    # Training          
    total_timesteps = self.cfg_train.get('total_timesteps',1000)    
    progress_bar = self.cfg_train.get('progress_bar',True)    
    tb_prefix =  utils.add_now_suffix(self.model_name)

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
    print(f'saving model to {model_name} trained with cfg {cfg_name}')
    self.model.save(model_name) #TODO try except 
    with open(cfg_name,'w') as f:
         yaml.dump(self.cfg, f)

  def set_checkpoint_dir(self):
    cfg= self.cfg
    model_full_path, cfg_name = utils.get_model_fullpath(cfg)    
    chkpt_dir_root = os.path.splitext(model_full_path)[0]        
    chkpt_dir_root = utils.add_now_suffix(chkpt_dir_root)
    i = -1
    chkpt_dir_exists = True        
    while chkpt_dir_exists:
        i = i+1    
        chkpt_dir = chkpt_dir_root+'__training'+str(i)
        chkpt_dir_exists = os.path.exists(chkpt_dir)
        if chkpt_dir_exists is True:
          is_empty = not any(os.scandir(chkpt_dir))
          if is_empty is True:
            print(f'using empty folder for checkpoints: {chkpt_dir}')
            break
        else:
          print(f'creating folder for checkpoints: {chkpt_dir}')
          os.makedirs(chkpt_dir) # TODO catch exception if problem writing this folder    
    
    cfg_name = os.path.join(chkpt_dir,'cfg0.yml')    
    return chkpt_dir, cfg_name

  def init_PPO(self):
    cfg_ppo = {}

    # Get options for ppo    
    cfg_algo0 = self.cfg_train.get('algo')
    cfg_algo  = cfg_algo0.copy()   
    if cfg_algo.get('ppo') is not None:                     
      cfg_ppo = copy.deepcopy(cfg_algo.get('ppo'))

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
  

