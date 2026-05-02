# Generic stuff
import numpy as np
import os
import sys
import importlib
import copy #for deep copy of dicts (*not* default !)
#import functools

# Gym and sb3 stuff
import gymnasium as gym
from stable_baselines3 import PPO,A2C,SAC,TD3,DQN,DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.wrappers import FlattenObservation

from mo_gymnasium.wrappers.vector import MOSyncVectorEnv, MORecordEpisodeStatistics

# rlrom stuff
from rlrom.testers import RLTester
import rlrom.utils as rlu
from rlrom.utils import policy_cfg2kargs
from rlrom.wrappers.specs_wrapper import wrap_env_specs

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
              
    env = wrap_env_specs(env, cfg)
            
    return env

def make_vec_envs(cfg, n_envs=8, use_subproc=False):
    """
    Create a vectorized environment with n_envs copies of make_env_train(cfg).
    Works for STL + RM wrappers.
    TODO: make that an option ? not fully compatible with Gymnasium, more like Gym 0.21 or 0.26
    see  https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    """
    def _make_env_fn():
        def _init():
            return make_env_train(cfg)
        return _init

    env_fns = [_make_env_fn() for _ in range(n_envs)]
    
    if n_envs == 1:
        # Just return a single env (not vectorized)
        env = make_env_train(cfg)
        env = VecMonitor(DummyVecEnv([lambda: env]))  # Wrap single env for logging
        return env
    else:
        vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
        vec_env = vec_env_cls(env_fns)
        vec_env = VecMonitor(vec_env) 
        return vec_env

class RlromCallback(BaseCallback):
  def __init__(self, verbose=0, cfg_main=dict(), chkpt_dir='', cfg_name=''):
    super().__init__(verbose)
    self.cfg = rlu.set_rec_cfg_field(cfg_main,render_mode=None)
    
    cfg_train = cfg_main.get('cfg_train')
    self.n_envs = int(cfg_train.get('n_envs',1))
    self.eval_freq = int(cfg_train.get('eval_freq', 1000))//self.n_envs        
    self.chkpt_dir = chkpt_dir
    self.chkpt_model_root_name = os.path.join(chkpt_dir, 'model_step_')
    self.chkpt_res_root_name = os.path.join(chkpt_dir, 'res_step_')
    
    cfg_filename= os.path.join(self.chkpt_dir,'cfg0.yml')
    
    print(f'n_envs: {self.n_envs}, Eval freq: {self.eval_freq}, Checkpoints folder: {self.chkpt_dir}')
    print(f'Saving configuration file to {cfg_filename}')
    
    with open(cfg_filename,'w') as f:
         rlu.yaml.dump(self.cfg, f)


  def _on_step(self):
    #print("step:", self.n_calls)
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      self.eval_and_save_model()           
    return True
    
  def  eval_and_save_model(self):
    model_filename = self.chkpt_model_root_name+str(self.n_calls*self.n_envs)
    print(f'saving model to {model_filename}...')
    self.model.save(model_filename)      
    res_filename = self.chkpt_res_root_name+str(self.n_calls*self.n_envs)+'.yml'            
    
    Tres = self.eval_policy()
    Tres.pop('episodes',[]) # TODO make a more generic save result thing, with options to keep episodes maybe
    print(f'saving test results to {res_filename}...')
    with open(res_filename,'w') as f:
       rlu.yaml.dump(Tres, f)

  def _on_training_end(self):
    self.eval_and_save_model()
    return super()._on_training_end()

  def eval_policy(self):
  # This might go eventually - log stuff from Tres into tensorboard      
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
    self.cfg = rlu.load_cfg(cfg)    
    self.cfg_train = self.cfg.get('cfg_train', {})
    self.model_use_specs = self.cfg.get('model_use_specs', False)
    self.env_name = self.cfg.get('env_name')
    self.model_name = self.cfg.get('model_name')
    #self.make_env=functools.partial(make_vec_envs, cfg, n_envs=8, use_subproc=False)#lambda: make_vec_envs(cfg, n_envs=2, use_subproc=True) #make_env_train(self.cfg)
    self.make_env= lambda: make_env_train(self.cfg)
    self.model = None
    
  def train(self):

    cfg_algo = self.cfg_train.get('algo')    
    
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
       
    model = self.init_rl_algo(cfg_algo) if self.model is None else self.model

    # Training
    total_timesteps = self.cfg_train.get('total_timesteps',1000)    
    progress_bar = self.cfg_train.get('progress_bar',True)    
    tb_prefix =  rlu.add_now_suffix(self.model_name)

    # algo_train_kwargs contains algorithm-specific configurations that are only
    # used when calling train(); these are used heavily in morl-baselines
    algo_train_kwargs = self.cfg_train.get('train_kwargs', {})
    if "ref_point" in algo_train_kwargs:
      algo_train_kwargs["ref_point"] = np.asarray(algo_train_kwargs["ref_point"])

    # Train the agent: Method is "learn" or "train" depending on whether it's a
    # SB3 or morl-baselines model
    assert hasattr(model, "learn") or hasattr(model, "train")
    if hasattr(model, "learn"):
      model.learn(
        total_timesteps = total_timesteps,
        callback = callbacks,
        tb_log_name = tb_prefix,
        progress_bar= progress_bar,
      )
    else:
      model.train(
        total_timesteps = total_timesteps,
        eval_env = self.make_env(),
        **algo_train_kwargs
      )
    
    # Saving the agent
    self.save_model()

    return model

  def save_model(self,path=None):
    model_name, cfg_name = rlu.get_model_fullpath(self.cfg)
    print(f'saving model to {model_name} trained with cfg {cfg_name}')
    self.model.save(model_name) #TODO try except 
    with open(cfg_name,'w') as f:
         rlu.yaml.dump(self.cfg, f)

  def set_checkpoint_dir(self):
    cfg= self.cfg
    model_full_path, cfg_name = rlu.get_model_fullpath(cfg)    
    chkpt_dir_root = os.path.splitext(model_full_path)[0]        
    chkpt_dir_root = rlu.add_now_suffix(chkpt_dir_root)
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

  def init_rl_algo(self, cfg_algo_name: dict):
    algo_name = None
    for name in rlu.ALGO_NAMES_CLASSES:
      if name in cfg_algo_name:
        algo_name = name
        break
    if algo_name is None:
      return None
    cfg_rl_algo = {}

    # Get options  
    cfg_algo0 = self.cfg_train.get('algo')
    cfg_algo  = cfg_algo0.copy()   
    if cfg_algo.get(algo_name) is not None:                     
      cfg_rl_algo = copy.deepcopy(cfg_algo.get(algo_name))

    env = self.make_env()
    is_multiobjective = env.get_wrapper_attr("multi_objective")
    del env

    # Policy     
    if 'policy' not in cfg_rl_algo and not is_multiobjective: # TODO: Handle multiobjective policy
      cfg_rl_algo['policy']= 'MlpPolicy'

    if 'policy_kwargs' in cfg_rl_algo:
      print('Reading policy_kwargs')
      cfg_rl_algo['policy_kwargs']= policy_cfg2kargs(cfg_rl_algo['policy_kwargs'])
    
    # Environments
    n_envs = int(self.cfg_train.get('n_envs',1))
    if n_envs>1 and not is_multiobjective: # TODO: Handle multiobjective vector envs, see MOSyncVectorEnv
       env = make_vec_env(self.make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)    
    else:
       env = self.make_env()

    print(cfg_rl_algo)
    algo_class = rlu.ALGO_NAMES_CLASSES[algo_name]
    self.model= algo_class(env=env, **cfg_rl_algo )

    return self.model
  

