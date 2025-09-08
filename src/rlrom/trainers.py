from stable_baselines3 import PPO #,A2C,SAC,TD3,DQN,DDPG
import rlrom.utils as utils
from rlrom.testers import RLTester
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
import yaml
import time
import gymnasium as gym
from rlrom.wrappers.stl_wrapper import stl_wrap_env


def make_env_train(cfg):

    env_name = cfg.get('env_name','')                   
    env = gym.make(env_name, render_mode=None)
    
    cfg_env = cfg.get('cfg_env',dict())
    if cfg_env != dict():
        env.unwrapped.configure(cfg_env)
      # wrap env with stl_wrapper. We'll have to check if not done already        
    cfg_specs = cfg.get('cfg_specs', None)
            
    if cfg_specs is not None:
        env = stl_wrap_env(env, cfg_specs)
            
    return env


class STLWrapperCallback(BaseCallback):
  def __init__(self, verbose=0, cfg_main=dict()):
    super().__init__(verbose)
    self.tester = RLTester(cfg_main, render_mode=None)
    self.tester.init_env()    
  

  def _on_step(self):
    return True
    
  def _on_rollout_end(self):
    # Access the rollout buffer
    buffer = self.model.rollout_buffer
    episodes = utils.get_episodes_from_rollout(buffer)
    num_ep = len(episodes)
                    
    res_rew = dict()
    res_rew_f_list = []    
    res_eval_f_list = []
    
    ep_lens = []
    for i in range(0,num_ep):        
      #print('----------------------------------------------------')
      #print('EPISODE', i, end=' ')            
      self.tester.env.set_episode_data(episodes[i])      
      # eval reward formulas and eval formulas
      res_rew, res_all_ep, res_rew_f_list, res_eval_f_list = self.tester.env.eval_specs_episode(
        episodes[i],res=res_rew,res_rew_f_list=res_rew_f_list, res_eval_f_list=res_eval_f_list)
      rewards = episodes[i]['rewards']
      ep_lens.append(len(rewards))
      #print()
    my_mean_ep_len = np.array(ep_lens).mean()
    print('Number of episodes:', num_ep, 'mean ep len:', my_mean_ep_len )
    self.logger.record('rollout/my_mean_ep_len', my_mean_ep_len)
      

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

    return True

class RLTrainer:
  def __init__(self, cfg):    
    self.cfg = utils.load_cfg(cfg)    
    self.cfg_train = self.cfg.get('cfg_train', {})
    pass   

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
          STLWrapperCallback(verbose=1, cfg_main=self.cfg)        
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

  def train_ppo(self,cfg, make_env,model_name, callbacks):
      
    # hyperparams, training configuration      
    n_envs = cfg.get('n_envs',1)
    batch_size = cfg.get('batch_size',128)
    neurons = cfg.get('neurons',128)
    learning_rate = float(cfg.get('learning_rate', '5e-4'))
    total_timesteps = cfg.get('total_timesteps',1000)
    cfg_tb = cfg.get('tensorboard',dict()) 
    tb_dir, tb_prefix= self.get_tb_dir(cfg_tb,model_name)
    
    policy_kwargs = dict(
      #activation_fn=th.nn.ReLU,
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
      n_epochs=10,
      learning_rate=learning_rate,
      gamma=0.9,
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