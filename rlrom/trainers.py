from stable_baselines3 import PPO #,A2C,SAC,TD3,DQN,DDPG
import rlrom.utils as utils
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import yaml
import time



def train_ppo(cfg, make_env, callbacks):
    cfg = utils.load_cfg(cfg)    
            
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
      callback = callbacks,
      tb_log_name=tb_prefix,
      progress_bar=True
    )

    # Saving the agent
    model_name, cfg_name = utils.get_model_fullpath(cfg)
    model.save(model_name) #TODO try except 
    with open(cfg_name,'w') as f:
         yaml.safe_dump(cfg, f)
    
    return model
