from context import *
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

import enum

import rlrom.wrappers.stl_wrapper
import stlrom
from rlrom.envs import *
import rlrom.utils
import time
import matplotlib.pyplot as plt

class EnvMode(enum.Enum):
    VANILLA=0
    TERM_SLOW=1

env_mode= EnvMode.VANILLA
collision_reward = -1
model_name = 'ppo_hw_van_high_col.zip'
total_timesteps = 100_000

def make_env(train=True, env_mode=env_mode, verbose=0):

    if train:
        env = gym.make("highway-fast-v0")
    else:
        env = gym.make("highway-v0", render_mode='human')

    env.unwrapped.configure({
            "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 100,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": collision_reward, 
                "right_lane_reward": 0,  
                "high_speed_reward": 1., 
                "lane_change_reward": 0, 
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,        
    })

    if env_mode==EnvMode.TERM_SLOW:
        cfg = cfg_envs['highway-env']
        driver= stlrom.STLDriver()
        driver.parse_string(cfg['specs'])        
        env = rlrom.wrappers.stl_wrapper.STLWrapper(env,driver,signals_map=cfg, terminal_formulas={'ego_slow_too_long'})

    if verbose>=1:
        pprint(cfg)
    return env

if __name__ == "__main__":
    n_cpu = 12
    batch_size = 64
    neurons = 128
    policy_kwargs = dict(
    #activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[neurons, neurons], qf=[neurons, neurons])
    )

    vec_env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
     "MlpPolicy",
     vec_env,
     device='cpu',
     policy_kwargs=policy_kwargs,
     n_steps=batch_size * 12 // n_cpu,
     batch_size=batch_size,
     n_epochs=10,
     learning_rate=5e-4,
     gamma=0.9,
     verbose=1,
     tensorboard_log="./highway_ppo/"
    )

# Train the agent
    model.learn(
    total_timesteps=total_timesteps,
    progress_bar=True
    )

    model.save(model_name)