import os
import yaml

supported_envs = []

supported_models = ['ppo', 'a2c', 'sac', 'td3', 'dqn', 'qrdqn', 'ddpg', 'trpo']

cfg_envs = {}

dir_cfgs = '../rlrom/cfgs/'
list=  os.listdir(dir_cfgs)
cfg_envs = {}
for c in list:
    cfg_name,_ = os.path.splitext(c)
    supported_envs.append(cfg_name)
    cfg_full_path = dir_cfgs + c
    with open(cfg_full_path, 'r') as file:
        dict_cfg = yaml.safe_load(file)
    cfg_envs[cfg_name] = dict_cfg