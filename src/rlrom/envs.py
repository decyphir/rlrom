import os
import yaml

supported_envs = []
supported_models = ['ppo', 'a2c', 'sac', 'td3', 'dqn', 'qrdqn', 'ddpg', 'trpo']
cfg_envs = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_cfgs = os.path.join(current_dir, '../rlrom/cfgs/')

list_cfg=  os.listdir(dir_cfgs)
cfg_envs = {}
for c in list_cfg:
    cfg_name,_ = os.path.splitext(c)
    supported_envs.append(cfg_name)
    cfg_full_path = dir_cfgs + c
    with open(cfg_full_path, 'r') as file:
        dict_cfg = yaml.safe_load(file)
    cfg_envs[cfg_name] = dict_cfg