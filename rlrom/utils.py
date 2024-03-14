from stable_baselines3 import PPO,A2C,SAC,TD3,DQN,DDPG
from sb3_contrib import TRPO, QRDQN
from huggingface_hub import HfApi, ModelFilter
from huggingface_sb3 import load_from_hub
from huggingface_sb3.naming_schemes import EnvironmentName, ModelName, ModelRepoId
import re

def load_ppo_model(env_name, repo_id):
    filename = ModelName('ppo', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
    }
    model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_a2c_model(env_name, repo_id):
    filename = ModelName('a2c', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = A2C.load(checkpoint, print_system_info=True)
    #model = A2C.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_sac_model(env_name, repo_id):
    filename = ModelName('sac', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = SAC.load(checkpoint, print_system_info=True)
    #model = SAC.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_td3_model(env_name, repo_id):
    filename = ModelName('td3', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = TD3.load(checkpoint, print_system_info=True)
    #model = TD3.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_dqn_model(env_name, repo_id):
    filename = ModelName('dqn', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = DQN.load(checkpoint, print_system_info=True)
    #model = DQN.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_qrdqn_model(env_name, repo_id):
    filename = ModelName('dqn', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = QRDQN.load(checkpoint, print_system_info=True)
    #model = DQN.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_ddpg_model(env_name, repo_id):
    filename = ModelName('ddpg', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    #model = DDPG.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    model = DDPG.load(checkpoint, print_system_info=True)
    return model

def load_trpo_model(env_name, repo_id):
    filename = ModelName('trpo', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    #model = TRPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    model = TRPO.load(checkpoint, print_system_info=True)
    return model

def load_model(env_name, repo_id):
    if 'ppo' in repo_id:
        return load_ppo_model(env_name, repo_id)
    elif 'a2c' in repo_id:
        return load_a2c_model(env_name, repo_id)
    elif 'sac' in repo_id:
        return load_sac_model(env_name, repo_id)
    elif 'td3' in repo_id:
        return load_td3_model(env_name, repo_id)
    elif 'dqn' in repo_id:
        return load_dqn_model(env_name, repo_id)
    elif 'qrdqn' in repo_id:
        return load_qrdqn_model(env_name, repo_id)
    elif 'ddpg' in repo_id:
        return load_ddpg_model(env_name, repo_id)
    elif 'trpo' in repo_id:
        return load_trpo_model(env_name, repo_id)
    else:
        return None


def find_models(env_name):
    api = HfApi()
    models_iter = api.list_models(filter=ModelFilter(tags=env_name))
    models = list(models_iter)
    return models, [model.modelId for model in models]


def get_layout_from_string(signals):
    # split signals string wrt linebreaks first
    signals = signals.split('\n')

    # then split wrt commas 
    signals = [signal.split(',') for signal in signals]

    # then strip 
    signals = [[s.strip() for s in signal] for signal in signals]

    # remove lists with only one empty string element
    signals = [signal for signal in signals if signal != ['']]

    # remove empty strings from lists
    signals = [[s for s in signal if s != ''] for signal in signals]

    return signals

def get_formulas(specs):
    # regular expression matching id variable in the specs at the beginning of a line followed by :=
    # then the rest of the line
    regex = r"^\s*\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*:="

    # find all variable id in the stl_string
    formulas = re.findall(regex, specs, re.MULTILINE)
    return formulas

