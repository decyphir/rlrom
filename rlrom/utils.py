from stable_baselines3 import PPO,A2C,SAC,TD3,DQN,DDPG
from sb3_contrib import TRPO, QRDQN
from huggingface_hub import HfApi
from huggingface_sb3 import load_from_hub
from huggingface_sb3.naming_schemes import EnvironmentName, ModelName, ModelRepoId
import re
import yaml
import os

# load cfg recursively 
def load_cfg(cfg, verbose=1):
    def recursive_load(cfg):
        for key, value in cfg.items():
            if key.startswith('cfg_') and isinstance(value, str) and value.endswith('.yml'):
                if verbose>=1:
                    print('loading field [', key, '] from file [', value, ']')
                if os.path.exists(value):
                    with open(value, 'r') as f:                        
                        cfg[key] = recursive_load(yaml.safe_load(f))
                else:
                    cfg[key] = value
                    print('WARNING: file', value,'not found!')
        return cfg

    if isinstance(cfg, str) and os.path.exists(cfg):
        with open(cfg, 'r') as f:
            cfg= yaml.safe_load(f)
    elif not isinstance(cfg, dict): 
        raise TypeError(f"Expected file name or dict.")
    
    return recursive_load(cfg)


def get_model_fullpath(cfg):
    # returns absolute path for model, as well as for yaml config (may not exist yet)
    model_path = cfg['cfg_train'].get('model_path', './models')
    model_name = cfg['cfg_train'].get('model_name', 'ppo_model')
    full_path = os.path.join(model_path, model_name+'.zip')

    if not os.path.exists(full_path):
        print(f"WARNING: Path does not exist: {full_path}")
    else:
        full_path= os.path.abspath(full_path)
    
    return full_path, full_path.replace('.zip', '.yml')


        
def load_ppo_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('ppo', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
    }
    model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    return model

def load_a2c_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('a2c', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = A2C.load(checkpoint, print_system_info=True)
    return model

def load_sac_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('sac', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = SAC.load(checkpoint, print_system_info=True)
    return model

def load_td3_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('td3', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = TD3.load(checkpoint, print_system_info=True)
    return model

def load_dqn_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('dqn', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = DQN.load(checkpoint, print_system_info=True)
    return model

def load_qrdqn_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('dqn', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)  
    model = QRDQN.load(checkpoint, print_system_info=True)
    return model

def load_ddpg_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('ddpg', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    model = DDPG.load(checkpoint, print_system_info=True)
    return model

def load_trpo_model(env_name, repo_id, filename=None):
    if filename is None:
        filename = ModelName('trpo', env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    model = TRPO.load(checkpoint, print_system_info=True)
    return model

def load_model(env_name, repo_id=None, filename=None):

    # checks if filename point to a valid file
    if filename is not None:
        try:
            with open(filename, 'r') as f:
                pass
            
            # try loading with PPO, A2C, SAC, TD3, DQN, QRDQN, DDPG, TRPO
            try:
                model= PPO.load(filename)
                print("loading PPO model succeeded")
                return model
            except:
                print("loading PPO model failed")
                pass

            try:
                model= A2C.load(filename)
                print("loading A2C model succeeded")
                return model
            except:
                print("loading A2C model failed")
                pass

            try:
                model= SAC.load(filename)
                print("loading SAC model succeeded")
                return model
            except:
                print("loading SAC model failed")                
                pass

            try:
                model= TD3.load(filename)
                print("loading TD3 model succeeded")
                return model                                
            except:
                print("loading TD3 model failed")
                pass

            try:                
                model= DQN.load(filename)
                print("loading DQN model succeeded")
                return model
            except:
                print("loading DQN model failed")
                pass    

            try:
                model= QRDQN.load(filename)
                print("loading QRDQN model succeeded")
                return model
            except:
                print("loading QRDQN model failed")
                pass

            try:
                model= DDPG.load(filename)
                print("loading DDPG model succeeded")
                return model                    
            except:
                print("loading DDPG model failed")
                pass

            try:
                model= TRPO.load(filename)
                print("loading TRPO model succeeded")
                return model
            except:
                print("loading TRPO model failed")
                pass            

        except FileNotFoundError:
            return None


    if 'ppo' in repo_id:
        return load_ppo_model(env_name, repo_id, filename=filename)
    elif 'a2c' in repo_id:
        return load_a2c_model(env_name, repo_id, filename=filename)
    elif 'sac' in repo_id:
        return load_sac_model(env_name, repo_id, filename=filename)
    elif 'td3' in repo_id:
        return load_td3_model(env_name, repo_id, filename=filename)
    elif 'dqn' in repo_id:
        return load_dqn_model(env_name, repo_id, filename=filename)
    elif 'qrdqn' in repo_id:
        return load_qrdqn_model(env_name, repo_id, filename=filename)
    elif 'ddpg' in repo_id:
        return load_ddpg_model(env_name, repo_id, filename=filename)
    elif 'trpo' in repo_id:
        return load_trpo_model(env_name, repo_id, filename=filename)
    else:
        return None

def find_models(env_name):
    api = HfApi()
    #models_iter = api.list_models(filter=ModelFilter(tags=env_name))
    models_iter = api.list_models(tags=env_name)
    models = list(models_iter)
    return models, [model.modelId for model in models]


def get_layout_from_string(signals_layout):
    out = []
    # split signals string wrt linebreaks first
    signals_rows = signals_layout.splitlines()

    # then strip and split wrt commas
    for line in signals_rows:        
        if line.strip() == '':
            continue
        else:
            out_row = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b(?:\([^)]*\))?', line)            
            out.append(out_row)        

    return out            

def parse_signal_spec(signal):
    # extract sig_name and args from signal_name(args)
    signal = signal.split('(')
    sig_name = signal[0]
    if len(signal) == 1:
        args = []
    else:
        args = [arg.strip() for arg in signal[1][:-1].split(',')]
    return sig_name, args


def get_formulas(specs):
    # regular expression matching id variable in the specs at the beginning of a line followed by :=
    # then the rest of the line
    regex = r"^\s*\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*:="

    # find all variable id in the stl_string
    formulas = re.findall(regex, specs, re.MULTILINE)
    return formulas

def parse_integer_set_spec(str):
    sp_str = str.split(',')
    idx_out = []
    for s in sp_str:
        s = s.strip()
        if s.isdigit():
            idx_out.append(int(s))
        else:
            [l, h] = s.split(':')
            if l.isdigit() and h.isdigit():
                range_idx = [ idx for idx in range(int(l),int(h)+1) ]
                idx_out = idx_out+range_idx
    return idx_out                
