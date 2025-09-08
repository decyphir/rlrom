from stable_baselines3 import PPO,A2C,SAC,TD3,DQN,DDPG
from sb3_contrib import TRPO, QRDQN
from huggingface_hub import HfApi
from huggingface_sb3 import load_from_hub
from huggingface_sb3.naming_schemes import EnvironmentName, ModelName, ModelRepoId
import re
import yaml
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# helper function to concat new values in a dict field array
def append_to_field_array(res, metric, val):
    vals = res.get(metric,None)
    if vals is None:
        res[metric]=np.array([val])
    else:
        vals = np.atleast_1d(vals)
        vals = np.append(vals,val)
        res[metric]= vals
    return res

def list_folders(folder, filter=''):
    try:
        # List all items in the given directory
        items = os.listdir(folder)
        # Filter out only the directories
        folders = [os.path.join(folder, item) for item in items 
                   if (os.path.isdir(os.path.join(folder, item)) and
                       filter in item)]
        
        return folders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def tb_extract_from_tag(file_path_list, tag='rollout/ep_rew_mean'):
# return a list of data dict with fields steps and values        

    if not(isinstance(file_path_list,list)):
        file_path_list = [file_path_list]
    
    all_data = []
    for file_path in file_path_list:
        l = os.listdir(file_path)
        event_file = file_path+'/'+l[0]    
    
        # Initialize the event accumulator
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        # Extract scalar values for the specified tag        
        data = dict()
        if tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(tag)
            data["steps"] = [event.step for event in scalar_events]
            data["values"] = [event.value for event in scalar_events]            
        
        all_data.append(data)

    return all_data    

def get_upper_values(all_data):
    # Assumes all_data in sync (i.e. same steps)

    all_values = []
    for v in all_data:
        all_values.append(v.get('values'))

    return np.max(all_values,axis=0)        

def get_lower_values(all_data):
# Assumes all_data in sync (i.e. same steps)

    all_values = []
    for v in all_data:
        all_values.append(v.get('values'))

    return np.min(all_values, axis=0)        

def get_mean_values(all_data):
    # Assumes all_data in sync (i.e. same steps)

    all_values = []
    for v in all_data:
        all_values.append(v.get('values'))

    return np.mean(all_values,axis=0)        

# load cfg recursively 
def load_cfg(cfg, verbose=1):
    def recursive_load(cfg):
        for key, value in cfg.items():
            #print('reading', key, 'with value', value)
            if isinstance(value, str) and value.endswith('.yml'):
                if verbose>=1:
                    print('loading field [', key, '] from YAML file [', value, ']')
                    with open(value, 'r') as f:                        
                        cfg[key] = recursive_load(yaml.safe_load(f))                
                else:
                    cfg[key] = value
                    print('WARNING: file', value,'not found!')
            elif isinstance(value, str) and value.endswith('.stl'):
                if verbose>=1:
                    print('loading field [', key, '] from STL file [', value, ']')            
                with open(value,'r') as F: 
                    cfg[key]= F.read()
            elif isinstance(value, dict):
                cfg[key]= recursive_load(value)

        return cfg

    if isinstance(cfg, str) and os.path.exists(cfg):
        with open(cfg, 'r') as f:
            cfg= yaml.safe_load(f)
    elif not isinstance(cfg, dict): 
        raise TypeError(f"Expected file name or dict.")
    
    return recursive_load(cfg)

def get_model_fullpath(cfg):
    # returns absolute path for model, as well as for yaml config (may not exist yet)
    # The yml file (second output), if it exists, contains the full configuration used 
    # to train the model
    model_path = cfg.get('model_path', './models')        
    model_name = cfg.get('model_name', 'random')
    full_path = os.path.join(model_path, model_name+'.zip')
    
    if model_name=='random':
        if os.path.exists(full_path):
            print(f"WARNING: Somehow a model was named 'random' (path: {full_path}). Rename it if you actually want to use it.")        
        full_path = 'random'
        cfg_full_path = None
    else:
        os.makedirs(model_path, exist_ok=True) # creates folder for model(s) if it does not exist
        full_path= os.path.abspath(full_path)
        cfg_full_path = full_path.replace('.zip', '.yml')
    
    return full_path, cfg_full_path

def find_model(env_name):
    return find_huggingface_models(env_name)

def find_huggingface_models(env_name, repo_contains='', algo=''):
    api = HfApi()
    models_iter = api.list_models(tags=env_name)
    
    models = []
    models_ids = []
    for model in models_iter:
        model_id = model.modelId
        if repo_contains.lower() in model_id.lower() and algo.lower() in model_id.lower(): 
            models.append(model)
            models_ids.append(model_id)
   
    return models, [model.modelId for model in models]

def load_model(env_name, repo_id=None, filename=None):

    if repo_id is None and filename is None:
        filename=env_name

    model = None
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
            print("File not found",filename)            

    if repo_id is not None:
        if 'ppo' in repo_id:
            model = load_ppo_model(env_name, repo_id, filename=filename)
        elif 'a2c' in repo_id:
            model = load_a2c_model(env_name, repo_id, filename=filename)
        elif 'sac' in repo_id:
            model = load_sac_model(env_name, repo_id, filename=filename)
        elif 'td3' in repo_id:
            model = load_td3_model(env_name, repo_id, filename=filename)
        elif 'dqn' in repo_id:
            model = load_dqn_model(env_name, repo_id, filename=filename)
        elif 'qrdqn' in repo_id:
            model = load_qrdqn_model(env_name, repo_id, filename=filename)
        elif 'ddpg' in repo_id:
            model = load_ddpg_model(env_name, repo_id, filename=filename)
        elif 'trpo' in repo_id:
            model = load_trpo_model(env_name, repo_id, filename=filename)
        else:
            model = None
    return model

def get_episodes_from_rollout(buffer):
# Takes a rollout buffer as produced by PPO and returns a list of complete episodes     
    episodes = []
    for env_idx in range(buffer.n_envs):
        env_dones = buffer.episode_starts[:, env_idx] # dones flags for each steps

        sz = np.shape(buffer.observations)        
        if sz[0]==buffer.buffer_size:
            env_obs = buffer.observations[:, env_idx]  # All steps for this env
        else:
            start_idx_env = env_idx*buffer.buffer_size 
            end_idx_env = env_idx*buffer.buffer_size + buffer.buffer_size
            env_obs = buffer.observations[start_idx_env:end_idx_env]  # All steps for this env
        
        sz = np.shape(buffer.actions)        
        if sz[0]==buffer.buffer_size:            
            env_actions = buffer.actions[:, env_idx]  # All actions for this env            
        else:
            start_idx_env = env_idx*buffer.buffer_size 
            end_idx_env = env_idx*buffer.buffer_size + buffer.buffer_size
            env_actions = buffer.actions[start_idx_env:end_idx_env]  # All actions for this env

        sz = np.shape(buffer.rewards)        
        if sz[0]==buffer.buffer_size:            
            env_rewards = buffer.rewards[:, env_idx]  # All rewards for this env            
        else:
            start_idx_env = env_idx*buffer.buffer_size 
            end_idx_env = env_idx*buffer.buffer_size + buffer.buffer_size
            env_rewards = buffer.rewards[start_idx_env:end_idx_env]  # All actions for this env
            
        
        # Split into episodes based on done flags
        episode_start = 0                
        for step in range(len(env_dones)):
            episode = dict()
            if env_dones[step] and step > 0:  # found episode boundary
                episode['observations'] = env_obs[episode_start:step]                
                episode['actions'] = env_actions[episode_start:step]                
                episode['rewards'] = env_rewards[episode_start:step]                
                episode['dones'] = env_dones[episode_start:step]                
                episodes.append(episode)
                episode_start = step                        
        # note we only want complete episodes, so we drop the last observations for each batch, if they don't end with a done

    return episodes

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

def get_symmetric_max(sig):
    npsig= np.array(sig)
    max_pos = npsig.max()
    min_neg = -npsig.min()
    return max(max_pos,min_neg)

# Auxiliary load functions        
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
