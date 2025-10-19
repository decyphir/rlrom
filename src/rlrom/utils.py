from stable_baselines3 import PPO,A2C,SAC,TD3,DQN,DDPG
from sb3_contrib import TRPO, QRDQN
from huggingface_hub import HfApi
from huggingface_sb3 import load_from_hub
from huggingface_sb3.naming_schemes import EnvironmentName, ModelName, ModelRepoId
import re
import os, sys, glob
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import importlib
from datetime import datetime, date
from ruamel.yaml import YAML
import torch as th
import polars as pl

yaml = YAML(typ='safe')
# Define a representer for NumPy arrays
yaml.representer.add_representer(np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist()))

# Define a representer for NumPy floats
yaml.representer.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))


def set_rec_cfg_field(cfg, **kargs):
    def rec_set(cfg, key, value):
        for item in cfg:
            if item==key:
                cfg[item]=value
            elif isinstance(cfg[item], dict):
                cfg[item]= rec_set(cfg[item],key,value)
        return cfg    
    for item in kargs:
        cfg = rec_set(cfg,item,kargs[item])
    return cfg



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

# load cfg recursively 
def load_cfg(cfg, verbose=1):
    def recursive_load(cfg):
        exclude_load_file = ['res_file']
        for key, value in cfg.items():
            #print('reading', key, 'with value', value)
            if isinstance(value, str) and value.endswith('.yml'):
                if verbose>=1:
                    if  key not in exclude_load_file:
                        print('loading field [', key, '] from YAML file [', value, ']')
                        with open(value, 'r') as f:                        
                            cfg[key] = recursive_load(yaml.load(f))                
                    else:
                        cfg[key] = value
                else:
                    cfg[key] = value
                    print('WARNING: file', value,'not found!')
            elif isinstance(value, str) and value.endswith('.stl'):
                if verbose>=1:
                    print('loading field [', key, '] from STL file [', value, ']')            
                with open(value,'r') as F: 
                    cfg[key]= F.read()
            elif key=='import_module':                
                to_import = cfg.get('import_module')                
                if to_import is not None:
                    imported = importlib.import_module(to_import)
                    print(f'Imported module {to_import}')

            elif isinstance(value, dict):
                cfg[key]= recursive_load(value)

        return cfg
    
    if isinstance(cfg, str) and os.path.exists(cfg):        
        # here cfg is a path and we found it so we can already load it at the top level
        cfg_file = cfg
        with open(cfg_file, 'r') as f:
            cfg= yaml.load(f)

        # now we should check if it has a path where we should be         
        this_cfg_pathdir = cfg.get('this_cfg_pathdir', '')
        if this_cfg_pathdir == '':
            # no cfg_pathdir is explicitly specified. We try the folder of cfg file 
            this_cfg_pathdir, _ = os.path.split(cfg_file)
            if this_cfg_pathdir == '':  
                this_cfg_pathdir ='.'            
        
        this_cfg_pathdir = os.path.abspath(this_cfg_pathdir)
        cfg['this_cfg_pathdir'] = this_cfg_pathdir                        
    elif not isinstance(cfg, dict): 
        raise TypeError(f"Expected file name or dict.")
    else:
        this_cfg_pathdir = cfg.get('this_cfg_pathdir', '.')
    # now we have defined this_cfg_pathdir. Might be that it don't exist. We issue warning in that case
    if os.path.exists(this_cfg_pathdir):        
        os.chdir(this_cfg_pathdir)      
    else:
        print(f'WARNING: {this_cfg_pathdir} does not exist, assume current folder for working directory.')
              
    if '' not in sys.path:
        sys.path.append('')
    if '.' not in sys.path:
        sys.path.append('.')
        
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

def add_now_suffix(s):
  dd = datetime.now()
  s_dd= dd.strftime("_%Y_%m_%d")
  return s+s_dd

def policy_cfg2kargs(cfg_policy):
  act_fn =  {
    "ReLU": th.nn.ReLU,
    "Tanh": th.nn.Tanh,
    "ELU": th.nn.ELU,
  }
  if 'activation_fn' in cfg_policy:
    if isinstance(cfg_policy['activation_fn'], str):
      cfg_policy['activation_fn']= act_fn[cfg_policy['activation_fn']]
  
  return cfg_policy

def list_trained_models(folder='./models'):
    list_models = []
    with os.scandir(folder) as d:
        for e in d:
            m, ext=  os.path.splitext(e.name)
            if ext=='.yml': 
                list_models.append(m)
    return list_models

def get_sys_info():
    os.sys    

def get_training_folders(cfg):
    cfg= load_cfg(cfg)
    mpath,_ = get_model_fullpath(cfg)
    patt = os.path.splitext(mpath)[0]
    globs = glob.glob(patt+'*')
    folders = []
    for d in globs:
        if os.path.isdir(d):
            pattern = patt+r"_\d{4}_\d{2}_\d{2}__training\d+"
            if re.match(pattern, d) is not None:
                folders.append(d)
    return folders 

def get_date_num_training(cfg, training_folder):
    mpath,_ = get_model_fullpath(cfg)    
    mpath = os.path.splitext(mpath)[0]
    if mpath is not None:
        s = training_folder.removeprefix(mpath+'_')                
        ds,training  = s.split('__training')
        y, m, d = ds.split('_')
        dt = date(int(y), int(m), int(d))     
    
    return dt, int(training)


def get_df_training(cfg, idx=-1):
    dfallt= get_df_all_training_files(cfg)
    df_files_lastt = dfallt.collect()['training_files'][idx]
    return get_df_load_training_res(df_files_lastt)


def get_df_all_trainings(cfg):
    dfallt= get_df_all_training_files(cfg)
    return get_df_load_all_training_res(dfallt)


def get_df_all_training_files(cfg):
    # returns a dataframe with all non empty folders containing checkpoints models and tests
    list_folders = get_training_folders(cfg)    
    dict_trainings = dict({'date':[], 'num':[], 'training_files':[], 'path':[]})

    for fd in list_folders:             
        l = os.scandir(fd)
        steps = []
        res_files = []
        model_files = []
        for f in l:
            if f.name.startswith('res_step_'):                
                step = f.name.removesuffix('.yml').removeprefix('res_step_')
                steps.append(int(step))
                res_files.append(os.path.join(fd,f.name))
                model_files.append(f.name.replace('res','model').replace('.yml','.zip'))

        if len(steps)>0:    
            dict_cp = { 'steps':steps, 
                        'res_files':res_files,
                        'model_files':model_files,
                        'path': fd}
            df_cp = pl.LazyFrame(dict_cp)
            
            dt, num = get_date_num_training(cfg,fd)
            dict_trainings['date'].append(dt)
            dict_trainings['num'].append(num)            
            dict_trainings['training_files'].append(df_cp.sort('steps'))                        
            dict_trainings['path'].append(fd)

    df = pl.LazyFrame(dict_trainings)
    df= df.sort('date', 'num')
    
    return df

def get_df_load_training_res(df_training_files, label='Training0'):
# load res files from a df_training_files (dataframe with list of res and model files)

    def load_result_fn(p):                    
        with open(p,'r') as f:
            res = yaml.load(f)    
        return res

    def get_dtype_from_res(res):
        df_res = pl.DataFrame(res)
        res_typ = df_res['res'].dtype
        res_all_ep_typ = df_res['res_all_ep'].dtype
        typ= pl.Struct([pl.Field('res', res_typ), pl.Field('res_all_ep', res_all_ep_typ)])
        return typ

    p = df_training_files.collect()['res_files'][-1]
    res = load_result_fn(p)
    typ = get_dtype_from_res(res)
    expr = pl.col('res_files').map_elements(load_result_fn, return_dtype=typ)
    out = df_training_files.with_columns(
         expr.alias('results')
    )
    out = out.unnest('results').unnest('res_all_ep').unnest('basics').unnest('eval_formulas')

    return out.with_columns(pl.lit(label).alias('label'))

def get_df_load_all_training_res(df_all_trainings):
# load res files for trainings found by get_df_all_trainings, concat them vertically

    df_all_training_res = None
    idx =0
    for r in df_all_trainings.collect()['training_files']:
        r = get_df_load_training_res(r,f'Training{idx}')
        if df_all_training_res is None:
            df_all_training_res = r
        else:
            df_all_training_res = pl.concat([   df_all_training_res,
                                                r])
        idx = idx+1
    return df_all_training_res
        
def get_df_mean_min_max_val(df, feature):
# returns mean, min and max values for a dataframe df of (steps,feature) concat vertically with label
    expr_min =  pl.col(feature).list.min().name.suffix('_min')
    expr_max =  pl.col(feature).list.max().name.suffix('_max')
    expr_mean = pl.col(feature).list.mean().name.suffix('_mean')

    df = df.select('label','steps',feature)
    df_enveloppe = df.group_by(pl.col('steps')).agg(pl.col(feature)
                            ).sort(pl.col('steps'))                        
    df_enveloppe = df_enveloppe.collect().select('steps',expr_mean, expr_min, expr_max)

    
    return df_enveloppe
