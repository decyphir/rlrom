from stable_baselines3 import PPO,A2C,SAC,TD3,DQN,DDPG
from morl_baselines.single_policy.ser.mo_ppo import MOPPO
from morl_baselines.single_policy.ser.nl_mo_ppo import NLMOPPO
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning
from morl_baselines.single_policy.ser.mosac_continuous_action import MOSAC
from morl_baselines.single_policy.ser.mosac_discrete_action import MOSACDiscrete
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning
from morl_baselines.multi_policy.morld.morld import MORLD
from morl_baselines.multi_policy.ipro.ipro_2d import IPRO2D
from morl_baselines.multi_policy.ipro.ipro import IPRO
from morl_baselines.multi_policy.capql.capql import CAPQL
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPILSContinuousAction
from morl_baselines.multi_policy.gpi_ls_jax.gpi_ls_continuous_action_jax import GPILSContinuousAction as GPILSContinuousActionJax
import sbx
from rlrom.extra_algos.tabular_q_learning import TabularQLearning
from sb3_contrib import TRPO, QRDQN, CrossQ
from huggingface_hub import HfApi
from huggingface_sb3 import load_from_hub
from huggingface_sb3.naming_schemes import EnvironmentName, ModelName, ModelRepoId
import re
import os, sys, glob, shutil
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import importlib
from datetime import datetime, date
from ruamel.yaml import YAML
import torch as th
import polars as pl
import pandas as pd

ALGO_NAMES_CLASSES = {
    # Stable Baselines 3 (single objective)
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "dqn": DQN,
    "ddpg": DDPG,
    # SBX variants (Stable Baselines Jax)
    "sbx_ppo": sbx.PPO,
    "sbx_sac": sbx.SAC,
    "sbx_td3": sbx.TD3,
    "sbx_dqn": sbx.DQN,
    "sbx_ddpg": sbx.DDPG,
    "sbx_crossq": sbx.CrossQ,
    # sb3-contrib (more RL algos)
    "trpo": TRPO,
    "crossq": CrossQ,
    "qrdqn": QRDQN,
    # Other algos
    "qlearning": TabularQLearning,
    # MORL single-policy
    "moppo": MOPPO,
    "nlmoppo": NLMOPPO,
    "moql": MOQLearning,
    "mosac": MOSAC,
    "mosac_discrete": MOSACDiscrete,
    "eupg": EUPG,
    # MORL multi-policy
    "pcn": PCN,
    "pgmorl": PGMORL,
    "pql": PQL,
    "mpmoql": MPMOQLearning,
    "morld": MORLD,
    "ipro2d": IPRO2D,
    "ipro": IPRO,
    "capql": CAPQL,
    "envelope": Envelope,
    "gpipd": GPIPD,
    "gpils": GPILS,
    "gpipd_continuous_action": GPIPDContinuousAction,
    "gpils_continuous_action": GPILSContinuousAction,
    "gpils_continuous_action_jax": GPILSContinuousActionJax
}

yaml = YAML(typ='safe')
# Define a representer for NumPy arrays
yaml.representer.add_representer(np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist()))

# Define a representer for NumPy floats
yaml.representer.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))

def disp_df_metric(df, m): 
    metric_values = df[m].to_numpy()
    print(m+':', metric_values)

def disp_df_formula_metrics(df, f):
    if isinstance(df[f].iloc[0], dict):
        dff = pd.DataFrame(df[f].tolist(), index=df.index)
    else:
        dff = df[[f]]
    print(f)
    metrics = dff.columns
    
    for m in metrics:
        print(' - ', end='')
        disp_df_metric(dff,m)


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
            for rl_algo_name, rl_algo_class in ALGO_NAMES_CLASSES.items():
                try:
                    model = rl_algo_class.load(filename)
                    print(f"loading {rl_algo_name.upper()} model succeeded")
                    return model
                except:
                    print(f"loading {rl_algo_name.upper()} model failed")
        except FileNotFoundError:
            print("File not found",filename)            

    if repo_id is not None:
        model = load_rl_model(env_name, repo_id, filename=filename,
                              custom_objects={
                                "learning_rate": 0.0,
                                "lr_schedule": lambda _: 0.0,
                                "clip_range": lambda _: 0.0,
                              } if "ppo" in repo_id else None)
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

# Auxiliary load function
def load_rl_model(env_name, repo_id, filename=None, custom_objects=None):
    algo_name = None
    for name in ALGO_NAMES_CLASSES:
        if name in repo_id:
            algo_name = name
            break
    if algo_name is None:
        return None
    if filename is None:
        filename = ModelName(algo_name, env_name)+'.zip'
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    algo_class = ALGO_NAMES_CLASSES[algo_name]
    model = algo_class.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
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

def list_training_folders(cfg):
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

#######################
### DataFraming results 
def get_training_folders(cfg, verbose=0):
    # returns a dataframe with all non empty folders containing checkpoints models and tests
    
    list_folders = list_training_folders(cfg)    
    dict_trainings = dict({'best_reward':[], 'date':[], 'num':[], 'training_files':[], 'path':[]})
    if list_folders:
        print('Scanning models in ',os.path.dirname(list_folders[0]))
    else:
        mpath,_ = get_model_fullpath(cfg)
        print('No model found in ', os.path.dirname(mpath))
    
    for fd in list_folders:             
        l = os.scandir(fd)
        steps = []
        res_files = []
        model_files = []
        best_reward = 0
        for f in l:
            if f.name.startswith('res_step_'):
                step = f.name.removesuffix('.yml').removeprefix('res_step_')
                with open(f.path, 'r') as fn:
                        res = yaml.load(fn)
                        mean_reward  = res['res_all_ep']['basics']['mean_ep_rew']
                        if mean_reward>best_reward:
                            best_reward=mean_reward   
                                
                steps.append(int(step))
                res_files.append(os.path.join(fd,f.name))
                model_files.append(f.name.replace('res','model').replace('.yml','.zip'))
                
        if len(steps)>0:    
            dict_cp = {
                'path': fd,
                'steps':steps, 
                'res_files':res_files,
                'model_files':model_files
            }
            df_cp = pd.DataFrame(dict_cp).sort_values('steps').reset_index(drop=True)
            dt, num = get_date_num_training(cfg,fd)
            dict_trainings['date'].append(dt)
            dict_trainings['num'].append(num)            
            dict_trainings['best_reward'].append(best_reward)
            dict_trainings['training_files'].append(df_cp)                        
            dict_trainings['path'].append(fd)
            if verbose>0:
                print(f'{os.path.basename(fd)}: {best_reward}')                            

    df = pd.DataFrame(dict_trainings)
    if not df.empty:
        df = df.sort_values(['date', 'num']).reset_index(drop=True)

    return df

def get_training_res(training_folders, training_idx=-1):
# load res files from a training_folders (dataframe with list of res and model files)

    if isinstance(training_folders, dict):
        cfg = training_folders
        training_folders = get_training_folders(cfg)

    def load_result_fn(p):                    
        with open(p, 'r') as f:
            res = yaml.load(f)    
        return res

    if training_idx == 'all': # load all of them 
        training_idx = training_folders.index

    if np.isscalar(training_idx):
        out = training_folders.iloc[training_idx]['training_files']
        results = pd.DataFrame(out['res_files'].apply(load_result_fn).tolist(), index=out.index)

        if 'res_all_ep' in results.columns:
            res_all_ep = pd.DataFrame(results['res_all_ep'].tolist(), index=out.index)
            results = results.drop(columns=['res_all_ep']).join(res_all_ep)

        if 'basics' in results.columns:
            basics = pd.DataFrame(results['basics'].tolist(), index=out.index)
            results = results.drop(columns=['basics']).join(basics)

        out = out.join(results)
        out['training_idx'] = training_idx
    else:
        out_list = []
        for idx in training_idx:
            out_list.append(get_training_res(training_folders,idx))
        out = pd.concat(out_list, ignore_index=True)

    return out


def get_df_mean_min_max_val(df, feature):
    df_envelope = df.groupby('steps')[feature].agg(
        **{
            f'{feature}_mean': 'mean',
            f'{feature}_min': 'min',
            f'{feature}_max': 'max'
        }
    ).reset_index().sort_values('steps').reset_index(drop=True)
    
    return df_envelope


def get_best_models(cfg,train_idx=-1):
    if isinstance(cfg, dict):
        df = get_training_res(cfg, train_idx)
    else:
        df=cfg
    dff = df[df["mean_ep_rew"] == df["mean_ep_rew"].max()].sort_values("steps")

    dff['mdl_path'] = dff['path']+'/'+dff["model_files"]
    dff['cfg_path'] = dff["path"] + "/cfg0.yml"
    mdl_file = dff['mdl_path'].iloc[0]
    cfg_file = dff['cfg_path'].iloc[0]
        
    return cfg_file, mdl_file, dff[['steps', 'mean_ep_rew', 'mdl_path', 'cfg_path']]

def get_training_cfg_path(cfg, train_idx=-1):
    df = get_training_res(cfg, train_idx)    
    dff = df[df["mean_ep_rew"] == df["mean_ep_rew"].max()].sort_values("steps")
    cfg_path = (dff["path"] + "/cfg0.yml").iloc[0]
    return cfg_path

def get_step_model(cfg,step, train_idx=-1):
    if isinstance(cfg, dict):
        df = get_training_res(cfg, train_idx)
    else:
        df=cfg
    dff = df[df['steps'] > step].sort_values('steps')
    
    mdl_file = (dff['path'] + '/' + dff['model_files']).iloc[0]
    cfg_file = (dff["path"] + "/cfg0.yml").iloc[0]

    return cfg_file, mdl_file
    
def set_active_model(cfg, cfg_file, mdl_file):
    mdl_path, cfg_path  = get_model_fullpath(cfg)

    target = mdl_file
    link = mdl_path
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)
    
    target = cfg_file
    link = cfg_path

    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)
    show_active_model(cfg)

def show_active_model(cfg):
    mdl_path, cfg_path  = get_model_fullpath(cfg)
    p = os.path.realpath(mdl_path)
    filename = os.path.basename(p)
    top_folder = os.path.basename(os.path.dirname(p))
    print(f"{top_folder}/{filename}")
    return p

## DEPRECATED
def get_df_all_trainings(cfg):
    dfallt= get_training_folders(cfg)
    return get_df_all_training_res(dfallt)

def get_df_all_training_res(df_all_trainings, select = None):
# load res files for trainings found by get_df_all_trainings, concat them vertically

    # In case inconsistent result structure prevent concat:
    # safe_select = ['label', 'steps', 'mean_ep_rew', 'mean_ep_len', 'res', 'res_files', 'model_files', 'path']
    
    res_list = []
    for idx, r in enumerate(df_all_trainings['training_files']):
        df_res = get_training_res(r, f'Training{idx}')
        if select is not None:
            cols = [c for c in select if c in df_res.columns]
            df_res = df_res[cols]
        res_list.append(df_res)
    if not res_list:
        return pd.DataFrame()
    return pd.concat(res_list, ignore_index=True)
    