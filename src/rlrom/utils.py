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

def _res_from_training_df(training_df):
    """Load and expand the res files of a single training.

    training_df: DataFrame with at least the columns 'steps' and 'res_files'
    (as found in the 'training_files' column of get_training_folders()), one
    row per checkpoint evaluation. Returns a DataFrame with the same rows
    plus the res file fields expanded: 'res' (dict), 'eval_formulas' (dict),
    'reward_formulas' (dict) and the 'basics' fields flattened (e.g.
    'mean_ep_rew'). A 'training_idx' label (training folder name) is added
    when a 'path' column is present.
    """
    def load_result_fn(p):
        with open(p, 'r') as f:
            return yaml.load(f)

    results = pd.DataFrame(training_df['res_files'].apply(load_result_fn).tolist(), index=training_df.index)

    if 'res_all_ep' in results.columns:
        res_all_ep = pd.DataFrame(results['res_all_ep'].tolist(), index=training_df.index)
        results = results.drop(columns=['res_all_ep']).join(res_all_ep)

    if 'basics' in results.columns:
        basics = pd.DataFrame(results['basics'].tolist(), index=training_df.index)
        results = results.drop(columns=['basics']).join(basics)

    out = training_df.join(results)
    if 'path' in training_df.columns:
        out['training_idx'] = os.path.basename(str(training_df['path'].iloc[0]))
    return out


def get_training_res(training_folders, training_idx=-1):
    """Load res files for one or several trainings, as a flat DataFrame.

    training_folders: a cfg dict (resolved through get_training_folders()),
    the DataFrame returned by get_training_folders(), or a single-training
    DataFrame (columns 'steps', 'res_files', ...).
    training_idx: an integer position (negative allowed), a list of
    positions, or 'all'. Ignored when a single-training DataFrame is given.
    """
    if isinstance(training_folders, dict):
        training_folders = get_training_folders(training_folders)

    # single training already selected
    if 'training_files' not in training_folders.columns:
        return _res_from_training_df(training_folders)

    if training_idx == 'all':
        idxs = list(training_folders.index)
    elif np.isscalar(training_idx):
        idxs = [training_idx]
    else:
        idxs = list(training_idx)

    out_list = []
    for i in idxs:
        out_list.append(_res_from_training_df(training_folders.iloc[i]['training_files']))
    return pd.concat(out_list, ignore_index=True)


def get_df_mean_min_max_val(df, feature):
    df_envelope = df.groupby('steps')[feature].agg(
        **{
            f'{feature}_mean': 'mean',
            f'{feature}_min': 'min',
            f'{feature}_max': 'max'
        }
    ).reset_index().sort_values('steps').reset_index(drop=True)
    
    return df_envelope


def add_model_paths(df):
    """Return a copy of a training results DataFrame with 'mdl_path' and
    'cfg_path' columns added (the checkpoint model file and the cfg0.yml of
    the training folder, for each row)."""
    df = df.copy()
    df['mdl_path'] = df['path'] + '/' + df['model_files']
    df['cfg_path'] = df['path'] + '/cfg0.yml'
    return df


def get_training_num(folder):
    """Return the integer training number of a training folder name or path
    (e.g. 'ppo_specs_2026_09_30__training12' -> 12), or None if it cannot be
    parsed."""
    m = re.search(r'__training(\d+)$', os.path.basename(str(folder)))
    return int(m.group(1)) if m else None


def short_model_path(p):
    """Return 'model_folder/model_name.ext' for a file path: the basename of
    the containing folder plus the basename of the file."""
    d, f = os.path.split(str(p))
    return os.path.basename(d) + '/' + f


def get_selection_df(cfg, trainings='last'):
    """Return the flat results DataFrame for a pool of trainings.

    cfg: a cfg dict, the DataFrame returned by get_training_folders(), or a
    results DataFrame (passed through unchanged).
    trainings: 'last' (most recent training), 'all', or a list of integer
    positions into the (date, num) sorted list of training folders.
    """
    if isinstance(cfg, dict):
        cfg = get_training_folders(cfg)
    if 'training_files' not in cfg.columns:
        return cfg  # already a results DataFrame
    if trainings == 'all':
        return get_df_all_training_res(cfg)
    if trainings == 'last':
        trainings = -1
    return get_training_res(cfg, trainings)


def get_model_ranking(df, metric='mean_ep_rew', mode='max', top=5):
    """Return the top checkpoint rows of a results DataFrame ranked by
    `metric` (see get_metric_series; mode 'max' or 'min'), with 'mdl_path'
    and 'cfg_path' columns added."""
    s = get_metric_series(df, metric)
    order = s.sort_values(ascending=(mode == 'min'), na_position='last').head(top).index
    return add_model_paths(df.loc[order])


def get_best_models(cfg, train_idx=-1, *, metric='mean_ep_rew', mode='max'):
    """Select the checkpoint with the best (or worst) value of a metric.

    cfg: a cfg dict (train_idx then selects the training pool, as in
    get_training_res) or a results DataFrame (train_idx ignored).
    metric: a flat name or 'formula/metric' pair (see get_metric_series).
    mode: 'max' (default) selects the highest value, 'min' the lowest.
    Ties are broken by the lowest steps (earliest checkpoint to reach the
    value). Rows with no value for the metric are ignored.

    Returns (cfg_file, mdl_file, dff) where dff is the DataFrame of the
    candidate rows (ties), with 'mdl_path' and 'cfg_path' columns added.
    """
    if mode not in ('max', 'min'):
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")
    if isinstance(cfg, dict):
        df = get_training_res(cfg, train_idx)
    else:
        df = cfg
    values = get_metric_series(df, metric).dropna()
    if values.empty:
        raise ValueError(f"metric '{metric}' has no value in the selected trainings")
    extremum = values.max() if mode == 'max' else values.min()
    dff = add_model_paths(df.loc[values[values == extremum].index]).sort_values('steps')
    mdl_file = dff['mdl_path'].iloc[0]
    cfg_file = dff['cfg_path'].iloc[0]
    return cfg_file, mdl_file, dff


def get_training_cfg_path(cfg, train_idx=-1):
    cfg_path, _, _ = get_best_models(cfg, train_idx)
    return cfg_path

def get_step_model(cfg, step, train_idx=-1):
    """Select the first checkpoint with steps > step. Same input handling as
    get_best_models. Returns (cfg_file, mdl_file)."""
    if isinstance(cfg, dict):
        df = get_training_res(cfg, train_idx)
    else:
        df = cfg
    dff = add_model_paths(df[df['steps'] > step]).sort_values('steps')
    if dff.empty:
        raise ValueError(f"no checkpoint with steps > {step}")
    return dff['cfg_path'].iloc[0], dff['mdl_path'].iloc[0]
    
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

def get_df_all_trainings(cfg):
    """Load the res files of all trainings found for cfg, vertically
    concatenated. See get_df_all_training_res()."""
    return get_df_all_training_res(get_training_folders(cfg))


def get_df_all_training_res(df_all_trainings, select=None):
    """Load the res files of the trainings in df_all_trainings (output of
    get_training_folders()), vertically concatenated. Each row is one
    checkpoint evaluation of one training, labeled by 'training_idx' (the
    training folder name)."""
    res_list = []
    for r in df_all_trainings['training_files']:
        df_res = get_training_res(r)
        if select is not None:
            cols = [c for c in select if c in df_res.columns]
            df_res = df_res[cols]
        res_list.append(df_res)
    if not res_list:
        return pd.DataFrame()
    return pd.concat(res_list, ignore_index=True)


_NON_METRIC_COLS = ('path', 'steps', 'res_files', 'model_files', 'res',
                    'eval_formulas', 'reward_formulas', 'training_idx')


def list_available_metrics(df):
    """List the metric names plottable from a training results DataFrame:
    the flat metric columns plus the 'formula/metric' pairs found in the
    'eval_formulas' and 'reward_formulas' dict columns."""
    metrics = [c for c in df.columns if c not in _NON_METRIC_COLS]
    for col in ('eval_formulas', 'reward_formulas'):
        if col in df.columns:
            for row in df[col]:
                if isinstance(row, dict):
                    for f, fm in row.items():
                        if isinstance(fm, dict):
                            metrics.extend(f'{f}/{m}' for m in fm)
    return sorted(set(metrics))


def get_metric_series(df, metric):
    """Return a numeric Series with the values of `metric` for each row of df.

    `metric` is either a flat column name (e.g. 'mean_ep_rew') or a
    'formula/metric' pair looked up in the 'eval_formulas' and
    'reward_formulas' dict columns (e.g. 'phi_left_goal/ratio_init_sat').
    """
    if metric in df.columns:
        return pd.to_numeric(df[metric], errors='coerce')

    if '/' in metric:
        formula, submetric = metric.split('/', 1)
        vals = []
        for _, row in df.iterrows():
            v = np.nan
            for col in ('eval_formulas', 'reward_formulas'):
                if col not in row.index:
                    continue
                d = row[col]
                if isinstance(d, dict):
                    fd = d.get(formula)
                    if isinstance(fd, dict) and submetric in fd:
                        v = fd[submetric]
                        break
            vals.append(v)
        return pd.Series(vals, index=df.index, dtype=float)

    raise KeyError(f"metric '{metric}' not found. Available metrics: {list_available_metrics(df)}")
    
# Re-export lightweight config helpers so existing callers of
# ``utils.load_cfg`` / ``utils.set_rec_cfg_field`` keep working.
# The implementations now live in :mod:`rlrom.cfg_utils` to avoid
# importing the heavy dependency stack for config-only operations.
from rlrom.cfg_utils import load_cfg, set_rec_cfg_field
