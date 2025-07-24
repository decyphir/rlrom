import gymnasium as gym
from gymnasium.utils.save_video import save_video
from gymnasium.spaces.utils import flatten_space
import numpy as np
import pandas as pd
import stlrom
import rlrom.envs as envs
import rlrom.utils as utils
from rlrom.wrappers.stl_wrapper import STLWrapper

from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
# itertools handles the cycling
import itertools



def stl_wrap_env(env, cfg_specs):
    driver= stlrom.STLDriver()
    stl_specs_str = cfg_specs['specs']
    driver.parse_string(stl_specs_str)
    obs_formulas = cfg_specs.get('obs_formulas',{})        
    reward_formulas = cfg_specs.get('reward_formulas',{})
    eval_formulas = cfg_specs.get('eval_formulas',{})
    end_formulas = cfg_specs.get('end_formulas',{})
    BigM = cfg_specs.get('BigM')

    env = STLWrapper(env,driver,
                     signals_map=cfg_specs, 
                     obs_formulas = obs_formulas,
                     reward_formulas = reward_formulas,
                     eval_formulas=eval_formulas,
                     end_formulas=end_formulas,
                     BigM=BigM)
    return env

def make_env_generic(cfg, render_mode='human'):

    env_name = cfg.get('env_name','highway-v0')                   
    env = gym.make(env_name, render_mode=render_mode)
    
    cfg_env = cfg.get('cfg_env',dict())
    if cfg_env != dict():
        env.unwrapped.configure(cfg_env)
      # wrap env with stl_wrapper. We'll have to check if not done already        
    cfg_specs = cfg.get('cfg_specs', None)
            
    if cfg_specs is not None:
        env = stl_wrap_env(env, cfg_specs)
            
    return env


class RLTester:
    def __init__(self,cfg, render_mode='human'):
        self.cfg = cfg        
        self.manual_control = True    
        self.env_name = cfg.get('env_name','highway-v0')                                
        self.env = None
        self.model = None
        self.test_results = []
        self.has_stl_wrapper = cfg.get('cfg_specs', None) is not None

    def load_model(self):
        cfg_env = self.cfg.get('cfg_env',dict())
        if cfg_env.get('manual_control', False):
            model = 'manual'
            print("INFO: manual_control set to True, stay alert.")            
        else:                                       
            model_name, _ = utils.get_model_fullpath(self.cfg)
            print("INFO: Loading model ", model_name)
            model= utils.load_model(model_name)
            self.manual_control = False
        self.model = model

    def _get_action(self, obs):        
        if self.has_stl_wrapper is not True:   # agent was trained without stl_wrapper, so we need to use wrapped_obs to predict action
            obs = self.env.wrapped_obs
        
        if self.manual_control is True:
            action = self.env.action_space.sample()
        else:
            action, _ = self.model.predict(obs)
        return action

    def init_env(self, render_mode=None):
        self.env = make_env_generic(self.cfg, render_mode=render_mode)

    def run_seed(self, seed=None, num_steps=100):

        # We actually might want to reload every time to en    
        # self.load_model()

        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()        
        for _ in range(num_steps):    
            action = self._get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)    
            if terminated:
                print('Terminated. Crash?')
                break    
        self.env.close()
        return 

    def run_cfg_test(self):
        cfg_test = self.cfg.get('cfg_test') 
        test_result = dict({'cfg':self.cfg})                        
        if cfg_test is not None:
            init_seed = cfg_test.get('init_seed',0)
            num_ep = cfg_test.get('num_ep',1)
            render = cfg_test.get('render', True)
            num_steps  = cfg_test.get('num_steps', 100)
            
            if render:
                render_mode = 'human'
            else:            
                render_mode = None
                cfg_env = self.cfg.get('cfg_env',dict())
                if cfg_env.get('manual_control', False):                    
                    print('WARNING: Manual control was set too True without render, that is dangerous. Setting to False')
                    self.cfg['cfg_env']['manual_control'] = False
                
            self.init_env(render_mode=render_mode)
            self.load_model()

            res = dict()
            res_all_ep = dict()
            res_rew_f_list = []
            res_eval_f_list = []
            episodes = []
            for seed in range(init_seed, init_seed+num_ep):
                self.run_seed(seed=seed, num_steps = num_steps)
                episodes.append(self.env.episode)
                res, res_all_ep, res_rew_f_list, res_eval_f_list  = self.env.eval_episode(res=res,
                                                                                          res_rew_f_list= res_rew_f_list,
                                                                                          res_eval_f_list= res_eval_f_list)
            test_result['episodes']= episodes
            test_result['res']= res
            test_result['res_all_ep']= res_all_ep
            test_result['res_rew_f_list']= res_rew_f_list
            test_result['res_eval_f_list']= res_eval_f_list
            self.test_results.append(test_result)

        return test_result



    def add_episode(self, cfg, episode):
        self.episodes.append((cfg,episode))


# Old version - we keep for now
class RLModelTester:  
    
    def __init__(self, env_name=None, model=None, cfg={}):
        self.reset()
        self.env_name = env_name
        self.model = model
        self.cfg = cfg
                
    def reset(self):
        self.env = None
        self.model = None
        self.real_time_step=1 # if available, duration of a simulation time step
        self.model_id = 'RandomAgent'
        self.runs = []
        self.evals = None
        self.trace_idx = None
        self.specs = None
        self.stl_driver = stlrom.STLDriver()

    def create_env(self, render_mode=None):
        if self.env_name is None:
            return "No environment found"
        else:
            if render_mode == 'video':
                self.env = gym.make(self.env_name, render_mode="rgb_array")
                self.record_video = True        
            else:
                self.env = gym.make(self.env_name, render_mode=render_mode)
                self.signals_names = envs.cfg_envs[self.env_name]['action_names']+envs.cfg_envs[self.env_name]['obs_names']+['reward']
                self.real_time_step = envs.cfg_envs[self.env_name]['real_time_step']
                
    def configure_env(self, cfg):
        print("Configuring env with ", cfg)
        if cfg is None:
            if self.env_name== 'highway-v0':
                print("Highway env, default configuration")                
                if self.model_id == 'Manual':
                    print("Manual control")
                    cfg = {
                    "manual_control":True,
                    "duration": 1000
                    }
                else:
                    cfg = {
                    "manual_control":False,
                    "duration": 1000
                    }
                self.env.configure(cfg)
        elif self.env is not None:
            self.env.configure(cfg)

                
## Manage models
    def find_hf_models(self):
        _,models=  utils.find_models(self.env_name)
        
        if len(models) == 0:
            self.models_list = None
        else:
            self.models_list = []
            count = 0
            for model in models:
                for model_type in envs.supported_models:
                    if model_type in model:
                        self.models_list.append(model)
                        count += 1
        print("Found ", count, " models for ", self.env_name)  
        return self.models_list

    def load_hf_model(self, model_id):
        if self.env_name is None:
            return "No environment found"
        else:
            if self.model is not None:
                self.model = None
                self.model_id = 'RandomAgent'
            self.model = utils.load_model(self.env_name, model_id)
            if self.model is not None:
                print("Model ", model_id, " loaded")
                self.model_id = model_id
            else:                
                print("Model not loaded")
            return self.model

    ## Test methods
    def add_eval(self,trace_idx, eval_name, value):
        
        if self.evals is None:
            return
        # checks if we have a record with seed and model_name already        
        record_idx = self.evals["trace_idx"]==trace_idx 
        if record_idx.any():
            self.evals.loc[record_idx, eval_name] = value
        else:
            return

    def test_seed(self, seed=0, num_steps=100, render_mode=None, lazy=True):         
        print("Testing seed ", seed, " with model ", self.model_id, " for ", num_steps, " steps", " lazy: ", lazy)                
        #print("current evals: ", self.evals)
        
        # check if we have already run this seed with this model
        if self.evals is not None:            
            record_idx = (self.evals["seed"]==seed) & (self.evals["model_name"]==self.model_id)
            
            if record_idx.any():
                if lazy:            
                    print('Lazy, done already run.')
                    record = self.evals[record_idx]
                    self.trace_idx= record['trace_idx'][0]
                    total_reward = record['total_reward'][0]
                    return total_reward            
        
        # configure env and model TODO: this is gonna need to be one function per env
        #self.create_env(render_mode=render_mode)
        obs, info = self.env.reset(seed=seed)
        print("seed", seed, " obs: ", obs, " info: ", info)
        #self.configure_env(cfg=None)

        # Compute the trace      
        trace= []
        time_step = 0
        total_reward = 0

        try:
            for _ in range(num_steps):

                if self.model is not None:
                    action, _ = self.model.predict(obs)
                else:
                    action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                trace.append([time_step, obs, action, next_obs, reward, terminated, truncated, info])
                if terminated or truncated:
                    break  
                time_step += self.real_time_step
                obs = next_obs
                total_reward += reward

        except Exception as e:
            print("Env crashed with: ", e)   

        self.env.close()
        
        # record the trace and init evals                
        
        trace_idx = self.add_trace(trace)
               
        self.trace_idx = trace_idx
        new_record = pd.DataFrame({'trace_idx':trace_idx, 'env_name':self.env_name,'model_name': [self.model_id],'seed': [seed], 'total_reward': [total_reward]})

        if self.evals is None:
            self.evals = new_record
        else:
            self.evals = pd.concat([self.evals, new_record])
        
        return total_reward

    
    ## STL monitoring
    # For now we make a non optimal use of the stlrom library
    # wherein we create a new driver each time.     

    def add_trace(self, trace):
        if self.runs is None:
            trace_idx = 0
            self.trace_idx=0
            self.runs = [trace]
            #self.df_signals = [self.get_dataframe_from_trace()]
        else:
            self.runs += [trace]
            #df = self.get_dataframe_from_trace()
            trace_idx= len(self.runs)-1
            #df = self.get_dataframe_from_trace(trace_idx)

        return trace_idx

    def monitor_trace(self, phi=None):
        # reset the stl driver data
        if phi is None or self.trace_idx is None or self.specs is None:
            return None
        else:
            
            saved_idx =self.trace_idx

            # checks if a trace_idx is specified
            if phi.endswith(')'):
                self.trace_idx = int(phi.split('(')[1][:-1]) # TODO checks int !             
                phi = phi.split('(')[0]

            self.stl_driver = stlrom.STLDriver()
            self.stl_driver.parse_string(self.specs)

            # load data in the driver
            time = np.array([trace_state[0] for trace_state in self.runs[self.trace_idx]])
            for trace_state in self.runs[self.trace_idx]:
                self.stl_driver.add_sample(self.get_sample(trace_state))

            # compute the robustness at each time step
            df_rob = None
            for t0 in time:
                robs = self.stl_driver.get_online_rob(phi, t0)
                rho = robs[0]
                if rho>0:
                    sat = 1
                elif rho==0:
                    sat = np.nan
                else:
                    sat = 0
                new_record = pd.DataFrame({'time': [t0], 'sat': [sat], 'rho': [rho]}) 
                if df_rob is None:
                    df_rob = new_record
                else:
                    df_rob = pd.concat([df_rob, new_record])

            self.trace_idx = saved_idx
            return df_rob

    def eval_spec(self, phi=None):
        if phi is None:
            return

        idx = 0        
        for r in self.runs:                    
            self.stl_driver = stlrom.STLDriver()
            self.stl_driver.parse_string(self.specs)
            trace = self.runs[idx]
            time = np.array([trace_state[0] for trace_state in trace])
            for trace_state in trace:
                self.stl_driver.add_sample(self.get_sample(trace_state))
            rob = self.stl_driver.get_online_rob(phi, time[0])
            print("Robustness for ", idx, " is ", rob[0])            
            self.add_eval(trace_idx=idx, eval_name=phi, value=rob[0])
            idx= idx +1
        
    ## Helper functions
    def get_video_filename(self):

        filename = self.env_name+'-'
        if self.model is not None:
            filename += self.model.__class__.__name__
        else:
            filename += 'random'

        return filename + '.mp4'

    ## Signal stuff
    def get_sample(self,trace_state):
        # returns a sample for stlrom from a trace state 
        # of the form (time, obs, action, next_obs, reward) 

        time = np.array([trace_state[0]])
        try:
            action = trace_state[2].flatten()
        except:
            action = trace_state[2]
        try:    
            obs = trace_state[1].flatten()
        except:
            obs = trace_state[1]

        reward = np.array([trace_state[4]])

        ss = np.concatenate((time, action, obs, reward))
        return ss

    def get_dataframe_from_trace(self, trace_idx=None, signals_names=None):

        if trace_idx is None:
            trace_idx = self.trace_idx

        if signals_names is None:
            signals_names = self.signals_names
        elif type(signals_names) is str:
            signals_names = [signals_names]
                
        df_signals = pd.DataFrame()
        df_signals['time'] = self.get_time(trace_idx)
        for signal in signals_names:
            df_signals[signal]= self.get_signal(trace_idx, signal)
        
        return df_signals

    def get_time(self, trace_idx=None):
        if trace_idx is None:
            trace_idx = self.trace_idx
        return [self.get_sample(trace_state)[0] for trace_state in self.runs[trace_idx]]   

    def get_signal(self, trace_idx=None, signal_name=''):
        if trace_idx is None:
            trace_idx = self.trace_idx
            
        if signal_name == 'reward':
            out =[self.get_sample(trace_state)[-1] for trace_state in self.runs[trace_idx]]    
        else:
            signal_index = self.signals_names.index(signal_name)
            out = [self.get_sample(trace_state)[signal_index+1] for trace_state in self.runs[trace_idx]]    
 
        return out

    def get_signal_string(self):
        return 'signal '+ ', '.join(self.signals_names)

    def set_current_trace(self, trace_idx):
        self.trace_idx = trace_idx

    ## Info 

    def get_env_info(self):
        info = "Environment: " + self.env_name + "\n" + "----------------------\n"
        info += "Observation space: " + str(self.env.observation_space) + "\n\n"
        info += "Action space: " + str(self.env.action_space) + "\n\n"
        info += "Signals: " + self.get_signal_string() + "\n\n"
        return info

## Plotting

    def get_fig(self, signals_layout):
        lay = utils.get_layout_from_string(signals_layout)
        status = "Plot ok. Hit reset on top right if not visible."            

        #f= figure(height=200)
        figs = []
        colors = itertools.cycle(palette)    

        for signal_list in enumerate(lay):
            f=None
            for signal in signal_list[1]:                
                #try: 
                    color=colors.__next__()                    
                    tr_idx = self.trace_idx
                    print(signal.strip())
                    if signal.strip().startswith("set_trace_idx(") or signal.strip().startswith("_tr("):            
                        tr_idx = int(signal.split('(')[1][:-1])                         
                        self.set_current_trace(tr_idx)                        
                    else: 
                        if f is None:
                            if figs == []:
                                f = figure(height=200)
                            else:
                                f = figure(height=200, x_range=figs[0][0].x_range)
                            figs.append([f])
                        if signal in self.signals_names or signal.split('(')[0] in self.signals_names:                        
                            df_sig = self.get_dataframe_from_trace(signal)
                            f.scatter(df_sig["time"], df_sig[signal], legend_label=signal+', trace_idx='+str(tr_idx), color=color)
                            f.line(df_sig["time"], df_sig[signal], legend_label=signal+', trace_idx='+str(tr_idx), color=color)
                        # else if signal is of the form rho(phi)
                        elif signal.startswith('rho(') or signal.startswith('rob('):
                            phi = signal.split('(')[1][:-1]
                            df_rob = self.monitor_trace(phi)
                            f.step(df_rob["time"], df_rob["rho"], legend_label=signal+', trace_idx='+str(tr_idx), color=color)
                        elif signal.startswith('sat('):
                            phi = signal.split('(')[1][:-1]
                            df_rob = self.monitor_trace(phi)
                            f.step(df_rob["time"], df_rob["sat"], legend_label=signal+', trace_idx='+str(tr_idx), color=color)
                        else: # try implicit rho(signal), i.e., signal is a formula name
                            df_rob = self.monitor_trace(signal)
                            f.step(df_rob["time"], df_rob["rho"], legend_label=signal+', trace_idx='+str(tr_idx), color=color)
                #except:
                #     status = "Warning: error getting values for " + signal
        fig = gridplot(figs, sizing_mode='stretch_width')        
        
        return fig, status
                
    
