import gymnasium as gym
from gymnasium.utils.save_video import save_video
from gymnasium.spaces.utils import flatten_space
import numpy as np
import pandas as pd
import stlrom
import rlrom.envs as envs
import rlrom.utils as utils

class RLModelTester:
    
    def __init__(self, env_name=None, model=None):
        self.reset()
        self.env_name = env_name
        self.model = model
        self.real_time_step=1 # if available, duration of a simulation time step
        
    def reset(self):
        self.env = None
        self.model = None
        self.runs = {}
        self.evals = {}
        self.trace = None
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
        self.signals_names = envs.cfg_envs[self.env_name]['signals_names']
        self.real_time_step = envs.cfg_envs[self.env_name]['real_time_step']
                

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
            self.model = utils.load_model(self.env_name, model_id)
            if self.model is not None:
                print("Model loaded")
            else:
                print("Model not loaded")
            return self.model

## Test methods
    def add_eval(self, eval_name, model, seed, value):
        # if evals is not defined, create it as a dictionary
        if not hasattr(self, 'evals'):
            self.evals = {}
        
        # if self.evals[eval_name] is not a dataframe, create it
        
        if self.evals.get(eval_name) is None:
            # create a dataframe with columns seed, model
            self.evals[eval_name] = pd.DataFrame(columns=['seed', model])
        else: # add column for model if not present
            if model not in self.evals[eval_name].columns:
                self.evals[eval_name][model] = np.nan
        # add value to the dataframe
        self.evals[eval_name].loc[seed, model] = value        
                
        
    def test_seed(self, seed=1, num_steps=100, render_mode=None, lazy=True):         
        
        # checks if run already exists        
        if not hasattr(self, 'runs'):
            self.runs = {}
                
        if lazy and (self.model, seed) in self.runs:            
            return self.evals['total_reward'].loc[seed,self.model]
        
        if self.env is None or render_mode is not None:
            self.create_env(render_mode=render_mode)

        trace= []
        obs, info = self.env.reset(seed=seed)
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
        self.runs[self.model, seed] = trace        
        self.add_eval('total_reward', self.model, seed, total_reward) 
        self.trace = trace

        return total_reward

## STL monitoring
    def monitor_trace(self, trace=None, phi=None):
        # reset the stl driver
        self.stl_driver.data =[]
        if trace is None:
            trace = self.trace
        if phi is None:
            return None
        
        for trace_state in trace:
            self.stl_driver.add_sample(self.get_sample(trace_state))
            
        robs = self.stl_driver.get_online_rob(phi)
        return robs
        
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
        action = trace_state[2].flatten()
        obs = trace_state[1].flatten()
        reward = np.array([trace_state[4]])
        
        ss = np.concatenate((time, action, obs, reward))
        return ss

    
    def get_dataframe_from_trace(self, df_signals=None):
        if df_signals is None:
            df_signals = pd.DataFrame()
        
        df_signals.drop(df_signals.index, inplace=True)
        df_signals['time'] = self.get_time()
        for signal in self.signals_names:
            df_signals[signal]= self.get_signal(signal)

        return df_signals

    def get_time(self):
        return [self.get_sample(trace_state)[0] for trace_state in self.trace]   

    def get_signal(self, signal_name):
        if signal_name == 'reward':
            return [self.get_sample(trace_state)[-1] for trace_state in self.trace]    
    
        signal_index = self.signals_names.index(signal_name)
        return [self.get_sample(trace_state)[signal_index+1] for trace_state in self.trace]    
    
    def get_signal_string(self):
        return 'signal '+ ', '.join(self.signals_names)

## Info 
    
    def get_env_info(self):
        info = "Environment: " + self.env_name + "\n" + "----------------------\n"
        info += "Observation space: " + str(self.env.observation_space) + "\n\n"
        info += "Action space: " + str(self.env.action_space) + "\n\n"
        info += "Signals: " + self.get_signal_string() + "\n\n"
        return info
        