import gymnasium as gym
from gymnasium.utils.save_video import save_video
from gymnasium.spaces.utils import flatten_space
import numpy as np
import pandas as pd
import stlrom
import rlrom.envs as envs
import rlrom.utils as utils

from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
# itertools handles the cycling
import itertools

class RLModelTester:
    
    def __init__(self, env_name=None, model=None):
        self.reset()
        self.env_name = env_name
        self.model = model
                
    def reset(self):
        self.env = None
        self.model = None
        self.real_time_step=1 # if available, duration of a simulation time step
        self.model_id = 'RandomAgent'
        self.runs = {}
        self.evals = None
        self.trace = None
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
                self.model_id = 'RandomAgent'
            self.model = utils.load_model(self.env_name, model_id)
            if self.model is not None:
                print("Model ", model_id, " loaded")
                self.model_id = model_id
            else:                
                print("Model not loaded")
            return self.model

    ## Test methods
    def add_eval(self,model_name, eval_name, seed, value):
        print('current evals: ', self.evals)
        print("Adding eval ", eval_name, " for seed ", seed, " with value ", value)

        if self.evals is None:
            self.evals = pd.DataFrame(columns=['seed', 'model_name', eval_name])

        # checks if we have a record with seed and model_name already
        
        record_idx = (self.evals["seed"]==seed) & (self.evals["model_name"]==model_name)
        if record_idx.any():
            self.evals.loc[record_idx, eval_name] = value
        else:
            new_record = pd.DataFrame({'seed': [seed], 'model_name': [model_name], eval_name: [value]})
            if self.evals is None:
                self.evals = new_record
            else:
                self.evals = pd.concat([ self.evals,  new_record ])

    def test_seed(self, seed=1, num_steps=100, render_mode=None, lazy=True):         
        print("Testing seed ", seed, " with model ", self.model_id, " for ", num_steps, " steps", " lazy: ", lazy)
        # checks if run already exists                
        if not hasattr(self, 'runs'):
            self.runs = {}

        print("current evals: ", self.evals)
        if lazy and (self.model_id, seed) in self.runs:            
            print("Found previous run for seed ", seed, " and model ", self.model_id)
            try:
                r_seed = self.evals[self.evals["seed"]==seed]
                r_eval = r_seed[r_seed["model_name"]==self.model_id]
                r = r_eval['total_reward']
            except:
                print("Error with seed ", seed, " and model ", self.model_id, " result not found")
                r = np.nan()
            return r

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
        self.runs[self.model_id, seed] = trace        
        self.add_eval(eval_name='total_reward',model_name=self.model_id, seed=seed, value=total_reward) 
        self.trace = trace

        return total_reward

    ## STL monitoring
    # For now we make a non optimal use of the stlrom library
    # whereas we create a new driver each time. 

    def monitor_trace(self, phi=None):
        # reset the stl driver data
        if phi is None or self.trace is None or self.specs is None:
            return None
        else:
            self.stl_driver = stlrom.STLDriver()
            self.stl_driver.parse_string(self.specs)

            # load data in the driver
            time = np.array([trace_state[0] for trace_state in self.trace])
            for trace_state in self.trace:
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

        return df_rob

    def eval_spec(self, phi=None):
        if phi is None:
            return
        
        trace_saved = self.trace
        for r in self.runs:
            self.stl_driver = stlrom.STLDriver()
            self.stl_driver.parse_string(self.specs)
            model_id = r[0]
            seed = r[1]
            self.trace = self.runs[r]
            time = np.array([trace_state[0] for trace_state in self.trace])
            for trace_state in self.trace:
                self.stl_driver.add_sample(self.get_sample(trace_state))
            rob = self.stl_driver.get_online_rob(phi, time[0])
            print("Robustness for ", r, " is ", rob[0])
            self.add_eval(model_name=model_id, eval_name=phi, seed=seed, value=rob[0])
        self.trace = trace_saved
        

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

    def get_dataframe_from_trace(self, signals_names=None):
        
        if signals_names is None:
            signals_names = self.signals_names
        elif type(signals_names) is str:
            signals_names = [signals_names]
                
        df_signals = pd.DataFrame()
        df_signals['time'] = self.get_time()
        for signal in signals_names:
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

    def set_current_trace(self, seed):
        self.trace = self.runs[self.model_id, seed]

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
        status = "Plot ok."            

        f= figure(height=200)
        figs = [[f]]
        colors = itertools.cycle(palette)    

        for signal_list in enumerate(lay):

            if signal_list[0]>0:   # second figure, create and append to figs
                f = figure(height=200, x_range=figs[0][0].x_range)
                figs.append([f])
            for signal in signal_list[1]:                
                try: 
                    if signal in self.signals_names:
                        df_sig = self.get_dataframe_from_trace(signal)
                        f.marker(df_sig["time"], df_sig[signal])
                        f.line(df_sig["time"], df_sig[signal], legend_label=signal, color=colors.__next__())
                    # else if signal is of the form rho(phi)
                    elif signal.startswith('rho(') or signal.startswith('rob('):
                        phi = signal.split('(')[1][:-1]
                        df_rob = self.monitor_trace(phi)
                        f.step(df_rob["time"], df_rob["rho"], legend_label=signal, color=colors.__next__())
                    elif signal.startswith('sat('):
                        phi = signal.split('(')[1][:-1]
                        df_rob = self.monitor_trace(phi)
                        f.step(df_rob["time"], df_rob["sat"], legend_label=signal, color=colors.__next__())
                    else: # try implicit rho(signal), i.e., signal is a formula name
                        df_rob = self.monitor_trace(signal)
                        f.step(df_rob["time"], df_rob["sat"], legend_label=signal, color=colors.__next__())
                except:
                     status = "Warning: error getting values for " + signal
            

        fig = gridplot(figs, sizing_mode='stretch_width')
        return fig, status
                
