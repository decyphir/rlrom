import gymnasium as gym
from gymnasium.utils.save_video import save_video
from gymnasium.spaces.utils import flatten_space
import numpy as np
import pandas as pd
import stlrom
import rlrom.envs as envs
import rlrom.utils as utils
from rlrom.wrappers.stl_wrapper import STLWrapper

import rlrom.plots
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
# itertools handles the cycling
import itertools
import os


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
        
        cfg = utils.load_cfg(cfg)

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
            res_rew_f_list = []
            res_eval_f_list = []
            episodes = []
            
            num_ep_so_far =0
            for seed in range(init_seed, init_seed+num_ep):
                self.run_seed(seed=seed, num_steps = num_steps)
                num_ep_so_far+=1
                print('.', end='')
                if num_ep_so_far%10==0:
                    print('|')
                episodes.append(self.env.episode)
                res, res_all_ep, res_rew_f_list, res_eval_f_list  = self.env.eval_episode(res=res,
                                                                                          res_rew_f_list= res_rew_f_list,
                                                                                          res_eval_f_list= res_eval_f_list)
            print()
            test_result['episodes']= episodes
            test_result['res']= res
            test_result['res_all_ep']= res_all_ep
            test_result['res_rew_f_list']= res_rew_f_list
            test_result['res_eval_f_list']= res_eval_f_list
            self.test_results.append(test_result)

        return test_result

    def eval_all_episodes(self, episodes):
        
        test_result = dict({'cfg':self.cfg})                        
        res = dict()        
        res_rew_f_list = []
        res_eval_f_list = []            
        for episode in episodes:                                
                res, res_all_ep, res_rew_f_list, res_eval_f_list  = self.env.eval_episode(episode=episode,
                                                                                          res=res,
                                                                                          res_rew_f_list= res_rew_f_list,
                                                                                          res_eval_f_list= res_eval_f_list)
        test_result['episodes']= episodes
        test_result['res']= res
        test_result['res_all_ep']= res_all_ep
        test_result['res_rew_f_list']= res_rew_f_list
        test_result['res_eval_f_list']= res_eval_f_list
        return  test_result

    def print_res_all_ep(self, test_result):
        res_all_ep = test_result['res_all_ep']
        print('mean_ep_len:', f"{res_all_ep['basics']['mean_ep_len']:.4g}", end=' | ')
        print('mean_ep_rew:', f"{res_all_ep['basics']['mean_ep_rew']:.4g}")
        #for f_name,_ in self.env.reward_formulas.items():
        #    print(f_name+':',f"{res_all_ep['reward_formulas'][f_name]['mean_sum']:.4g}",end=' | ')      
        for f_name,f_cfg in self.env.eval_formulas.items():
            is_local_formula = f_cfg.get('eval_all_steps', True)
            if is_local_formula:                                                
                print(f_name, end=': ')
                print("sum=",f"{res_all_ep['eval_formulas'][f_name]['mean_sum']:.4g}",end=' | ')
                print("mean=",f"{res_all_ep['eval_formulas'][f_name]['mean_mean']:.4g}",end=' | ')
                print("num_sat=",f"{res_all_ep['eval_formulas'][f_name]['mean_num_sat']:.4g}")                
            else:           
                print(f_name+": ratio_sat=",f"{res_all_ep['eval_formulas'][f_name]['ratio_init_sat']:.4g}")                      
                    


    def get_fig(self, signals_layout, ep_idx=0, test_result=-1):
    # plots stuff in bokeh from episodes in test_results
        
        if isinstance(test_result, int):
            if self.test_results == []: 
                print('PROBLEM: No result computed yet.')
            else:
                test_result = self.test_results[test_result]
        episodes = test_result['episodes']
        current_ep = episodes[ep_idx] 
        self.init_env()
        self.env.set_episode_data(current_ep)
        num_ep = len(episodes)            
        lay = rlrom.plots.get_layout_from_string(signals_layout)
        status = "Plot ok. Hit reset on top right if not visible."            

        #f= figure(height=200)
        figs = []
        colors = itertools.cycle(palette)    

        for signal_list in enumerate(lay):
            f=None
            for signal in signal_list[1]:                
                try: 
                    color=colors.__next__()                                        
                    print(signal.strip())                    
                    if signal.strip().startswith("set_ep_idx(") or signal.strip().startswith("_ep("):            
                        ep_idx = int(signal.split('(')[1][:-1])
                        current_ep = episodes[ep_idx]                         
                        self.env.set_episode_data(current_ep)
                    else: 
                        if f is None:
                            if figs == []:
                                f = figure(height=200)
                            else:
                                f = figure(height=200, x_range=figs[0][0].x_range)
                            figs.append([f])

                        ttime = self.env.get_time()                        
                        labl = signal                                                
                        if num_ep>1:                                                            
                            labl += ', ep '+str(ep_idx)

                        sig_values, sig_type = self.env.get_values_from_str(signal)
                        if sig_type == 'val':
                            f.scatter(ttime, sig_values, legend_label=labl, color=color)
                            f.line(ttime, sig_values, legend_label=labl, color=color)
                        elif sig_type == 'rob':
                            f.step(ttime, sig_values, legend_label=labl, color=color)                        

                        elif sig_type == 'sat':
                            f.step(ttime, sig_values, legend_label=labl, color=color)                        

                except:
                     status = "Warning: error getting values for " + signal
        fig = gridplot(figs, sizing_mode='stretch_width')        

        return fig, status

