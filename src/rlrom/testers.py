import numpy as np

import gymnasium as gym
from gymnasium import spaces
import rlrom.utils as utils
from rlrom.wrappers.stl_wrapper import stl_wrap_env
from rlrom.utils import append_to_field_array as add_metric

import rlrom.plots
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette
# itertools handles the cycling
import itertools
import sys


def make_env_test(cfg):

    if 'make_env_test' in cfg:
      # recover and execute the make_env_test custom function
      context = sys.modules[cfg['import_module']]
      custom_make_env = getattr(context, cfg['make_env_test'])        
      env = custom_make_env(cfg)      
    else:  
      # default
      env_name = cfg.get('env_name','')                           
      cfg_test = cfg.get('cfg_test',{})
    
      if 'render_mode' in cfg_test:
        render_mode = cfg_test.get('render_mode', 'human')
        if render_mode=='None':
            render_mode=None
      else: 
        render_mode=None
      
      env = gym.make(env_name, render_mode=render_mode)
    
    cfg_specs = cfg.get('cfg_specs', None)            
    if cfg_specs is not None:
        env = stl_wrap_env(env, cfg_specs)
        env = gym.wrappers.FlattenObservation(env)        
    return env

    
class RLTester:
    def __init__(self,cfg):
        
        cfg = utils.load_cfg(cfg)
        self.cfg = cfg        
        self.manual_control = False
        self.env_name = cfg.get('env_name')
        self.env = None
        self.model = None
        self.test_results = []
        self.has_stl_wrapper = cfg.get('cfg_specs', None) is not None
        self.model_use_specs = cfg.get('model_use_specs', False)  # if False, model will use observation from the wrapped environment
        
    def load_model(self):
        
        cfg_env = self.cfg.get('cfg_env',dict())
        if cfg_env.get('manual_control', False):
            model = 'manual'
            print("INFO: manual_control set to True, stay alert.")            
        else:
            self.manual_control = False
            model_path = self.cfg.get('model_path', './models')
            model_name = self.cfg.get('model_name', 'random')
            if model_path=='huggingface':
                repo_id = model_name
                env_name = self.cfg.get('env_name')
                model = utils.load_model(env_name, repo_id)
            else:
                model_name, _ = utils.get_model_fullpath(self.cfg)
                if model_name=='random':
                    model='random'
                else:
                    print("INFO: Loading model ", model_name)
                    model= utils.load_model(model_name)
            
        self.model = model

    def _get_action(self, obs):        
        
        if self.manual_control is True or self.model=='random':
            action = self.env.action_space.sample()
        else:
            if self.has_stl_wrapper:  
                if self.model_use_specs is False:
                    # agent was trained without stl_wrapper, so we need to use wrapped_obs to predict action                    
                    obs = self.env.env.last_obs['unwrapped'] # TODO might break with more than one layer of wrapper
                else:
                    pass # use the obs that we were given                
            action, _ = self.model.predict(obs)
        return action

    def init_env(self, **kargs):      # **kargs allows to override self.cfg fields without changing self.cfg
        cfg = self.cfg        
        cfg= utils.set_rec_cfg_field(cfg,**kargs)                    
        self.env = make_env_test(cfg)

    def run_seed(self, seed=None, num_steps=100, reload_model=False):

        self.init_env()
        
        # We actually might want to reload every time to enforce determinism     
        if reload_model:
             self.load_model()
        
        if self.has_stl_wrapper is False:
            episode = {'observations':[], 'actions':[],'rewards':[], 'dones':[], 'last_obs':None}

        if seed is not None:
            last_obs, info = self.env.reset(seed=seed)
        else:
            last_obs, info = self.env.reset()        
        
        for _ in range(num_steps):    

            action = self._get_action(last_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)    

            # we collect episode here if no stl_wrapper
            if self.has_stl_wrapper is False:
                episode['observations'].append(last_obs)               
                episode['last_obs'] = obs
                episode['actions'].append(action)
                episode['rewards'].append(reward)
                episode['dones'].append(terminated)
                
            last_obs=obs
            if terminated:                
                break    
        
        if self.has_stl_wrapper:
            episode = self.env.get_wrapper_attr('episode')
        self.env.close()
        return episode

    def run_cfg_test(self, reload_model=True):
        cfg_test = self.cfg.get('cfg_test') 
        test_result = dict({'cfg':self.cfg})                        
        if cfg_test is not None:
            init_seed = cfg_test.get('init_seed',0)
            num_ep = cfg_test.get('num_ep',1)            
            num_steps  = cfg_test.get('num_steps', 100)
            reload_model_every_ep = cfg_test.get('reload_model_every_ep',False)
                
            test_result = {'episodes':[], 'res':{}}
            num_ep_so_far=0
            if reload_model: 
                self.load_model() 
            
            for seed in range(init_seed, init_seed+num_ep):
                episode = self.run_seed(seed=seed, num_steps=num_steps, reload_model=reload_model_every_ep)
                num_ep_so_far+=1
                print('.', end='')
                if num_ep_so_far%10==0:
                    print('|')
                test_result = self.eval_episode(episode, test_result)
            print()
            self.test_results.append(test_result)
            
        return test_result
    
    def eval_episode(self, episode, test_result):
        test_result['episodes'].append(episode)                
        res = test_result.get('res',{})
            
        if self.has_stl_wrapper:
            res_rew_f_list = test_result.get('res_rew_f_list',[])
            res_eval_f_list = test_result.get('res_eval_f_list',[])
            eval_specs_episode = self.env.get_wrapper_attr('eval_specs_episode')
            res, res_all_ep, res_rew_f_list, res_eval_f_list  = eval_specs_episode(res=res,
                                                                                res_rew_f_list= res_rew_f_list,
                                                                                res_eval_f_list= res_eval_f_list)
        else: # only episode length and sum reward
            rewards = episode['rewards']
            ep_len = len(rewards)                    
            res = add_metric(res,'ep_len',ep_len)
            ep_rew=0        
            for step in range(0,ep_len):  # computes sum of rewards
                ep_rew +=  rewards[step]
            res= add_metric(res,'ep_rew',ep_rew)            
            
            # synthetic over all past episodes
            res_all_ep = dict({'basics':{}, 'reward_formulas':dict(), 'eval_formulas':dict()})            
            res_all_ep['basics']['mean_ep_len'] = np.double(res['ep_len']).mean()
            res_all_ep['basics']['mean_ep_rew'] = res['ep_rew'].mean()
            test_result['res'] = res
                
        test_result['res_all_ep'] = res_all_ep

        return test_result

    def eval_all_episodes(self, episodes):
        
        test_result = dict({'cfg':self.cfg})                        
        res = dict()        
        res_rew_f_list = []
        res_eval_f_list = []            
        for episode in episodes:                                
                res, res_all_ep, res_rew_f_list, res_eval_f_list  = self.env.env.eval_episode(episode=episode,
                                                                                          res=res,
                                                                                          res_rew_f_list= res_rew_f_list,
                                                                                          res_eval_f_list= res_eval_f_list)
        test_result['episodes']= episodes
        test_result['res']= res
        test_result['res_all_ep']= res_all_ep
        test_result['res_rew_f_list']= res_rew_f_list
        test_result['res_eval_f_list']= res_eval_f_list
        return test_result

    def print_res_all_ep(self, test_result):
        res_all_ep = test_result['res_all_ep']
        print('mean_ep_len:', f"{res_all_ep['basics']['mean_ep_len']:.4g}", end=' | ')
        print('mean_ep_rew:', f"{res_all_ep['basics']['mean_ep_rew']:.4g}")
        #for f_name,_ in self.env.reward_formulas.items():
        #    print(f_name+':',f"{res_all_ep['reward_formulas'][f_name]['mean_sum']:.4g}",end=' | ')      
        if self.has_stl_wrapper:
            eval_formulas = self.env.get_wrapper_attr('eval_formulas')
            for f_name,f_cfg in eval_formulas.items():
                if f_cfg is None:
                    f_cfg = {}                
                is_local_formula = f_cfg.get('eval_all_steps', False)
                if is_local_formula:                                                
                    print(f_name, end=': ')
                    print("sum=",f"{res_all_ep['eval_formulas'][f_name]['mean_sum']:.4g}",end=' | ')
                    print("mean=",f"{res_all_ep['eval_formulas'][f_name]['mean_mean']:.4g}",end=' | ')
                    print("num_sat=",f"{res_all_ep['eval_formulas'][f_name]['mean_num_sat']:.4g}")                
                else:           
                    print(f_name+": ratio_sat=",f"{res_all_ep['eval_formulas'][f_name]['ratio_init_sat']:.4g}")                      
                    
    def find_huggingface_models(self, repo_contains='', algo=''):
        env_name = self.cfg.get('env_name')
        if env_name is None:
            models = []
            models_ids = []
        else:
            models, models_ids = utils.find_huggingface_models(env_name, repo_contains=repo_contains, algo=algo) 
        return models, models_ids

    def get_fig(self, signals_layout, ep_idx=0, test_result=-1):
    # plots stuff in bokeh from episodes in test_results
        
        if isinstance(test_result, int):
            if self.test_results == []: 
                print('PROBLEM: No result computed yet.')
            else:
                test_result = self.test_results[test_result]
        episodes = test_result['episodes']
        current_ep = episodes[ep_idx]         
        self.init_env(render_mode=None)
        #self.init_env()
        self.env.env.set_episode_data(current_ep)
        num_ep = len(episodes)            
        lay = rlrom.plots.get_layout_from_string(signals_layout)
        status = "Plot ok. Hit reset on top right if not visible."            

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

                        ttime = self.env.env.get_time()                        
                        labl = signal                                                
                        if num_ep>1:                                                            
                            labl += ', ep '+str(ep_idx)

                        sig_values, sig_type = self.env.env.get_values_from_str(signal)
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
        self.env.close()
        return fig, status

