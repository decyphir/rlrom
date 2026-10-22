# Generic stuff
import numpy as np
import itertools
import os,sys,copy
import time

import matplotlib.pyplot as plt

# Plotting (?)

# Gym and co
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

#rlrom
import rlrom.utils as rlu
from rlrom.wrappers.spec_wrapper import wrap_env_specs
from rlrom.utils import append_to_field_array as add_metric
from rlrom.utils import yaml
import rlrom.plots as rlp

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
      if env_name.lower().startswith("minigrid"):          
        env = FlattenObservation(env) 

    env = wrap_env_specs(env, cfg)
    return env

class RLTester:

    def __init__(self,cfg):
        
        cfg = rlu.load_cfg(cfg)
        self.cfg = cfg        
        self.keyboard_control = False
        self.env_name = cfg.get('env_name')
        self.env = None
        self.model = None
        self.callbacks = []
        self.fig_layout = None
        self.test_results = []
        self.has_spec_wrapper = cfg.get('cfg_specs', None) is not None
        if self.has_spec_wrapper:
            self.model_use_specs = cfg.get('model_use_specs', False)  # if False, model will use observation from the wrapped environment            
            cfg_specs = cfg.get('cfg_specs')
            if 'obs_formulas' in cfg_specs.keys():
                num_obs_formulas = len(cfg_specs.get('obs_formulas'))
            else:
                num_obs_formulas = 0
            if num_obs_formulas==0: 
                self.model_use_specs = True
        self.has_rm_wrapper = cfg.get('cfg_rm', None) is not None
        

        self.model_action_function = None # if None, model.predict() is used; otherwise, model_action_function(model, obs) is called
        
    def load_model(self, model_file=None):
        
        cfg_env = self.cfg.get('cfg_env',dict())
        if cfg_env.get('keyboard_control', False):
            model = 'keyboard'
            print("INFO: manual_control set to True, stay alert.")    
            self.keyboard_control = True        
        else:            
            self.manual_control = False
            model_path = self.cfg.get('model_path', './models')
            model_name = self.cfg.get('model_name', 'random')
                
            if model_path=='huggingface':
                repo_id = model_name
                env_name = self.cfg.get('env_name')
                model = rlu.load_model(env_name, repo_id)
            elif model_file is not None:
                print("INFO: Loading model file ", model_file)
                model= rlu.load_model(model_file)           
            else:
                model_name, _ = rlu.get_model_fullpath(self.cfg)
                if model_name=='random':
                    model='random'
                else:
                    print("INFO: Loading model ", model_name)
                    model= rlu.load_model(model_name)            
        self.model = model
        return model

    def _get_action(self, obs):        
        
        if self.keyboard_control is True:
            keyboard_control = self.env.get_wrapper_attr('keyboard_control')
            action = None 
            while action==None: # wait for action to come (note: we'll want to change that for real time control)
                action = keyboard_control(obs)           
        elif isinstance(self.model, str):  # includes "random" and "manual_control" which is bypassed anyway for hw env 
            action = self.env.action_space.sample()            
        else:
            if self.has_spec_wrapper:                  
                if self.model_use_specs is False:
                    # agent was trained without spec_wrapper, so we need to use wrapped_obs to predict action                    
                    last_obs= self.env.get_wrapper_attr('last_obs')
                    obs = last_obs['unwrapped']
                else:
                    pass # use the obs that we were given                
            action = self.model.predict(obs)[0] if self.model_action_function is None \
                else self.model_action_function(self.model, obs)
        return action

    def init_env(self, **kargs):      # **kargs allows to override self.cfg fields without changing self.cfg
        cfg = self.cfg        
        cfg= rlu.set_rec_cfg_field(cfg,**kargs)                    
        self.env = make_env_test(cfg)

    def run_episode(self, seed=None, num_steps=100, 
                    init_env = True, load_model=True,
                    step_pause=0, 
                    ):

        if init_env:
            self.init_env()

        if self.fig_layout is not None:
            rl_fig = rlp.RLFig(self, self.fig_layout)
            rl_fig.fig.show()
            time.sleep(.1)  # TODO probably cfg that 
        else:
            rl_fig = None

        # Live-updating figures requested in cfg_test.plot (live: True)
        self._live_plot = None
        live_items = self._live_plot_items()
        if live_items:
            live_window = None
            for item in live_items:
                if item.get('live_window') is not None:
                    live_window = float(item['live_window'])
                    break
            lp = rlp.LivePlot(self, live_items, live_window=live_window)
            lp.create()
            self._live_plot = lp
        # We actually might want to reload every time to enforce determinism     
        if load_model:
             self.load_model()
        
        if self.has_spec_wrapper is False:
            episode = {'observations':[], 'actions':[],'rewards':[], 'dones':[], 'last_obs':None}

        if seed is not None:
            last_obs, info = self.env.reset(seed=seed)
        else:
            last_obs, info = self.env.reset()        
        
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):    

            action = self._get_action(last_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            # we collect episode here if no spec_wrapper
            if self.has_spec_wrapper is False:
                episode['observations'].append(last_obs)               
                episode['last_obs'] = obs
                episode['actions'].append(action)
                episode['rewards'].append(reward)
                episode['dones'].append(terminated)            
                
            last_obs=obs

            # callbacks
            for c in self.callbacks:
                c()
            if rl_fig is not None:
                rl_fig.update()
            if self._live_plot is not None:
                self._live_plot.step()
                if self._live_plot.quit:
                    break

            if terminated:                
                break    
        
            step += 1
            if step >= num_steps :
                truncated = True
                        
        if self.has_spec_wrapper:
            episode = self.env.get_wrapper_attr('episode')

        if self._live_plot is not None:
            self._live_plot.close()
            self._live_plot = None
        self.env.close()
        return episode
    
    def _live_plot_items(self):
        """cfg_test.plot items with live: True (None if none)."""
        cfg_test = self.cfg.get('cfg_test') or {}
        plot_cfgs = cfg_test.get('plot')
        if plot_cfgs == 'None':
            plot_cfgs = None
        if not plot_cfgs:
            return None
        live_items = [p for p in plot_cfgs if p.get('live')]
        return live_items or None

    def run_cfg_test(self, reload_model=True):
        
        cfg_test = self.cfg.get('cfg_test') 
        test_result = dict({'cfg':self.cfg})                        

        if cfg_test is not None:
            init_seed = int(cfg_test.get('init_seed',0))
            num_ep = int(cfg_test.get('num_ep',1))            
            num_steps  = int(cfg_test.get('num_steps', 100))
            reload_model_every_ep = cfg_test.get('reload_model_every_ep',False)
            model_file = cfg_test.get('model_file')    
            test_result = {'episodes':[], 'res':{}}
            num_ep_so_far=0
            if reload_model: 
                self.load_model(model_file=model_file) 
            
            for seed in range(init_seed, init_seed+num_ep):
                episode = self.run_episode(seed=seed, num_steps=num_steps, 
                                        load_model=reload_model_every_ep, 
                                        init_env=True) # TODO is this necessary ? I mean, to init_env each time ?
                num_ep_so_far+=1
                print('.', end='')
                if num_ep_so_far%10==0:
                    print('|')
                test_result = self.eval_episode(episode, test_result)
                        
            print()
            self.test_results.append(test_result)

            # Saving if requested
            res_file = cfg_test.get('res_file')
            if res_file is not None:
                Tres = copy.deepcopy(test_result) #TODO could do better surely, should make option to keep traces or not
                Tres.pop('episodes', []) 
                with open(res_file,'w') as f:
                   yaml.dump(Tres, f)
            # Optional plotting defined in cfg_test['plot']
            plot_cfgs = cfg_test.get('plot')
            if plot_cfgs == 'None':  # --set-params passes strings
                plot_cfgs = None
            if plot_cfgs:
                live_items = self._live_plot_items() or []
                for p in plot_cfgs:
                    if p in live_items:
                        # already displayed during run_episode; only save out:
                        out_file = p.get('out')
                        if out_file:
                            layout_lines = []
                            for ax in p.get('ax', []):
                                if isinstance(ax, (list, tuple)):
                                    layout_lines.append(', '.join(str(sig) for sig in ax))
                                else:
                                    layout_lines.append(str(ax))
                            fig, _ = self.get_fig("\n".join(layout_lines), ep_idx=0, test_result=test_result)
                            fig.savefig(out_file, dpi=150, bbox_inches='tight')
                            plt.close(fig)  # not displayed: it was shown live already
                            print(f"Plot saved to {out_file}")
                        continue
                    title = p.get('title')
                    ax_cfg = p.get('ax', [])
                    # Each item of 'ax' is one axis: either a comma-separated
                    # string of signal names, or a list/tuple of signal names.
                    layout_lines = []
                    for ax in ax_cfg:
                        if isinstance(ax, (list, tuple)):
                            layout_lines.append(', '.join(str(sig) for sig in ax))
                        else:
                            layout_lines.append(str(ax))
                    layout_str = "\n".join(layout_lines)
                    fig, status = self.get_fig(layout_str, ep_idx=0, test_result=test_result)
                    if title:
                        fig.suptitle(title)
                    out_file = p.get('out')
                    if out_file:
                        fig.savefig(out_file, dpi=150, bbox_inches='tight')
                        print(f"Plot saved to {out_file}")
                    print(status)
                # Show all figures together (a single non-per-loop show() opens
                # every built window at once instead of blocking per figure).
                plt.show()

        return test_result
    
    def eval_episode(self, episode, test_result):
        test_result['episodes'].append(episode)                
        res = test_result.get('res',{})
            
        if self.has_spec_wrapper:
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
        eval_episode_fn= self.env.get_wrapper_attr('eval_episode')
        for episode in episodes:                                
                res, res_all_ep, res_rew_f_list, res_eval_f_list  = eval_episode_fn(episode=episode,
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
        if self.has_spec_wrapper:
            eval_formulas = self.env.get_wrapper_attr('eval_formulas')
            for f_name,f_cfg in eval_formulas.items():
                if f_cfg is None:
                    f_cfg = {}                
                is_local_formula = f_cfg.get('eval_all_steps', False) or f_cfg.get('online', False)
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
            models, models_ids = rlu.find_huggingface_models(env_name, repo_contains=repo_contains, algo=algo) 
        return models, models_ids

    def get_fig(self, signals_layout, ep_idx=0, test_result=-1):
    # plots episode signals with matplotlib, using stlrom Signal.plot
    # (discrete signals are encoded with derivative 0 and rendered as steps)

        if isinstance(test_result, int):
            if self.test_results == []:
                print('PROBLEM: No result computed yet.')
            else:
                test_result = self.test_results[test_result]
        episodes = test_result['episodes']
        current_ep = episodes[ep_idx]
        self.init_env(render_mode=None)
        set_ep = self.env.get_wrapper_attr('set_episode_data')
        set_ep(current_ep)
        num_ep = len(episodes)
        lay = rlp.get_layout_from_string(signals_layout)
        status = "Plot ok."

        n_ax = max(len(lay), 1)
        fig, axs = plt.subplots(n_ax, 1, sharex=True, figsize=(9, 2.5*n_ax), squeeze=False)
        colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        get_sig_fn = self.env.get_wrapper_attr('get_signal_from_str')
        for iax, signals in enumerate(lay):
            ax = axs[iax, 0]
            for signal in signals:
                s = signal.strip()
                try:
                    if s.startswith("set_ep_idx(") or s.startswith("_ep("):
                        ep_idx = int(s.split('(')[1][:-1])
                        current_ep = episodes[ep_idx]
                        set_ep(current_ep)
                        continue
                    sig = get_sig_fn(s)
                    labl = s
                    if num_ep > 1:
                        labl += f', ep {ep_idx}'
                    sig.plot(label=labl, ax=ax, color=next(colors), draw_canvas=False)
                except Exception as e:
                    status = f"Warning: error getting values for {signal}: {e}"
        self.env.close()
        return fig, status

    def retest_checkoints_models(self,df_idx=-1, **kargs):
        df = rlu.get_df_training(self.cfg, idx=df_idx)

        cfg_tmp = copy.deepcopy(self.cfg)  
        cfg_test = cfg_tmp['cfg_test']
        for r in df.collect().iter_rows(named=True):
            print(f'Retesting for step {r['steps']}, i.e., model {r['res_files']}')
            cfg_test['model_file']= os.path.join(r['path'],r['model_files'])
            cfg_test['res_file'] = r['res_files']        
            
            cfg_tmp = rlu.set_rec_cfg_field(cfg_tmp,cfg_test=cfg_test)
            cfg_tmp = rlu.set_rec_cfg_field(cfg_tmp,render_mode=None,**kargs)
            T = RLTester(cfg_tmp)
            T.run_cfg_test()
    