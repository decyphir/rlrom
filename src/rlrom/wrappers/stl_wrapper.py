import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stlrom
import matplotlib.pyplot as plt
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette

from rlrom.utils import append_to_field_array as add_metric

def stl_wrap_env(env, cfg_specs):
    driver= stlrom.STLDriver()
    stl_specs_str = cfg_specs.get('specs','')
    if stl_specs_str=='':
        stl_specs_str = 'signal'
        first = True
        for a in cfg_specs.get('action_names',{}):
            if first:
                stl_specs_str += ' '+ a
                first = False
            else:
                stl_specs_str += ','+ a
        for o in cfg_specs.get('obs_names',{}):
            if first:
                stl_specs_str += ' '+ o
                first = False
            else:
                stl_specs_str += ','+ o
        stl_specs_str += ',reward'                                 

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

class STLWrapper(gym.Wrapper):
    
    def __init__(self,env,
                 stl_driver, 
                 signals_map={},
                 obs_formulas={},  
                 reward_formulas={},
                 eval_formulas={},
                 end_formulas={},
                 BigM=None
                 ):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.real_time_step = 1
        self.time_step = 0     # integer current time step 
        self.current_time = 0  # real time (=time_step*real_time_step)  for formula evaluation
        self.stl_driver = stl_driver                        
        self.obs_formulas = obs_formulas
        self.reward_formulas = reward_formulas
        self.eval_formulas = eval_formulas
        self.end_formulas= end_formulas
        self.episode={'observations':[], 'actions':[],'rewards':[], 'dones':[]}

        self.signals_map={}
        if signals_map=={}:
            # assumes 1 action and n obs
            signals = stl_driver.get_signals_names().split()
            i_sig=0

            for sig in signals:
                if i_sig==0:
                    self.signals_map[sig] = 'action'
                elif i_sig<len(signals)-1:    
                    self.signals_map[sig] = f'obs[{i_sig-1}]'
                else:
                    self.signals_map[sig] = 'reward'
                i_sig+=1

        elif type(signals_map)==dict: 
            if 'action_names' in signals_map:
                for a_name,a_ref in signals_map['action_names'].items():                    
                    self.signals_map[a_name]=a_ref            
            if 'obs_names' in signals_map:
                for o_name,o_ref in signals_map['obs_names'].items():                     
                    self.signals_map[o_name]=o_ref
            if 'aux_sig_names' in signals_map:
                for o_name,o_ref in signals_map['aux_sig_names'].items():                     
                    self.signals_map[o_name]=o_ref

            self.signals_map['reward'] = 'reward'
        else: # assumes all is fine (TODO? catch bad signals_map here)    
            self.signals_map=signals_map
        
        low = self.observation_space.__getattribute__("low") 
        high = self.observation_space.__getattribute__("high")
        if BigM is None:
            BigM = stlrom.Signal.get_BigM()
        self.observation_space = spaces.Box(np.append(low,  [-BigM]*len(self.obs_formulas)), 
                                            np.append(high, [BigM]*len(self.obs_formulas)),        
                                            dtype=np.float32)
        
        idx_obs_f = self.observation_space.shape[0]-len(self.obs_formulas) # adding mapping from stl signal to obs array, now flat 
        for f_name, f_opt in obs_formulas.items():             
            f_hor = f_opt.get('past_horizon',0)
            obs_name = 'obs_'+f_name+'_hor_'+str(f_hor)
            obs_name = f_opt.get('obs_name', obs_name)
            ref_in_obs = 'obs['+str(idx_obs_f)+']'
            self.signals_map[obs_name]= ref_in_obs
            idx_obs_f +=1

    def step(self, action):
    
        # steps the wrapped env
        obs, reward, terminated, truncated, info = self.env.step(action)                
        
        # store wrapped obs
        self.wrapped_obs = obs
                
        # collect the sample for monitoring 
        s = self.get_sample(self.prev_obs, action, reward) 
        
        # add sample and compute robustness
        self.stl_driver.add_sample(s)        
        
        idx_formula = 0
        rob = [0]*len(self.obs_formulas)
        for f_name, f_opt in self.obs_formulas.items():             
            robs_f,_ = self.eval_formula_cfg(f_name,f_opt)        
            rob[idx_formula] = robs_f # forget about low and high rob for now
            idx_formula+=1     
         
        for f_name, f_opt in self.end_formulas.items():             
            _, res = self.eval_formula_cfg(f_name,f_opt)          
            if res['lower_rob'] > 0:
                print('Episode terminated because of formula', f_name)
                terminated = True

        new_reward = reward                         
        # add stl robustness to reward
        for f_name, f_opt in self.reward_formulas.items():             
            robs_f,_ = self.eval_formula_cfg(f_name,f_opt)
            w = f_opt.get('weight',1)        
            new_reward += w*robs_f   
        
        # update current time
        self.time_step += 1
        self.current_time += self.real_time_step
        
        # return obs with added robustness
        new_obs = np.append(obs, rob)
                        
        self.prev_obs = new_obs

        self.episode['observations'].append(new_obs)               
        self.episode['actions'].append(action)
        self.episode['rewards_wrapped'].append(reward)
        self.episode['rewards'].append(new_reward)
        self.episode['dones'].append(terminated)
            

        if terminated: 
            self.env.reset()

        return new_obs, new_reward, terminated, truncated, info

    def get_sample(self,obs,action,reward):        
        # get a sample for stl driver, converts obs, action,reward into (t, signals_values)
        s = np.zeros(len(self.signals_map)+1-len(self.obs_formulas))
        s[0] = self.current_time
        i_sig = 0
        for key, value in self.signals_map.items():
            i_sig = i_sig+1
            if i_sig>len(s)-1:
                break
            #print(key, value)
            s[i_sig] = eval(value)

        return s

    def reset(self, **kwargs):        
        self.time_step = 0
        self.current_time = 0
        obs0, info = self.env.reset(**kwargs)
        self.wrapped_obs = obs0        
        robs0 = self.reset_monitor()
        obs = np.append(obs0, robs0)
        self.prev_obs = obs
        self.episode={'observations':[], 'actions':[],'rewards':[], 'rewards_wrapped':[],'dones':[]}        
        return obs, info

    def reset_monitor(self):        
        self.stl_driver.data = [] 
        return [0]*len(self.obs_formulas) 
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.reset(seed=seed)
    
    def plot_signal(self, signal, fig=None,label=None,  color=None, online=False, past_horizon=0, linestyle='-', booleanize=False):
    # signal should be part of the "signal" declaration or a valid formula id 
     
        if self.stl_driver.data == []:
            raise ValueError("No data to plot.")
                 
        time = self.get_time()
        sig_values = self.get_sig(signal)
        if sig_values is None:
            if signal in self.formulas:
                sig_values = self.get_rob(signal, online=online,past_horizon=past_horizon)
                signal_index = self.formulas.index(signal)+len(self.signals_map)        
            elif isinstance(signal, np.ndarray) and signal.shape == (len(self.get_time()),):
                sig_values = signal
            elif isinstance(signal, stlrom.Signal):
                pass
            else:
                try:
                    sig_values = self.get_rob(signal, online=online,past_horizon=past_horizon)
                except Exception as e:
                    raise ValueError(f"Name '{signal}' not in signals_map nor in parsed formulas")

        if booleanize:
            sig_values = (sig_values >0).astype(int)

        if fig is None:
             fig = figure(height=200)

        fig.set_xlabel('Time')
        fig.grid(True)

        fig.step(time, sig_values)
        if color is None:
            l, = fig.step(time, sig_values, label=label,linestyle=linestyle)
            color = l.get_color()
        else:
            l = fig.step(time, sig_values, color=color,linestyle=linestyle)
        
        if label is None:
            label=signal
            
        l.set_label(label)
        fig.legend()

        return fig

    def get_time(self):
        if self.stl_driver.data == []:
            raise ValueError("No data to plot.")
        
        return [s[0] for s in self.stl_driver.data]
        
    def get_sig(self, sig_name):
        sig = None        
        sig_expr = self.signals_map.get(sig_name,[])
        if sig_expr != []:                     
            observations = self.episode['observations']
            actions = self.episode['actions']
            rewards = self.episode['rewards']            
            rewards_wrapped = self.episode.get('rewards_wrapped',rewards) 
            step = 0
            sig=[]            
            while step<len(observations):
                obs = observations[step]
                action = actions[step]
                reward = rewards[step]
                reward_wrapped = rewards_wrapped[step]                
                sig.append(eval(sig_expr))                
                step +=1
        return sig

    def get_values_from_str(self, str):
        sig_type = 'val'
        env_signal_names = self.signals_map.keys()
                    
        if str in env_signal_names or str.split('(')[0] in env_signal_names:                        
            sig_val = self.get_sig(str)                        
        elif str.startswith('rho(') or str.startswith('rob('):
            arg_rho = str.split('(')[1][:-1]
            arg_rho = arg_rho.split(',')            
            past_horizon = 0            
            phi = arg_rho[0]
            if len(arg_rho)>1:                          
                past_horizon = -float(arg_rho[1])            
            sig_val = self.get_rob(phi, horizon=past_horizon, online=False)                                                                                                
            sig_type = 'rob'
        elif str.startswith('sat('):
            arg_rho = str.split('(')[1][:-1]
            arg_rho = arg_rho.split(',')
            past_horizon = 0
            phi = arg_rho[0]
            if len(arg_rho)>1:                
                past_horizon = -float(arg_rho[1])
            sig_val = self.get_rob(phi, horizon=past_horizon, online=False)
            sig_val = (sig_val >0).astype(int)         
            sig_type = 'sat'
        else: # try implicit rho(str), i.e., str is a formula name
            sig_val = self.get_rob(str, online=False)                                                                                            
            sig_type = 'rob'                        
        
        return sig_val, sig_type

    def get_rob(self, formula, horizon=0, online=True):
    # Compute robustness signal. If online is true, then 
    # compute it at each time as if future was not known, 
    # otherwise uses all data for all computation
    # MAYBE careful with eval_formula_cfg

        if self.stl_driver.data == []:
            raise ValueError("No data/episode was computed.")
        
        monitor = self.stl_driver.get_monitor(formula)
        
        rob = np.zeros(len(self.stl_driver.data))
        if online:
            monitor.data=[]
        step = 0
        for s in self.stl_driver.data:
            t0 = max(0,step*self.real_time_step-horizon)
            if online:
                monitor.add_sample(s)
            rob[step] = monitor.eval_rob(t0)
            step= step+1
        return rob

    def set_episode_data(self, episode):
        self.reset_monitor()
        self.episode = episode
        observations = self.episode['observations']
        actions = self.episode['actions']
        rewards = self.episode['rewards']            
        rewards_wrapped = self.episode.get('rewards_wrapped',rewards) 
        self.time_step = 0
        self.current_time=0
        
        while self.time_step<len(observations):
            obs = observations[self.time_step]
            action = actions[self.time_step]
            reward_wrapped = rewards_wrapped[self.time_step]                
            s = self.get_sample(obs, action, reward_wrapped)            
            self.stl_driver.add_sample(s)            
            self.time_step +=1
            self.current_time += self.real_time_step

        
    def eval_formula_cfg(self, f_name, f_opt, res=dict()):
    # eval a formula based on f_opt configuration options AT CURRENT STEP

        if f_opt is None:
            f_opt={}

        f_hor = f_opt.get('past_horizon',0)            
        t0 = f_opt.get('t0',max(0, self.current_time-f_hor))
        robs = self.stl_driver.get_online_rob(f_name, t0)
        val = robs[0]
        if res==dict():
            res['estimate_rob'] = np.array(robs[0])
            res['lower_rob'] = np.array(robs[1])
            res['upper_rob'] = np.array(robs[2])                        
            sat = 1 if robs[0]>0 else 0
            res['sat'] = np.array(sat)
        else:
            res['estimate_rob'] = np.append(res['estimate_rob'],robs[0])
            res['lower_rob'] = np.append(res['lower_rob'],robs[1])
            res['upper_rob'] = np.append(res['upper_rob'],robs[2])                        
            sat = 1 if robs[0]>0 else 0
            res['sat'] = np.append(res['sat'],sat)
        
        upper_bound = f_opt.get('upper_bound',np.inf)
        lower_bound = f_opt.get('lower_bound',-np.inf)
        val = max(val, lower_bound)
        val = min(val, upper_bound)

        return val, res
            
        
    def eval_specs_episode(self, episode=None, res=dict(), res_rew_f_list=[], res_eval_f_list=[]):
    # computes different metrics to evaluate an episode. If res contains values already, concatenate
    # returns top level metrics, and robustness and stuff at all steps for reward formulas and eval formulas    
    
        
        if episode is None:
            episode= self.episode            
        else:
            self.episode = episode
                
        rewards = episode['rewards']
        rewards_wrapped = episode.get('rewards_wrapped',rewards)         

        # episode length
        ep_len = len(rewards)        
        res = add_metric(res,'ep_len',ep_len)
        
        # cumulative reward
        ep_rew=0        
        for step in range(0,ep_len):            
            ep_rew +=  rewards[step]
        res= add_metric(res,'ep_rew',ep_rew)

        res_all_ep = dict({'basics':{}, 'reward_formulas':dict(), 'eval_formulas':dict()})            
        res_all_ep['basics']['mean_ep_len'] = np.double(res['ep_len']).mean()
        res_all_ep['basics']['mean_ep_rew'] = res['ep_rew'].mean()
        # maybe a mean mean reward ?

        observations = episode['observations']
        actions = episode['actions']            
        if self.reward_formulas != dict():
            self.reset_monitor()
            self.current_time=0
            self.time_step = 0

            
            res_f = dict()
            for f_name,f_cfg in self.reward_formulas.items():
                res_f[f_name] = []
            
            while self.time_step<len(observations):
                obs = observations[self.time_step]
                action = actions[self.time_step]
                rew_wrapped = rewards_wrapped[self.time_step]                
                s = self.get_sample(obs, action, rew_wrapped)            
                self.stl_driver.add_sample(s)

                for f_name,f_cfg in self.reward_formulas.items():
                    v, _ = self.eval_formula_cfg(f_name,f_cfg)    
                    res_f[f_name] = np.append(res_f[f_name],v)                                        
                    if f_name not in res:
                        res[f_name] = dict() 

                self.time_step +=1
                self.current_time += self.real_time_step
            
            # Synthesize 
            for f_name,f_cfg in self.reward_formulas.items():
                w = f_cfg.get('weight', 1)
                res[f_name] = add_metric(res[f_name], 'mean', w*res_f[f_name].mean())
                
                sum_f = w*res_f[f_name].sum()
                res[f_name] = add_metric(res[f_name], 'sum', sum_f)
                
                num_sat = (res_f[f_name]>0).sum()
                res[f_name] = add_metric(res[f_name], 'num_sat', num_sat)

            res_rew_f_list.append(res_f)            

        # eval formulas - for those, we evaluate off-line, after the trace has been completely computed
            
        if self.eval_formulas != dict():
                        
            self.current_time=0
            self.time_step = 0            

            res_f = dict()
            for f_name,f_cfg in self.eval_formulas.items():
                res_f[f_name] = []
            
            while self.time_step<len(observations):
                
                for f_name,f_cfg in self.eval_formulas.items():
                    v, _ = self.eval_formula_cfg(f_name,f_cfg)    
                    res_f[f_name] = np.append(res_f[f_name],v)                                        
                    if f_name not in res:
                        res[f_name] = dict() 

                self.time_step +=1
                self.current_time += self.real_time_step
            
            # Synthesize: we compute all metrics - maybe we choose in cfg  (TODO)
            for f_name,f_cfg in self.eval_formulas.items():                
                if f_cfg is None:
                    f_cfg = {}
                eval_all_steps = f_cfg.get('eval_all_steps', False)                
                if eval_all_steps:
                    w = f_cfg.get('weight', 1)
                    res[f_name] = add_metric(res[f_name], 'mean', w*res_f[f_name].mean())               
                    sum_f = w*res_f[f_name].sum()
                    res[f_name] = add_metric(res[f_name], 'sum', sum_f)
                    num_sat = (res_f[f_name]>0).sum()
                    res[f_name] = add_metric(res[f_name], 'num_sat', num_sat)                
                else:
                    init_rob = res_f[f_name][0]
                    res[f_name] = add_metric(res[f_name], 'init_rob', init_rob)
                    init_sat = 1 if res_f[f_name][0]>0 else 0
                    res[f_name] = add_metric(res[f_name], 'init_sat', init_sat)                
                
            res_eval_f_list.append(res_f)
                        
            for f_name,f_cfg in self.reward_formulas.items():
              if f_cfg is None:
                    f_cfg = {}                
              if isinstance(res[f_name], dict):                    
                res_all_ep['reward_formulas'][f_name] = dict()
                res_all_ep['reward_formulas'][f_name]['mean_rob'] = res[f_name]['mean'].mean()
                res_all_ep['reward_formulas'][f_name]['mean_num_sat'] = res[f_name]['num_sat'].mean()
                res_all_ep['reward_formulas'][f_name]['mean_sum'] = res[f_name]['sum'].mean()

            num_ep = len(res['ep_len'])
            for f_name,f_cfg in self.eval_formulas.items():
              if f_cfg is None:
                    f_cfg = {}                
              if isinstance(res[f_name], dict):                    
                res_all_ep['eval_formulas'][f_name] = dict()                
                eval_all_steps = f_cfg.get('eval_all_steps', False)
                if eval_all_steps:
                    res_all_ep['eval_formulas'][f_name]['mean_sum'] = res[f_name]['sum'].mean()
                    res_all_ep['eval_formulas'][f_name]['mean_mean'] = res[f_name]['mean'].mean()
                    res_all_ep['eval_formulas'][f_name]['mean_num_sat'] = res[f_name]['num_sat'].mean()
                else:
                    res_all_ep['eval_formulas'][f_name]['ratio_init_sat'] = (res[f_name]['init_sat']>0).sum()/num_ep
                    res_all_ep['eval_formulas'][f_name]['mean_init_rob'] = res[f_name]['init_rob'].mean()

                        

        return res, res_all_ep, res_rew_f_list, res_eval_f_list
    
    