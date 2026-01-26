import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stlrom
from rlrom.utils import append_to_field_array as add_metric
import importlib

def stl_wrap_env(env, cfg):  # keeping this for compatibility no reason
    env = STLWrapper(env,cfg)   
    return env

class STLWrapper(gym.Wrapper): 
    
    def __init__(self,env,cfg):        
        
        gym.Wrapper.__init__(self, env)        
        self.env = env        
        
        # Parsing and adding STL formulas
        stl_driver= stlrom.STLDriver()        
        cfg_specs = cfg.get('cfg_specs',{})
        stl_specs_str = cfg_specs.get('specs','') # keeping for backward compat
        stl_specs_str = cfg_specs.get('stl_specs',stl_specs_str) 
        
        if stl_specs_str=='':    # default stl signals declaration. Not sure why it is here.
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
    
        stl_driver.parse_string(stl_specs_str)        
        self.real_time_step = cfg_specs.get('real_time_step',1)
        self.obs_formulas = cfg_specs.get('obs_formulas',{})        
        self.reward_formulas = cfg_specs.get('reward_formulas',{})
        self.eval_formulas = cfg_specs.get('eval_formulas',{})
        self.end_formulas = cfg_specs.get('end_formulas',{})
        self.debug_signals=  cfg_specs.get('debug_signals',False)
        self.debug_formulas= cfg_specs.get('debug_formulas',False)
        self.BigM = cfg_specs.get('BigM')
        self.semantics= 'Boolean'  # TODO option to change. For now Boolean is simpler to interpret rewards
        self.stl_driver = stl_driver                        

        self.time_step = 0     # integer current time step 
        self.current_time = 0  # real time (=time_step*real_time_step)  for formula evaluation
        self.episode={}
        to_import = cfg.get('import_module')                
        if to_import is not None:
            import_module = importlib.import_module(to_import)
        else:
            import_module = None

        self.import_module= import_module
        
        # define signals_map: map signal names to their expressions given in cfg_specs
        cfg_specs= cfg.get('cfg_specs')
        self.signals_map={}
        if 'action_names' in cfg_specs:
            for a_name,a_ref in cfg_specs['action_names'].items():                                    
                self.signals_map[a_name]=a_ref            
        if 'obs_names' in cfg_specs:
            for o_name,o_ref in cfg_specs['obs_names'].items():                     
                self.signals_map[o_name]=o_ref
        if 'aux_sig_names' in cfg_specs: # TODO document/test/example this
            for o_name,o_ref in cfg_specs['aux_sig_names'].items():                     
                self.signals_map[o_name]=o_ref
        self.signals_map['reward'] = 'reward'

        # signals in specs
        self.signals_specs = stl_driver.get_signals_names().split()               
        self.signals_specs_idx = {}  # index map of signals in stl samples (t, s1, s2, etc).
        idx = 1
        for s in self.signals_specs:
            self.signals_specs_idx[s] = idx
            idx+=1

        num_obs_formulas = len(self.obs_formulas)
        if self.BigM is None:
            self.BigM = stlrom.Signal.get_BigM()
                
        obs_formula_space = spaces.Box(np.array([-self.BigM]*num_obs_formulas), np.array([self.BigM]*num_obs_formulas))
        dict_obs = {'unwrapped': env.observation_space, 'obs_formulas': obs_formula_space}        
        self.observation_space =  spaces.Dict(dict_obs)        

        # adding mapping from stl signal to its config
        for f_name, f_opt in self.obs_formulas.items():                                     
            if f_opt is None:
                f_opt=dict()            
            obs_name = f_opt.get('obs_name', f_name)
            self.signals_map[obs_name]= f_opt

    def reset(self, **kwargs):        
        self.time_step = 0
        self.current_time = 0
        obs0, info = self.env.reset(**kwargs)
        self.wrapped_obs = obs0        
        robs0 = self.reset_monitor()
        obs = dict()
        obs['unwrapped'] = obs0
        obs['obs_formulas'] = robs0
        self.last_obs = obs
        self.episode={'observations':[], 'actions':[],'rewards':[], 'rewards_wrapped':[],'dones':[], 'last_obs':[obs], 
                      'stl_data':[]}        
        # initialize res_f. It will store monitoring of formulas computed during the episode
        res_f={}
        for f_name, _ in self.obs_formulas.items():                     
            res_f[f_name]=[]  
        for f_name, _ in self.reward_formulas.items():                     
            res_f[f_name]=[]  
        for f_name, _ in self.end_formulas.items():                     
            res_f[f_name]=[]  
        self.episode['res_f'] = res_f
        return obs, info

    def reset_monitor(self):        
        self.stl_driver.data = [] 
        return [0]*len(self.obs_formulas) 

    def step(self, action):
        
        # steps the wrapped env
        obs, reward, terminated, truncated, info = self.env.step(action)                
        # collect the sample for monitoring 
        s = self.get_sample(obs, action, reward)
        self.episode['stl_data'].append(s)               
        
        # add sample and compute robustness
        self.stl_driver.add_sample(s)        
        idx_formula = 0
        num_obs_formulas= len(self.obs_formulas)
        robs = [0]*num_obs_formulas
        for f_name, f_opt in self.obs_formulas.items():                                     
            robs_f,_ = self.eval_formula_cfg(f_name,f_opt)        
            robs[idx_formula] = robs_f # forget about low and high robs for now                        
            self.episode['res_f'][f_name].append(robs_f)
            idx_formula+=1     
            if self.debug_formulas is True:                
                if idx_formula==1:
                    print('obs formulas', end='  --  ')                
                print(f_name+f': {robs_f:.3}', end=' ')
                if idx_formula==num_obs_formulas:
                    print('')
         
        for f_name, f_opt in self.end_formulas.items():                     
            _, eval_res = self.eval_formula_cfg(f_name,f_opt)                  
            self.episode['res_f'][f_name].append(robs_f)
            if eval_res['lower_rob'] > 0:
                print('Episode terminated because of formula', f_name)
                terminated = True

        new_reward = reward                         
        # add stl robustness to reward
        for f_name, f_opt in self.reward_formulas.items():                         
            robs_f,_ = self.eval_formula_cfg(f_name,f_opt)            
            self.episode['res_f'][f_name].append(robs_f)
            w = f_opt.get('weight',1)        
            new_reward += w*robs_f   
        
        # update current time
        self.time_step += 1
        self.current_time += self.real_time_step
        
        # return obs with added robustness
        
        new_obs = dict()
        new_obs['unwrapped'] = obs
        new_obs['obs_formulas'] = robs

        self.episode['observations'].append(self.last_obs)               
        self.episode['actions'].append(action)
        self.episode['rewards_wrapped'].append(reward)
        self.episode['rewards'].append(new_reward)
        self.episode['dones'].append(terminated)
        self.episode['last_obs'] = new_obs
        self.last_obs = new_obs    
        
        return new_obs, new_reward, terminated, truncated, info

    def get_sample(self,obs,action,reward):        
        # get a sample for stl driver, converts obs, action,reward into (t, signals_values)
   
        s = np.zeros(len(self.signals_specs)+1)
        s[0] = self.current_time
        if self.debug_signals is True:
                print()
                print(f't:{s[0]:.3}',end=' ')
        i_sig = 0
        
        for sig in self.signals_specs:
            value= self.signals_map[sig]
            i_sig = i_sig+1           
            s[i_sig] = eval(value)
            if self.debug_signals is True:
                print(f'{sig}: {s[i_sig]:.3}', end=' ')
        if self.debug_signals is True:
            print()
        return s
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.reset(seed=seed)
    
    def get_time(self):
        if self.stl_driver.data == []:
            raise ValueError("No data to plot.")
        
        return [s[0] for s in self.stl_driver.data]
        
    def get_sig(self, sig_name):
        # recover a signal computed during an episode, either observation, reward, reward formula, 
        sig_val=[]

        if sig_name in self.signals_specs:
            idx = self.signals_specs_idx[sig_name]            
            for s in self.episode['stl_data']:
                sig_val.append(s[idx])
        elif sig_name in self.episode['res_f']:
            sig_val = self.episode['res_f'][sig_name]               
        return sig_val

    def get_values_from_str(self, str):
        
        env_signal_names = self.signals_map.keys() 
        sig_type= 'val'                   
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
        stl_data = self.episode['stl_data']
        observations = self.episode['observations']
        self.time_step = 0
        self.current_time=0
        
        while self.time_step<len(observations):
            i_step = self.time_step
            s = stl_data[i_step]
            self.stl_driver.add_sample(s)            
            self.time_step +=1
            self.current_time += self.real_time_step
   
    def eval_formula_cfg(self, f_name, f_opt, eval_res=None):
    # eval a formula based on f_opt configuration options AT CURRENT STEP
    # it uses whatever is in the data of stl_driver at this point
    
        if f_opt is None:
            f_opt={}        

        f_hor = f_opt.get('past_horizon',0.)   
        t0 = f_opt.get('t0',0.)
        tend = f_opt.get('tend',self.current_time)
        online = f_opt.get('eval_all_steps', False) or f_opt.get('online', False) # eval_all_steps and online are equivalent. online should prevail.
        if online is True:
            t0 = max(t0, tend-f_hor)
                    
        robs = self.stl_driver.get_online_rob(f_name, t0)
        val = robs[0]
        if eval_res is None:
            eval_res= dict()
            eval_res['estimate_rob'] = np.array(robs[0])
            eval_res['lower_rob'] = np.array(robs[1])
            eval_res['upper_rob'] = np.array(robs[2])                        
            sat = 1 if robs[0]>0 else 0
            eval_res['sat'] = np.array(sat)
        else:            
            eval_res['estimate_rob'] = np.append(eval_res['estimate_rob'],robs[0])
            eval_res['lower_rob'] = np.append(eval_res['lower_rob'],robs[1])
            eval_res['upper_rob'] = np.append(eval_res['upper_rob'],robs[2])                        
            sat = 1 if robs[0]>0 else 0
            eval_res['sat'] = np.append(eval_res['sat'],sat)
        
        semantics = f_opt.get('semantics', 'rob')
        if semantics == 'lower_rob':
            val = robs[1]
        elif semantics == 'upper_rob':
            val = robs[2]
        elif semantics == 'bool':
            if robs[1]>0:
                val=1
            else:
                val=0
        
        upper_bound = f_opt.get('upper_bound',np.inf)
        lower_bound = f_opt.get('lower_bound',-np.inf)
        val = max(val, lower_bound)
        val = min(val, upper_bound)
        
        if self.debug_formulas>=2:
            print(f'{f_name} at t0={t0} (current_time:{tend}, f_hor:{f_hor}): [ {robs[1]}  <=  {val} <= {robs[2]}]')
        
        return val, eval_res
               
    def eval_specs_episode(self, episode=None, res=dict(), res_rew_f_list=[], res_eval_f_list=[]):
    # computes different metrics to evaluate an episode. If res contains values already, concatenate
    # returns top level metrics, and robustness and stuff at all steps for reward formulas and eval formulas    
        
        if episode is None:
            episode= self.episode            
        else:
            self.episode = episode
                
        rewards = episode['rewards']        

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
        
        # rewards formulas: here we "replay" the trace, reusing stl_data, but reinjecting each sample step by step to get true online evaluation
        # WHY ???
        stl_data = episode['stl_data']            
        len_episode = len(stl_data)
        if self.reward_formulas != dict():
            
            res_f = episode['res_f']
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
            
            # eval formulas at all steps 
            while self.time_step<len_episode:
                
                # eval all formulas for this step - possible optimization here is to do only first step when eval_all_steps is false
                for f_name,f_cfg in self.eval_formulas.items():
                    # if formula is new, create field in res for global evaluation
                    if f_name not in res:                        
                        res[f_name] = dict() 
                    
                    # compute and append formula eval for this step 
                    v, _ = self.eval_formula_cfg(f_name,f_cfg)    
                    res_f[f_name] = np.append(res_f[f_name],v)                                        
                    
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
    
