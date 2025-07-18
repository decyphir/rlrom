import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stlrom
import matplotlib.pyplot as plt
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.palettes import Dark2_5 as palette

class STLWrapper(gym.Wrapper):
    
    def __init__(self,env,
                 stl_driver, 
                 signals_map={},
                 obs_formulas=[],  
                 reward_formulas=[],
                 end_formulas=[],
                 BigM=None
                 ):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.real_time_step = 1
        self.time_step = 0        
        self.stl_driver = stl_driver                        
        self.obs_formulas = obs_formulas
        self.reward_formulas = reward_formulas
        self.end_formulas= end_formulas
        self.episode={'observations':[], 'actions':[],'rewards':[], 'dones':[]}

        self.signals_map={}
        if signals_map=={}:
            # assumes 1 action and n obs
            signals = stl_driver.get_signals_names().split()
            i_sig=0

            for sig in signals:
                if i_sig==0:
                    self.signals_map[sig] = 'action[0]'
                elif i_sig<len(signals)-1:    
                    self.signals_map[sig] = f'obs[{i_sig-1}]'
                else:
                    self.signals_map[sig] = 'reward'
                i_sig+=1

        elif type(signals_map)==dict: 
            if 'action_names' in signals_map:
                for a in signals_map['action_names']:
                    a_name,a_ref = next(iter(a.items()))
                    self.signals_map[a_name]=a_ref
            
            if 'obs_names' in signals_map:
                for o in signals_map['obs_names']:
                    o_name,o_ref = next(iter(o.items()))
                    self.signals_map[o_name]=o_ref
            if 'aux_sig_names' in signals_map:
                for o in signals_map['aux_sig_names']:
                    o_name,o_ref = next(iter(o.items()))
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
        for f in obs_formulas:
            f_name, f_opt = next(iter(f.items()))
            f_hor = f_opt.get('hor',0)
            obs_name = 'obs_'+f_name+'_hor_'+str(f_hor)
            obs_name = f_opt.get('obs_name', obs_name)
            ref_in_obs = 'obs['+str(idx_obs_f)+']'
            self.signals_map[obs_name]= ref_in_obs
            idx_obs_f +=1
        
    def eval_rob(self,f_name, f_opt):                
        f_hor = f_opt.get('hor',0)            
        t0_f = max(0, self.time_step-f_hor)
        robs_f = self.stl_driver.get_online_rob(f_name, t0_f)
        return robs_f
        
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
        for obs_f in self.obs_formulas:
            f_name, f_opt = next(iter(obs_f.items()))
            robs_f = self.eval_rob(f_name,f_opt)        
            rob[idx_formula] = robs_f[0] # forget about low and high rob for now
            idx_formula+=1     
         
        for end_f in self.end_formulas:
            f_name, f_opt = next(iter(end_f.items()))
            robs_f = self.eval_rob(f_name,f_opt)          
            #if self.time_step >20 and self.time_step % 5 == 0:
            #    print('t,rob:',self.time_step,robs_f[1])
            if robs_f[1] > 0:
                print('Episode terminated because of formula', f_name)
                terminated = True

        new_reward = reward                         
        # add stl robustness to reward
        for rew_f in self.reward_formulas:
            f_name, f_opt = next(iter(rew_f.items()))
            robs_f = self.eval_rob(f_name,f_opt)
            w = f_opt.get('weight',1)        
            new_reward += w*robs_f[0]   
        
        # update current time
        self.time_step += 1
        
        # return obs with added robustness
        new_obs = np.append(obs, rob)
                        
        self.prev_obs = new_obs

        self.episode['observations'].append(new_obs)               
        self.episode['actions'].append(action)
        self.episode['rewards'].append(new_reward)
        self.episode['dones'].append(terminated)
        if terminated: self.env.reset()
        return new_obs, new_reward, terminated, truncated, info

    def get_sample(self,obs,action,reward):        
        # get a sample for stl driver, converts obs, action,reward into (t, signals_values)
        s = np.zeros(len(self.signals_map)+1-len(self.obs_formulas))
        s[0] = self.time_step*self.real_time_step
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
        obs0, info = self.env.reset(**kwargs)
        self.wrapped_obs = obs0        
        robs0 = self.reset_monitor(obs0)
        obs = np.append(obs0, robs0)
        self.prev_obs = obs
        self.episode={'observations':[], 'actions':[],'rewards':[], 'dones':[]}        
        return obs, info

    def reset_monitor(self, obs0):        
        self.stl_driver.data = [] 
        return [0]*len(self.obs_formulas) 
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.reset(seed=seed)
    
    def plot_signal(self, signal, fig=None,label=None,  color=None, online=False, horizon=0, linestyle='-', booleanize=False):
    # signal should be part of the "signal" declaration or a valid formula id 
     
        if self.stl_driver.data == []:
            raise ValueError("No data to plot.")
                 
        time = self.get_time()
        sig_values = self.get_sig(signal)
        if sig_values is None:
            if signal in self.formulas:
                sig_values = self.get_rob(signal, online=online,horizon=horizon)
                signal_index = self.formulas.index(signal)+len(self.signals_map)        
            elif isinstance(signal, np.ndarray) and signal.shape == (len(self.get_time()),):
                sig_values = signal
            elif isinstance(signal, stlrom.Signal):
                pass
            else:
                try:
                    sig_values = self.get_rob(signal, online=online,horizon=horizon)
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
            step = 0
            sig=[]            
            while step<len(observations):
                obs = observations[step]
                action = actions[step]
                reward = rewards[step]                
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
            hor = 0            
            phi = arg_rho[0]
            if len(arg_rho)>1:                          
                hor = -float(arg_rho[1])            
            sig_val = self.get_rob(phi, horizon=hor, online=False)                                                                                                
            sig_type = 'rob'
        elif str.startswith('sat('):
            arg_rho = str.split('(')[1][:-1]
            arg_rho = arg_rho.split(',')
            hor = 0
            phi = arg_rho[0]
            if len(arg_rho)>1:                
                hor = -float(arg_rho[1])
            sig_val = self.get_rob(phi, horizon=hor, online=False)
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

        if self.stl_driver.data == []:
            raise ValueError("No data/episode was computed.")
        
        monitor = self.stl_driver.get_monitor(formula)
        
        rob = np.zeros(len(self.stl_driver.data))
        if online:
            monitor.data=[]
        step = 0
        for s in self.stl_driver.data:
            t0 = max(0,step-horizon)
            if online:
                monitor.add_sample(s)
            rob[step] = monitor.eval_rob(t0)
            step= step+1
        return rob




