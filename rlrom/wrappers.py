import gymnasium as gym
from gymnasium import spaces
import numpy as np
import stlrom
import matplotlib.pyplot as plt


class STLWrapper(gym.Wrapper):
    
    def __init__(self,env,stl_driver,formulas=[], horizon=[],signals_map={}, terminal_formulas=[]):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.timestep = 0
        
        self.stl_driver = stl_driver                        
        self.formulas = formulas

        self.horizon=horizon
        if horizon==[]:
            self.horizon=[0]*len(self.formulas)
        
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
            self.signals_map['reward'] = 'reward'
        else: # assumes all is fine (TODO? catch bad signals_map here)    
            self.signals_map=signals_map
            
        self.terminal_formulas=terminal_formulas

        low = self.observation_space.__getattribute__("low") 
        high = self.observation_space.__getattribute__("high")
        
        BigM = stlrom.Signal.get_BigM()
        self.observation_space = spaces.Box(np.append(low,  [-BigM]*len(self.formulas)), 
                                            np.append(high, [BigM]*len(self.formulas)),        
                                            dtype=np.float32)
        
    def step(self, action):
    
        # steps the wrapped env
        obs, reward, terminated, truncated, info = self.env.step(action)                
        
        # collect the sample for monitoring 
        s = self.get_sample(self.prev_obs, action, reward) 
        
        # add sample and compute robustness
        self.stl_driver.add_sample(s)        
        
        idx_formula = 0
        rob = [0]*len(self.formulas)
        for f in self.formulas:
           t0_f = max(0, self.timestep-self.horizon[idx_formula])
           robs_f = self.stl_driver.get_online_rob(f, t0_f)
           rob[idx_formula] = robs_f[0] # forget about low and high rob for now     
        
        for f in self.terminal_formulas:
            #print('checking terminal formula:', f)
            t0_f = 0
            robs_f = self.stl_driver.get_online_rob(f, t0_f)
            if robs_f[1]>0:
               terminated = True
               print('Terminal Formula ', f, ' is true, episode is done.') 

        # update current time
        self.timestep += 1
        self.prev_obs = obs
        
        # return obs with added robustness
        new_obs = np.append(obs, rob)
        new_reward = reward                 
        
        if terminated: self.env.reset()
        return new_obs, new_reward, terminated, truncated, info

    
    def get_sample(self,obs,action,reward):        
        s = np.zeros(len(self.signals_map)+1)
        s[0] = self.timestep
        i_sig = 0
        for key, value in self.signals_map.items():
            i_sig = i_sig+1
            s[i_sig] = eval(value) 
        return s

    def reset(self, **kwargs):
        self.timestep = 0
        obs0, info = self.env.reset(**kwargs)
        self.prev_obs = obs0
        
        robs0 = self.reset_monitor(obs0)
        obs = np.append(obs0, robs0)
        #obs = np.concatenate((obs0, robs0))
        return obs, info

    def reset_monitor(self, obs0):        
        self.stl_driver.data = [] 
        return [0]*len(self.formulas) 
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.reset(seed=seed)
    

    def plot_signal(self, signal, fig=None,label=None,  color=None, online=False, horizon=0, linestyle='-'):
    # signal should be part of the "signal" declaration or a valid formula id 
     
        if self.stl_driver.data == []:
            raise ValueError("No data to plot.")
                 
        time = self.get_time()

        if signal in self.signals_map:
            signal_index = list(self.signals_map.keys()).index(signal)+1        
            sig_values = [s[signal_index] for s in self.stl_driver.data]
            if label is None:
                label=signal
        elif signal in self.formulas:
            sig_values = self.get_rob(signal, online=online,horizon=horizon)
            signal_index = self.formulas.index(signal)+len(self.signals_map)        
            if label is None:
                label=signal
        elif isinstance(signal, np.ndarray) and signal.shape == (len(self.get_time()),):
            sig_values = signal
        elif isinstance(signal, stlrom.Signal):
            pass
        else:
            try:
                sig_values = self.get_rob(signal, online=online,horizon=horizon)
            except Exception as e:
               raise ValueError(f"Name '{signal}' not in signals_map nor in parsed formulas")

        if fig is None:
            fig = plt.figure(figsize=(15, 5)).gca()

        fig.set_xlabel('Time')
        fig.grid(True)

        fig.plot(time, sig_values)
        if color is None:
            l, = fig.plot(time, sig_values, label=label,linestyle=linestyle)
            color = l.get_color()
        else:
            l = fig.plot(time, sig_values, color=color,linestyle=linestyle)
        
        if label is not None:
            l.set_label(label)
        fig.legend()

        return fig

    def get_time(self):
        if self.stl_driver.data == []:
            raise ValueError("No data to plot.")
        
        return [s[0] for s in self.stl_driver.data]
        

    def get_rob(self, formula, horizon=None, online=True):
    # Compute robustness signal. If online is true, then 
    # compute it at each time as if future was not known, 
    # otherwise uses all data for all computation

        if self.stl_driver.data == []:
            raise ValueError("No data/episode was computed.")

        #if not (formula in self.formulas):
        #    raise ValueError("Name '{formula}' not in formulas")

        if (horizon is None):
            if formula in self.formulas:
                index_formula = self.formulas.index(formula)
                horizon = self.horizon[index_formula]
            else:
               horizon = 0
               #print(f"Warning: Horizon for formula '{formula}' is set to 0.")

        
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





