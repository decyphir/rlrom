import gymnasium as gym
import numpy as np
import rlrom.utils as utils

class RewardMachine: 
    def __init__(self, cfg_rm):
        self_current_state= 0   
        self.cfg_rm = cfg_rm
        
    def reset(self):
        return self.u0


class RewardMachineWrapper(gym.Wrapper):
    
    def __init__(self, env, cfg_rm):
        
        super().__init__(env)
        self.env = env
        self.cfg_rm = cfg_rm        
        self.states_idx_map={}
        self.num_states, self.u_0, self.u_t = self._load_reward_machine()
        
        if self.cfg_rm['in_observation']:
            old_shape = env.observation_space["unwrapped"].shape[0]
            new_shape = old_shape+1 # add RM feature

            self.observation_space = gym.spaces.Box(
                low=0,
                high=255, 
                shape=(new_shape,),
                dtype=env.observation_space["unwrapped"].dtype)
        
    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs["unwrapped"]
        u_in = self.u_in
        
        # Update the values of the reward machine        
        u_out, rm_reward = self.get_rm_transition(u_in)

        if u_out == self.u_t:
            terminated = True
        if self.cfg_rm['in_observation']:
            obs = self._augmented_obs(obs)
        
        if self.cfg_rm.get('debug_rm',False):
            rm_id = self.cfg_rm.get('rm_id','rm_doe')
            print(f'Machine {rm_id}: u_in: {u_in} u_out: {u_out} reward: {rm_reward}')

        self.u_in= u_out
        
        return obs, rm_reward*self.unwrapped._reward(), terminated, truncated, info

    def reset(self, *, seed=None, options=None):                
        self.reset_rm()
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs["unwrapped"]        
        if self.cfg_rm['in_observation']:
            obs = self._augmented_obs(obs)
        return obs, info

    def reset_rm(self):
        self.u_in = self.u_0
        

    def get_rm_transition(self, u_in):
        get_rob= self.env.get_wrapper_attr('get_rob')
        transitions = self.cfg_rm['transitions']
        
        states = self.cfg_rm['states']
        for s in states:
            if u_in == s['id']:
                reward = s['reward']
                break
        priority = 0
        for t in transitions:
            formula_name = t["condition"]
            if u_in == t["from"] and get_rob(formula_name)[-1] > 0 :
                u_out = t['to']
                reward = t["reward"]
        return u_out, reward

    def _augmented_obs(self, obs):
        rm_state = [self.states_idx_map[self.u_in]]  
        return np.concatenate([obs, rm_state])

    def _load_reward_machine(self):
        num_states = len(self.cfg_rm['states'])
        u_0 = next((s['id'] for s in self.cfg_rm['states'] if s.get('initial')), None)
        u_t = next((s['id'] for s in self.cfg_rm['states'] if s.get('final')), None)
        
        i_state=0
        for s in self.cfg_rm['states']:
            self.states_idx_map[s['id']] = i_state
            self.states_idx_map[i_state] = s['id']
            i_state += 1
        #returns a list of all possible transitions and the number of states of the reward machine
        return num_states, u_0, u_t
