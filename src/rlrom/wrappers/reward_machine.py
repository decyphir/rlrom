"""Wrapper for transforming the reward."""
import gymnasium as gym
from gymnasium import spaces
#from tensorflow.python.ops.numpy_ops.np_dtypes import float32
from collections import Counter
import minigrid.wrappers
import numpy as np
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import ResizeObservation
import minigrid
from minigrid.wrappers import FlatObsWrapper

class RewardMachine(gym.Wrapper):
    """Transform the reward via reward machine.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, 
        the :attr:`reward_range` of the wrapped environment will be incorrect.

    """

    def __init__(self, env, rm_filename):
        """Initialize the :class:`RewardMachine` wrapper with an environment and the file that contains the reward machine.

        Args:
            env: The environment to apply the wrapper
            rm_filename: The location and the filename.txt of the reward machine
        """
        super().__init__(env)
        self.env = env
        self.timestep = 0
        self.rm_list, self.num_states, self.u_0, self.ut = self._load_reward_machine(f"./{rm_filename}")
        self.key_pickups = 0
        self.last_carring = None
        self.has_key = False
        
        '''        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space'''

    def step(self, action):
        """Modify the step function
        :param action: same as the original step
        :return: observation with the reward machine state, the new reward value from the
        reward machine, terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        u_in = self.u_in
        # Check if the agent is carrying the key
        carrying = self.env.unwrapped.carrying
        self.has_key = False ; door = False ; open = False
        if carrying and carrying.type == "key":
            self.has_key = True
            # Detect key pickup by comparing last_carrying and current
            if carrying != self.last_carrying:
                self.key_pickups += 1
        self.last_carrying = carrying
        # Add key pickup count to info
        info["key_pickups"] = self.key_pickups

        #Check if the agent is in front and the door and can open the door
        front_pos = self.env.unwrapped.front_pos
        obj_in_front = self.env.unwrapped.grid.get(*front_pos)
        if obj_in_front:
            if obj_in_front.type == "door" and obj_in_front.is_open:
                open = True
        # Update the values of the reward machine
        self.u_in, rm_reward = self.get_rm_transition(u_in, self.has_key, open)
        if self.u_in == self.ut:
            terminated = True
            self.env.reset()
        self.timestep += 1
        if reward == 0:
            reward = 1
        return obs, rm_reward*reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.u_in = self.u_0
        self.key_pickups = 0
        self.last_carrying = None
        return self.env.reset(seed=seed, options=options)


    def get_rm_transition(self, u_in, has_key, open):
        # Evaluate the new state given the input state with the true specification in the reward machine
        u_out = 0 
        for transition in self.rm_list:
            if u_in == transition[0]:
                p = transition[2]
                if eval(p): # boolean variable
                    
                    u_out = transition[1]
                    reward = float(transition[3])
                    break
                else:
                    u_out = u_in
        return u_out, reward

    def _load_reward_machine(self, file):
        f_copy = open(file)
        len_file = len(f_copy.readlines())
        f = open(file)
        u_0 = int(f.readline()[0]) # Initial state. First line of txt file
        terminal_state = f.readline().split('[')[1].split(']')[0] # Terminal state(s). Second line of txt file
        terminal_state = terminal_state.split(",")
        try:
            ut = [int(i) for i in terminal_state]
        except ValueError:
            ut = None # If there is no terminal state

        rm_list = []
        for line in range(len_file - 2): # Rest of the lines in txt file --> [ui, uo, dnf, reward_env]
            rm_line = f.readline().replace("\n", "").split(",")
            rm_list.append(rm_line)
        num_states = [] # Count number of different states in the RM and change the type str of the RM states to int
        for U in rm_list:
            U[:2] = [int(u) for u in U[:2]]
            num_states += U[:2]
        num_states = len(Counter(num_states).keys())
        #returns a list of all possible transitions and the number of states of the reward machine
        return rm_list, num_states, u_0, ut