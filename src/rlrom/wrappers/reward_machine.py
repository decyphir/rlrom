"""Wrapper for transforming the reward."""
import gymnasium as gym
from gymnasium import spaces
#from tensorflow.python.ops.numpy_ops.np_dtypes import float32
from collections import Counter
import numpy as np
import math

class RewardMachine(gym.Wrapper):
    """Transform the reward via reward machine.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> env = gym.make("CartPole-v1")
        >>> env = RewardMachine(env)
        >>> _ = env.reset()
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    def __init__(self, env, rm_filename):
        """Initialize the :class:`RewardMachine` wrapper with an environment and the file that contains the reward machine.

        Args:
            env: The environment to apply the wrapper
            rm_filename: The location and the filename of the reward machine
        """
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.timestep = 0
        self.rm_list, self.num_states , self.ut = self._load_reward_machine(f"./{rm_filename}")
        self.u_in = 0
        self.task = 0

    def step(self, action):
        """Modify the step function
        :param action: same as the original step
        :return: observation with the reward machine state and robustness, and the new reward value from the
        reward machine, terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.timestep += 1
        reward_env = reward
        u_in = self.u_in
        self.u_in, rm_reward = self.get_rm_transition(u_in, action, self.timestep, obs[0],reward_env)
        
        if terminated: self.env.reset()
        return obs, rm_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.timestep = 0
        self.u_in = 0
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.reset(seed=seed)

    def get_rm_transition(self, u_in, action, ts, x, reward_env):
        # Evaluate the new state given the input state with the true specification in the reward machine
        u_out = 0 ; reward = reward_env
        for transition in self.rm_list:
            if transition[3] == "reward_env":
                transition[3] = reward_env
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
        self.u0 = int(f.readline()[0]) # Initial state. First line of txt file
        terminal_state = f.readline().split('[')[1].split(']')[0] # Terminal state(s). Second line of txt file
        terminal_state = terminal_state.split(",")
        try:
            self.ut = [int(i) for i in terminal_state]
        except ValueError:
            self.ut = None # If there is no terminal state

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
        return rm_list, num_states, self.ut


MAX_NUM_STATE = 100 # Max number of state for reward machine - we might want do reconsider if we go into composing and stuff
class RewardMachineObservation(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.u0 = 0
        self.ut = None
        
        low = self.observation_space.__getattribute__("low") 
        high = self.observation_space.__getattribute__("high")
        
        self.observation_space = spaces.Box(np.append(low, self.u0), 
                                            np.append(high, MAX_NUM_STATE),        
                                            dtype=np.float32)
        
        #self.observation_space = spaces.Box(np.append(low, np.array([self.u0, -1000])), 
        #                                    np.append(high,np.array([MAX_NUM_STATE, 1000])),        
        #                                    dtype=np.float32)
        

    def observation(self, observation):
        # observation with rm state 
        observation = np.append(observation , self.get_wrapper_attr('u_in'))

        # Observation only rm state and dummy 0. after
        # observation = np.append(observation , np.array([self.get_wrapper_attr('u_in'),0.]))
        
        return observation
    

