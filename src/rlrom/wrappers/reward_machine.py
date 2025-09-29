import gymnasium as gym
import numpy as np
import rlrom.utils as utils

class RewardMachine(gym.Wrapper):
    """Transform the reward via reward machine.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, 
        the :attr:`reward_range` of the wrapped environment will be incorrect.

    """

    def __init__(self, env):
        """Initialize the :class:`RewardMachine` wrapper with an environment and the file that contains the reward machine.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env = env
        self.rm, self.num_states, self.u_0, self.u_t = self._load_reward_machine()
        if self.rm['in_observation']:
            old_shape = env.observation_space.shape[0]
            new_shape = old_shape + 1  # add RM feature

            self.observation_space = gym.spaces.Box(
                low=0,
                high=255, 
                shape=(new_shape,),
                dtype=env.observation_space.dtype)
        

    def step(self, action):
        """Modify the step function
        :param action: same as the original step
        :return: observation with the reward machine state, the new reward value from the
        reward machine, terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        u_in = self.u_in
        # Update the values of the reward machine
        self.u_in, rm_reward = self.get_rm_transition(u_in)
        if self.u_in == self.u_t:
            terminated = True
            self.env.reset()
        if reward == 0:
            reward = 1
        if self.rm['in_observation']:
            obs = self._augmented_obs(obs)
        return obs, rm_reward*reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.u_in = self.u_0
        obs, info = self.env.reset(seed=seed, options=options)
        if self.rm['in_observation']:
            obs = self._augmented_obs(obs)
        return obs, info


    def get_rm_transition(self, u_in):
        #Check if the agent is in front and the door and can open the door

        print("open", self.env.unwrapped.door.is_open)
        conditions = {"open_door": self.env.unwrapped.door.is_open,
                      "has_key": self.env.unwrapped.key,
                      "not has_key": not self.env.unwrapped.key,
                      }
        
        transitions = self.rm['transitions']
        for t in transitions:
            if u_in == t["from"] and conditions[t["condition"]]:
                u_out = t["to"]
                reward = t["reward"]
                for state in self.rm['states']:
                    if state['id'] == u_out and 'final' in state and state['final'] == True:
                        return u_out, reward
        print("Current state", u_out, "reward",  reward)
        return u_out, reward

    def _augmented_obs(self, obs):
        rm_state = [int(self.u_in[1:])]  
        print("rm_state", rm_state)
        return np.concatenate([obs, rm_state])

    def _load_reward_machine(self):
        cfg = utils.load_cfg("cfg_rm.yml")
        rm = cfg.get("reward_machine")
        rm = cfg["reward_machine"]
        num_states = len(rm['states'])

        u_0 = next((s['id'] for s in rm['states'] if s.get('initial')), None)
        u_t = next((s['id'] for s in rm['states'] if s.get('final')), None)

        #returns a list of all possible transitions and the number of states of the reward machine
        return rm, num_states, u_0, u_t
