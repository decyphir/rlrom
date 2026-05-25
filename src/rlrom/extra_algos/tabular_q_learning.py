from collections import defaultdict
import numpy as np
import gymnasium as gym


class TabularQLearning:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        gamma=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay_steps=None,
        verbose=1,
        seed=None,
        **kwargs,
    ):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "Action space must be Discrete"
        self.env = env
        if seed is not None:
            np.random.seed(seed)
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
        self.lr = learning_rate
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.verbose = verbose
        self.n_actions = env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        self.global_step = 0

    def _update_epsilon(self):
        if self.epsilon_decay_steps is None:
            self.epsilon = self.final_epsilon
            return
        steps_fraction = min(self.global_step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.initial_epsilon + steps_fraction * (
            self.final_epsilon - self.initial_epsilon
        )

    def _get_max_q_value_action(self, observation):
        observation = tuple(observation)
        q_values = self.q_table[observation]
        max_actions = np.flatnonzero(q_values == q_values.max())
        return int(np.random.choice(max_actions))

    def predict(self, observation, deterministic=True):
        if deterministic:
            action = self._get_max_q_value_action(observation)
        else:
            # epsilon-greedy
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self._get_max_q_value_action(observation)
        return np.array([action])

    def learn(self, total_timesteps=10000, **kwargs):
        obs, _ = self.env.reset()
        obs = tuple(obs)
        episode_reward = 0
        episode = 0
        for _ in range(total_timesteps):
            self._update_epsilon()
            self.global_step += 1
            action = self.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = tuple(next_obs)
            done = terminated or truncated
            best_next_action = self._get_max_q_value_action(next_obs)
            td_target = reward + (1 - terminated) * (
                self.gamma * self.q_table[next_obs][best_next_action]
            )
            td_error = td_target - self.q_table[obs][action]
            self.q_table[obs][action] += self.lr * td_error
            obs = next_obs
            episode_reward += reward
            if done:
                if self.verbose:
                    print(
                        f"episode={episode} "
                        f"reward={episode_reward:.2f} "
                        f"epsilon={self.epsilon:.3f} "
                        f"q_table_size={len(self.q_table)}"
                    )
                obs, _ = self.env.reset()
                obs = tuple(obs)
                episode_reward = 0
                episode += 1
        return self

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path, env):
        raise NotImplementedError
