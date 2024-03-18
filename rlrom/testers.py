import gymnasium as gym
from gymnasium.utils.save_video import save_video
from gymnasium.spaces.utils import flatten_space
import numpy as np

class RLModelTester:
    
    def __init__(self, env_name=None, model=None, render_mode=None):
        self.env = None
        self.env_name = env_name
        self.model = model
        self.render_mode = render_mode
        self.set_signals_names_from_env()
        self.real_time_step=1 # if available, duration of a simulation time step

    def create_env(self):
            
        if self.env_name is None:
            return "No environment found"
        else:
            if self.render_mode == 'video':
                self.env = gym.make(self.env_name, render_mode="rgb_array")
                self.record_video = True
            else:
                self.env = gym.make(self.env_name, render_mode=self.render_mode)
        self.set_signals_names_from_env()
        
    def test_random(self, seed=1, num_steps=100): 
        if self.env is None:
            self.create_env()

        self.trace= []
        obs, info = self.env.reset(seed=seed)
        time_step = 0
        total_reward = 0
        try:
            for _ in range(num_steps):
                
                if self.model is not None:
                    action, _ = self.model.predict(obs)
                else:
                    action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.trace.append([time_step, obs, action, next_obs, reward, terminated, truncated, info])
                if terminated or truncated:
                    break  
                time_step += self.real_time_step
                obs = next_obs
                total_reward += reward

        except Exception as e:
            print("Env crashed with: ", e)   
        self.env.close()
        return total_reward


## Helper functions
    def get_video_filename(self):
        
        filename = self.env_name+'-'
        if self.model is not None:
            filename += self.model.__class__.__name__
        else:
            filename += 'random'
        
        return filename + '.mp4'

## Signal stuff

    def get_signal_state(self,trace_state):
        time = np.array([trace_state[0]])
        action = trace_state[2].flatten()
        obs = trace_state[1].flatten()
        reward = np.array([trace_state[4]])
        
        ss = np.concatenate((time, action, obs, reward))
        return ss

    def get_signal(self, signal_name):
        if signal_name == 'reward':
            return [self.get_signal_state(trace_state)[-1] for trace_state in self.trace]    
    
        signal_index = self.signals_names.index(signal_name)
        return [self.get_signal_state(trace_state)[signal_index+1] for trace_state in self.trace]    
    
    def get_df_signals(self, df_signals=None):
        import pandas as pd
        if df_signals is None:
            df_signals = pd.DataFrame()
        
        df_signals.drop(df_signals.index, inplace=True)
        df_signals['time'] = self.get_time()
        for signal in self.signals_names:
            df_signals[signal]= self.get_signal(signal)

        return df_signals

    def get_time(self):
        return [self.get_signal_state(trace_state)[0] for trace_state in self.trace]   
    
    def set_signals_names_from_env(self):

        if self.env_name == 'Pendulum-v1':
            self.signals_names = [ "torque", "cos_theta",  "sin_theta","theta_dot","reward"]
            self.real_time_step = .05
        elif self.env_name == 'CartPole-v1':
            self.signals_names = [ "push","cart_pos", "cart_speed", "pole_angle", "pole_speed", "reward"]
            self.real_time_step = .05
        elif self.env_name == 'MountainCar-v0':
            self.signals_names = [ "push","car_pos", "car_speed", "reward"]            
        elif self.env_name == 'Acrobot-v1':
            self.signals_names = [ "torque", "cos_theta1",  "sin_theta1", "cos_theta2",  "sin_theta2", "theta_dot1", "theta_dot2", "reward"]
            self.real_time_step = .05
        elif self.env_name == 'LunarLander-v2':
            self.signals_names = [ "fire", "x", "y", "x_dot", "y_dot", "angle", "angle_dot", "reward"]
        elif self.env_name == 'BipedalWalker-v3':
            self.signals_names = [ "x", "u", "reward"]
        elif self.env_name == 'BipedalWalkerHardcore-v3':
            self.signals_names = [ "x", "u", "reward"]
        elif self.env_name == 'CarRacing-v2':
            self.signals_names = [ "x", "u", "reward"]
        elif self.env_name == 'LunarLanderContinuous-v2':
            self.signals_names = [ "fire", "x", "y", "x_dot", "y_dot", "angle", "angle_dot", "reward"]
        elif self.env_name == 'MountainCarContinuous-v0':
            self.signals_names = [ "push","car_pos", "car_speed", "reward"]

    def get_signal_string(self):
        return 'signal '+ ', '.join(self.signals_names)

## Info 
    
    def get_env_info(self):
        info = "Environment: " + self.env_name + "\n" + "----------------------\n"
        info += "Observation space: " + str(self.env.observation_space) + "\n\n"
        info += "Action space: " + str(self.env.action_space) + "\n\n"
        info += "Signals: " + self.get_signal_string() + "\n\n"
        return info
        