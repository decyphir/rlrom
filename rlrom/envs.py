supported_envs = ['Pendulum-v1',
                 'CartPole-v1',
                 'MountainCar-v0', 
                 'Acrobot-v1', 
                 'LunarLander-v2',
                 'BipedalWalker-v3',
                 'BipedalWalkerHardcore-v3', 
                 'CarRacing-v2', 
                 'LunarLanderContinuous-v2', 
                 'MountainCarContinuous-v0',
                 'highway-v0']

supported_models = ['ppo', 'a2c', 'sac', 'td3', 'dqn', 'qrdqn', 'ddpg', 'trpo']

cfg_envs = {}

cfg_envs['Pendulum-v1'] = {'env_name': 'Pendulum-v1',
                'signals_names': [ "torque", "cos_theta",  "sin_theta","theta_dot","reward"],
                'real_time_step': .05}

cfg_envs['CartPole-v1'] = {'env_name': 'CartPole-v1',
                'signals_names': [ "push","cart_pos", "cart_speed", "pole_angle", "pole_speed", "reward"],
                'real_time_step': .05}
                                           
cfg_envs['MountainCar-v0'] = {'env_name': 'MountainCar-v0',
                     'signals_names': [ "push","car_pos", "car_speed", "reward"],
                     'real_time_step': 1}

cfg_envs['Acrobot-v1'] = {'env_name': 'Acrobot-v1',
                'signals_names': [ "torque", "cos_theta1",  "sin_theta1", "cos_theta2",  "sin_theta2", "theta_dot1", "theta_dot2", "reward"],
                'real_time_step': .05}


cfg_envs['LunarLander-v2'] = {'env_name': 'LunarLander-v2',
                    'signals_names': [ "fire", "x", "y", "x_dot", "y_dot", "angle", "angle_dot", "reward"],
                    'real_time_step': .05}

cfg_envs['BipedalWalker-v3'] = {'env_name': 'BipedalWalker-v3',
                        'signals_names': [ "x", "u", "reward"],
                        'real_time_step': 1.}


cfg_envs['BipedalWalkerHardcore-v3'] = {'env_name': 'BipedalWalkerHardcore-v3',
                                'signals_names': [ "x", "u", "reward"],
                                'real_time_step': 1.}

cfg_envs['CarRacing-v2'] = {'env_name': 'CarRacing-v2',
                    'signals_names': [ "x", "u", "reward"],
                    'real_time_step': 1.}

cfg_envs['LunarLanderContinuous-v2'] = {'env_name': 'LunarLanderContinuous-v2',
                                'signals_names': [ "fire", "x", "y", "x_dot", "y_dot", "angle", "angle_dot", "reward"],
                                'real_time_step': 1.}


cfg_envs['MountainCarContinuous-v0'] = {'env_name': 'MountainCarContinuous-v0',
                                'signals_names': [ "push","car_pos", "car_speed", "reward"],
                                'real_time_step': 1.}

cfg_envs['highway-v0'] = {'env_name': 'highway-v0',
                'signals_names': ["x","u", "reward"],
                'real_time_step': 1.}

