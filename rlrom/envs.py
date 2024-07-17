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

pendul_specs = """signal torque, cos_theta, sin_theta, theta_dot, reward 
sin_theta_small := abs(sin_theta[t]) < .2
cos_theta_high  := cos_theta[t] > .8 
up_right := sin_theta_small and cos_theta_high

goal_stable := alw_[0,1] up_right
phi_goal := ev_[0,4] goal_stable
"""

pendul_plot = """torque
goal_stable
reward
"""

cfg_envs['Pendulum-v1'] = {'env_name': 'Pendulum-v1',
                'action_names': ['torque'],
                'obs_names': ["cos_theta",  "sin_theta","theta_dot","reward"],                
                'specs': pendul_specs,
                'plots': pendul_plot,
                'real_time_step': .05}

cfg_envs['CartPole-v1'] = {'env_name': 'CartPole-v1',
                           'action_names': ['push'],
                'obs_names': [ "cart_pos", "cart_speed", "pole_angle", "pole_speed"],
                'real_time_step': .05}
                                           
cfg_envs['MountainCar-v0'] = {'env_name': 'MountainCar-v0',
                     'action_names': ['push'],
                     'obs_names': [ "car_pos", "car_speed"],
                     'real_time_step': 1}

cfg_envs['Acrobot-v1'] = {'env_name': 'Acrobot-v1',
                'action_names': ['torque'],
                'obs_names': ["cos_theta1",  "sin_theta1", "cos_theta2",  "sin_theta2", "theta_dot1", "theta_dot2"],
                'real_time_step': .05}

cfg_envs['LunarLander-v2'] = {'env_name': 'LunarLander-v2',
                    'action_names': ['fire'],
                    'obs_names': ["x", "y", "x_dot", "y_dot", "angle", "angle_dot"],
                    'real_time_step': .05}

cfg_envs['BipedalWalker-v3'] = {'env_name': 'BipedalWalker-v3',
                        'action_names': ['hip_torque', 'knee_torque'],
                        'obs_names': [ "x"],
                        'real_time_step': 1.}

cfg_envs['BipedalWalkerHardcore-v3'] = {'env_name': 'BipedalWalkerHardcore-v3',
                                'action_names': ['hip_torque', 'knee_torque'],
                                'obs_names': [ "x", "u"],
                                'real_time_step': 1.}

cfg_envs['CarRacing-v2'] = {'env_name': 'CarRacing-v2',
                    'action_names': ['steer', 'gas', 'brake'],
                    'obs_names': [ "x"],
                    'real_time_step': 1.}

cfg_envs['LunarLanderContinuous-v2'] = {'env_name': 'LunarLanderContinuous-v2',
                                'action_names': ['fire'],
                                'obs_names': ["x", "y", "x_dot", "y_dot", "angle", "angle_dot"],
                                'real_time_step': 1.}

cfg_envs['MountainCarContinuous-v0'] = {'env_name': 'MountainCarContinuous-v0',
                                'action_names': ['push'],
                                'obs_names': ["car_pos", "car_speed"],
                                'real_time_step': 1.}

# highway-env

highway_specs = """signal action, ego_presence, ego_x, ego_y, ego_vx, ego_vy, car1_presence, car1_x, car1_y, car1_vx, car1_vy, car2_presence, car2_x, car2_y, car2_vx, car2_vy, reward
ego_slow := ego_vx[t] < 21
ego_fast:= ego_vx[t] > 29
"""
highway_plots = """_tr(0)
ego_vx
sat(ego_slow)
sat(ego_fast)"""

cfg_envs['highway-v0'] = {'env_name': 'highway-v0',
                'action_names': ['action'],
                'obs_names': ["ego_presence","ego_x", "ego_y", "ego_vx", "ego_vy",
                                  "car1_presence","car1_x", "car1_y", "car1_vx", "car1_vy",
                                  "car2_presence","car2_x", "car2_y", "car2_vx", "car2_vy"
                                  ],
                'specs': highway_specs,
                'plots': highway_plots,
                'real_time_step': 1.}

