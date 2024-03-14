"""
This is the rlrom module.

It contains methods for robust online monitoring of reinforcement learning models.
"""

supported_envs = ['Pendulum-v1',
                 'CartPole-v1',
                 'MountainCar-v0', 
                 'Acrobot-v1', 
                 'LunarLander-v2',
                 'BipedalWalker-v3',
                 'BipedalWalkerHardcore-v3', 
                 'CarRacing-v2', 
                 'LunarLanderContinuous-v2', 
                 'MountainCarContinuous-v0']

__all__ = ['RLModelTester', 'supported_envs']

from .testers import RLModelTester